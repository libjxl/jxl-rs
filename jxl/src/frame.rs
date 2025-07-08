// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    bit_reader::BitReader,
    entropy_coding::decode::Histograms,
    error::Result,
    features::{
        epf::create_sigma_image, noise::Noise, patches::PatchesDictionary, spline::Splines,
    },
    headers::{
        FileHeader, Orientation,
        color_encoding::ColorSpace,
        encodings::UnconditionalCoder,
        extra_channels::{ExtraChannel, ExtraChannelInfo},
        frame_header::{Encoding, FrameHeader, Toc, TocNonserialized},
        permutation::Permutation,
    },
    image::{Image, Rect},
    render::{
        RenderPipeline, RenderPipelineBuilder, SimpleRenderPipeline, SimpleRenderPipelineBuilder,
        stages::*,
    },
    util::{CeilLog2, Xorshift128Plus, tracing_wrappers::*},
};
use adaptive_lf_smoothing::adaptive_lf_smoothing;
use block_context_map::BlockContextMap;
use coeff_order::decode_coeff_orders;
use color_correlation_map::ColorCorrelationParams;
use group::decode_vardct_group;
use modular::{FullModularImage, ModularStreamId, Tree};
use modular::{decode_hf_metadata, decode_vardct_lf};
use quant_weights::DequantMatrices;
use quantizer::LfQuantFactors;
use quantizer::QuantizerParams;
use transform_map::*;

use std::sync::Arc;

mod adaptive_lf_smoothing;
mod block_context_map;
mod coeff_order;
pub mod color_correlation_map;
mod group;
pub mod modular;
mod quant_weights;
pub mod quantizer;
pub mod transform_map;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Section {
    LfGlobal,
    Lf { group: usize },
    HfGlobal,
    Hf { group: usize, pass: usize },
}

#[allow(dead_code)]
pub struct LfGlobalState {
    patches: Option<PatchesDictionary>,
    splines: Option<Splines>,
    noise: Option<Noise>,
    lf_quant: LfQuantFactors,
    pub quant_params: Option<QuantizerParams>,
    block_context_map: Option<BlockContextMap>,
    color_correlation_params: Option<ColorCorrelationParams>,
    tree: Option<Tree>,
    modular_global: FullModularImage,
}

#[allow(dead_code)]
pub struct PassState {
    coeff_orders: Vec<Permutation>,
    histograms: Histograms,
}

pub struct HfGlobalState {
    num_histograms: u32,
    passes: Vec<PassState>,
    #[allow(dead_code)]
    dequant_matrices: DequantMatrices,
    hf_coefficients: Option<(Image<i32>, Image<i32>, Image<i32>)>,
}

#[derive(Clone, Debug)]
pub struct ReferenceFrame {
    pub frame: Vec<Image<f32>>,
    pub saved_before_color_transform: bool,
}

impl ReferenceFrame {
    #[cfg(test)]
    pub fn blank(
        width: usize,
        height: usize,
        num_channels: usize,
        saved_before_color_transform: bool,
    ) -> Result<Self> {
        let frame = (0..num_channels)
            .map(|_| Image::new_constant((width, height), 0.0))
            .collect::<Result<_>>()?;
        Ok(Self {
            frame,
            saved_before_color_transform,
        })
    }
}

#[derive(Debug)]
pub struct DecoderState {
    pub(super) file_header: FileHeader,
    pub(super) reference_frames: [Option<ReferenceFrame>; Self::MAX_STORED_FRAMES],
    pub(super) lf_frames: [Option<[Image<f32>; 3]>; 4],
    pub xyb_output_linear: bool,
    pub enable_output: bool,
    pub render_spotcolors: bool,
}

impl DecoderState {
    pub const MAX_STORED_FRAMES: usize = 4;

    pub fn new(file_header: FileHeader) -> Self {
        Self {
            file_header,
            reference_frames: [None, None, None, None],
            lf_frames: [None, None, None, None],
            xyb_output_linear: true,
            enable_output: true,
            render_spotcolors: true,
        }
    }

    pub fn extra_channel_info(&self) -> &Vec<ExtraChannelInfo> {
        &self.file_header.image_metadata.extra_channel_info
    }

    pub fn reference_frame(&self, i: usize) -> Option<&ReferenceFrame> {
        assert!(i < Self::MAX_STORED_FRAMES);
        self.reference_frames[i].as_ref()
    }
}

pub struct HfMetadata {
    ytox_map: Image<i8>,
    ytob_map: Image<i8>,
    pub raw_quant_map: Image<i32>,
    pub transform_map: Image<u8>,
    pub epf_map: Image<u8>,
    used_hf_types: u32,
}

pub struct Frame {
    header: FrameHeader,
    toc: Toc,
    modular_color_channels: usize,
    lf_global: Option<LfGlobalState>,
    hf_global: Option<HfGlobalState>,
    lf_image: Option<[Image<f32>; 3]>,
    quant_lf: Image<u8>,
    hf_meta: Option<HfMetadata>,
    decoder_state: DecoderState,
    render_pipeline: Option<SimpleRenderPipeline>,
}

pub struct FrameOutput {
    pub decoder_state: Option<DecoderState>,
    pub channels: Option<Vec<Image<f32>>>,
}

impl Frame {
    pub fn new(br: &mut BitReader, decoder_state: DecoderState) -> Result<Self> {
        let mut frame_header = FrameHeader::read_unconditional(
            &(),
            br,
            &decoder_state.file_header.frame_header_nonserialized(),
        )?;
        frame_header.postprocess(&decoder_state.file_header.frame_header_nonserialized());
        let num_toc_entries = frame_header.num_toc_entries();
        let toc = Toc::read_unconditional(
            &(),
            br,
            &TocNonserialized {
                num_entries: num_toc_entries as u32,
            },
        )
        .unwrap();
        br.jump_to_byte_boundary()?;
        Self::from_header_and_toc(frame_header, toc, decoder_state)
    }

    pub fn from_header_and_toc(
        frame_header: FrameHeader,
        toc: Toc,
        decoder_state: DecoderState,
    ) -> Result<Self> {
        let modular_color_channels = if frame_header.encoding == Encoding::VarDCT {
            0
        } else if decoder_state
            .file_header
            .image_metadata
            .color_encoding
            .color_space
            == ColorSpace::Gray
        {
            1
        } else {
            3
        };
        let size_blocks = frame_header.size_blocks();
        let lf_image = if frame_header.encoding == Encoding::VarDCT {
            if frame_header.has_lf_frame() {
                decoder_state.lf_frames[frame_header.lf_level as usize].clone()
            } else {
                Some([
                    Image::new(size_blocks)?,
                    Image::new(size_blocks)?,
                    Image::new(size_blocks)?,
                ])
            }
        } else {
            None
        };
        let quant_lf = Image::new(size_blocks)?;
        let size_color_tiles = (size_blocks.0.div_ceil(8), size_blocks.1.div_ceil(8));
        let hf_meta = if frame_header.encoding == Encoding::VarDCT {
            Some(HfMetadata {
                ytox_map: Image::new(size_color_tiles)?,
                ytob_map: Image::new(size_color_tiles)?,
                raw_quant_map: Image::new(size_blocks)?,
                transform_map: Image::new_with_default(
                    size_blocks,
                    HfTransformType::INVALID_TRANSFORM,
                )?,
                epf_map: Image::new(size_blocks)?,
                used_hf_types: 0,
            })
        } else {
            None
        };
        Ok(Self {
            header: frame_header,
            modular_color_channels,
            toc,
            lf_global: None,
            hf_global: None,
            lf_image,
            quant_lf,
            hf_meta,
            decoder_state,
            render_pipeline: None,
        })
    }

    pub fn toc(&self) -> &Toc {
        &self.toc
    }

    pub fn header(&self) -> &FrameHeader {
        &self.header
    }

    pub fn total_bytes_in_toc(&self) -> usize {
        self.toc.entries.iter().map(|x| *x as usize).sum()
    }

    /// Given a bit reader pointing at the end of the TOC, returns a vector of `BitReader`s, each
    /// of which reads a specific section.
    pub fn sections<'a>(&self, br: &'a mut BitReader) -> Result<Vec<BitReader<'a>>> {
        debug!(toc = ?self.toc);
        let ret = self
            .toc
            .entries
            .iter()
            .scan(br, |br, count| Some(br.split_at(*count as usize)))
            .collect::<Result<Vec<_>>>()?;
        if !self.toc.permuted {
            return Ok(ret);
        }
        let mut inv_perm = vec![0; ret.len()];
        for (i, pos) in self.toc.permutation.iter().enumerate() {
            inv_perm[*pos as usize] = i;
        }
        let mut shuffled_ret = ret.clone();
        for (br, pos) in ret.into_iter().zip(inv_perm.into_iter()) {
            shuffled_ret[pos] = br;
        }
        Ok(shuffled_ret)
    }

    #[instrument(level = "debug", skip(self), ret)]
    pub fn get_section_idx(&self, section: Section) -> usize {
        if self.header.num_toc_entries() == 1 {
            0
        } else {
            match section {
                Section::LfGlobal => 0,
                Section::Lf { group } => 1 + group,
                Section::HfGlobal => self.header.num_lf_groups() + 1,
                Section::Hf { group, pass } => {
                    2 + self.header.num_lf_groups() + self.header.num_groups() * pass + group
                }
            }
        }
    }

    #[instrument(level = "debug", skip_all)]
    pub fn decode_lf_global(&mut self, br: &mut BitReader) -> Result<()> {
        debug!(section_size = br.total_bits_available());
        assert!(self.lf_global.is_none());
        trace!(pos = br.total_bits_read());

        let patches = if self.header.has_patches() {
            info!("decoding patches");
            Some(PatchesDictionary::read(
                br,
                self.header.width as usize,
                self.header.height as usize,
                self.decoder_state.extra_channel_info().len(),
                &self.decoder_state.reference_frames,
            )?)
        } else {
            None
        };

        let splines = if self.header.has_splines() {
            info!("decoding splines");
            Some(Splines::read(br, self.header.width * self.header.height)?)
        } else {
            None
        };

        let noise = if self.header.has_noise() {
            info!("decoding noise");
            Some(Noise::read(br)?)
        } else {
            None
        };

        let lf_quant = LfQuantFactors::new(br)?;
        debug!(?lf_quant);

        let quant_params = if self.header.encoding == Encoding::VarDCT {
            info!("decoding VarDCT quantizer params");
            Some(QuantizerParams::read(br)?)
        } else {
            None
        };
        debug!(?quant_params);

        let block_context_map = if self.header.encoding == Encoding::VarDCT {
            info!("decoding block context map");
            Some(BlockContextMap::read(br)?)
        } else {
            None
        };
        debug!(?block_context_map);

        let color_correlation_params = if self.header.encoding == Encoding::VarDCT {
            info!("decoding color correlation params");
            Some(ColorCorrelationParams::read(br)?)
        } else {
            None
        };
        debug!(?color_correlation_params);

        let tree = if br.read(1)? == 1 {
            let size_limit = (1024
                + self.header.width as usize
                    * self.header.height as usize
                    * (self.modular_color_channels
                        + self.decoder_state.extra_channel_info().len())
                    / 16)
                .min(1 << 22);
            Some(Tree::read(br, size_limit)?)
        } else {
            None
        };

        let modular_global = FullModularImage::read(
            &self.header,
            &self.decoder_state.file_header.image_metadata,
            self.modular_color_channels,
            &tree,
            br,
        )?;

        self.lf_global = Some(LfGlobalState {
            patches,
            splines,
            noise,
            lf_quant,
            quant_params,
            block_context_map,
            color_correlation_params,
            tree,
            modular_global,
        });

        Ok(())
    }

    #[instrument(level = "debug", skip(self, br))]
    pub fn decode_lf_group(&mut self, group: usize, br: &mut BitReader) -> Result<()> {
        debug!(section_size = br.total_bits_available());
        let lf_global = self.lf_global.as_mut().unwrap();
        if self.header.encoding == Encoding::VarDCT && !self.header.has_lf_frame() {
            info!("decoding VarDCT LF with group id {}", group);
            decode_vardct_lf(
                group,
                &self.header,
                &self.decoder_state.file_header.image_metadata,
                &lf_global.tree,
                lf_global.color_correlation_params.as_ref().unwrap(),
                lf_global.quant_params.as_ref().unwrap(),
                &lf_global.lf_quant,
                lf_global.block_context_map.as_ref().unwrap(),
                self.lf_image.as_mut().unwrap(),
                &mut self.quant_lf,
                br,
            )?;
        }
        lf_global.modular_global.read_stream(
            ModularStreamId::ModularLF(group),
            &self.header,
            &lf_global.tree,
            br,
        )?;
        if self.header.encoding == Encoding::VarDCT {
            info!("decoding HF metadata with group id {}", group);
            let hf_meta = self.hf_meta.as_mut().unwrap();
            decode_hf_metadata(
                group,
                &self.header,
                &self.decoder_state.file_header.image_metadata,
                &lf_global.tree,
                hf_meta,
                br,
            )?;
        }
        Ok(())
    }

    #[instrument(level = "debug", skip_all)]
    pub fn decode_hf_global(&mut self, br: &mut BitReader) -> Result<()> {
        debug!(section_size = br.total_bits_available());
        if self.header.encoding == Encoding::Modular {
            return Ok(());
        }
        let lf_global = self.lf_global.as_mut().unwrap();
        let mut dequant_matrices = DequantMatrices::decode(&self.header, lf_global, br)?;
        dequant_matrices.ensure_computed(self.hf_meta.as_ref().unwrap().used_hf_types)?;
        let block_context_map = lf_global.block_context_map.as_mut().unwrap();
        let num_histo_bits = self.header.num_groups().ceil_log2();
        let num_histograms: u32 = br.read(num_histo_bits)? as u32 + 1;
        info!(
            "Processing HFGlobal section with {} passes and {} histograms",
            self.header.passes.num_passes, num_histograms
        );
        let mut passes: Vec<PassState> = vec![];
        #[allow(unused_variables)]
        for i in 0..self.header.passes.num_passes as usize {
            let used_orders = match br.read(2)? {
                0 => 0x5f,
                1 => 0x13,
                2 => 0,
                _ => br.read(coeff_order::NUM_ORDERS)?,
            } as u32;
            debug!(used_orders);
            let coeff_orders = decode_coeff_orders(used_orders, br)?;
            assert_eq!(coeff_orders.len(), 3 * coeff_order::NUM_ORDERS);
            let num_contexts = num_histograms as usize * block_context_map.num_ac_contexts();
            info!(
                "Deconding histograms for pass {} with {} contexts",
                i, num_contexts
            );
            let histograms = Histograms::decode(num_contexts, br, true)?;
            debug!("Found {} histograms", histograms.num_histograms());
            passes.push(PassState {
                coeff_orders,
                histograms,
            });
        }
        let hf_coefficients = if passes.len() <= 1 {
            None
        } else {
            let xs = FrameHeader::GROUP_DIM * FrameHeader::GROUP_DIM;
            let ys = self.header.num_groups();
            Some((
                Image::new((xs, ys))?,
                Image::new((xs, ys))?,
                Image::new((xs, ys))?,
            ))
        };
        self.hf_global = Some(HfGlobalState {
            num_histograms,
            passes,
            dequant_matrices,
            hf_coefficients,
        });
        Ok(())
    }

    pub fn build_render_pipeline(
        decoder_state: &DecoderState,
        frame_header: &FrameHeader,
        lf_global: &LfGlobalState,
        epf_sigma: &Option<Arc<Image<f32>>>,
    ) -> Result<SimpleRenderPipeline> {
        let num_channels = frame_header.num_extra_channels as usize + 3;
        let num_temp_channels = if frame_header.has_noise() { 3 } else { 0 };
        let metadata = &decoder_state.file_header.image_metadata;
        let mut pipeline = SimpleRenderPipelineBuilder::new(
            num_channels + num_temp_channels,
            frame_header.size_upsampled(),
            frame_header.upsampling.ilog2() as usize,
            frame_header.log_group_dim(),
        );
        if frame_header.encoding == Encoding::Modular {
            if decoder_state.file_header.image_metadata.xyb_encoded {
                pipeline =
                    pipeline.add_stage(ConvertModularXYBToF32Stage::new(0, &lf_global.lf_quant))?
            } else {
                for i in 0..3 {
                    pipeline =
                        pipeline.add_stage(ConvertModularToF32Stage::new(i, metadata.bit_depth))?;
                }
            }
        }
        for i in 3..num_channels {
            pipeline = pipeline.add_stage(ConvertModularToF32Stage::new(i, metadata.bit_depth))?;
        }

        for c in 0..3 {
            if frame_header.hshift(c) != 0 {
                pipeline = pipeline.add_stage(HorizontalChromaUpsample::new(c))?;
            }
            if frame_header.vshift(c) != 0 {
                pipeline = pipeline.add_stage(VerticalChromaUpsample::new(c))?;
            }
        }

        let filters = &frame_header.restoration_filter;
        if filters.gab {
            pipeline = pipeline
                .add_stage(GaborishStage::new(
                    0,
                    filters.gab_x_weight1,
                    filters.gab_x_weight2,
                ))?
                .add_stage(GaborishStage::new(
                    1,
                    filters.gab_y_weight1,
                    filters.gab_y_weight2,
                ))?
                .add_stage(GaborishStage::new(
                    2,
                    filters.gab_b_weight1,
                    filters.gab_b_weight2,
                ))?;
        }

        let rf = &frame_header.restoration_filter;
        if rf.epf_iters >= 3 {
            pipeline = pipeline.add_stage(Epf0Stage::new(
                rf.epf_pass0_sigma_scale,
                rf.epf_border_sad_mul,
                rf.epf_channel_scale,
                epf_sigma.as_ref().unwrap().clone(),
            ))?
        }
        if rf.epf_iters >= 1 {
            pipeline = pipeline.add_stage(Epf1Stage::new(
                1.0,
                rf.epf_border_sad_mul,
                rf.epf_channel_scale,
                epf_sigma.as_ref().unwrap().clone(),
            ))?
        }
        if rf.epf_iters >= 2 {
            pipeline = pipeline.add_stage(Epf2Stage::new(
                rf.epf_pass2_sigma_scale,
                rf.epf_border_sad_mul,
                rf.epf_channel_scale,
                epf_sigma.as_ref().unwrap().clone(),
            ))?
        }

        let late_ec_upsample = frame_header.upsampling > 1
            && frame_header
                .ec_upsampling
                .iter()
                .all(|x| *x == frame_header.upsampling);

        if !late_ec_upsample {
            let transform_data = &decoder_state.file_header.transform_data;
            for (ec, ec_up) in frame_header.ec_upsampling.iter().enumerate() {
                if *ec_up > 1 {
                    pipeline = match *ec_up {
                        2 => pipeline.add_stage(Upsample2x::new(transform_data, 3 + ec)),
                        4 => pipeline.add_stage(Upsample4x::new(transform_data, 3 + ec)),
                        8 => pipeline.add_stage(Upsample8x::new(transform_data, 3 + ec)),
                        _ => unreachable!(),
                    }?;
                }
            }
        }

        if frame_header.has_patches() {
            // TODO(szabadka): Avoid cloning everything.
            pipeline = pipeline.add_stage(PatchesStage {
                patches: lf_global.patches.clone().unwrap(),
                extra_channels: metadata.extra_channel_info.clone(),
                decoder_state: Arc::new(decoder_state.reference_frames.to_vec()),
            })?
        }

        if frame_header.has_splines() {
            pipeline = pipeline.add_stage(SplinesStage::new(
                lf_global.splines.clone().unwrap(),
                frame_header.size(),
                &lf_global.color_correlation_params.unwrap_or_default(),
            ))?
        }

        if frame_header.upsampling > 1 {
            let transform_data = &decoder_state.file_header.transform_data;
            let nb_channels = if late_ec_upsample {
                3 + frame_header.ec_upsampling.len()
            } else {
                3
            };
            for c in 0..nb_channels {
                pipeline = match frame_header.upsampling {
                    2 => pipeline.add_stage(Upsample2x::new(transform_data, c)),
                    4 => pipeline.add_stage(Upsample4x::new(transform_data, c)),
                    8 => pipeline.add_stage(Upsample8x::new(transform_data, c)),
                    _ => unreachable!(),
                }?;
            }
        }

        if frame_header.has_noise() {
            pipeline = pipeline
                .add_stage(ConvolveNoiseStage::new(num_channels))?
                .add_stage(ConvolveNoiseStage::new(num_channels + 1))?
                .add_stage(ConvolveNoiseStage::new(num_channels + 2))?
                .add_stage(AddNoiseStage::new(
                    *lf_global.noise.as_ref().unwrap(),
                    lf_global.color_correlation_params.unwrap_or_default(),
                    num_channels,
                ))?;
        }
        if frame_header.lf_level != 0 {
            for i in 0..3 {
                pipeline = pipeline.add_stage(SaveStage::<f32>::new(
                    SaveStageType::Lf,
                    i,
                    frame_header.size_upsampled(),
                    1.0,
                    Orientation::Identity,
                )?)?;
            }
        }
        if frame_header.can_be_referenced && frame_header.save_before_ct {
            for i in 0..num_channels {
                pipeline = pipeline.add_stage(SaveStage::<f32>::new(
                    SaveStageType::Reference,
                    i,
                    frame_header.size_upsampled(),
                    1.0,
                    Orientation::Identity,
                )?)?;
            }
        }

        let mut linear = false;
        if frame_header.do_ycbcr {
            pipeline = pipeline.add_stage(YcbcrToRgbStage::new(0))?;
        } else if decoder_state.file_header.image_metadata.xyb_encoded {
            let intensity_target = 255.0;
            let opsin = &decoder_state
                .file_header
                .transform_data
                .opsin_inverse_matrix;
            // TODO(sboukortt): support more output colorspaces than just sRGB
            pipeline = pipeline.add_stage(XybToLinearSrgbStage::new(
                0,
                opsin.clone(),
                intensity_target,
            ))?;
            if decoder_state.xyb_output_linear {
                linear = true;
            } else {
                pipeline = pipeline.add_stage(FromLinearStage::new(0, TransferFunction::Srgb))?;
            }
        }

        if frame_header.needs_blending() {
            if linear {
                pipeline = pipeline.add_stage(FromLinearStage::new(0, TransferFunction::Srgb))?;
                linear = false;
            }
            pipeline = pipeline.add_stage(BlendingStage::new(
                frame_header,
                &decoder_state.file_header,
                &decoder_state.reference_frames,
            )?)?;
            pipeline = pipeline.add_stage(ExtendToImageDimensionsStage::new(
                frame_header,
                &decoder_state.file_header,
                &decoder_state.reference_frames,
            )?)?;
        }
        let image_size = &decoder_state.file_header.size;
        let image_size = (image_size.xsize() as usize, image_size.ysize() as usize);

        if frame_header.can_be_referenced && !frame_header.save_before_ct {
            if linear {
                pipeline = pipeline.add_stage(FromLinearStage::new(0, TransferFunction::Srgb))?;
                linear = false;
            }
            for i in 0..num_channels {
                pipeline = pipeline.add_stage(SaveStage::<f32>::new(
                    SaveStageType::Reference,
                    i,
                    image_size,
                    1.0,
                    Orientation::Identity,
                )?)?;
            }
        }

        if decoder_state.render_spotcolors {
            for (i, info) in decoder_state
                .file_header
                .image_metadata
                .extra_channel_info
                .iter()
                .enumerate()
            {
                if info.ec_type == ExtraChannel::SpotColor {
                    pipeline =
                        pipeline.add_stage(SpotColorStage::new(i, info.spot_color.unwrap()))?;
                }
            }
        }

        if frame_header.is_visible() {
            let color_space = decoder_state
                .file_header
                .image_metadata
                .color_encoding
                .color_space;
            let num_color_channels = if color_space == ColorSpace::Gray {
                1
            } else {
                3
            };
            for i in (0..num_color_channels).chain(3..num_channels) {
                if decoder_state.render_spotcolors
                    && i > 3
                    && decoder_state.file_header.image_metadata.extra_channel_info[i - 3].ec_type
                        == ExtraChannel::SpotColor
                {
                    continue;
                }
                if decoder_state.file_header.image_metadata.xyb_encoded
                    && decoder_state.xyb_output_linear
                    && !linear
                {
                    pipeline = pipeline
                        .add_stage(ToLinearStage::new(0, ToLinearTransferFunction::Srgb))?;
                    linear = true;
                }
                pipeline = pipeline.add_stage(SaveStage::<f32>::new(
                    SaveStageType::Output,
                    i,
                    image_size,
                    255.0,
                    metadata.orientation,
                )?)?;
            }
        }
        pipeline.build()
    }

    pub fn prepare_render_pipeline(&mut self) -> Result<()> {
        let lf_global = self.lf_global.as_mut().unwrap();
        let epf_sigma = if self.header.restoration_filter.epf_iters > 0 {
            let sigma_image = create_sigma_image(&self.header, lf_global, &self.hf_meta)?;
            Some(Arc::new(sigma_image))
        } else {
            None
        };

        let render_pipeline =
            Self::build_render_pipeline(&self.decoder_state, &self.header, lf_global, &epf_sigma)?;
        self.render_pipeline = Some(render_pipeline);
        if self.decoder_state.enable_output {
            lf_global.modular_global.process_output(
                0,
                0,
                &self.header,
                self.render_pipeline.as_mut().unwrap(),
            )?;
            for group in 0..self.header.num_lf_groups() {
                lf_global.modular_global.process_output(
                    1,
                    group,
                    &self.header,
                    self.render_pipeline.as_mut().unwrap(),
                )?;
            }
        }
        Ok(())
    }

    pub fn finalize_lf(&mut self) -> Result<()> {
        if self.header.should_do_adaptive_lf_smoothing() {
            let lf_global = self.lf_global.as_mut().unwrap();
            let lf_quant = &lf_global.lf_quant;
            let inv_quant_lf = lf_global.quant_params.as_mut().unwrap().inv_quant_lf();
            adaptive_lf_smoothing(
                [
                    inv_quant_lf * lf_quant.quant_factors[0],
                    inv_quant_lf * lf_quant.quant_factors[1],
                    inv_quant_lf * lf_quant.quant_factors[2],
                ],
                self.lf_image.as_mut().unwrap(),
            )
        } else {
            Ok(())
        }
    }

    #[instrument(level = "debug", skip(self, br))]
    pub fn decode_hf_group(&mut self, group: usize, pass: usize, br: &mut BitReader) -> Result<()> {
        debug!(section_size = br.total_bits_available());
        if self.header.has_noise() {
            // TODO(sboukortt): consider making this a dedicated stage
            let num_channels = self.header.num_extra_channels as usize + 3;
            self.render_pipeline.as_mut().unwrap().fill_input_channels(
                &[num_channels, num_channels + 1, num_channels + 2],
                group,
                1,
                |rects| {
                    let group_dim = self.header.group_dim() as u32;
                    let xsize_groups = self.header.size_groups().0;
                    let gx = (group % xsize_groups) as u32;
                    let gy = (group / xsize_groups) as u32;
                    // TODO(sboukortt): test upsampling+noise
                    let upsampling = self.header.upsampling;
                    let x0 = gx * upsampling * group_dim;
                    let y0 = gy * upsampling * group_dim;
                    // TODO(sboukortt): actual frame indices for the first two
                    let mut rng = Xorshift128Plus::new_with_seeds(1, 0, x0, y0);
                    let bits_to_float = |bits: u32| f32::from_bits((bits >> 9) | 0x3F800000);
                    for rect in rects {
                        let (xsize, ysize) = rect.size();

                        const FLOATS_PER_BATCH: usize =
                            Xorshift128Plus::N * size_of::<u64>() / size_of::<f32>();
                        let mut batch = [0u64; Xorshift128Plus::N];

                        for y in 0..ysize {
                            let row = rect.row(y);
                            for batch_index in 0..xsize.div_ceil(FLOATS_PER_BATCH) {
                                rng.fill(&mut batch);
                                let batch_size =
                                    (xsize - batch_index * FLOATS_PER_BATCH).min(FLOATS_PER_BATCH);
                                for i in 0..batch_size {
                                    let x = FLOATS_PER_BATCH * batch_index + i;
                                    let k = i / 2;
                                    let high_bytes = i % 2 != 0;
                                    let bits = if high_bytes {
                                        ((batch[k] & 0xFFFFFFFF00000000) >> 32) as u32
                                    } else {
                                        (batch[k] & 0xFFFFFFFF) as u32
                                    };
                                    row[x] = bits_to_float(bits);
                                }
                            }
                        }
                    }
                    Ok(())
                },
            )?;
        }
        let lf_global = self.lf_global.as_mut().unwrap();
        if self.header.encoding == Encoding::VarDCT {
            info!("Decoding VarDCT group {group}, pass {pass}");
            let hf_global = self.hf_global.as_mut().unwrap();
            let hf_meta = self.hf_meta.as_mut().unwrap();
            let pixels = decode_vardct_group(
                group,
                pass,
                &self.header,
                lf_global,
                hf_global,
                hf_meta,
                &self.lf_image,
                &self.quant_lf,
                &self
                    .decoder_state
                    .file_header
                    .transform_data
                    .opsin_inverse_matrix
                    .quant_biases,
                br,
            )?;
            if self.decoder_state.enable_output
                && pass + 1 == self.header.passes.num_passes as usize
            {
                for c in [0, 1, 2] {
                    self.render_pipeline.as_mut().unwrap().fill_input_channels(
                        &[c],
                        group,
                        1,
                        |rects| {
                            rects[0].copy_from(pixels[c].as_rect().rect(Rect {
                                origin: (0, 0),
                                size: rects[0].size(),
                            })?)
                        },
                    )?;
                }
            }
        }
        lf_global.modular_global.read_stream(
            ModularStreamId::ModularHF { group, pass },
            &self.header,
            &lf_global.tree,
            br,
        )?;
        if self.decoder_state.enable_output {
            lf_global.modular_global.process_output(
                2 + pass,
                group,
                &self.header,
                self.render_pipeline.as_mut().unwrap(),
            )?;
        }
        Ok(())
    }

    pub fn finalize(mut self) -> Result<FrameOutput> {
        let mut output_frame_data = Vec::<Image<f32>>::new();
        let mut reference_frame_data = Vec::<Image<f32>>::new();
        let mut lf_frame_data = [
            Image::<f32>::new((0, 0))?,
            Image::<f32>::new((0, 0))?,
            Image::<f32>::new((0, 0))?,
        ];

        let mut lf_chan = 0;
        if let Some(render_pipeline) = self.render_pipeline {
            for stage in render_pipeline
                .into_stages()
                .into_iter()
                .filter_map(|x| x.downcast::<SaveStage<f32>>().ok())
            {
                match stage.stage_type {
                    SaveStageType::Output => {
                        output_frame_data.push(stage.into_buffer());
                    }
                    SaveStageType::Reference => {
                        reference_frame_data.push(stage.into_buffer());
                    }
                    SaveStageType::Lf => {
                        lf_frame_data[lf_chan] = stage.into_buffer();
                        lf_chan += 1;
                    }
                }
            }
        }

        if self.header.can_be_referenced {
            info!("Saving frame in slot {}", self.header.save_as_reference);
            self.decoder_state.reference_frames[self.header.save_as_reference as usize] =
                Some(ReferenceFrame {
                    frame: reference_frame_data,
                    saved_before_color_transform: self.header.save_before_ct,
                });
        }

        if self.header.lf_level != 0 {
            self.decoder_state.lf_frames[(self.header.lf_level - 1) as usize] = Some(lf_frame_data);
        }
        let channels = if self.header.is_visible() {
            Some(output_frame_data)
        } else {
            None
        };
        let decoder_state = if self.header.is_last {
            None
        } else {
            Some(self.decoder_state)
        };
        let frame_output = FrameOutput {
            decoder_state,
            channels,
        };
        Ok(frame_output)
    }
}

#[cfg(test)]
mod test {
    use std::panic;

    use crate::{
        container::ContainerParser,
        decode::{DecodeOptions, decode_jxl_codestream},
        error::Error,
        features::spline::Point,
        util::test::assert_almost_eq,
    };
    use test_log::test;

    use super::Frame;

    #[allow(clippy::type_complexity)]
    fn read_frames(
        image: &[u8],
        callback: &mut dyn FnMut(&Frame) -> Result<(), Error>,
    ) -> Result<(), Error> {
        let codestream = ContainerParser::collect_codestream(image).unwrap();
        let mut options = DecodeOptions::new();
        options.frame_callback = Some(callback);
        decode_jxl_codestream(options, &codestream)?;
        Ok(())
    }

    #[test]
    fn splines() -> Result<(), Error> {
        let mut num_frames = 0;
        let mut verify_frame = |frame: &Frame| {
            let lf_global = frame.lf_global.as_ref().unwrap();
            let splines = lf_global.splines.as_ref().unwrap();
            assert_eq!(splines.quantization_adjustment, 0);
            let expected_starting_points = [Point { x: 9.0, y: 54.0 }].to_vec();
            assert_eq!(splines.starting_points, expected_starting_points);
            assert_eq!(splines.splines.len(), 1);
            let spline = splines.splines[0].clone();
            let expected_control_points = [
                (109, 105),
                (-130, -261),
                (-66, 193),
                (227, -52),
                (-170, 290),
            ]
            .to_vec();
            assert_eq!(spline.control_points.clone(), expected_control_points);

            const EXPECTED_COLOR_DCT: [[i32; 32]; 3] = [
                {
                    let mut row = [0; 32];
                    row[0] = 168;
                    row[1] = 119;
                    row
                },
                {
                    let mut row = [0; 32];
                    row[0] = 9;
                    row[2] = 7;
                    row
                },
                {
                    let mut row = [0; 32];
                    row[0] = -10;
                    row[1] = 7;
                    row
                },
            ];
            assert_eq!(spline.color_dct, EXPECTED_COLOR_DCT);
            const EXPECTED_SIGMA_DCT: [i32; 32] = {
                let mut dct = [0; 32];
                dct[0] = 4;
                dct[7] = 2;
                dct
            };
            assert_eq!(spline.sigma_dct, EXPECTED_SIGMA_DCT);
            num_frames += 1;
            Ok(())
        };
        read_frames(
            include_bytes!("../resources/test/splines.jxl"),
            &mut verify_frame,
        )?;
        assert_eq!(num_frames, 1);
        Ok(())
    }

    #[test]
    fn noise() -> Result<(), Error> {
        let mut num_frames = 0;
        let mut verify_frame = |frame: &Frame| {
            let lf_global = frame.lf_global.as_ref().unwrap();
            let noise = lf_global.noise.as_ref().unwrap();
            let want_noise = [
                0.000000, 0.000977, 0.002930, 0.003906, 0.005859, 0.006836, 0.008789, 0.010742,
            ];
            for (index, noise_param) in want_noise.iter().enumerate() {
                assert_almost_eq!(noise.lut[index], *noise_param, 1e-6);
            }
            num_frames += 1;
            Ok(())
        };
        read_frames(
            include_bytes!("../resources/test/8x8_noise.jxl"),
            &mut verify_frame,
        )?;
        assert_eq!(num_frames, 1);
        Ok(())
    }

    #[test]
    fn patches() -> Result<(), Error> {
        let mut num_frames = 0;
        let mut verify_frame = |frame: &Frame| {
            if num_frames == 0 {
                assert!(!frame.header().has_patches());
                assert!(frame.header().can_be_referenced);
            } else if num_frames == 1 {
                assert!(frame.header().has_patches());
                assert!(!frame.header().can_be_referenced);
            }
            num_frames += 1;
            Ok(())
        };
        read_frames(
            include_bytes!("../resources/test/grayscale_patches_modular.jxl"),
            &mut verify_frame,
        )?;
        assert_eq!(num_frames, 2);
        Ok(())
    }
}
