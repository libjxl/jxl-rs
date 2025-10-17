// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use super::{
    block_context_map::BlockContextMap,
    coeff_order::decode_coeff_orders,
    color_correlation_map::ColorCorrelationParams,
    group::decode_vardct_group,
    modular::{FullModularImage, ModularStreamId, Tree, decode_hf_metadata, decode_vardct_lf},
    quant_weights::DequantMatrices,
    quantizer::{LfQuantFactors, QuantizerParams},
    transform_map::*,
};
use crate::{
    GROUP_DIM,
    bit_reader::BitReader,
    entropy_coding::decode::Histograms,
    error::Result,
    features::{noise::Noise, patches::PatchesDictionary, spline::Splines},
    frame::{
        DecoderState, Frame, HfGlobalState, HfMetadata, LfGlobalState, PassState, coeff_order,
    },
    headers::{
        color_encoding::ColorSpace,
        frame_header::{Encoding, FrameHeader, Toc},
    },
    image::Image,
    render::RenderPipeline,
    util::{CeilLog2, Xorshift128Plus, tracing_wrappers::*},
};

impl Frame {
    pub fn from_header_and_toc(
        frame_header: FrameHeader,
        toc: Toc,
        decoder_state: DecoderState,
    ) -> Result<Self> {
        let image_metadata = &decoder_state.file_header.image_metadata;
        let is_modular_gray = !frame_header.do_ycbcr
            && !image_metadata.xyb_encoded
            && image_metadata.color_encoding.color_space == ColorSpace::Gray;
        let modular_color_channels = if frame_header.encoding == Encoding::VarDCT {
            0
        } else if is_modular_gray {
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

        let reference_frame_data = if frame_header.can_be_referenced {
            let image_size = &decoder_state.file_header.size;
            let image_size = (image_size.xsize() as usize, image_size.ysize() as usize);
            let sz = if frame_header.save_before_ct {
                frame_header.size_upsampled()
            } else {
                image_size
            };

            let num_ref_channels = 3 + image_metadata.extra_channel_info.len();
            Some(
                (0..num_ref_channels)
                    .map(|_| Image::new(sz))
                    .collect::<Result<Vec<_>>>()?,
            )
        } else {
            None
        };

        let lf_frame_data = if frame_header.lf_level != 0 {
            Some(
                (0..3)
                    .map(|_| Image::new(frame_header.size_upsampled()))
                    .collect::<Result<Vec<_>, _>>()?
                    .try_into()
                    .unwrap(),
            )
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
            reference_frame_data,
            lf_frame_data,
        })
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
            let xs = GROUP_DIM * GROUP_DIM;
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
    #[instrument(level = "debug", skip(self, br))]
    pub fn decode_hf_group(&mut self, group: usize, pass: usize, br: &mut BitReader) -> Result<()> {
        debug!(section_size = br.total_bits_available());
        if self.header.has_noise() {
            // TODO(sboukortt): consider making this a dedicated stage
            let num_channels = self.header.num_extra_channels as usize + 3;

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
            let rp = self.render_pipeline.as_mut().unwrap();
            for i in 0..3 {
                let mut buf = rp.get_buffer_for_group(num_channels + i, group)?;
                let mut rect = buf.as_rect_mut();
                let (xsize, ysize) = rect.size();
                const FLOATS_PER_BATCH: usize =
                    Xorshift128Plus::N * std::mem::size_of::<u64>() / std::mem::size_of::<f32>();
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

                rp.set_buffer_for_group(num_channels + i, group, 1, buf);
            }
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
                for (c, p) in pixels.into_iter().enumerate() {
                    self.render_pipeline
                        .as_mut()
                        .unwrap()
                        .set_buffer_for_group(c, group, 1, p);
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
}
