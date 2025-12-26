// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::{collections::BTreeSet, sync::Arc};

use crate::{
    entropy_coding::decode::Histograms,
    error::Result,
    features::{noise::Noise, patches::PatchesDictionary, spline::Splines},
    headers::{
        FileHeader,
        extra_channels::ExtraChannelInfo,
        frame_header::{Encoding, FrameHeader},
        permutation::Permutation,
        toc::Toc,
    },
    image::Image,
    util::tracing_wrappers::*,
};
use adaptive_lf_smoothing::adaptive_lf_smoothing;
use block_context_map::BlockContextMap;
use color_correlation_map::ColorCorrelationParams;
use modular::{FullModularImage, Tree};
use quant_weights::DequantMatrices;
#[cfg(feature = "jpeg-reconstruction")]
use quant_weights::QuantEncoding;
use quantizer::{LfQuantFactors, QuantizerParams};

mod adaptive_lf_smoothing;
mod block_context_map;
mod coeff_order;
pub mod color_correlation_map;
pub mod decode;
mod group;
pub mod modular;
mod quant_weights;
pub mod quantizer;
pub mod render;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Section {
    LfGlobal,
    Lf { group: usize },
    HfGlobal,
    Hf { group: usize, pass: usize },
}

pub struct LfGlobalState {
    patches: Option<Arc<PatchesDictionary>>,
    splines: Option<Splines>,
    noise: Option<Noise>,
    lf_quant: LfQuantFactors,
    pub quant_params: Option<QuantizerParams>,
    block_context_map: Option<BlockContextMap>,
    color_correlation_params: Option<ColorCorrelationParams>,
    tree: Option<Tree>,
    modular_global: FullModularImage,
}

pub struct PassState {
    coeff_orders: Vec<Permutation>,
    histograms: Histograms,
}

pub struct HfGlobalState {
    num_histograms: u32,
    passes: Vec<PassState>,
    dequant_matrices: DequantMatrices,
    hf_coefficients: Option<(Image<i32>, Image<i32>, Image<i32>)>,
}

#[derive(Debug)]
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
            .map(|_| Image::new((width, height)))
            .collect::<Result<_>>()?;
        Ok(Self {
            frame,
            saved_before_color_transform,
        })
    }
    #[cfg(test)]
    pub fn random<R: rand::Rng>(
        mut rng: &mut R,
        width: usize,
        height: usize,
        num_channels: usize,
        saved_before_color_transform: bool,
    ) -> Result<Self> {
        let frame = (0..num_channels)
            .map(|_| Image::new_random((width, height), &mut rng))
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
    pub(super) reference_frames: Arc<[Option<ReferenceFrame>; Self::MAX_STORED_FRAMES]>,
    pub(super) lf_frames: [Option<[Image<f32>; 3]>; 4],
    pub render_spotcolors: bool,
    #[cfg(test)]
    pub use_simple_pipeline: bool,
    pub visible_frame_index: usize,
    pub nonvisible_frame_index: usize,
    pub high_precision: bool,
    pub premultiply_output: bool,
}

impl DecoderState {
    pub const MAX_STORED_FRAMES: usize = 4;

    pub fn new(file_header: FileHeader) -> Self {
        Self {
            file_header,
            reference_frames: Arc::new([None, None, None, None]),
            lf_frames: [None, None, None, None],
            render_spotcolors: true,
            #[cfg(test)]
            use_simple_pipeline: false,
            visible_frame_index: 0,
            nonvisible_frame_index: 0,
            high_precision: false,
            premultiply_output: false,
        }
    }

    pub fn extra_channel_info(&self) -> &Vec<ExtraChannelInfo> {
        &self.file_header.image_metadata.extra_channel_info
    }

    pub fn reference_frame(&self, i: usize) -> Option<&ReferenceFrame> {
        assert!(i < Self::MAX_STORED_FRAMES);
        self.reference_frames[i].as_ref()
    }

    #[cfg(test)]
    pub fn set_use_simple_pipeline(&mut self, u: bool) {
        self.use_simple_pipeline = u;
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

// Re-export JpegDctCoefficients for JPEG reconstruction
#[cfg(feature = "jpeg-reconstruction")]
pub use crate::jpeg::JpegDctCoefficients;

pub struct Frame {
    header: FrameHeader,
    toc: Toc,
    color_channels: usize,
    lf_global: Option<LfGlobalState>,
    hf_global: Option<HfGlobalState>,
    lf_image: Option<[Image<f32>; 3]>,
    quant_lf: Image<u8>,
    hf_meta: Option<HfMetadata>,
    decoder_state: DecoderState,
    #[cfg(test)]
    use_simple_pipeline: bool,
    #[cfg(test)]
    render_pipeline: Option<Box<dyn std::any::Any>>,
    #[cfg(not(test))]
    render_pipeline: Option<Box<crate::render::LowMemoryRenderPipeline>>,
    reference_frame_data: Option<Vec<Image<f32>>>,
    lf_frame_data: Option<[Image<f32>; 3]>,
    lf_global_was_rendered: bool,
    /// Reusable buffers for VarDCT group decoding.
    vardct_buffers: Option<group::VarDctBuffers>,
    // Last pass rendered so far for each HF group.
    last_rendered_pass: Vec<Option<usize>>,
    // Groups that should be rendered on the next call to flush().
    groups_to_flush: BTreeSet<usize>,
    /// Storage for raw DCT coefficients (for JPEG reconstruction)
    #[cfg(feature = "jpeg-reconstruction")]
    pub jpeg_coefficients: Option<JpegDctCoefficients>,
    /// Whether to preserve DCT coefficients for JPEG reconstruction
    #[cfg(feature = "jpeg-reconstruction")]
    preserve_jpeg_coefficients: bool,
}

impl Frame {
    pub fn toc(&self) -> &Toc {
        &self.toc
    }

    pub fn header(&self) -> &FrameHeader {
        &self.header
    }

    pub fn total_bytes_in_toc(&self) -> usize {
        self.toc.entries.iter().map(|x| *x as usize).sum()
    }

    /// Enable coefficient preservation for JPEG reconstruction.
    /// This also initializes the coefficient storage if not already done.
    #[cfg(feature = "jpeg-reconstruction")]
    pub fn set_preserve_jpeg_coefficients(&mut self, preserve: bool) {
        self.preserve_jpeg_coefficients = preserve;
        if preserve && self.jpeg_coefficients.is_none() {
            self.init_jpeg_coefficients();
        }
    }

    /// Initialize JPEG coefficient storage based on frame dimensions.
    #[cfg(feature = "jpeg-reconstruction")]
    fn init_jpeg_coefficients(&mut self) {
        let (width, height) = self.header.size_upsampled();
        let num_components = if self.color_channels == 1 { 1 } else { 3 };
        let component_map = if num_components == 1 { [1usize, 1, 1] } else { [1usize, 0, 2] };
        let mut component_blocks = Vec::with_capacity(num_components);
        for &vardct_chan in component_map.iter().take(num_components) {
            let hshift = self.header.hshift(vardct_chan);
            let vshift = self.header.vshift(vardct_chan);
            let denom_x = 8usize << hshift;
            let denom_y = 8usize << vshift;
            let blocks_x = (width + denom_x - 1) / denom_x;
            let blocks_y = (height + denom_y - 1) / denom_y;
            component_blocks.push((blocks_x, blocks_y));
        }

        self.jpeg_coefficients =
            Some(JpegDctCoefficients::new(width, height, &component_blocks));
    }

    /// Check if coefficient preservation is enabled.
    #[cfg(feature = "jpeg-reconstruction")]
    pub fn preserve_jpeg_coefficients(&self) -> bool {
        self.preserve_jpeg_coefficients
    }

    /// Get the stored JPEG coefficients (if any).
    #[cfg(feature = "jpeg-reconstruction")]
    pub fn jpeg_coefficients(&self) -> Option<&JpegDctCoefficients> {
        self.jpeg_coefficients.as_ref()
    }

    /// Take ownership of the stored JPEG coefficients.
    #[cfg(feature = "jpeg-reconstruction")]
    pub fn take_jpeg_coefficients(&mut self) -> Option<JpegDctCoefficients> {
        self.jpeg_coefficients.take()
    }

    #[cfg(feature = "jpeg-reconstruction")]
    pub fn jpeg_raw_quant_table(&self) -> Option<(&[i32], f32)> {
        let hf_global = self.hf_global.as_ref()?;
        let encoding = hf_global.dequant_matrices.encodings().get(0)?;
        match encoding {
            QuantEncoding::Raw { qtable, qtable_den } => Some((qtable.as_slice(), *qtable_den)),
            _ => None,
        }
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

    pub fn finalize(mut self) -> Result<Option<DecoderState>> {
        // First, drop the render pipeline to ensure that no other references to the reference
        // frames are around.
        self.render_pipeline = None;
        // Save reference frame if this frame can be referenced and was actually decoded.
        // If reference_frame_data is None (frame was skipped), we don't save it.
        // Subsequent frames referencing this slot may fail.
        if self.header.can_be_referenced
            && let Some(frame_data) = self.reference_frame_data
        {
            info!("Saving frame in slot {}", self.header.save_as_reference);
            let rf = Arc::get_mut(&mut self.decoder_state.reference_frames)
                .expect("remaining references to reference_frames");
            rf[self.header.save_as_reference as usize] = Some(ReferenceFrame {
                frame: frame_data,
                saved_before_color_transform: self.header.save_before_ct,
            });
        }

        if self.header.lf_level != 0 {
            self.decoder_state.lf_frames[(self.header.lf_level - 1) as usize] = self.lf_frame_data;
        }
        let decoder_state = if self.header.is_last {
            None
        } else {
            Some(self.decoder_state)
        };
        Ok(decoder_state)
    }

    fn modular_color_channels(&self) -> usize {
        if self.header.encoding == Encoding::VarDCT {
            0
        } else {
            self.color_channels
        }
    }
}

#[cfg(test)]
mod test {
    use std::panic;

    use crate::{
        error::{Error, Result},
        features::spline::Point,
        util::test::assert_almost_abs_eq,
    };
    use test_log::test;

    use super::Frame;

    fn decode(
        bytes: &[u8],
        verify: impl Fn(&Frame, usize) -> Result<()> + 'static,
    ) -> Result<usize> {
        crate::api::tests::decode(bytes, usize::MAX, false, Some(Box::new(verify))).map(|x| x.0)
    }

    #[test]
    fn splines() -> Result<(), Error> {
        let verify_frame = move |frame: &Frame, _| {
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
            Ok(())
        };
        assert_eq!(
            decode(
                include_bytes!("../../resources/test/splines.jxl"),
                verify_frame
            )?,
            1
        );
        Ok(())
    }

    #[test]
    fn noise() -> Result<(), Error> {
        let verify_frame = |frame: &Frame, _| {
            let lf_global = frame.lf_global.as_ref().unwrap();
            let noise = lf_global.noise.as_ref().unwrap();
            let want_noise = [
                0.000000, 0.000977, 0.002930, 0.003906, 0.005859, 0.006836, 0.008789, 0.010742,
            ];
            for (index, noise_param) in want_noise.iter().enumerate() {
                assert_almost_abs_eq(noise.lut[index], *noise_param, 1e-6);
            }
            Ok(())
        };
        assert_eq!(
            decode(
                include_bytes!("../../resources/test/8x8_noise.jxl"),
                verify_frame,
            )?,
            1
        );
        Ok(())
    }

    #[test]
    fn patches() -> Result<(), Error> {
        let verify_frame = |frame: &Frame, frame_index| {
            if frame_index == 0 {
                assert!(!frame.header().has_patches());
                assert!(frame.header().can_be_referenced);
            } else if frame_index == 1 {
                assert!(frame.header().has_patches());
                assert!(!frame.header().can_be_referenced);
            }
            Ok(())
        };
        assert_eq!(
            decode(
                include_bytes!("../../resources/test/grayscale_patches_modular.jxl"),
                verify_frame,
            )?,
            2
        );
        Ok(())
    }

    #[test]
    fn multiple_lf_420() -> Result<(), Error> {
        let verify_frame = |frame: &Frame, _| {
            assert!(frame.header().is420());
            let Some(lf_image) = &frame.lf_image else {
                panic!("no lf_image");
            };
            for y in 0..146 {
                let sample_cb_row = lf_image[0].row(y);
                let sample_cr_row = lf_image[2].row(y);
                for x in 0..146 {
                    let sample_cb = sample_cb_row[x];
                    let sample_cr = sample_cr_row[x];
                    let no_chroma = sample_cb == 0.0 && sample_cr == 0.0;
                    if y < 128 || x < 128 {
                        assert!(!no_chroma);
                    } else {
                        assert!(no_chroma);
                    }
                }
            }
            Ok(())
        };
        decode(
            include_bytes!("../../resources/test/multiple_lf_420.jxl"),
            verify_frame,
        )?;
        Ok(())
    }

    #[test]
    fn xyb_grayscale_patches() -> Result<(), Error> {
        let verify_frame = |frame: &Frame, frame_index| {
            if frame_index == 0 {
                assert_eq!(
                    frame.header.frame_type,
                    crate::headers::frame_header::FrameType::ReferenceOnly,
                );
                assert_eq!(
                    frame.header.encoding,
                    crate::headers::frame_header::Encoding::Modular,
                );
                assert_eq!(frame.modular_color_channels(), 3);
            } else {
                assert!(frame.header.has_patches());
                assert_eq!(frame.modular_color_channels(), 0);
            }
            Ok(())
        };
        assert_eq!(
            decode(
                include_bytes!("../../resources/test/grayscale_patches_var_dct.jxl"),
                verify_frame,
            )?,
            2
        );
        Ok(())
    }
}
