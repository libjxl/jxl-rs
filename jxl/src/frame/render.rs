// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::any::Any;
use std::sync::Arc;

use crate::api::JxlCms;
use crate::api::JxlColorType;
use crate::api::JxlDataFormat;
use crate::api::JxlOutputBuffer;
use crate::error::{Error, Result};
use crate::features::epf::create_sigma_image;
use crate::headers::frame_header::Encoding;
use crate::headers::{Orientation, color_encoding::ColorSpace, extra_channels::ExtraChannel};
use crate::render::{
    LowMemoryRenderPipeline, RenderPipeline, RenderPipelineBuilder, SimpleRenderPipeline, stages::*,
};
use crate::{
    api::JxlPixelFormat,
    frame::{DecoderState, Frame, LfGlobalState},
    headers::frame_header::FrameHeader,
    image::Image,
};

macro_rules! pipeline {
    ($frame: expr, $pipeline: ident, $op: expr) => {
        if $frame.use_simple_pipeline {
            let $pipeline = $frame
                .render_pipeline
                .as_mut()
                .unwrap()
                .downcast_mut::<SimpleRenderPipeline>()
                .unwrap();
            $op
        } else {
            let $pipeline = $frame
                .render_pipeline
                .as_mut()
                .unwrap()
                .downcast_mut::<LowMemoryRenderPipeline>()
                .unwrap();
            $op
        }
    };
}

pub(crate) use pipeline;

impl Frame {
    pub fn render_frame_output(
        &mut self,
        api_buffers: &mut Option<&mut [JxlOutputBuffer<'_>]>,
        pixel_format: &JxlPixelFormat,
    ) -> Result<()> {
        if self.render_pipeline.is_none() {
            // We don't yet have any output ready (as the pipeline would be initialized otherwise),
            // so exit without doing anything.
            return Ok(());
        }

        let mut buffers: Vec<Option<JxlOutputBuffer>> = Vec::new();

        macro_rules! buffers_from_api {
            ($get_next: expr) => {
                if pixel_format.color_data_format.is_some() {
                    buffers.push($get_next);
                }

                for fmt in &pixel_format.extra_channel_format {
                    if fmt.is_some() {
                        buffers.push($get_next);
                    }
                }
            };
        }

        if let Some(api_buffers) = api_buffers {
            let mut api_buffers_iter = api_buffers.iter_mut();
            buffers_from_api!(Some(JxlOutputBuffer::from_output_buffer(
                api_buffers_iter.next().unwrap(),
            )));
        } else {
            buffers_from_api!(None);
        }

        if let Some(ref_images) = &mut self.reference_frame_data {
            buffers.extend(
                ref_images
                    .iter_mut()
                    .map(|img| Some(JxlOutputBuffer::from_image(img))),
            );
        };

        if let Some(lf_images) = &mut self.lf_frame_data {
            buffers.extend(
                lf_images
                    .iter_mut()
                    .map(|img| Some(JxlOutputBuffer::from_image(img))),
            );
        };

        if buffers.iter().any(|b| b.is_some()) {
            pipeline!(self, p, p.do_render(&mut buffers[..])?);
        }
        Ok(())
    }

    pub(crate) fn build_render_pipeline<T: RenderPipeline>(
        decoder_state: &DecoderState,
        frame_header: &FrameHeader,
        lf_global: &LfGlobalState,
        epf_sigma: &Option<Arc<Image<f32>>>,
        pixel_format: &JxlPixelFormat,
    ) -> Result<Box<T>> {
        let num_channels = frame_header.num_extra_channels as usize + 3;
        let num_temp_channels = if frame_header.has_noise() { 3 } else { 0 };
        let metadata = &decoder_state.file_header.image_metadata;
        let mut pipeline = RenderPipelineBuilder::<T>::new(
            num_channels + num_temp_channels,
            frame_header.size_upsampled(),
            frame_header.upsampling.ilog2() as usize,
            frame_header.log_group_dim(),
            frame_header.passes.num_passes as usize,
        );

        if frame_header.encoding == Encoding::Modular {
            if decoder_state.file_header.image_metadata.xyb_encoded {
                pipeline = pipeline
                    .add_inout_stage(ConvertModularXYBToF32Stage::new(0, &lf_global.lf_quant))?
            } else {
                for i in 0..3 {
                    pipeline = pipeline
                        .add_inout_stage(ConvertModularToF32Stage::new(i, metadata.bit_depth))?;
                }
            }
        }
        for i in 3..num_channels {
            pipeline =
                pipeline.add_inout_stage(ConvertModularToF32Stage::new(i, metadata.bit_depth))?;
        }

        for c in 0..3 {
            if frame_header.hshift(c) != 0 {
                pipeline = pipeline.add_inout_stage(HorizontalChromaUpsample::new(c))?;
            }
            if frame_header.vshift(c) != 0 {
                pipeline = pipeline.add_inout_stage(VerticalChromaUpsample::new(c))?;
            }
        }

        let filters = &frame_header.restoration_filter;
        if filters.gab {
            pipeline = pipeline
                .add_inout_stage(GaborishStage::new(
                    0,
                    filters.gab_x_weight1,
                    filters.gab_x_weight2,
                ))?
                .add_inout_stage(GaborishStage::new(
                    1,
                    filters.gab_y_weight1,
                    filters.gab_y_weight2,
                ))?
                .add_inout_stage(GaborishStage::new(
                    2,
                    filters.gab_b_weight1,
                    filters.gab_b_weight2,
                ))?;
        }

        let rf = &frame_header.restoration_filter;
        if rf.epf_iters >= 3 {
            pipeline = pipeline.add_inout_stage(Epf0Stage::new(
                rf.epf_pass0_sigma_scale,
                rf.epf_border_sad_mul,
                rf.epf_channel_scale,
                epf_sigma.as_ref().unwrap().clone(),
            ))?
        }
        if rf.epf_iters >= 1 {
            pipeline = pipeline.add_inout_stage(Epf1Stage::new(
                1.0,
                rf.epf_border_sad_mul,
                rf.epf_channel_scale,
                epf_sigma.as_ref().unwrap().clone(),
            ))?
        }
        if rf.epf_iters >= 2 {
            pipeline = pipeline.add_inout_stage(Epf2Stage::new(
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
                        2 => pipeline.add_inout_stage(Upsample2x::new(transform_data, 3 + ec)),
                        4 => pipeline.add_inout_stage(Upsample4x::new(transform_data, 3 + ec)),
                        8 => pipeline.add_inout_stage(Upsample8x::new(transform_data, 3 + ec)),
                        _ => unreachable!(),
                    }?;
                }
            }
        }

        if frame_header.has_patches() {
            // TODO(szabadka): Avoid cloning everything.
            pipeline = pipeline.add_inplace_stage(PatchesStage {
                patches: lf_global.patches.clone().unwrap(),
                extra_channels: metadata.extra_channel_info.clone(),
                decoder_state: Arc::new(decoder_state.reference_frames.to_vec()),
            })?
        }

        if frame_header.has_splines() {
            pipeline = pipeline.add_inplace_stage(SplinesStage::new(
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
                    2 => pipeline.add_inout_stage(Upsample2x::new(transform_data, c)),
                    4 => pipeline.add_inout_stage(Upsample4x::new(transform_data, c)),
                    8 => pipeline.add_inout_stage(Upsample8x::new(transform_data, c)),
                    _ => unreachable!(),
                }?;
            }
        }

        if frame_header.has_noise() {
            pipeline = pipeline
                .add_inout_stage(ConvolveNoiseStage::new(num_channels))?
                .add_inout_stage(ConvolveNoiseStage::new(num_channels + 1))?
                .add_inout_stage(ConvolveNoiseStage::new(num_channels + 2))?
                .add_inplace_stage(AddNoiseStage::new(
                    *lf_global.noise.as_ref().unwrap(),
                    lf_global.color_correlation_params.unwrap_or_default(),
                    num_channels,
                ))?;
        }

        let num_regular_output_buffers = frame_header.num_extra_channels as usize + 1;
        assert_eq!(
            pixel_format.extra_channel_format.len(),
            frame_header.num_extra_channels as usize
        );

        assert!(frame_header.lf_level == 0 || !frame_header.can_be_referenced);

        if frame_header.lf_level != 0 {
            for i in 0..3 {
                pipeline = pipeline.add_save_stage(
                    &[i],
                    Orientation::Identity,
                    num_regular_output_buffers + i,
                    JxlColorType::Grayscale,
                    JxlDataFormat::f32(),
                )?;
            }
        }
        if frame_header.can_be_referenced && frame_header.save_before_ct {
            for i in 0..num_channels {
                pipeline = pipeline.add_save_stage(
                    &[i],
                    Orientation::Identity,
                    num_regular_output_buffers + i,
                    JxlColorType::Grayscale,
                    JxlDataFormat::f32(),
                )?;
            }
        }

        let mut linear = false;
        let output_color_info = OutputColorInfo::from_header(&decoder_state.file_header)?;
        if frame_header.do_ycbcr {
            pipeline = pipeline.add_inplace_stage(YcbcrToRgbStage::new(0))?;
        } else if decoder_state.file_header.image_metadata.xyb_encoded {
            pipeline = pipeline.add_inplace_stage(XybStage::new(0, output_color_info.clone()))?;
            if decoder_state.xyb_output_linear {
                linear = true;
            } else {
                pipeline = pipeline
                    .add_inplace_stage(FromLinearStage::new(0, output_color_info.tf.clone()))?;
            }
        }

        if frame_header.needs_blending() {
            if linear {
                pipeline = pipeline
                    .add_inplace_stage(FromLinearStage::new(0, output_color_info.tf.clone()))?;
                linear = false;
            }
            pipeline = pipeline.add_inplace_stage(BlendingStage::new(
                frame_header,
                &decoder_state.file_header,
                &decoder_state.reference_frames,
            )?)?;
            // TODO(veluca): we might not need to add an extend stage if the image size is
            // compatible with the frame size.
            pipeline = pipeline.add_extend_stage(ExtendToImageDimensionsStage::new(
                frame_header,
                &decoder_state.file_header,
                &decoder_state.reference_frames,
            )?)?;
        }

        if frame_header.can_be_referenced && !frame_header.save_before_ct {
            if linear {
                pipeline = pipeline
                    .add_inplace_stage(FromLinearStage::new(0, output_color_info.tf.clone()))?;
                linear = false;
            }
            for i in 0..num_channels {
                pipeline = pipeline.add_save_stage(
                    &[i],
                    Orientation::Identity,
                    num_regular_output_buffers + i,
                    JxlColorType::Grayscale,
                    JxlDataFormat::f32(),
                )?;
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
                    pipeline = pipeline
                        .add_inplace_stage(SpotColorStage::new(i, info.spot_color.unwrap()))?;
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
            let alpha_in_color = if pixel_format.color_type.has_alpha() {
                decoder_state
                    .file_header
                    .image_metadata
                    .extra_channel_info
                    .iter()
                    .enumerate()
                    .find(|x| x.1.ec_type == ExtraChannel::Alpha)
                    .map(|x| x.0 + 3)
            } else {
                None
            };
            if pixel_format.color_type.is_grayscale() && num_color_channels == 3 {
                return Err(Error::NotGrayscale);
            }
            if decoder_state.file_header.image_metadata.xyb_encoded
                && decoder_state.xyb_output_linear
                && !linear
            {
                pipeline = pipeline
                    .add_inplace_stage(ToLinearStage::new(0, output_color_info.tf.clone()))?;
            }
            let color_source_channels: &[usize] =
                match (pixel_format.color_type.is_grayscale(), alpha_in_color) {
                    (true, None) => &[0],
                    (true, Some(c)) => &[0, c],
                    (false, None) => &[0, 1, 2],
                    (false, Some(c)) => &[0, 1, 2, c],
                };
            if let Some(df) = &pixel_format.color_data_format {
                pipeline = pipeline.add_save_stage(
                    color_source_channels,
                    metadata.orientation,
                    0,
                    pixel_format.color_type,
                    *df,
                )?;
            }
            for i in 0..frame_header.num_extra_channels as usize {
                if let Some(df) = &pixel_format.extra_channel_format[i] {
                    pipeline = pipeline.add_save_stage(
                        &[3 + i],
                        metadata.orientation,
                        1 + i,
                        JxlColorType::Grayscale,
                        *df,
                    )?;
                }
            }
        }
        pipeline.build()
    }

    pub fn prepare_render_pipeline(
        &mut self,
        pixel_format: &JxlPixelFormat,
        _cms: Option<&dyn JxlCms>,
    ) -> Result<()> {
        let lf_global = self.lf_global.as_mut().unwrap();
        let epf_sigma = if self.header.restoration_filter.epf_iters > 0 {
            let sigma_image = create_sigma_image(&self.header, lf_global, &self.hf_meta)?;
            Some(Arc::new(sigma_image))
        } else {
            None
        };

        let render_pipeline = if self.use_simple_pipeline {
            Self::build_render_pipeline::<SimpleRenderPipeline>(
                &self.decoder_state,
                &self.header,
                lf_global,
                &epf_sigma,
                pixel_format,
            )? as Box<dyn Any>
        } else {
            Self::build_render_pipeline::<LowMemoryRenderPipeline>(
                &self.decoder_state,
                &self.header,
                lf_global,
                &epf_sigma,
                pixel_format,
            )? as Box<dyn Any>
        };
        self.render_pipeline = Some(render_pipeline);

        if self.decoder_state.enable_output {
            let mut pass_to_pipeline = |chan, group, num_passes, image| {
                pipeline!(
                    self,
                    p,
                    p.set_buffer_for_group(chan, group, num_passes, image)
                );
            };
            lf_global
                .modular_global
                .process_output(0, 0, &self.header, &mut pass_to_pipeline)?;
            for group in 0..self.header.num_lf_groups() {
                lf_global.modular_global.process_output(
                    1,
                    group,
                    &self.header,
                    &mut pass_to_pipeline,
                )?;
            }
        }
        Ok(())
    }
}
