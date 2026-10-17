// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::sync::Arc;

use super::internal::{RenderPipelineShared, Stage};
use super::stages::ExtendToImageDimensionsStage;
use super::{
    RenderPipeline, RenderPipelineDownsampleStage, RenderPipelineInOutStage,
    RenderPipelineInPlaceStage,
};
use crate::api::{JxlColorType, JxlDataFormat};
use crate::error::{Error, Result};
use crate::headers::Orientation;
use crate::headers::bit_depth::BitDepth;
use crate::image::{BufferRecycler, DataTypeTag};
use crate::render::internal::ChannelInfo;
use crate::render::save::{SaveChannelType, SaveStage};
use crate::render::stages::{ConvertModular16ToF32Stage, ConvertModularToF32Stage};
use crate::util::ShiftRightCeil;
use crate::util::sync::atomic::{AtomicBool, Ordering};
use crate::util::tracing_wrappers::*;

pub struct RenderPipelineBuilder<Pipeline: RenderPipeline> {
    shared: RenderPipelineShared<Pipeline::Buffer>,
    channel_types: Vec<(DataTypeTag, Option<BitDepth>)>,
}

impl<Pipeline: RenderPipeline> RenderPipelineBuilder<Pipeline> {
    #[instrument(level = "debug")]
    pub(super) fn new_with_chunk_size(
        channel_types: &[(DataTypeTag, Option<BitDepth>)],
        size: (usize, usize),
        downsampling_shift: usize,
        mut log_group_size: usize,
        chunk_size: usize,
        buffer_recycler: Arc<BufferRecycler>,
    ) -> Self {
        info!("creating render pipeline");
        assert!(chunk_size <= u16::MAX as usize);
        assert_ne!(chunk_size, 0);
        let num_channels = channel_types.len();
        // The number of pixels that a group encompasses in the final, upsampled image along one
        // dimension is effectively multiplied by the upsampling factor.
        log_group_size += downsampling_shift;
        Self {
            shared: RenderPipelineShared {
                channel_info: vec![
                    channel_types
                        .iter()
                        .map(|&(ty, _)| ChannelInfo {
                            ty,
                            downsample: (0, 0),
                        })
                        .collect(),
                ],
                input_size: size,
                log_group_size,
                group_count: (size.0.shrc(log_group_size), size.1.shrc(log_group_size)),
                stages: vec![],
                group_chan_complete: (0..(size.0.shrc(log_group_size)
                    * size.1.shrc(log_group_size)))
                    .map(|_| (0..num_channels).map(|_| AtomicBool::new(false)).collect())
                    .collect(),
                chunk_size,
                extend_stage_index: None,
                channel_is_used: vec![false; num_channels],
                buffer_recycler,
            },
            channel_types: channel_types.to_vec(),
        }
    }

    pub(super) fn add_stage_internal(mut self, mut stage: Stage<Pipeline::Buffer>) -> Self {
        let output_type = stage.output_type();
        for (c, (cur_ty, bd)) in self.channel_types.iter_mut().enumerate() {
            if !stage.uses_channel(c) {
                continue;
            }
            if matches!(*cur_ty, DataTypeTag::I16 | DataTypeTag::I32)
                && let Some(in_bd) = *bd
            {
                if let Stage::Save(s) = &mut stage
                    && !in_bd.floating_point_sample()
                {
                    let in_bits = in_bd.bits_per_sample() as u8;
                    let can_convert_inline = match s.data_format {
                        JxlDataFormat::U8 {
                            bit_depth: out_bits,
                        } => out_bits.is_multiple_of(in_bits),
                        JxlDataFormat::F16 { .. } => *cur_ty == DataTypeTag::I16,
                        _ => false,
                    };
                    if can_convert_inline {
                        let save_ty = if *cur_ty == DataTypeTag::I16 {
                            SaveChannelType::I16 { bit_depth: in_bits }
                        } else {
                            SaveChannelType::I32 { bit_depth: in_bits }
                        };
                        s.set_channel_type(c, save_ty);
                    }
                }
                if stage.input_type(c) == DataTypeTag::F32 {
                    let conv = if *cur_ty == DataTypeTag::I16 {
                        Pipeline::box_inout_stage(ConvertModular16ToF32Stage::new(c, in_bd))
                    } else {
                        Pipeline::box_inout_stage(ConvertModularToF32Stage::new(c, in_bd))
                    };
                    self.shared.stages.push(Stage::InOut(conv));
                    *cur_ty = DataTypeTag::F32;
                }
            }
            if let Some(out_ty) = output_type {
                *cur_ty = out_ty;
            }
        }
        self.shared.stages.push(stage);
        self
    }

    pub fn new(
        channel_types: &[(DataTypeTag, Option<BitDepth>)],
        size: (usize, usize),
        downsampling_shift: usize,
        log_group_size: usize,
        buffer_recycler: Arc<BufferRecycler>,
    ) -> Self {
        Self::new_with_chunk_size(
            channel_types,
            size,
            downsampling_shift,
            log_group_size,
            1 << (log_group_size + downsampling_shift),
            buffer_recycler,
        )
    }

    pub fn add_save_stage(
        self,
        channels: &[usize],
        orientation: Orientation,
        output_buffer_index: usize,
        color_type: JxlColorType,
        data_format: JxlDataFormat,
        fill_opaque_alpha: bool,
    ) -> Self {
        let stage = SaveStage::new(
            channels,
            orientation,
            output_buffer_index,
            color_type,
            data_format,
            fill_opaque_alpha,
        );
        self.add_stage_internal(Stage::Save(stage))
    }

    pub fn add_extend_stage(self, extend: ExtendToImageDimensionsStage) -> Self {
        self.add_stage_internal(Stage::Extend(extend))
    }

    pub fn add_inplace_stage<S: RenderPipelineInPlaceStage>(self, stage: S) -> Self {
        self.add_stage_internal(Stage::InPlace(Pipeline::box_inplace_stage(stage)))
    }

    pub fn add_inout_stage<S: RenderPipelineInOutStage>(self, stage: S) -> Self {
        self.add_stage_internal(Stage::InOut(Pipeline::box_inout_stage(stage)))
    }

    pub fn add_downsample_stage<S: RenderPipelineDownsampleStage>(self, stage: S) -> Self {
        self.add_stage_internal(Stage::Downsample(Pipeline::box_downsample_stage(stage)))
    }

    #[instrument(skip_all, err)]
    pub fn build(mut self) -> Result<Box<Pipeline>> {
        let mut stage_is_used = vec![false; self.shared.stages.len()];
        let num_channels = self.shared.num_channels();
        // Prune unused stages.
        for i in (0..self.shared.stages.len()).rev() {
            let stage = &self.shared.stages[i];
            if matches!(stage, Stage::Save(_)) {
                for c in 0..num_channels {
                    if stage.uses_channel(c) {
                        self.shared.channel_is_used[c] = true;
                    }
                }
            }
            for c in 0..num_channels {
                if stage.uses_channel(c) {
                    stage_is_used[i] |= self.shared.channel_is_used[c];
                }
            }
            // For now, mark stages with a shift as used.
            // This is because they change channel information, which can then
            // trigger downstream failures.
            // TODO(veluca): we should probably compute channel information *before*
            // we do the pruning.
            if stage.shift() != (0, 0) || stage.downsample() != (0, 0) {
                stage_is_used[i] = true;
            }
            if stage_is_used[i] {
                for c in 0..num_channels {
                    if self.shared.stages[i].uses_channel(c) {
                        self.shared.channel_is_used[c] = true;
                    }
                }
            }
        }
        self.shared.stages = self
            .shared
            .stages
            .into_iter()
            .zip(stage_is_used)
            .filter_map(|(s, used)| used.then_some(s))
            .collect();

        let mut has_downsample = false;
        for (i, stage) in self.shared.stages.iter().enumerate() {
            if has_downsample && matches!(stage, Stage::InOut(_)) {
                return Err(Error::PipelineInvalidStageAfterDownsample(
                    stage.to_string(),
                ));
            }
            has_downsample |= matches!(stage, Stage::Downsample(_));
            let output_type = stage.output_type();
            let shift = stage.shift();
            let downsample = stage.downsample();
            if [shift.0, shift.1, downsample.0, downsample.1]
                .iter()
                .any(|&s| s as u32 >= usize::BITS - 1)
            {
                return Err(Error::ArithmeticOverflow);
            }
            let border = stage.border();
            let is_extend = matches!(stage, Stage::Extend(_));

            let current_info = self.shared.channel_info.last().unwrap().clone();
            debug!(
                last_stage_channel_info = ?current_info,
                extend_stage_index= ?self.shared.extend_stage_index,
                "adding stage '{stage}'",
            );
            let mut after_info = vec![];
            for (c, info) in current_info.iter().enumerate() {
                if !stage.uses_channel(c) {
                    after_info.push(ChannelInfo {
                        ty: info.ty,
                        downsample: (0, 0),
                    });
                } else {
                    let input_type = stage.input_type(c);
                    if info.ty != input_type {
                        return Err(Error::PipelineChannelTypeMismatch(
                            stage.to_string(),
                            c,
                            input_type,
                            info.ty,
                        ));
                    }
                    after_info.push(ChannelInfo {
                        ty: output_type.unwrap_or(input_type),
                        downsample: shift,
                    });
                }
            }
            if self.shared.extend_stage_index.is_some()
                && (shift != (0, 0)
                    || downsample != (0, 0)
                    || border != (0, 0)
                    || is_extend
                    || matches!(stage, Stage::Downsample(_)))
            {
                return Err(Error::PipelineInvalidStageAfterExtend(stage.to_string()));
            }
            if is_extend {
                self.shared.extend_stage_index = Some(i);
            }
            debug!(
                new_channel_info = ?after_info,
                extend_stage_index= ?self.shared.extend_stage_index,
                "added stage '{stage}'",
            );
            self.shared.channel_info.push(after_info);
        }

        let channel_info = &mut self.shared.channel_info;
        // All upsampling precedes downsampling, so only upsampling shifts determine
        // the input resolution. Assign reduced stage resolutions in the forward pass.
        let mut input_downsamples = vec![(0u8, 0u8); num_channels];
        for s in (0..self.shared.stages.len()).rev() {
            for (chan, input_downsample) in input_downsamples.iter_mut().enumerate() {
                let shift = channel_info[s + 1][chan].downsample;
                input_downsample.0 = input_downsample
                    .0
                    .checked_add(shift.0)
                    .ok_or(Error::ArithmeticOverflow)?;
                input_downsample.1 = input_downsample
                    .1
                    .checked_add(shift.1)
                    .ok_or(Error::ArithmeticOverflow)?;
            }
        }
        for (chan, &downsample) in input_downsamples.iter().enumerate() {
            channel_info[0][chan].downsample = downsample;
        }
        for (s, stage) in self.shared.stages.iter().enumerate() {
            let mut save_downsample = None;
            let mut stage_downsample = None;
            for chan in 0..num_channels {
                let current = channel_info[s][chan].downsample;
                if stage.uses_channel(chan) {
                    if let Some(expected) = stage_downsample {
                        if expected != current {
                            return Err(Error::PipelineUnsupported(format!(
                                "stage {stage} uses channels with different resolutions"
                            )));
                        }
                    } else {
                        stage_downsample = Some(current);
                    }
                    if matches!(stage, Stage::Extend(_)) && current != (0, 0) {
                        return Err(Error::PipelineUnsupported(
                            "extend requires full-resolution channels".into(),
                        ));
                    }
                }
                if matches!(stage, Stage::Save(_)) && stage.uses_channel(chan) {
                    if let Some(expected) = save_downsample {
                        if expected != current {
                            return Err(Error::SaveDifferentDownsample(expected, current));
                        }
                    } else {
                        save_downsample = Some(current);
                    }
                }
                let shift = channel_info[s + 1][chan].downsample;
                let reduction = if stage.uses_channel(chan) {
                    stage.downsample()
                } else {
                    (0, 0)
                };
                let next = (
                    (current.0 - shift.0)
                        .checked_add(reduction.0)
                        .ok_or(Error::ArithmeticOverflow)?,
                    (current.1 - shift.1)
                        .checked_add(reduction.1)
                        .ok_or(Error::ArithmeticOverflow)?,
                );
                if next.0 as u32 >= usize::BITS - 1 || next.1 as u32 >= usize::BITS - 1 {
                    return Err(Error::ArithmeticOverflow);
                }
                channel_info[s + 1][chan].downsample = next;
            }
        }
        #[cfg(feature = "tracing")]
        {
            for (s, (current_info, stage)) in channel_info
                .iter()
                .zip(self.shared.stages.iter())
                .enumerate()
            {
                debug!("final channel info before stage {s} '{stage}': {current_info:?}");
            }
            debug!(
                "final channel info after all stages {:?}",
                channel_info.last().unwrap()
            );
        }

        for (c, is_used) in self.shared.channel_is_used.iter().enumerate() {
            if !is_used {
                for g in self.shared.group_chan_complete.iter_mut() {
                    g[c].store(true, Ordering::Relaxed);
                }
            }
        }

        Ok(Box::new(Pipeline::new_from_shared(self.shared)?))
    }
}
