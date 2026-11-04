// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(clippy::needless_range_loop)]

use std::fmt::Debug;

use row_buffers::RowBuffer;

use super::RenderPipeline;
use super::internal::{RenderPipelineShared, RunInOutStage, RunInPlaceStage};
use crate::api::JxlOutputBuffer;
use crate::error::Result;
use crate::image::{DataTypeTag, Image, ImageDataType, Rect};
use crate::render::buffer_splitter::{BufferSplitter, SaveStageBufferInfo};
use crate::render::internal::Stage;
use crate::render::low_memory_pipeline::input_buffers::InputBuffers;
use crate::render::{ErasedLocalState, MAX_BORDER};
use crate::util::sync::atomic::Ordering;
use crate::util::tracing_wrappers::*;
use crate::util::{PerThreadStorage, ShiftRightCeil};

mod downsample;
mod group_scheduler;
mod helpers;
mod input_buffers;
mod render_group;
pub mod row_buffers;
mod run_stage;
mod save;

#[derive(Clone, Copy, PartialEq, Eq)]
struct RowBufferConfig {
    data_type: DataTypeTag,
    next_y_border: usize,
    y_shift: usize,
    x_shift: usize,
    row_len: usize,
}

impl RowBufferConfig {
    fn merge(&mut self, other: &Self) {
        assert_eq!(self.data_type, other.data_type);
        self.next_y_border = self.next_y_border.max(other.next_y_border);
        self.y_shift = self.y_shift.max(other.y_shift);
        self.x_shift = self.x_shift.max(other.x_shift);
        self.row_len = self.row_len.max(other.row_len);
    }

    fn allocate(&self) -> Result<RowBuffer> {
        RowBuffer::new(
            self.data_type,
            self.next_y_border,
            self.y_shift,
            self.x_shift,
            self.row_len,
        )
    }
}

struct LowMemoryRenderPipelinePerThread {
    row_buffers: Vec<Vec<RowBuffer>>,
    save_scratch: Option<[RowBuffer; 4]>,
    // Local states of each stage, if any.
    local_states: Vec<Option<Box<ErasedLocalState>>>,
}

impl Debug for LowMemoryRenderPipelinePerThread {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<scratch>")
    }
}

impl LowMemoryRenderPipelinePerThread {
    fn ensure_populated(&mut self, p: &LowMemoryRenderPipeline) -> Result<()> {
        if !self.row_buffers.is_empty() {
            return Ok(());
        }
        self.row_buffers = p
            .row_buffer_configs
            .iter()
            .map(|stage_cfgs| stage_cfgs.iter().map(RowBufferConfig::allocate).collect())
            .collect::<Result<_>>()?;
        let scratch_len = p.shared.chunk_size + 2 * p.border_size.0;
        self.save_scratch = Some([
            RowBuffer::new(DataTypeTag::F32, 0, 0, 0, scratch_len)?,
            RowBuffer::new(DataTypeTag::F32, 0, 0, 0, scratch_len)?,
            RowBuffer::new(DataTypeTag::F32, 0, 0, 0, scratch_len)?,
            RowBuffer::new(DataTypeTag::F32, 0, 0, 0, scratch_len)?,
        ]);
        self.local_states = p
            .shared
            .stages
            .iter()
            .map(|x| x.init_local_state())
            .collect::<Result<_>>()?;
        Ok(())
    }
}

pub struct LowMemoryRenderPipeline {
    shared: RenderPipelineShared<RowBuffer>,
    per_thread_data: PerThreadStorage<LowMemoryRenderPipelinePerThread>,
    input_buffers: InputBuffers,
    row_buffer_configs: Vec<Vec<RowBufferConfig>>,
    save_buffer_info: Vec<Option<SaveStageBufferInfo>>,
    // The input buffer that each channel of each stage should use.
    // This is indexed both by stage index (0 corresponds to input data, 1 to stage[0], etc) and by
    // index *of those channels that are used*.
    stage_input_buffer_index: Vec<Vec<(usize, usize)>>,
    // Tracks whether we already rendered the padding around the core frame (if any).
    padding_was_rendered: bool,
    // The amount of pixels that each stage needs to *output* around the current group to
    // run future stages correctly.
    stage_output_border_pixels: Vec<(usize, usize)>,
    // The amount of pixels that we need to read (for every channel) in non-edge groups to run all
    // stages correctly.
    input_border_pixels: Vec<(usize, usize)>,
    // Size of the border, in image (i.e. non-downsampled) pixels.
    border_size: (usize, usize),
    // For every stage, the downsampling level of *any* channel that the stage uses at that point.
    // Note that this must be equal across all the used channels.
    downsampling_for_stage: Vec<(usize, usize)>,
    downsample_align: (usize, usize),
    // Sorted indices to call get_distinct_indices.
    sorted_buffer_indices: Vec<Vec<(usize, usize, usize)>>,
}

impl RenderPipeline for LowMemoryRenderPipeline {
    type Buffer = RowBuffer;

    fn new_from_shared(shared: RenderPipelineShared<Self::Buffer>) -> Result<Self> {
        let nc = shared.num_channels();
        let mut previous_output: Vec<_> = (0..nc).map(|x| (0usize, x)).collect();
        let mut stage_input_buffer_index = vec![];
        let mut next_border_and_cur_downsample = vec![vec![]];

        for ci in shared.channel_info[0].iter() {
            next_border_and_cur_downsample[0].push((0, ci.downsample));
        }

        // For each stage, compute in which stage its input was buffered (the previous stage
        // with a separate output). Also, compute for each such stage and channel the border
        // with which its output is used, to allocate buffers of the correct size.
        for (i, stage) in shared.stages.iter().enumerate() {
            stage_input_buffer_index.push(previous_output.clone());
            next_border_and_cur_downsample.push(vec![]);
            if stage.output_type().is_some() {
                for (chan, (ps, pc)) in previous_output.iter_mut().enumerate() {
                    if !stage.uses_channel(chan) {
                        continue;
                    }

                    let border = if let Stage::Downsample(s) = stage {
                        // Make `RowBuffer` allocate sufficient rows for a downsample block.
                        // NOTE(tirr-c): In this way, `RowBuffer` will allocate more rows than
                        // necessary.
                        (1u8 << s.shift()) - 1
                    } else {
                        stage.border().1
                    };

                    next_border_and_cur_downsample[*ps][*pc].0 = border;
                    *ps = i + 1;
                    *pc = next_border_and_cur_downsample[i + 1].len();
                    next_border_and_cur_downsample[i + 1]
                        .push((0, shared.channel_info[i + 1][chan].downsample));
                }
            }
        }

        // Compute information to be used to compute sub-rects for "save" stages to operate on
        // rects.
        let mut save_buffer_info = vec![];
        'stage: for (i, (s, ci)) in shared
            .stages
            .iter()
            .zip(shared.channel_info.iter())
            .enumerate()
        {
            let Stage::Save(s) = s else {
                continue;
            };
            for (c, ci) in ci.iter().enumerate() {
                if s.uses_channel(c) {
                    let info = SaveStageBufferInfo {
                        downsample: ci.downsample,
                        orientation: s.orientation,
                        byte_size: s.bytes_per_pixel(),
                        after_extend: shared.extend_stage_index.is_some_and(|e| i > e),
                    };
                    while save_buffer_info.len() <= s.output_buffer_index {
                        save_buffer_info.push(None);
                    }
                    save_buffer_info[s.output_buffer_index] = Some(info);
                    continue 'stage;
                }
            }
        }

        // Compute the amount of border pixels needed per channel, per stage.
        let mut border_pixels = vec![(0usize, 0usize); nc];
        let mut border_pixels_per_stage = vec![];
        for s in shared.stages.iter().rev() {
            let mut stage_max = (0, 0);
            for (c, bp) in border_pixels.iter().enumerate() {
                if s.uses_channel(c) {
                    stage_max.0 = stage_max.0.max(bp.0);
                    stage_max.1 = stage_max.1.max(bp.1);
                }
            }
            let input_border = if let Stage::Downsample(s) = s {
                let shift = s.shift();
                let block_size = 1 << shift;
                // Adding `block_size - 1` to the y-border ensures that the whole downsampling
                // block is ready before running the stage.
                (
                    stage_max.0 << shift,
                    (stage_max.1 << shift) + (block_size - 1),
                )
            } else {
                (
                    stage_max.0.shrc(s.shift().0) + s.border().0 as usize,
                    stage_max.1.shrc(s.shift().1) + s.border().1 as usize,
                )
            };
            for (c, bp) in border_pixels.iter_mut().enumerate() {
                if s.uses_channel(c) {
                    *bp = input_border;
                }
            }
            border_pixels_per_stage.push(stage_max);
        }
        border_pixels_per_stage.reverse();

        let has_downsampling = shared
            .stages
            .iter()
            .any(|s| matches!(s, Stage::Downsample(_)));
        if !has_downsampling {
            assert!(border_pixels_per_stage[0].0 <= MAX_BORDER);
        }

        let downsampling_for_stage: Vec<_> = shared
            .stages
            .iter()
            .zip(shared.channel_info.iter())
            .map(|(s, ci)| {
                let downsamplings: Vec<_> = (0..nc)
                    .filter_map(|c| {
                        if s.uses_channel(c) {
                            Some(ci[c].downsample)
                        } else {
                            None
                        }
                    })
                    .collect();
                for &d in downsamplings.iter() {
                    assert_eq!(d, downsamplings[0]);
                }
                (downsamplings[0].0 as usize, downsamplings[0].1 as usize)
            })
            .collect();

        let mut downsample_align = (0, 0);
        for (info, &used) in shared.channel_info[0].iter().zip(&shared.channel_is_used) {
            if used {
                downsample_align.0 = downsample_align.0.max(info.downsample.0 as usize);
                downsample_align.1 = downsample_align.1.max(info.downsample.1 as usize);
            }
        }
        for (stage, &(dx, dy)) in shared.stages.iter().zip(&downsampling_for_stage) {
            if let Stage::Downsample(stage) = stage {
                let shift = stage.shift() as usize;
                downsample_align.0 = downsample_align.0.max(dx + shift);
                downsample_align.1 = downsample_align.1.max(dy + shift);
            }
        }

        let default_channels: Vec<usize> = (0..nc).collect();
        for (s, ibi) in stage_input_buffer_index.iter_mut().enumerate() {
            let mut filtered = vec![];
            // For SaveStage, use s.channels to get correct output ordering (e.g., BGRA).
            let channels = if let Stage::Save(save_stage) = &shared.stages[s] {
                save_stage.channels.as_slice()
            } else {
                default_channels.as_slice()
            };
            for &c in channels {
                if shared.stages[s].uses_channel(c) {
                    filtered.push(ibi[c]);
                }
            }
            *ibi = filtered;
        }

        let sorted_buffer_indices = (0..shared.stages.len())
            .map(|s| {
                let mut v: Vec<_> = stage_input_buffer_index[s]
                    .iter()
                    .enumerate()
                    .map(|(i, (outer, inner))| (*outer, *inner, i))
                    .collect();
                v.sort_unstable();
                v
            })
            .collect();

        let mut border_size = (0, 0);
        for c in 0..nc {
            border_size.0 = border_size
                .0
                .max(border_pixels[c].0 << shared.channel_info[0][c].downsample.0);
            border_size.1 = border_size
                .1
                .max(border_pixels[c].1 << shared.channel_info[0][c].downsample.1);
        }
        border_size.0 = border_size.0.next_multiple_of(1 << downsample_align.0);
        border_size.1 = border_size.1.next_multiple_of(1 << downsample_align.1);

        let left_padding_bytes = RowBuffer::x0_byte_offset();
        let mut row_buffer_configs = vec![];
        let mut initial_configs = vec![];
        for chan in 0..nc {
            let dsx = shared.channel_info[0][chan].downsample.0;
            let data_type = shared.channel_info[0][chan].ty;
            let required_left_border_bytes = border_pixels[chan].0 * data_type.size();
            if has_downsampling && required_left_border_bytes > left_padding_bytes {
                return Err(crate::error::Error::DownsampleTooHigh);
            }

            initial_configs.push(RowBufferConfig {
                data_type,
                next_y_border: next_border_and_cur_downsample[0][chan].0 as usize,
                y_shift: 0,
                x_shift: 0,
                row_len: (shared.chunk_size + 2 * border_size.0) >> dsx,
            });
        }
        row_buffer_configs.push(initial_configs);

        for (i, stage) in shared.stages.iter().enumerate() {
            let mut stage_configs = vec![];
            for &(next_y_border, (dsx, _)) in next_border_and_cur_downsample[i + 1].iter() {
                let data_type = stage.output_type().unwrap();
                let required_left_border_bytes = border_pixels_per_stage[i].0 * data_type.size();
                if has_downsampling && required_left_border_bytes > left_padding_bytes {
                    return Err(crate::error::Error::DownsampleTooHigh);
                }

                stage_configs.push(RowBufferConfig {
                    data_type,
                    next_y_border: next_y_border as usize,
                    y_shift: stage.shift().1.max(0) as usize,
                    x_shift: stage.shift().0.max(0) as usize,
                    row_len: (shared.chunk_size + 2 * border_size.0) >> dsx,
                });
            }
            row_buffer_configs.push(stage_configs);
        }

        // Unify buffer configurations for stages with separate outputs across all their channels
        // to a fixed point.
        loop {
            let mut changed = false;
            for (s, stage) in shared.stages.iter().enumerate().rev() {
                if stage.output_type().is_some() {
                    if let Some(mut unified_out) = row_buffer_configs[s + 1].first().copied() {
                        for cfg in &row_buffer_configs[s + 1] {
                            unified_out.merge(cfg);
                        }
                        for cfg in &mut row_buffer_configs[s + 1] {
                            if *cfg != unified_out {
                                *cfg = unified_out;
                                changed = true;
                            }
                        }
                    }

                    let inputs = &stage_input_buffer_index[s];
                    if let Some(&(first_o, first_i)) = inputs.first() {
                        let mut unified_in = row_buffer_configs[first_o][first_i];
                        for &(o, i) in inputs {
                            unified_in.merge(&row_buffer_configs[o][i]);
                        }
                        for &(o, i) in inputs {
                            if row_buffer_configs[o][i] != unified_in {
                                row_buffer_configs[o][i] = unified_in;
                                changed = true;
                            }
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }

        Ok(Self {
            input_buffers: InputBuffers::new(nc, shared.group_count)?,
            stage_input_buffer_index,
            row_buffer_configs,
            per_thread_data: PerThreadStorage::new(|| LowMemoryRenderPipelinePerThread {
                row_buffers: vec![],
                save_scratch: None,
                local_states: vec![],
            }),
            padding_was_rendered: false,
            save_buffer_info,
            stage_output_border_pixels: border_pixels_per_stage,
            border_size,
            input_border_pixels: border_pixels,
            shared,
            downsampling_for_stage,
            downsample_align,
            sorted_buffer_indices,
        })
    }

    fn input_downsample(&self, channel: usize) -> (u8, u8) {
        self.shared.channel_info[0][channel].downsample
    }

    #[instrument(skip_all, err)]
    fn get_buffer<T: ImageDataType>(&self, channel: usize) -> Result<Image<T>> {
        let sz = self.shared.group_size_for_channel(channel, T::DATA_TYPE_ID);
        self.shared.buffer_recycler.get_buffer(sz)
    }

    fn set_buffer_for_group<T: ImageDataType>(
        &self,
        channel: usize,
        group_id: usize,
        complete: bool,
        buf: Image<T>,
        buffer_splitter: &BufferSplitter,
    ) -> Result<()> {
        if self.shared.channel_is_used[channel] {
            debug!(
                "filling data for group {}, channel {}, using type {:?}",
                group_id,
                channel,
                T::DATA_TYPE_ID,
            );
            let num_ready = self
                .input_buffers
                .get(group_id)
                .set_buffer(channel, buf.into_raw());
            self.shared.group_chan_complete[group_id][channel].store(complete, Ordering::Relaxed);

            assert!(num_ready <= self.shared.num_used_channels());
            if num_ready != self.shared.num_used_channels() {
                return Ok(());
            }

            self.render_with_new_group(group_id, buffer_splitter)?;
        } else {
            self.shared.buffer_recycler.recycle_buffer(buf);
        }
        Ok(())
    }

    fn check_buffer_sizes(&self, buffers: &mut [Option<JxlOutputBuffer>]) -> Result<()> {
        // Check that buffer sizes are correct.
        let mut size = self.shared.input_size;
        for (i, s) in self.shared.stages.iter().enumerate() {
            match s {
                Stage::Extend(e) => size = e.image_size,
                Stage::Save(s) => {
                    let (dx, dy) = self.downsampling_for_stage[i];
                    s.check_buffer_size(
                        if self.shared.extend_stage_index.is_some_and(|e| i > e) {
                            size
                        } else {
                            (size.0.shrc(dx), size.1.shrc(dy))
                        },
                        buffers[s.output_buffer_index].as_ref(),
                    )?
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn render_outside_frame(&mut self, buffer_splitter: &BufferSplitter) -> Result<()> {
        if self.shared.extend_stage_index.is_none() || self.padding_was_rendered {
            return Ok(());
        }
        self.padding_was_rendered = true;
        // TODO(veluca): consider pre-computing those strips at pipeline construction and making
        // smaller strips.
        let e = self.shared.extend_stage_index.unwrap();
        let (dx, dy) = self.downsampling_for_stage[e];
        let chunk_size = self.shared.chunk_size.shrc(dx);
        let Stage::Extend(e) = &self.shared.stages[e] else {
            unreachable!("extend stage is not an extend stage");
        };
        let frame_end = (
            e.frame_origin.0 + self.shared.input_size.0.shrc(dx) as isize,
            e.frame_origin.1 + self.shared.input_size.1.shrc(dy) as isize,
        );
        // Split the full image area in 4 strips: left and right of the frame, and above and below.
        // Use the row-buffer width at Extend's resolution.
        let mut strips = vec![];
        // Above (including left and right)
        if e.frame_origin.1 > 0 {
            let xend = e.image_size.0;
            let yend = (e.frame_origin.1 as usize).min(e.image_size.1);
            for x in (0..xend).step_by(chunk_size) {
                let xe = (x + chunk_size).min(xend);
                strips.push((x..xe, 0..yend));
            }
        }
        // Below
        if frame_end.1 < e.image_size.1 as isize {
            let ystart = frame_end.1.max(0) as usize;
            let yend = e.image_size.1;
            let xend = e.image_size.0;
            for x in (0..xend).step_by(chunk_size) {
                let xe = (x + chunk_size).min(xend);
                strips.push((x..xe, ystart..yend));
            }
        }
        // Left
        if e.frame_origin.0 > 0 {
            let ystart = e.frame_origin.1.max(0) as usize;
            let yend = (frame_end.1 as usize).min(e.image_size.1);
            let xend = (e.frame_origin.0 as usize).min(e.image_size.0);
            for x in (0..xend).step_by(chunk_size) {
                let xe = (x + chunk_size).min(xend);
                strips.push((x..xe, ystart..yend));
            }
        }
        // Right
        if frame_end.0 < e.image_size.0 as isize {
            let xstart = frame_end.0.max(0) as usize;
            let xend = e.image_size.0;
            let ystart = e.frame_origin.1.max(0) as usize;
            let yend = (frame_end.1 as usize).min(e.image_size.1);
            for x in (xstart..xend).step_by(chunk_size) {
                let xe = (x + chunk_size).min(xend);
                strips.push((x..xe, ystart..yend));
            }
        }
        let full_image_size = e.image_size;

        // TODO(veluca): parallelize here.
        let mut data = self.per_thread_data.get();
        data.ensure_populated(self)?;

        for (xrange, yrange) in strips {
            let rect_to_render = Rect {
                origin: (xrange.start, yrange.start),
                size: (xrange.clone().count(), yrange.clone().count()),
            };
            if rect_to_render.size.0 == 0 || rect_to_render.size.1 == 0 {
                continue;
            }
            let mut local_buffers = buffer_splitter.get_local_buffers(
                &self.save_buffer_info,
                Rect {
                    origin: (rect_to_render.origin.0 << dx, rect_to_render.origin.1 << dy),
                    size: (rect_to_render.size.0 << dx, rect_to_render.size.1 << dy),
                },
                true,
                (full_image_size.0 << dx, full_image_size.1 << dy),
                (full_image_size.0 << dx, full_image_size.1 << dy),
                (0, 0),
            );
            self.render_outside_frame_internal(&mut data, xrange, yrange, &mut local_buffers)?;
        }
        Ok(())
    }

    fn mark_group_to_rerender(&mut self, g: usize) {
        let all_finalized = (0..self.shared.num_channels())
            .filter(|&c| self.shared.channel_is_used[c])
            .all(|c| self.shared.group_chan_complete[g][c].load(Ordering::Relaxed));
        // The caller will feed back in all the non-ready channels.
        if !all_finalized {
            self.input_buffers.mark_not_ready(g);
        }
    }

    fn box_inout_stage<S: super::RenderPipelineInOutStage>(
        stage: S,
    ) -> Box<dyn RunInOutStage<Self::Buffer>> {
        Box::new(stage)
    }

    fn box_inplace_stage<S: super::RenderPipelineInPlaceStage>(
        stage: S,
    ) -> Box<dyn RunInPlaceStage<Self::Buffer>> {
        Box::new(stage)
    }

    fn used_channel_mask(&self) -> &[bool] {
        &self.shared.channel_is_used
    }
}
