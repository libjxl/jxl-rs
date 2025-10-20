// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::any::Any;
use std::usize;

use row_buffers::RowBuffer;

use crate::BLOCK_DIM;
use crate::api::JxlOutputBuffer;
use crate::error::Result;
use crate::headers::Orientation;
use crate::image::{Image, ImageDataType, Rect};
use crate::render::internal::Stage;
use crate::simd::CACHE_LINE_BYTE_SIZE;
use crate::util::{ShiftRightCeil, tracing_wrappers::*};

use super::RenderPipeline;
use super::internal::{RenderPipelineShared, RunInOutStage, RunInPlaceStage};

mod render_group;
pub(super) mod row_buffers;
mod run_stage;
mod save;

const MAX_OVERALL_BORDER: usize = 16; // probably an overestimate.

const _: () = assert!(MAX_OVERALL_BORDER * 8 <= CACHE_LINE_BYTE_SIZE * 2);

struct InputBuffer {
    // One buffer per channel.
    data: Vec<Option<Box<dyn Any>>>,
    completed_passes: usize,
}

struct SaveStageBufferInfo {
    buffer_index: usize,
    downsample: (u8, u8),
    orientation: Orientation,
}

pub struct LowMemoryRenderPipeline {
    shared: RenderPipelineShared<RowBuffer>,
    input_buffers: Vec<InputBuffer>,
    row_buffers: Vec<Vec<RowBuffer>>,
    // The input buffer that each channel of each stage should use. 0 corresponds to input data.
    stage_input_buffer_index: Vec<Vec<usize>>,
    // Tracks whether we already rendered the padding around the core frame (if any).
    padding_was_rendered: bool,
    // sorted by buffer_index; all values of buffer_index are distinct.
    save_buffer_info: Vec<SaveStageBufferInfo>,
    // The amount of pixels we need to load from neighbouring groups in each dimension.
    border_pixels: (usize, usize),
    // Local states of each stage, if any.
    local_states: Vec<Option<Box<dyn Any>>>,
}

impl RenderPipeline for LowMemoryRenderPipeline {
    type Buffer = RowBuffer;

    fn new_from_shared(shared: RenderPipelineShared<Self::Buffer>) -> Result<Self> {
        let mut input_buffers = vec![];
        for _ in 0..shared.group_chan_ready_passes.len() {
            input_buffers.push(InputBuffer {
                data: vec![],
                completed_passes: 0,
            });
            for _ in 0..shared.group_chan_ready_passes[0].len() {
                input_buffers.last_mut().unwrap().data.push(None);
            }
        }
        let nc = shared.channel_info[0].len();
        let mut previous_inout = vec![0usize; nc];
        let mut stage_input_buffer_index = vec![];
        let mut used_with_border = vec![vec![0u8; nc]; shared.stages.len() + 1];

        // For each stage, compute in which stage its input was buffered (the previous InOut
        // stage). Also, compute for each InOut stage and channel the border with which the stage
        // output is used; this will used to allocate buffers of the correct size.
        for (i, stage) in shared.stages.iter().enumerate() {
            stage_input_buffer_index.push(previous_inout.clone());
            if let Stage::InOut(p) = stage {
                for chan in 0..nc {
                    if !p.uses_channel(chan) {
                        continue;
                    }
                    used_with_border[previous_inout[chan]][chan] = p.border().1;
                    previous_inout[chan] = i + 1;
                }
            }
        }

        let mut initial_buffers = vec![];
        for chan in 0..nc {
            initial_buffers.push(RowBuffer::new(
                shared.channel_info[0][chan].ty.unwrap(),
                used_with_border[0][chan] as usize,
                0,
                shared.chunk_size >> shared.channel_info[0][chan].downsample.0,
            )?);
        }
        let mut row_buffers = vec![initial_buffers];

        // Allocate buffers.
        for (i, stage) in shared.stages.iter().enumerate() {
            let mut stage_buffers = vec![];
            if let Stage::InOut(p) = stage {
                for chan in 0..nc {
                    if !p.uses_channel(chan) {
                        continue;
                    }
                    stage_buffers.push(RowBuffer::new(
                        p.output_type(),
                        used_with_border[i + 1][chan] as usize,
                        p.shift().1 as usize,
                        shared.chunk_size >> shared.channel_info[i + 1][chan].downsample.0,
                    )?);
                }
            }
            row_buffers.push(stage_buffers);
        }
        // Compute information to be used to compute sub-rects for "save" stages to operate on
        // rects.
        let mut save_buffer_info = vec![];
        'stage: for (s, ci) in shared.stages.iter().zip(shared.channel_info.iter()) {
            let Stage::Save(s) = s else {
                continue;
            };
            for (c, ci) in ci.iter().enumerate() {
                if s.uses_channel(c) {
                    save_buffer_info.push(SaveStageBufferInfo {
                        buffer_index: s.output_buffer_index,
                        downsample: ci.downsample,
                        orientation: s.orientation,
                    });
                    continue 'stage;
                }
            }
        }
        save_buffer_info.sort_by_key(|x| x.buffer_index);

        // Compute the amount of border pixels needed per channel.
        let mut border_pixels = vec![(0usize, 0usize); nc];
        for s in shared.stages.iter().rev() {
            for c in 0..nc {
                if !s.uses_channel(c) {
                    continue;
                }
                border_pixels[c].0 = border_pixels[c].0.shrc(s.shift().0) + s.border().0 as usize;
                border_pixels[c].1 = border_pixels[c].1.shrc(s.shift().1) + s.border().1 as usize;
            }
        }
        let border_pixels = (
            border_pixels.iter().map(|x| x.0).max().unwrap(),
            border_pixels.iter().map(|x| x.1).max().unwrap(),
        );
        Ok(Self {
            input_buffers,
            stage_input_buffer_index,
            row_buffers,
            padding_was_rendered: false,
            save_buffer_info,
            border_pixels,
            local_states: shared
                .stages
                .iter()
                .map(|x| x.init_local_state())
                .collect::<Result<_>>()?,
            shared,
        })
    }

    #[instrument(skip_all, err)]
    fn get_buffer_for_group<T: ImageDataType>(
        &mut self,
        channel: usize,
        group_id: usize,
    ) -> Result<Image<T>> {
        let sz = self
            .shared
            .group_size_for_channel(channel, group_id, T::DATA_TYPE_ID);
        if let Some(buf) = self.input_buffers[group_id].data[channel].take() {
            let img: Image<T> = *buf
                .downcast()
                .expect("inconsistent usage of pipeline buffers");
            let bsz = img.size();
            assert!(sz.0 <= bsz.0);
            assert!(sz.1 <= bsz.1);
            assert!(sz.0 + BLOCK_DIM > bsz.0);
            assert!(sz.1 + BLOCK_DIM > bsz.1);
            return Ok(img);
        }
        Image::<T>::new(sz)
    }

    fn set_buffer_for_group<T: ImageDataType>(
        &mut self,
        channel: usize,
        group_id: usize,
        num_passes: usize,
        buf: Image<T>,
    ) {
        debug!(
            "filling data for group {}, channel {}, using type {:?}",
            group_id,
            channel,
            T::DATA_TYPE_ID,
        );
        let sz = self
            .shared
            .group_size_for_channel(channel, group_id, T::DATA_TYPE_ID);
        let bsz = buf.size();
        assert!(sz.0 <= bsz.0);
        assert!(sz.1 <= bsz.1);
        assert!(sz.0 + BLOCK_DIM > bsz.0);
        assert!(sz.1 + BLOCK_DIM > bsz.1);
        self.input_buffers[group_id].data[channel] = Some(Box::new(buf));
        self.shared.group_chan_ready_passes[group_id][channel] += num_passes;
    }

    fn do_render(&mut self, buffers: &mut [Option<JxlOutputBuffer>]) -> Result<()> {
        if self.shared.extend_stage_index.is_some() {
            // TODO(veluca): implement this case.
            unimplemented!()
        }
        // First, render all groups that have made progress.
        // TODO(veluca): this could potentially be quadratic for huge images that receive a group
        // at a time. Take care of that.
        for g in 0..self.shared.group_chan_ready_passes.len() {
            let ready_passes = self.shared.group_chan_ready_passes[g]
                .iter()
                .copied()
                .min()
                .unwrap();
            if self.input_buffers[g].completed_passes < ready_passes {
                let (gx, gy) = self.shared.group_position(g);
                let mut fully_ready_passes = ready_passes;
                if self.border_pixels.0 != 0 && self.border_pixels.1 != 0 {
                    for dy in -1..=1 {
                        let igy = gy as isize + dy;
                        if igy < 0 || igy >= self.shared.group_count.1 as isize {
                            continue;
                        }
                        for dx in -1..=1 {
                            let igx = gx as isize + dx;
                            if igx < 0 || igx >= self.shared.group_count.0 as isize {
                                continue;
                            }
                            let ig = (igy as usize) * self.shared.group_count.0 + igx as usize;
                            let ready_passes = self.shared.group_chan_ready_passes[ig]
                                .iter()
                                .copied()
                                .min()
                                .unwrap();
                            fully_ready_passes = fully_ready_passes.min(ready_passes);
                        }
                    }
                }
                if self.input_buffers[g].completed_passes >= fully_ready_passes {
                    continue;
                }
                debug!(
                    "new ready passes for group {gx},{gy} ({} completed, \
                    {ready_passes} ready, {fully_ready_passes} ready including neighbours)",
                    self.input_buffers[g].completed_passes
                );

                // Prepare output buffers for the group.
                let mut local_buffers = vec![];
                local_buffers.try_reserve(buffers.len())?;
                for _ in 0..buffers.len() {
                    local_buffers.push(None::<JxlOutputBuffer>);
                }
                let mut buffer_iter = buffers.iter_mut().enumerate();
                for bi in self.save_buffer_info.iter() {
                    let Some(buf) = (loop {
                        let Some((i, b)) = buffer_iter.next() else {
                            panic!("Invalid save_buffer_info");
                        };
                        if i != bi.buffer_index {
                            continue;
                        }
                        break b;
                    }) else {
                        continue;
                    };
                    let size = (
                        1 << (self.shared.log_group_size - bi.downsample.0 as usize),
                        1 << (self.shared.log_group_size - bi.downsample.1 as usize),
                    );
                    let origin = (size.0 * gx, size.1 * gy);
                    let image_size = (
                        self.shared.input_size.0 >> bi.downsample.0,
                        self.shared.input_size.1 >> bi.downsample.1,
                    );
                    let size = (
                        (size.0 + origin.0).min(image_size.0) - origin.0,
                        (size.1 + origin.1).min(image_size.1) - origin.1,
                    );
                    local_buffers[bi.buffer_index] = Some(
                        buf.subrect(
                            bi.orientation
                                .display_rect(Rect { size, origin }, image_size),
                        ),
                    );
                }

                self.render_group((gx, gy), &mut local_buffers)?;

                self.input_buffers[g].completed_passes = fully_ready_passes;
            }
        }

        // Clear buffers that will not be used again.
        for g in 0..self.shared.group_chan_ready_passes.len() {
            let (gx, gy) = self.shared.group_position(g);
            let mut neigh_complete_passes = self.input_buffers[g].completed_passes;
            if self.border_pixels.0 != 0 && self.border_pixels.1 != 0 {
                for dy in -1..=1 {
                    let igy = gy as isize + dy;
                    if igy < 0 || igy >= self.shared.group_count.1 as isize {
                        continue;
                    }
                    for dx in -1..=1 {
                        let igx = gx as isize + dx;
                        if igx < 0 || igx >= self.shared.group_count.0 as isize {
                            continue;
                        }
                        let ig = (igy as usize) * self.shared.group_count.0 + igx as usize;
                        neigh_complete_passes = self.input_buffers[ig]
                            .completed_passes
                            .min(neigh_complete_passes);
                    }
                }
            }
            if self.shared.num_passes <= neigh_complete_passes {
                for b in self.input_buffers[g].data.iter_mut() {
                    *b = None;
                }
            }
        }

        Ok(())
    }

    fn num_groups(&self) -> usize {
        self.shared.num_groups()
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
}
