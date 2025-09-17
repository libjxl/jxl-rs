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
use crate::image::{Image, ImageDataType};
use crate::render::internal::{RenderPipelineStageType, Stage};
use crate::simd::CACHE_LINE_BYTE_SIZE;
use crate::util::tracing_wrappers::*;

use super::internal::{RenderPipelineShared, RunStage};
use super::{RenderPipeline, RenderPipelineStage};

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
    stage_index: usize,
    buffer_index: usize,
    downsample: (u8, u8),
}

pub struct LowMemoryRenderPipeline {
    shared: RenderPipelineShared<RowBuffer>,
    input_buffers: Vec<InputBuffer>,
    row_buffers: Vec<Vec<RowBuffer>>,
    // The input buffer that each channel of each stage should use (none for input data).
    stage_input_buffer_index: Vec<Vec<Option<usize>>>,
    // Tracks whether we already rendered the padding around the core frame (if any).
    padding_was_rendered: bool,
    // sorted by buffer_index; all values of buffer_index are distinct.
    save_buffer_info: Vec<SaveStageBufferInfo>,
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
        let mut previous_inout = vec![None::<usize>; nc];
        let mut stage_input_buffer_index = vec![];
        let mut row_buffers = vec![];
        let mut used_with_border = vec![vec![0u8; nc]; shared.stages.len()];

        // For each stage, compute in which stage its input was buffered (the previous InOut
        // stage). Also, compute for each InOut stage and channel the border with which the stage
        // output is used; this will used to allocate buffers of the correct size.
        for (i, stage) in shared.stages.iter().enumerate() {
            stage_input_buffer_index.push(previous_inout.clone());
            if let Stage::Process(p) = stage
                && p.stage_type() == RenderPipelineStageType::InOut
            {
                for chan in 0..nc {
                    if !p.uses_channel(chan) {
                        continue;
                    }
                    if let Some(prev) = previous_inout[chan] {
                        used_with_border[prev][chan] = p.border().1;
                    }
                    previous_inout[chan] = Some(i);
                }
            }
        }

        // Allocate buffers.
        for (i, stage) in shared.stages.iter().enumerate() {
            let mut stage_buffers = vec![];
            if let Stage::Process(p) = stage
                && p.stage_type() == RenderPipelineStageType::InOut
            {
                for chan in 0..nc {
                    if !p.uses_channel(chan) {
                        continue;
                    }
                    stage_buffers.push(RowBuffer::new(
                        p.output_type().unwrap(),
                        used_with_border[i][chan] as usize,
                        p.shift().1 as usize,
                        shared.chunk_size >> shared.channel_info[i][chan].downsample.0,
                    )?);
                }
            }
            row_buffers.push(stage_buffers);
        }
        // Compute information to be used to compute sub-rects for "save" stages to operate on
        // rects.
        let mut save_buffer_info = vec![];
        'stage: for ((i, s), ci) in shared
            .stages
            .iter()
            .enumerate()
            .zip(shared.channel_info.iter())
        {
            let Stage::Save(s) = s else {
                continue;
            };
            for (c, ci) in ci.iter().enumerate() {
                if s.uses_channel(c) {
                    save_buffer_info.push(SaveStageBufferInfo {
                        stage_index: i,
                        buffer_index: s.output_buffer_index,
                        downsample: ci.downsample,
                    });
                    continue 'stage;
                }
            }
        }
        save_buffer_info.sort_by_key(|x| x.buffer_index);
        Ok(Self {
            shared,
            input_buffers,
            stage_input_buffer_index,
            row_buffers,
            padding_was_rendered: false,
            save_buffer_info,
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
            unimplemented!()
        }
        // First, render all groups that have made progress.
        // TODO(veluca): this could potentially be quadratic for huge images that receive a group
        // at a time. Take care of that.
        for (g, gi) in self.shared.group_chan_ready_passes.iter().enumerate() {
            let ready_passes = gi.iter().copied().min().unwrap();
            if self.input_buffers[g].completed_passes < ready_passes {
                let (gx, gy) = self.shared.group_position(g);
                let mut fully_ready_passes = usize::MAX;
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
                if self.input_buffers[g].completed_passes >= fully_ready_passes {
                    continue;
                }
                debug!(
                    "new ready passes for group {gx},{gy} ({} completed, \
                    {ready_passes} ready, {fully_ready_passes} ready including neighbours)",
                    self.input_buffers[g].completed_passes
                );

                // TODO: extract buffers and call render_buffer.
                todo!();

                self.input_buffers[g].completed_passes = fully_ready_passes;
            }
        }

        // Clear buffers that will not be used again.
        for g in 0..self.shared.group_chan_ready_passes.len() {
            let (gx, gy) = self.shared.group_position(g);
            let mut neigh_complete_passes = usize::MAX;
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
            if self.shared.num_passes >= neigh_complete_passes {
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

    fn box_stage<S: RenderPipelineStage>(stage: S) -> Box<dyn RunStage<Self::Buffer>> {
        Box::new(stage)
    }
}
