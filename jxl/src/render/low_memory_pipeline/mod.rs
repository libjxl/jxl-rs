// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::any::Any;

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

mod helpers;
mod render_group;
pub(super) mod row_buffers;
mod run_stage;
mod save;

struct InputBuffer {
    // One buffer per channel.
    data: Vec<Option<Box<dyn Any>>>,
    completed_passes: usize,
}

struct SaveStageBufferInfo {
    buffer_index: usize,
    downsample: (u8, u8),
    orientation: Orientation,
    byte_size: usize,
    after_extend: bool,
}

pub struct LowMemoryRenderPipeline {
    shared: RenderPipelineShared<RowBuffer>,
    input_buffers: Vec<InputBuffer>,
    row_buffers: Vec<Vec<RowBuffer>>,
    // The input buffer that each channel of each stage should use.
    // This is indexed both by stage index (0 corresponds to input data, 1 to stage[0], etc) and by
    // channel index (as only used channels have a buffer).
    stage_input_buffer_index: Vec<Vec<(usize, usize)>>,
    // Tracks whether we already rendered the padding around the core frame (if any).
    padding_was_rendered: bool,
    // sorted by buffer_index; all values of buffer_index are distinct.
    save_buffer_info: Vec<SaveStageBufferInfo>,
    // The amount of pixels that each stage needs to *output* around the current group to
    // run future stages correctly.
    stage_output_border_pixels: Vec<(usize, usize)>,
    // The amount of pixels that we need to read (for every channel) in non-edge groups to run all
    // stages correctly.
    input_border_pixels: Vec<(usize, usize)>,
    has_nontrivial_border: bool,
    // For every stage, the downsampling level of *any* channel that the stage uses at that point.
    // Note that this must be equal across all the used channels.
    downsampling_for_stage: Vec<(usize, usize)>,
    // Local states of each stage, if any.
    local_states: Vec<Option<Box<dyn Any>>>,
}

fn extract_local_buffers<'a>(
    buffers: &'a mut [Option<JxlOutputBuffer>],
    save_buffer_info: &[SaveStageBufferInfo],
    get_rect: impl Fn(&SaveStageBufferInfo) -> Option<Rect>,
    frame_size: (usize, usize),
    full_image_size: (usize, usize),
    frame_origin: (isize, isize),
) -> Result<Vec<Option<JxlOutputBuffer<'a>>>> {
    let mut local_buffers = vec![];
    local_buffers.try_reserve(buffers.len())?;
    for _ in 0..buffers.len() {
        local_buffers.push(None::<JxlOutputBuffer>);
    }
    let mut buffer_iter = buffers.iter_mut().enumerate();
    for bi in save_buffer_info.iter() {
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
        let Some(Rect { origin, size }) = get_rect(&bi) else {
            continue;
        };
        let frame_size = (
            frame_size.0.shrc(bi.downsample.0),
            frame_size.1.shrc(bi.downsample.1),
        );
        let size = (
            (size.0 + origin.0).min(frame_size.0) - origin.0,
            (size.1 + origin.1).min(frame_size.1) - origin.1,
        );
        let mut rect = Rect { size, origin };
        if bi.after_extend {
            // clip this rect to its visible area in the full image (in full image coordinates).
            let origin = (
                rect.origin.0 as isize + frame_origin.0,
                rect.origin.1 as isize + frame_origin.1,
            );
            let end = (
                origin.0 + rect.size.0 as isize,
                origin.1 + rect.size.1 as isize,
            );
            let origin = (origin.0.max(0) as usize, origin.1.max(0) as usize);
            let end = (
                end.0.min(full_image_size.0 as isize).max(0) as usize,
                end.1.min(full_image_size.1 as isize).max(0) as usize,
            );
            if origin.0 >= end.0 || origin.1 >= end.1 {
                // rect would be empty
                continue;
            }
            rect = Rect {
                origin,
                size: (end.0 - origin.0, end.1 - origin.1),
            };
        }
        let rect = bi.orientation.display_rect(rect, full_image_size);
        let rect = Rect {
            origin: (rect.origin.0 * bi.byte_size, rect.origin.1),
            size: (rect.size.0 * bi.byte_size, rect.size.1),
        };
        local_buffers[bi.buffer_index] = Some(buf.subrect(rect));
    }
    Ok(local_buffers)
}

impl LowMemoryRenderPipeline {
    fn render_padding(&mut self, buffers: &mut [Option<JxlOutputBuffer>]) -> Result<()> {
        // TODO(veluca): consider pre-computing those strips at pipeline construction and making
        // smaller strips.
        let e = self.shared.extend_stage_index.unwrap();
        let Stage::Extend(e) = &self.shared.stages[e] else {
            unreachable!("extend stage is not an extend stage");
        };
        // Split the full image area in 4 strips: left and right of the frame, and above and below.
        // We divide each part further in strips of width self.shared.chunk_size.
        let mut strips = vec![];
        if e.frame_origin.0 > 0 {
            let xend = e.frame_origin.0 as usize;
            for x in (0..xend).step_by(self.shared.chunk_size) {
                let xe = (x + self.shared.chunk_size).min(xend);
                strips.push((x..xe, 0..e.image_size.1));
            }
        }
        if e.frame_origin.1 > 0 {
            let xstart = e.frame_origin.0.max(0) as usize;
            let xend = (e.frame_origin.0 + self.shared.input_size.0 as isize) as usize;
            for x in (xstart..xend).step_by(self.shared.chunk_size) {
                let xe = (x + self.shared.chunk_size).min(xend);
                strips.push((x..xe, 0..e.frame_origin.1 as usize));
            }
        }
        if e.frame_origin.1 + (self.shared.input_size.1 as isize) < e.image_size.1 as isize {
            let ystart = (e.frame_origin.1 + (self.shared.input_size.1 as isize)).max(0) as usize;
            let yend = e.image_size.1;
            let xstart = e.frame_origin.0.max(0) as usize;
            let xend = (e.frame_origin.0 + self.shared.input_size.0 as isize) as usize;
            for x in (xstart..xend).step_by(self.shared.chunk_size) {
                let xe = (x + self.shared.chunk_size).min(xend);
                strips.push((x..xe, ystart..yend));
            }
        }
        if e.frame_origin.0 + (self.shared.input_size.0 as isize) < e.image_size.0 as isize {
            let xstart = (e.frame_origin.0 + (self.shared.input_size.0 as isize)).max(0) as usize;
            let xend = e.image_size.0;
            for x in (xstart..xend).step_by(self.shared.chunk_size) {
                let xe = (x + self.shared.chunk_size).min(xend);
                strips.push((x..xe, 0..e.image_size.1));
            }
        }
        let full_image_size = e.image_size;
        for (xrange, yrange) in strips {
            let mut local_buffers = extract_local_buffers(
                buffers,
                &self.save_buffer_info,
                |bi| {
                    if bi.after_extend {
                        assert_eq!(bi.downsample, (0, 0));
                        Some(Rect {
                            origin: (xrange.start, yrange.start),
                            size: (xrange.clone().count(), yrange.clone().count()),
                        })
                    } else {
                        None
                    }
                },
                full_image_size,
                full_image_size,
                (0, 0),
            )?;
            self.render_outside_frame(xrange, yrange, &mut local_buffers)?;
        }
        Ok(())
    }
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
        let mut previous_inout: Vec<_> = (0..nc).map(|x| (0usize, x)).collect();
        let mut stage_input_buffer_index = vec![];
        let mut next_border_and_cur_downsample = vec![vec![]];

        for ci in shared.channel_info[0].iter() {
            next_border_and_cur_downsample[0].push((0, ci.downsample));
        }

        // For each stage, compute in which stage its input was buffered (the previous InOut
        // stage). Also, compute for each InOut stage and channel the border with which the stage
        // output is used; this will used to allocate buffers of the correct size.
        for (i, stage) in shared.stages.iter().enumerate() {
            stage_input_buffer_index.push(previous_inout.clone());
            next_border_and_cur_downsample.push(vec![]);
            if let Stage::InOut(p) = stage {
                for chan in 0..nc {
                    if !p.uses_channel(chan) {
                        continue;
                    }
                    let (ps, pc) = previous_inout[chan];
                    next_border_and_cur_downsample[ps][pc].0 = p.border().1;
                    previous_inout[chan] = (i + 1, next_border_and_cur_downsample[i + 1].len());
                    next_border_and_cur_downsample[i + 1]
                        .push((0, shared.channel_info[i + 1][chan].downsample));
                }
            }
        }

        let mut initial_buffers = vec![];
        for chan in 0..nc {
            initial_buffers.push(RowBuffer::new(
                shared.channel_info[0][chan].ty.unwrap(),
                next_border_and_cur_downsample[0][chan].0 as usize,
                0,
                shared.chunk_size >> shared.channel_info[0][chan].downsample.0,
            )?);
        }
        let mut row_buffers = vec![initial_buffers];

        // Allocate buffers.
        for (i, stage) in shared.stages.iter().enumerate() {
            let mut stage_buffers = vec![];
            for (next_y_border, (dsx, _)) in next_border_and_cur_downsample[i + 1].iter() {
                stage_buffers.push(RowBuffer::new(
                    stage.output_type().unwrap(),
                    *next_y_border as usize,
                    stage.shift().1 as usize,
                    shared.chunk_size >> *dsx,
                )?);
            }
            row_buffers.push(stage_buffers);
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
                    save_buffer_info.push(SaveStageBufferInfo {
                        buffer_index: s.output_buffer_index,
                        downsample: ci.downsample,
                        orientation: s.orientation,
                        byte_size: s.data_format.bytes_per_sample() * s.channels.len(),
                        after_extend: shared.extend_stage_index.is_some_and(|e| i > e),
                    });
                    continue 'stage;
                }
            }
        }
        save_buffer_info.sort_by_key(|x| x.buffer_index);

        // Compute the amount of border pixels needed per channel, per stage.
        let mut border_pixels = vec![(0usize, 0usize); nc];
        let mut border_pixels_per_stage = vec![];
        for s in shared.stages.iter().rev() {
            let mut stage_max = (0, 0);
            for c in 0..nc {
                if !s.uses_channel(c) {
                    continue;
                }
                stage_max.0 = stage_max.0.max(border_pixels[c].0);
                stage_max.1 = stage_max.1.max(border_pixels[c].1);

                border_pixels[c].0 = border_pixels[c].0.shrc(s.shift().0) + s.border().0 as usize;
                border_pixels[c].1 = border_pixels[c].1.shrc(s.shift().1) + s.border().1 as usize;
            }
            border_pixels_per_stage.push(stage_max);
        }
        border_pixels_per_stage.reverse();

        for c in 0..nc {
            let (bx, _) = border_pixels_per_stage[0];
            assert!(bx * shared.channel_info[0][c].ty.unwrap().size() <= 2 * CACHE_LINE_BYTE_SIZE);
        }

        let downsampling_for_stage = shared
            .stages
            .iter()
            .zip(shared.channel_info.iter())
            .map(|(s, ci)| {
                let dowsamplings: Vec<_> = (0..nc)
                    .filter_map(|c| {
                        if s.uses_channel(c) {
                            Some(ci[c].downsample)
                        } else {
                            None
                        }
                    })
                    .collect();
                for &d in dowsamplings.iter() {
                    assert_eq!(d, dowsamplings[0]);
                }
                (dowsamplings[0].0 as usize, dowsamplings[0].1 as usize)
            })
            .collect();

        Ok(Self {
            input_buffers,
            stage_input_buffer_index,
            row_buffers,
            padding_was_rendered: false,
            save_buffer_info,
            stage_output_border_pixels: border_pixels_per_stage,
            has_nontrivial_border: border_pixels.iter().any(|x| *x != (0, 0)),
            input_border_pixels: border_pixels,
            local_states: shared
                .stages
                .iter()
                .map(|x| x.init_local_state())
                .collect::<Result<_>>()?,
            shared,
            downsampling_for_stage,
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
        // Check that buffer sizes are correct.
        {
            let mut size = self.shared.input_size;
            for (i, s) in self.shared.stages.iter().enumerate() {
                match s {
                    Stage::Extend(e) => size = e.image_size,
                    Stage::Save(s) => {
                        let (dx, dy) = self.downsampling_for_stage[i];
                        s.check_buffer_size(
                            (size.0 >> dx, size.1 >> dy),
                            buffers[s.output_buffer_index].as_ref(),
                        )?
                    }
                    _ => {}
                }
            }
        }

        if self.shared.extend_stage_index.is_some() && !self.padding_was_rendered {
            self.padding_was_rendered = true;
            self.render_padding(buffers)?;
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
                // Here we assume that we never need more than one group worth of border.
                if self.has_nontrivial_border {
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
                let (origin, size) = if let Some(e) = self.shared.extend_stage_index {
                    let Stage::Extend(e) = &self.shared.stages[e] else {
                        unreachable!("extend stage is not an extend stage");
                    };
                    (e.frame_origin, e.image_size)
                } else {
                    ((0, 0), self.shared.input_size)
                };
                let mut local_buffers = extract_local_buffers(
                    buffers,
                    &self.save_buffer_info,
                    |bi| {
                        let size = (
                            1 << (self.shared.log_group_size - bi.downsample.0 as usize),
                            1 << (self.shared.log_group_size - bi.downsample.1 as usize),
                        );
                        let origin = (size.0 * gx, size.1 * gy);
                        Some(Rect { origin, size })
                    },
                    self.shared.input_size,
                    size,
                    origin,
                )?;

                self.render_group((gx, gy), &mut local_buffers)?;

                self.input_buffers[g].completed_passes = fully_ready_passes;
            }
        }

        // Clear buffers that will not be used again.
        for g in 0..self.shared.group_chan_ready_passes.len() {
            let (gx, gy) = self.shared.group_position(g);
            let mut neigh_complete_passes = self.input_buffers[g].completed_passes;
            if self.has_nontrivial_border {
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
