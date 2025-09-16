// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::any::Any;

use crate::{
    BLOCK_DIM,
    api::JxlOutputBuffer,
    error::Result,
    image::{Image, ImageDataType},
    render::internal::ChannelInfo,
    simd::round_up_size_to_two_cache_lines,
    util::{ShiftRightCeil, tracing_wrappers::*},
};

use super::{
    RenderPipeline, RenderPipelineExtendStage, RenderPipelineInOutStage,
    RenderPipelineInPlaceStage, RenderPipelineStage,
    internal::{RenderPipelineRunStage, RenderPipelineShared, RunStage, Stage},
};

mod save;

/// A RenderPipeline that waits for all input of a pass to be ready before doing any rendering, and
/// prioritizes simplicity over memory usage and computational efficiency.
/// Eventually meant to be used only for verification purposes.
pub struct SimpleRenderPipeline {
    shared: RenderPipelineShared<Image<f64>>,
    input_buffers: Vec<Image<f64>>,
}

fn clone_images<T: ImageDataType>(images: &[Image<T>]) -> Result<Vec<Image<T>>> {
    images.iter().map(|x| x.as_rect().to_image()).collect()
}

impl RenderPipeline for SimpleRenderPipeline {
    type Buffer = Image<f64>;

    fn new_from_shared(shared: RenderPipelineShared<Self::Buffer>) -> Result<Self> {
        let input_buffers = shared.channel_info[0]
            .iter()
            .map(|x| {
                let xsize = shared.input_size.0.shrc(x.downsample.0);
                let ysize = shared.input_size.1.shrc(x.downsample.1);
                Image::new((xsize, ysize))
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            shared,
            input_buffers,
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
        let goffset = self.shared.group_offset(group_id);
        let ChannelInfo { ty, downsample } = self.shared.channel_info[0][channel];
        let off = (goffset.0 >> downsample.0, goffset.1 >> downsample.1);
        debug!(?sz, input_buffers_sz=?self.input_buffers[channel].size(), offset=?off, ?downsample, ?goffset);
        let bsz = buf.size();
        assert!(sz.0 <= bsz.0);
        assert!(sz.1 <= bsz.1);
        assert!(sz.0 + BLOCK_DIM > bsz.0);
        assert!(sz.1 + BLOCK_DIM > bsz.1);
        let ty = ty.unwrap();
        assert_eq!(ty, T::DATA_TYPE_ID);
        for y in 0..sz.1 {
            for x in 0..sz.0 {
                self.input_buffers[channel].as_rect_mut().row(y + off.1)[x + off.0] =
                    buf.as_rect().row(y)[x].to_f64();
            }
        }
        self.shared.group_chan_ready_passes[group_id][channel] += num_passes;
    }

    #[instrument(skip_all, err)]
    fn do_render(&mut self, buffers: &mut [Option<JxlOutputBuffer>]) -> Result<()> {
        let ready_passes = self
            .shared
            .group_chan_ready_passes
            .iter()
            .flat_map(|x| x.iter())
            .copied()
            .min()
            .unwrap();
        if ready_passes <= self.shared.completed_passes {
            debug!(
                "no more ready passes ({} completed, {ready_passes} ready)",
                self.shared.completed_passes
            );
            return Ok(());
        }
        debug!(
            "new ready passes ({} completed, {ready_passes} ready)",
            self.shared.completed_passes
        );

        let mut current_buffers = clone_images(&self.input_buffers)?;

        let mut current_size = self.shared.input_size;

        for (i, stage) in self.shared.stages.iter().enumerate() {
            debug!("running stage {i}: {stage}");
            let mut output_buffers = clone_images(&current_buffers)?;
            match stage {
                Stage::Process(stage) => {
                    // Replace buffers of different sizes.
                    if stage.shift() != (0, 0) || stage.new_size(current_size) != current_size {
                        current_size = stage.new_size(current_size);
                        for (c, info) in self.shared.channel_info[i + 1].iter().enumerate() {
                            if stage.uses_channel(c) {
                                let xsize = current_size.0.shrc(info.downsample.0);
                                let ysize = current_size.1.shrc(info.downsample.1);
                                debug!("reallocating channel {c} to new size {xsize}x{ysize}");
                                output_buffers[c] = Image::new((xsize, ysize))?;
                            }
                        }
                    }
                    let input_buf: Vec<_> = current_buffers
                        .iter()
                        .enumerate()
                        .filter(|x| stage.uses_channel(x.0))
                        .map(|x| x.1)
                        .collect();
                    let mut output_buf: Vec<_> = output_buffers
                        .iter_mut()
                        .enumerate()
                        .filter(|x| stage.uses_channel(x.0))
                        .map(|x| x.1)
                        .collect();
                    let mut state = stage.init_local_state()?;
                    stage.run_stage_on(
                        self.shared.chunk_size,
                        &input_buf,
                        &mut output_buf,
                        state.as_deref_mut(),
                    );
                }
                Stage::Save(stage) => {
                    stage.save_simple(&output_buffers, buffers)?;
                }
            }
            current_buffers = output_buffers;
        }

        self.shared.completed_passes = ready_passes;

        Ok(())
    }

    fn num_groups(&self) -> usize {
        self.shared.num_groups()
    }

    fn box_stage<S: RenderPipelineStage>(stage: S) -> Box<dyn RunStage<Self::Buffer>> {
        Box::new(stage)
    }
}

impl<T: ImageDataType> RenderPipelineRunStage<Image<f64>> for RenderPipelineInPlaceStage<T> {
    #[instrument(skip_all)]
    fn run_stage_on<S: RenderPipelineStage<Type = Self>>(
        stage: &S,
        chunk_size: usize,
        input_buffers: &[&Image<f64>],
        output_buffers: &mut [&mut Image<f64>],
        mut state: Option<&mut dyn Any>,
    ) {
        debug!("running inplace stage '{stage}' in simple pipeline");
        let numc = input_buffers.len();
        if numc == 0 {
            return;
        }
        assert_eq!(output_buffers.len(), numc);
        let size = input_buffers[0].size();
        for b in input_buffers.iter() {
            assert_eq!(size, b.size());
        }
        for b in output_buffers.iter() {
            assert_eq!(size, b.size());
        }
        let mut buffer =
            vec![vec![T::default(); round_up_size_to_two_cache_lines::<T>(chunk_size)]; numc];
        for y in 0..size.1 {
            for x in (0..size.0).step_by(chunk_size) {
                let xsize = size.0.min(x + chunk_size) - x;
                debug!("position: {x}x{y} xsize: {xsize}");
                for c in 0..numc {
                    let in_rect = input_buffers[c].as_rect();
                    let in_row = in_rect.row(y);
                    for ix in 0..xsize {
                        buffer[c][ix] = T::from_f64(in_row[x + ix]);
                    }
                }
                let mut row: Vec<_> = buffer.iter_mut().map(|x| x as &mut [T]).collect();
                stage.process_row_chunk((x, y), xsize, &mut row, state.as_deref_mut());
                for c in 0..numc {
                    let mut out_rect = output_buffers[c].as_rect_mut();
                    let out_row = out_rect.row(y);
                    for ix in 0..xsize {
                        out_row[x + ix] = buffer[c][ix].to_f64();
                    }
                }
            }
        }
    }
}

impl<
    InputT: ImageDataType,
    OutputT: ImageDataType,
    const BORDER_X: u8,
    const BORDER_Y: u8,
    const SHIFT_X: u8,
    const SHIFT_Y: u8,
> RenderPipelineRunStage<Image<f64>>
    for RenderPipelineInOutStage<InputT, OutputT, BORDER_X, BORDER_Y, SHIFT_X, SHIFT_Y>
{
    #[instrument(skip_all)]
    fn run_stage_on<S: RenderPipelineStage<Type = Self>>(
        stage: &S,
        chunk_size: usize,
        input_buffers: &[&Image<f64>],
        output_buffers: &mut [&mut Image<f64>],
        mut state: Option<&mut dyn Any>,
    ) {
        assert_ne!(chunk_size, 0);
        debug!("running inout stage '{stage}' in simple pipeline");
        let numc = input_buffers.len();
        if numc == 0 {
            return;
        }
        assert_eq!(output_buffers.len(), numc);
        let input_size = input_buffers[0].size();
        let output_size = output_buffers[0].size();
        for c in 1..numc {
            assert_eq!(input_size, input_buffers[c].size());
            assert_eq!(output_size, output_buffers[c].size());
        }
        debug!(
            "input_size = {input_size:?} output_size = {output_size:?} SHIFT_X = {SHIFT_X} \
		SHIFT_Y = {SHIFT_Y} BORDER_X = {BORDER_X} BORDER_Y = {BORDER_Y} numc = {numc}"
        );
        assert_eq!(input_size.0, output_size.0.div_ceil(1 << SHIFT_X));
        assert_eq!(input_size.1, output_size.1.div_ceil(1 << SHIFT_Y));
        let mut buffer_in = vec![
            vec![
                vec![
                    InputT::default();
                    // Double rounding make sure that we always have enough buffer for reading a whole SIMD lane.
                    round_up_size_to_two_cache_lines::<OutputT>(
                        round_up_size_to_two_cache_lines::<OutputT>(chunk_size)
                            + BORDER_X as usize * 2
                    )
                ];
                BORDER_Y as usize * 2 + 1
            ];
            numc
        ];
        let mut buffer_out = vec![
            vec![
                vec![
                    OutputT::default();
                    round_up_size_to_two_cache_lines::<OutputT>(chunk_size)
                        << SHIFT_X
                ];
                1 << SHIFT_Y
            ];
            numc
        ];

        let mirror = |mut v: i64, size: i64| {
            while v < 0 || v >= size {
                if v < 0 {
                    v = -v - 1;
                }
                if v >= size {
                    v = size + (size - v) - 1;
                }
            }
            v as usize
        };
        for y in 0..input_size.1 {
            for x in (0..input_size.0).step_by(chunk_size) {
                let border_x = BORDER_X as i64;
                let border_y = BORDER_Y as i64;
                let xsize = input_size.0.min(x + chunk_size) - x;
                let xs = xsize as i64;
                debug!("position: {x}x{y} xsize: {xsize}");
                for c in 0..numc {
                    let in_rect = input_buffers[c].as_rect();
                    for iy in -border_y..=border_y {
                        let imgy = mirror(y as i64 + iy, input_size.1 as i64);
                        let in_row = in_rect.row(imgy);
                        let buf_in_row = &mut buffer_in[c][(iy + border_y) as usize];
                        for ix in (-border_x..0).chain(xs..xs + border_x) {
                            let imgx = mirror(x as i64 + ix, input_size.0 as i64);
                            buf_in_row[(ix + border_x) as usize] = InputT::from_f64(in_row[imgx]);
                        }
                        for ix in 0..xsize {
                            buf_in_row[ix + border_x as usize] = InputT::from_f64(in_row[x + ix]);
                        }
                    }
                }
                let buffer_in_ref: Vec<Vec<_>> = buffer_in
                    .iter()
                    .map(|x| x.iter().map(|x| x as &[_]).collect())
                    .collect();
                let mut buffer_out_ref: Vec<Vec<_>> = buffer_out
                    .iter_mut()
                    .map(|x| x.iter_mut().map(|x| x as &mut [_]).collect::<Vec<_>>())
                    .collect();
                let mut row: Vec<_> = buffer_in_ref
                    .iter()
                    .zip(buffer_out_ref.iter_mut())
                    .map(|(itin, itout)| (itin as &[_], itout as &mut [_]))
                    .collect();
                stage.process_row_chunk((x, y), xsize, &mut row, state.as_deref_mut());
                let stripe_xsize = (xsize << SHIFT_X).min(output_size.0 - (x << SHIFT_X));
                let stripe_ysize = (1usize << SHIFT_Y).min(output_size.1 - (y << SHIFT_Y));
                for c in 0..numc {
                    let mut out_rect = output_buffers[c].as_rect_mut();
                    for iy in 0..stripe_ysize {
                        let out_row = out_rect.row((y << SHIFT_Y) + iy);
                        for ix in 0..stripe_xsize {
                            out_row[(x << SHIFT_X) + ix] = buffer_out[c][iy][ix].to_f64();
                        }
                    }
                }
            }
        }
    }
}

impl<T: ImageDataType> RenderPipelineRunStage<Image<f64>> for RenderPipelineExtendStage<T> {
    #[instrument(skip_all)]
    fn run_stage_on<S: RenderPipelineStage<Type = Self>>(
        stage: &S,
        chunk_size: usize,
        input_buffers: &[&Image<f64>],
        output_buffers: &mut [&mut Image<f64>],
        mut state: Option<&mut dyn Any>,
    ) {
        debug!("running extend stage '{stage}' in simple pipeline");
        let numc = input_buffers.len();
        if numc == 0 {
            return;
        }
        assert_eq!(output_buffers.len(), numc);
        let input_size = input_buffers[0].size();
        let output_size = output_buffers[0].size();
        for c in 1..numc {
            assert_eq!(input_size, input_buffers[c].size());
            assert_eq!(output_size, output_buffers[c].size());
        }
        assert_eq!(output_size, stage.new_size(input_size));
        let origin = stage.original_data_origin();
        assert!(origin.0 <= output_size.0 as isize);
        assert!(origin.1 <= output_size.1 as isize);
        assert!(origin.0 + input_size.0 as isize >= 0);
        assert!(origin.1 + input_size.1 as isize >= 0);
        let origin = stage.original_data_origin();
        debug!("input_size = {input_size:?} output_size = {output_size:?} origin = {origin:?}");
        // Compute the input rectangle
        let x0 = origin.0.max(0) as usize;
        let y0 = origin.1.max(0) as usize;
        let x1 = (origin.0 + input_size.0 as isize).min(output_size.0 as isize) as usize;
        let y1 = (origin.1 + input_size.1 as isize).min(output_size.1 as isize) as usize;
        debug!("x0 = {x0} x1 = {x1} y0 = {y0} y1 = {y1}");
        let in_x0 = (x0 as isize - origin.0) as usize;
        let in_x1 = (x1 as isize - origin.0) as usize;
        let in_y0 = (y0 as isize - origin.1) as usize;
        let in_y1 = (y1 as isize - origin.1) as usize;
        debug!("in_x0 = {in_x0} in_x1 = {in_x1} in_y0 = {in_y0} in_y1 = {in_y1}");
        // First, copy the data in the middle.
        for c in 0..numc {
            for in_y in in_y0..in_y1 {
                debug!("copy row: {in_y}");
                let in_row = input_buffers[c].as_rect().row(in_y);
                let y = (in_y as isize + origin.1) as usize;
                output_buffers[c].as_rect_mut().row(y)[x0..x1]
                    .copy_from_slice(&in_row[in_x0..in_x1]);
            }
        }
        // Fill in rows above and below the original data.
        let mut buffer =
            vec![vec![T::default(); round_up_size_to_two_cache_lines::<T>(chunk_size)]; numc];
        for y in (0..y0).chain(y1..output_size.1) {
            for x in (0..output_size.0).step_by(chunk_size) {
                let xsize = output_size.0.min(x + chunk_size) - x;
                debug!("position above/below: ({x},{y}) xsize: {xsize}");
                let mut row: Vec<_> = buffer.iter_mut().map(|x| x as &mut [T]).collect();
                stage.process_row_chunk((x, y), xsize, &mut row, state.as_deref_mut());
                for c in 0..numc {
                    for ix in 0..xsize {
                        output_buffers[c].as_rect_mut().row(y)[x + ix] = buffer[c][ix].to_f64();
                    }
                }
            }
        }
        // Fill in left and right of the original data.
        for y in y0..y1 {
            for (x, xsize) in (0..x0)
                .step_by(chunk_size)
                .map(|x| (x, x0.min(x + chunk_size) - x))
                .chain(
                    (x1..output_size.0)
                        .step_by(chunk_size)
                        .map(|x| (x, output_size.0.min(x + chunk_size) - x)),
                )
            {
                let mut row: Vec<_> = buffer.iter_mut().map(|x| x as &mut [T]).collect();
                debug!("position on the side: ({x},{y}) xsize: {xsize}");
                stage.process_row_chunk((x, y), xsize, &mut row, state.as_deref_mut());
                for c in 0..numc {
                    for ix in 0..xsize {
                        output_buffers[c].as_rect_mut().row(y)[x + ix] = buffer[c][ix].to_f64();
                    }
                }
            }
        }
    }
}
