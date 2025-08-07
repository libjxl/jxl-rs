// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::any::Any;

use crate::{
    error::{Error, Result},
    image::{DataTypeTag, Image, ImageDataType},
    render::internal::RenderPipelineStageInfo,
    util::ShiftRightCeil,
    util::tracing_wrappers::*,
};

use super::{
    RenderPipeline, RenderPipelineBuilder, RenderPipelineExtendStage, RenderPipelineInOutStage,
    RenderPipelineInPlaceStage, RenderPipelineInspectStage, RenderPipelineStage,
    internal::RenderPipelineStageType,
};

#[derive(Clone, Debug)]
struct ChannelInfo {
    ty: Option<DataTypeTag>,
    downsample: (u8, u8),
}

pub struct SimpleRenderPipelineBuilder {
    pipeline: SimpleRenderPipeline,
    can_shift: bool,
}

impl SimpleRenderPipelineBuilder {
    #[instrument(level = "debug")]
    pub(super) fn new_with_chunk_size(
        num_channels: usize,
        size: (usize, usize),
        downsampling_shift: usize,
        mut log_group_size: usize,
        chunk_size: usize,
    ) -> Self {
        info!("creating simple pipeline");
        assert!(chunk_size <= u16::MAX as usize);
        assert_ne!(chunk_size, 0);
        // The number of pixels that a group encompasses in the final, upsampled image along one dimension is effectively multiplied by the upsampling factor.
        log_group_size += downsampling_shift;
        SimpleRenderPipelineBuilder {
            pipeline: SimpleRenderPipeline {
                channel_info: vec![vec![
                    ChannelInfo {
                        ty: None,
                        downsample: (0, 0)
                    };
                    num_channels
                ]],
                input_size: size,
                log_group_size,
                xgroups: size.0.shrc(log_group_size),
                stages: vec![],
                group_chan_ready_passes: vec![
                    vec![0; num_channels];
                    size.0.shrc(log_group_size)
                        * size.1.shrc(log_group_size)
                ],
                completed_passes: 0,
                input_buffers: vec![],
                chunk_size,
            },
            can_shift: true,
        }
    }
}

impl RenderPipelineBuilder for SimpleRenderPipelineBuilder {
    type RenderPipeline = SimpleRenderPipeline;

    fn new(
        num_channels: usize,
        size: (usize, usize),
        downsampling_shift: usize,
        log_group_size: usize,
    ) -> Self {
        Self::new_with_chunk_size(num_channels, size, downsampling_shift, log_group_size, 256)
    }

    #[instrument(skip_all, err)]
    fn add_stage<Stage: RenderPipelineStage>(mut self, stage: Stage) -> Result<Self> {
        let current_info = self.pipeline.channel_info.last().unwrap().clone();
        debug!(
            last_stage_channel_info = ?current_info,
            can_shift = self.can_shift,
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
                if let Some(ty) = info.ty
                    && ty != Stage::Type::INPUT_TYPE
                {
                    return Err(Error::PipelineChannelTypeMismatch(
                        stage.to_string(),
                        c,
                        Stage::Type::INPUT_TYPE,
                        ty,
                    ));
                }
                after_info.push(ChannelInfo {
                    ty: Some(Stage::Type::OUTPUT_TYPE.unwrap_or(Stage::Type::INPUT_TYPE)),
                    downsample: Stage::Type::SHIFT,
                });
            }
        }
        if !self.can_shift && Stage::Type::SHIFT != (0, 0) {
            return Err(Error::PipelineShiftAfterExpand(stage.to_string()));
        }
        if Stage::Type::TYPE == RenderPipelineStageType::Extend {
            self.can_shift = false;
        }
        debug!(
            new_channel_info = ?after_info,
            can_shift = self.can_shift,
            "added stage '{stage}'",
        );
        self.pipeline.channel_info.push(after_info);
        self.pipeline.stages.push(Box::new(stage));
        Ok(self)
    }

    #[instrument(skip_all, err)]
    fn build(mut self) -> Result<Self::RenderPipeline> {
        let channel_info = &mut self.pipeline.channel_info;
        let num_channels = channel_info[0].len();
        let mut cur_downsamples = vec![(0u8, 0u8); num_channels];
        for (s, stage) in self.pipeline.stages.iter().enumerate().rev() {
            let [current_info, next_info, ..] = &mut channel_info[s..] else {
                unreachable!()
            };
            for chan in 0..num_channels {
                let cur_chan = &mut current_info[chan];
                let next_chan = &mut next_info[chan];
                if stage.uses_channel(chan) {
                    assert_eq!(Some(stage.output_type()), next_chan.ty);
                }
                if cur_chan.ty.is_none() {
                    cur_chan.ty = if stage.uses_channel(chan) {
                        Some(stage.input_type())
                    } else {
                        next_chan.ty
                    }
                }
                // Arithmetic overflows here should be very uncommon, so custom error variants
                // are probably unwarranted.
                let cur_downsample = &mut cur_downsamples[chan];
                let next_downsample = &mut next_chan.downsample;
                let next_total_downsample = *cur_downsample;
                cur_downsample.0 = cur_downsample
                    .0
                    .checked_add(next_downsample.0)
                    .ok_or(Error::ArithmeticOverflow)?;
                cur_downsample.1 = cur_downsample
                    .1
                    .checked_add(next_downsample.1)
                    .ok_or(Error::ArithmeticOverflow)?;
                *next_downsample = next_total_downsample;
            }
        }
        for (chan, cur_downsample) in cur_downsamples.iter().enumerate() {
            channel_info[0][chan].downsample = *cur_downsample;
        }
        #[cfg(feature = "tracing")]
        {
            for (s, (current_info, stage)) in channel_info
                .iter()
                .zip(self.pipeline.stages.iter())
                .enumerate()
            {
                debug!("final channel info before stage {s} '{stage}': {current_info:?}");
            }
            debug!(
                "final channel info after all stages {:?}",
                channel_info.last().unwrap()
            );
        }

        // Ensure all channels have been used, so that we know the types of all buffers at all
        // stages.
        for (c, chinfo) in channel_info.iter().flat_map(|x| x.iter().enumerate()) {
            if chinfo.ty.is_none() {
                return Err(Error::PipelineChannelUnused(c));
            }
        }

        let input_buffers: Result<_> = channel_info[0]
            .iter()
            .map(|x| {
                let xsize = self.pipeline.input_size.0.shrc(x.downsample.0);
                let ysize = self.pipeline.input_size.1.shrc(x.downsample.1);
                Image::new((xsize, ysize))
            })
            .collect();
        self.pipeline.input_buffers = input_buffers?;

        Ok(self.pipeline)
    }
}

/// A RenderPipeline that waits for all input of a pass to be ready before doing any rendering, and
/// prioritizes simplicity over memory usage and computational efficiency.
/// Eventually meant to be used only for verification purposes.
pub struct SimpleRenderPipeline {
    channel_info: Vec<Vec<ChannelInfo>>,
    input_size: (usize, usize),
    log_group_size: usize,
    xgroups: usize,
    stages: Vec<Box<dyn RunStage>>,
    group_chan_ready_passes: Vec<Vec<usize>>,
    completed_passes: usize,
    input_buffers: Vec<Image<f64>>,
    chunk_size: usize,
}

fn clone_images<T: ImageDataType>(images: &[Image<T>]) -> Result<Vec<Image<T>>> {
    images.iter().map(|x| x.as_rect().to_image()).collect()
}

impl SimpleRenderPipeline {
    #[instrument(skip_all, err)]
    fn do_render(&mut self) -> Result<()> {
        let ready_passes = self
            .group_chan_ready_passes
            .iter()
            .flat_map(|x| x.iter())
            .copied()
            .min()
            .unwrap();
        if ready_passes <= self.completed_passes {
            debug!(
                "no more ready passes ({} completed, {ready_passes} ready)",
                self.completed_passes
            );
            return Ok(());
        }
        debug!(
            "new ready passes ({} completed, {ready_passes} ready)",
            self.completed_passes
        );
        self.completed_passes = ready_passes;

        let mut current_buffers = clone_images(&self.input_buffers)?;

        let mut current_size = self.input_size;

        for (i, stage) in self.stages.iter().enumerate() {
            debug!("running stage {i}: {stage}");
            let mut output_buffers = clone_images(&current_buffers)?;
            // Replace buffers of different sizes.
            if stage.shift() != (0, 0) || stage.new_size(current_size) != current_size {
                current_size = stage.new_size(current_size);
                for (c, info) in self.channel_info[i + 1].iter().enumerate() {
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
                self.chunk_size,
                &input_buf,
                &mut output_buf,
                state.as_deref_mut(),
            );
            current_buffers = output_buffers;
        }

        Ok(())
    }
}

impl RenderPipeline for SimpleRenderPipeline {
    type Builder = SimpleRenderPipelineBuilder;

    #[instrument(skip_all, err)]
    fn fill_input_channels<T: ImageDataType>(
        &mut self,
        channels: &[usize],
        group_id: usize,
        num_filled_passes: usize,
        fill_fn: impl FnOnce(&mut [crate::image::ImageRectMut<T>]) -> Result<()>,
    ) -> Result<()> {
        debug!(
            "filling data for group {}, channels {:?}, using type {:?}",
            group_id,
            channels,
            T::DATA_TYPE_ID,
        );
        let group = (group_id % self.xgroups, group_id / self.xgroups);
        let goffset = (
            group.0 << self.log_group_size,
            group.1 << self.log_group_size,
        );
        let gsize = (
            self.input_size
                .0
                .min(goffset.0 + (1 << self.log_group_size))
                - goffset.0,
            self.input_size
                .1
                .min(goffset.1 + (1 << self.log_group_size))
                - goffset.1,
        );
        debug!(?gsize, ?group, ?group_id);
        let mut images = vec![];
        for c in channels.iter().copied() {
            let ChannelInfo { ty, downsample } = self.channel_info[0][c];
            let ty = ty.unwrap();
            assert_eq!(goffset.0 % (1 << downsample.0), 0);
            assert_eq!(goffset.1 % (1 << downsample.1), 0);
            if ty == T::DATA_TYPE_ID {
                images.push(Image::<T>::new((
                    gsize.0.shrc(downsample.0),
                    gsize.1.shrc(downsample.1),
                ))?);
            } else {
                panic!(
                    "Invalid pipeline usage: incorrect channel type, expected {:?}, found {ty:?}",
                    T::DATA_TYPE_ID
                );
            }
        }
        let mut images: Vec<_> = images.iter_mut().map(|x| x.as_rect_mut()).collect();
        fill_fn(&mut images)?;
        for (i, c) in channels.iter().copied().enumerate() {
            let ChannelInfo { ty, downsample } = self.channel_info[0][c];
            let ty = ty.unwrap();
            let off = (goffset.0 >> downsample.0, goffset.1 >> downsample.1);
            assert_eq!(ty, T::DATA_TYPE_ID);
            for y in 0..gsize.1.shrc(downsample.1) {
                for x in 0..gsize.0.shrc(downsample.0) {
                    self.input_buffers[c].as_rect_mut().row(y + off.1)[x + off.0] =
                        images[i].as_rect().row(y)[x].to_f64();
                }
            }
            self.group_chan_ready_passes[group_id][c] += num_filled_passes;
        }
        self.do_render()
    }

    fn into_stages(self) -> Vec<Box<dyn std::any::Any>> {
        self.stages.into_iter().map(|x| x.as_any()).collect()
    }

    fn num_groups(&self) -> usize {
        self.xgroups * self.input_size.1.shrc(self.log_group_size)
    }
}

pub trait RenderPipelineRunStage {
    fn run_stage_on<S: RenderPipelineStage<Type = Self>>(
        stage: &S,
        chunk_size: usize,
        input_buffers: &[&Image<f64>],
        output_buffers: &mut [&mut Image<f64>],
        state: Option<&mut dyn Any>,
    );
}

impl<T: ImageDataType> RenderPipelineRunStage for RenderPipelineInspectStage<T> {
    #[instrument(skip_all)]
    fn run_stage_on<S: RenderPipelineStage<Type = Self>>(
        stage: &S,
        chunk_size: usize,
        input_buffers: &[&Image<f64>],
        _output_buffers: &mut [&mut Image<f64>],
        mut state: Option<&mut dyn Any>,
    ) {
        debug!("running input stage '{stage}' in simple pipeline");
        let numc = input_buffers.len();
        if numc == 0 {
            return;
        }
        let size = input_buffers[0].size();
        for b in input_buffers.iter() {
            assert_eq!(size, b.size());
        }
        let mut buffer = vec![vec![T::default(); chunk_size]; numc];
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
                let mut row: Vec<_> = buffer.iter().map(|x| x as &[T]).collect();
                stage.process_row_chunk((x, y), xsize, &mut row, state.as_deref_mut());
            }
        }
    }
}

impl<T: ImageDataType> RenderPipelineRunStage for RenderPipelineInPlaceStage<T> {
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
        let mut buffer = vec![vec![T::default(); chunk_size]; numc];
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
> RenderPipelineRunStage
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
                vec![InputT::default(); chunk_size + BORDER_X as usize * 2];
                BORDER_Y as usize * 2 + 1
            ];
            numc
        ];
        let mut buffer_out =
            vec![vec![vec![OutputT::default(); chunk_size << SHIFT_X]; 1 << SHIFT_Y]; numc];

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

impl<T: ImageDataType> RenderPipelineRunStage for RenderPipelineExtendStage<T> {
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
        let mut buffer = vec![vec![T::default(); chunk_size]; numc];
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

trait RunStage: Any + std::fmt::Display {
    fn run_stage_on(
        &self,
        chunk_size: usize,
        input_buffers: &[&Image<f64>],
        output_buffers: &mut [&mut Image<f64>],
        state: Option<&mut dyn Any>,
    );
    fn init_local_state(&self) -> Result<Option<Box<dyn Any>>>;
    fn shift(&self) -> (u8, u8);
    fn new_size(&self, size: (usize, usize)) -> (usize, usize);
    fn uses_channel(&self, c: usize) -> bool;
    fn as_any(self: Box<Self>) -> Box<dyn Any>;
    fn input_type(&self) -> DataTypeTag;
    fn output_type(&self) -> DataTypeTag;
}

impl<T: RenderPipelineStage> RunStage for T {
    fn run_stage_on(
        &self,
        chunk_size: usize,
        input_buffers: &[&Image<f64>],
        output_buffers: &mut [&mut Image<f64>],
        state: Option<&mut dyn Any>,
    ) {
        T::Type::run_stage_on(self, chunk_size, input_buffers, output_buffers, state)
    }

    fn init_local_state(&self) -> Result<Option<Box<dyn Any>>> {
        T::init_local_state(self)
    }

    fn shift(&self) -> (u8, u8) {
        T::Type::SHIFT
    }

    fn new_size(&self, size: (usize, usize)) -> (usize, usize) {
        self.new_size(size)
    }

    fn uses_channel(&self, c: usize) -> bool {
        self.uses_channel(c)
    }
    fn as_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }
    fn input_type(&self) -> DataTypeTag {
        T::Type::INPUT_TYPE
    }
    fn output_type(&self) -> DataTypeTag {
        T::Type::OUTPUT_TYPE.unwrap_or(T::Type::INPUT_TYPE)
    }
}
