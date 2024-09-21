// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    error::{Error, Result},
    image::{DataTypeTag, Image, ImageDataType},
    render::internal::RenderPipelineStageInfo,
    util::ShiftRightCeil,
};

use super::{
    internal::RenderPipelineStageType, RenderPipeline, RenderPipelineBuilder,
    RenderPipelineExtendStage, RenderPipelineInOutStage, RenderPipelineInPlaceStage,
    RenderPipelineInputStage, RenderPipelineStage,
};

#[derive(Clone)]
struct ChannelInfo {
    ty: Option<DataTypeTag>,
    downsample: (u8, u8),
}

pub struct SimpleRenderPipelineBuilder {
    pipeline: SimpleRenderPipeline,
    can_shift: bool,
}

impl RenderPipelineBuilder for SimpleRenderPipelineBuilder {
    type RenderPipeline = SimpleRenderPipeline;
    fn new(num_channels: usize, size: (usize, usize), log_group_size: usize) -> Self {
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
                group_ready_passes: vec![
                    0;
                    size.0.shrc(log_group_size) * size.1.shrc(log_group_size)
                ],
                completed_passes: 0,
                input_buffers: vec![],
            },
            can_shift: true,
        }
    }
    fn add_stage<Stage: RenderPipelineStage>(mut self, stage: Stage) -> Result<Self> {
        let current_info = self.pipeline.channel_info.last().unwrap().clone();
        let mut after_info = vec![];
        for (c, info) in current_info.iter().enumerate() {
            if !stage.uses_channel(c) {
                after_info.push(ChannelInfo {
                    ty: info.ty,
                    downsample: (0, 0),
                });
            } else {
                if let Some(ty) = info.ty {
                    if ty != Stage::Type::INPUT_TYPE {
                        return Err(Error::PipelineChannelTypeMismatch(
                            stage.name(),
                            c,
                            Stage::Type::INPUT_TYPE,
                            ty,
                        ));
                    }
                }
                after_info.push(ChannelInfo {
                    ty: Some(Stage::Type::OUTPUT_TYPE.unwrap_or(Stage::Type::INPUT_TYPE)),
                    downsample: Stage::Type::SHIFT,
                });
            }
        }
        if !self.can_shift && Stage::Type::SHIFT != (0, 0) {
            return Err(Error::PipelineShiftAfterExpand(stage.name()));
        }
        if Stage::Type::TYPE == RenderPipelineStageType::Extend {
            self.can_shift = false;
        }
        self.pipeline.stages.push(Box::new(stage));
        Ok(self)
    }

    fn build(mut self) -> Result<Self::RenderPipeline> {
        let channel_info = &mut self.pipeline.channel_info;
        let num_channels = channel_info[0].len();
        for stage in (0..self.pipeline.stages.len()).rev() {
            let mut stage_iter = channel_info.iter_mut().skip(stage);
            let current_info = stage_iter.next().unwrap();
            let next_info = stage_iter.next().unwrap();
            for chan in 0..num_channels {
                let cur_chan = &mut current_info[chan];
                let next_chan = &next_info[chan];
                if cur_chan.ty.is_none() {
                    cur_chan.ty = next_chan.ty;
                }
                // Arithmetic overflows here should be very uncommon, so custom error variants
                // are probably unwarranted.
                let cur_downsample = &mut cur_chan.downsample;
                let next_downsample = next_chan.downsample;
                cur_downsample.0 = cur_downsample
                    .0
                    .checked_add(next_downsample.0)
                    .ok_or(Error::ArithmeticOverflow)?;
                cur_downsample.1 = cur_downsample
                    .1
                    .checked_add(next_downsample.1)
                    .ok_or(Error::ArithmeticOverflow)?;
            }
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
                Image::new(xsize, ysize)
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
    group_ready_passes: Vec<usize>,
    completed_passes: usize,
    input_buffers: Vec<Image<f64>>,
}

fn clone_images<T: ImageDataType>(images: &[Image<T>]) -> Result<Vec<Image<T>>> {
    images.iter().map(|x| x.as_rect().to_image()).collect()
}

impl SimpleRenderPipeline {
    fn do_render(&mut self) -> Result<()> {
        let ready_passes = self.group_ready_passes.iter().copied().min().unwrap();
        if ready_passes <= self.completed_passes {
            return Ok(());
        }
        self.completed_passes = ready_passes;

        let mut current_buffers = clone_images(&self.input_buffers)?;

        let mut current_size = self.input_size;

        for (i, stage) in self.stages.iter_mut().enumerate() {
            let mut output_buffers = clone_images(&current_buffers)?;
            // Replace buffers of different sizes.
            if stage.shift() != (0, 0) || stage.new_size(current_size) != current_size {
                current_size = stage.new_size(current_size);
                for (c, info) in self.channel_info[i].iter().enumerate() {
                    if stage.uses_channel(c) {
                        let xsize = current_size.0.shrc(info.downsample.0);
                        let ysize = current_size.1.shrc(info.downsample.1);
                        output_buffers[c] = Image::new(xsize, ysize)?;
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
            stage.run_stage_on(&input_buf, &mut output_buf);
            current_buffers = output_buffers;
        }

        Ok(())
    }
}

impl RenderPipeline for SimpleRenderPipeline {
    type Builder = SimpleRenderPipelineBuilder;
    fn fill_input_two_types<
        T1: crate::image::ImageDataType,
        T2: crate::image::ImageDataType,
        F1,
        F2,
    >(
        &mut self,
        group_fill_info: Vec<super::GroupFillInfo<(F1, F2)>>,
    ) -> Result<()>
    where
        F1: FnOnce(&mut [crate::image::ImageRectMut<T1>]) -> Result<()> + Send,
        F2: FnOnce(&mut [crate::image::ImageRectMut<T2>]) -> Result<()> + Send,
    {
        for group_info in group_fill_info {
            let group = (
                group_info.group_id % self.xgroups,
                group_info.group_id / self.xgroups,
            );
            let goffset = (
                group.0 << self.log_group_size,
                group.1 << self.log_group_size,
            );
            let gsize = (
                self.input_size
                    .0
                    .min((goffset.0 + 1) << self.log_group_size)
                    - goffset.0,
                self.input_size
                    .1
                    .min((goffset.1 + 1) << self.log_group_size)
                    - goffset.1,
            );
            let mut images1 = vec![];
            let mut images2 = vec![];
            let mut ch_idx = vec![];
            for ChannelInfo { ty, downsample } in self.channel_info[0].iter() {
                let ty = ty.unwrap();
                assert_eq!(goffset.0 % (1 << downsample.0), 0);
                assert_eq!(goffset.1 % (1 << downsample.1), 0);
                if ty == T1::DATA_TYPE_ID {
                    ch_idx.push(images1.len());
                    images1.push(Image::<T1>::new(
                        gsize.0.shrc(downsample.0),
                        gsize.1.shrc(downsample.1),
                    )?);
                } else if ty == T2::DATA_TYPE_ID {
                    ch_idx.push(images2.len());
                    images2.push(Image::<T2>::new(
                        gsize.0.shrc(downsample.0),
                        gsize.1.shrc(downsample.1),
                    )?);
                } else {
                    panic!("Invalid pipeline usage: channels of unknown type");
                }
            }
            {
                let mut images1: Vec<_> = images1.iter_mut().map(|x| x.as_rect_mut()).collect();
                let mut images2: Vec<_> = images2.iter_mut().map(|x| x.as_rect_mut()).collect();
                group_info.fill_fn.0(&mut images1)?;
                group_info.fill_fn.1(&mut images2)?;
            }
            for (c, ChannelInfo { ty, downsample }) in self.channel_info[0].iter().enumerate() {
                let ty = ty.unwrap();
                let off = (goffset.0 >> downsample.0, goffset.1 >> downsample.1);
                if ty == T1::DATA_TYPE_ID {
                    for y in 0..gsize.1.shrc(downsample.1) {
                        for x in 0..gsize.0.shrc(downsample.0) {
                            self.input_buffers[c].as_rect_mut().row(y + off.1)[x + off.0] =
                                images1[ch_idx[c]].as_rect().row(y)[x].to_f64();
                        }
                    }
                } else if ty == T2::DATA_TYPE_ID {
                    for y in 0..gsize.1.shrc(downsample.1) {
                        for x in 0..gsize.0.shrc(downsample.0) {
                            self.input_buffers[c].as_rect_mut().row(y + off.1)[x + off.0] =
                                images2[ch_idx[c]].as_rect().row(y)[x].to_f64();
                        }
                    }
                }
            }
            self.group_ready_passes[group_info.group_id] += group_info.num_filled_passes;
        }

        self.do_render()
    }
}

pub trait RenderPipelineRunStage {
    fn run_stage_on<S: RenderPipelineStage<Type = Self>>(
        stage: &S,
        input_buffers: &[&Image<f64>],
        output_buffers: &mut [&mut Image<f64>],
    );
}

const CHUNK_SIZE: usize = 256;

impl<T: ImageDataType> RenderPipelineRunStage for RenderPipelineInputStage<T> {
    fn run_stage_on<S: RenderPipelineStage<Type = Self>>(
        stage: &S,
        input_buffers: &[&Image<f64>],
        _output_buffers: &mut [&mut Image<f64>],
    ) {
        let numc = input_buffers.len();
        if numc == 0 {
            return;
        }
        let size = input_buffers[0].size();
        for b in input_buffers.iter() {
            assert_eq!(size, b.size());
        }
        let mut buffer = vec![[T::default(); CHUNK_SIZE]; numc];
        for y in 0..size.1 {
            for x in (0..size.0).step_by(256) {
                let xsize = size.0.min(x + CHUNK_SIZE) - x;
                for c in 0..numc {
                    for ix in 0..xsize {
                        buffer[c][ix] = T::from_f64(input_buffers[c].as_rect().row(y)[x + ix]);
                    }
                }
                let mut row: Vec<_> = buffer.iter().map(|x| x as &[T]).collect();
                stage.process_row_chunk((x, y), xsize, &mut row);
            }
        }
    }
}

impl<T: ImageDataType> RenderPipelineRunStage for RenderPipelineInPlaceStage<T> {
    fn run_stage_on<S: RenderPipelineStage<Type = Self>>(
        stage: &S,
        input_buffers: &[&Image<f64>],
        output_buffers: &mut [&mut Image<f64>],
    ) {
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
        let mut buffer = vec![[T::default(); CHUNK_SIZE]; numc];
        for y in 0..size.1 {
            for x in (0..size.0).step_by(CHUNK_SIZE) {
                let xsize = size.0.min(x + CHUNK_SIZE) - x;
                for c in 0..numc {
                    for ix in 0..xsize {
                        buffer[c][ix] = T::from_f64(input_buffers[c].as_rect().row(y)[x + ix]);
                    }
                }
                let mut row: Vec<_> = buffer.iter_mut().map(|x| x as &mut [T]).collect();
                stage.process_row_chunk((x, y), xsize, &mut row);
                for c in 0..numc {
                    for ix in 0..xsize {
                        output_buffers[c].as_rect_mut().row(y)[x + ix] = buffer[c][ix].to_f64();
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
    fn run_stage_on<S: RenderPipelineStage<Type = Self>>(
        stage: &S,
        input_buffers: &[&Image<f64>],
        output_buffers: &mut [&mut Image<f64>],
    ) {
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
        assert_eq!(input_size.0 << SHIFT_X, output_size.0);
        assert_eq!(input_size.1 << SHIFT_Y, output_size.1);
        let mut buffer_in = vec![
            vec![
                vec![InputT::default(); CHUNK_SIZE + BORDER_X as usize * 2];
                BORDER_Y as usize * 2 + 1
            ];
            numc
        ];
        let mut buffer_out =
            vec![vec![vec![OutputT::default(); CHUNK_SIZE << SHIFT_X]; 1 << SHIFT_Y]; numc];

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
            for x in (0..input_size.0).step_by(CHUNK_SIZE) {
                let border_x = BORDER_X as i64;
                let border_y = BORDER_Y as i64;
                let xsize = input_size.0.min(x + CHUNK_SIZE) - x;
                for c in 0..numc {
                    for iy in -border_y..=border_y {
                        let imgy = mirror(y as i64 + iy, input_size.1 as i64);
                        for ix in -border_x..xsize as i64 + border_x {
                            let imgx = mirror(x as i64 + ix, input_size.0 as i64);
                            buffer_in[c][(iy + border_y) as usize][(ix + border_x) as usize] =
                                InputT::from_f64(input_buffers[c].as_rect().row(imgy)[imgx]);
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
                stage.process_row_chunk((x, y), xsize, &mut row);
                for c in 0..numc {
                    for iy in 0..1usize << SHIFT_Y {
                        for ix in 0..input_size.0 << SHIFT_X {
                            output_buffers[c].as_rect_mut().row(y + iy)[x + ix] =
                                buffer_out[c][iy][ix].to_f64();
                        }
                    }
                }
            }
        }
    }
}

impl<T: ImageDataType> RenderPipelineRunStage for RenderPipelineExtendStage<T> {
    fn run_stage_on<S: RenderPipelineStage<Type = Self>>(
        stage: &S,
        input_buffers: &[&Image<f64>],
        output_buffers: &mut [&mut Image<f64>],
    ) {
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
        // First, copy the data in the middle.
        let origin = stage.original_data_origin();
        for c in 0..numc {
            for y in 0..input_size.1 {
                let in_row = input_buffers[c].as_rect().row(y);
                output_buffers[c].as_rect_mut().row(origin.1 + y)
                    [origin.0..origin.0 + in_row.len()]
                    .copy_from_slice(in_row);
            }
        }
        // Fill in rows above and below the original data.
        let mut buffer = vec![[T::default(); CHUNK_SIZE]; numc];
        for y in (0..origin.1).chain(origin.1 + input_size.1..output_size.1) {
            for x in (0..output_size.0).step_by(CHUNK_SIZE) {
                let xsize = output_size.0.min(x + CHUNK_SIZE) - x;
                let mut row: Vec<_> = buffer.iter_mut().map(|x| x as &mut [T]).collect();
                stage.process_row_chunk((x, y), xsize, &mut row);
                for c in 0..numc {
                    for ix in 0..xsize {
                        output_buffers[c].as_rect_mut().row(y)[x + ix] = buffer[c][ix].to_f64();
                    }
                }
            }
        }
        // Fill in left and right of the original data.
        for y in origin.1..origin.1 + input_size.1 {
            for (x, xsize) in (0..output_size.0)
                .step_by(CHUNK_SIZE)
                .map(|x| (x, origin.0.min(x + CHUNK_SIZE) - x))
                .chain(
                    (origin.0 + input_size.0..output_size.0)
                        .step_by(CHUNK_SIZE)
                        .map(|x| (x, output_size.0.min(x + CHUNK_SIZE) - x)),
                )
            {
                let mut row: Vec<_> = buffer.iter_mut().map(|x| x as &mut [T]).collect();
                stage.process_row_chunk((x, y), xsize, &mut row);
                for c in 0..numc {
                    for ix in 0..xsize {
                        output_buffers[c].as_rect_mut().row(y)[x + ix] = buffer[c][ix].to_f64();
                    }
                }
            }
        }
    }
}

trait RunStage {
    fn run_stage_on(&self, input_buffers: &[&Image<f64>], output_buffers: &mut [&mut Image<f64>]);
    fn shift(&self) -> (u8, u8);
    fn new_size(&self, size: (usize, usize)) -> (usize, usize);
    fn uses_channel(&self, c: usize) -> bool;
}

impl<T: RenderPipelineStage> RunStage for T {
    fn run_stage_on(&self, input_buffers: &[&Image<f64>], output_buffers: &mut [&mut Image<f64>]) {
        T::Type::run_stage_on(self, input_buffers, output_buffers)
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
}
