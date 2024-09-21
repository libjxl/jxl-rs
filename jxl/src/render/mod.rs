// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    error::Result,
    image::{ImageDataType, ImageRectMut},
};

/// These are the only stages that are assumed to have observable effects, i.e. calls to process_row
/// for other stages may be omitted if it can be shown they can't affect any Input stage process_row
/// call that happens inside image boundaries.
pub trait RenderPipelineInputStage<InputT: ImageDataType> {
    fn uses_channel(&self, c: usize) -> bool;

    /// `row` has as many entries as the number of used channels for the stage, i.e. the channels
    /// for which `uses_channel` returns true.
    fn process_row_chunk(&mut self, position: (usize, usize), xsize: usize, row: &[&[InputT]]);
}

/// Modifies channels in-place.
pub trait RenderPipelineInPlaceStage<T: ImageDataType> {
    fn uses_channel(&self, c: usize) -> bool;
    /// `row` has as many entries as the number of used channels for the stage, i.e. the channels
    /// for which `uses_channel` returns true.
    fn process_row_chunk(&mut self, position: (usize, usize), xsize: usize, row: &mut [&mut [T]]);
}

/// Modifies data and writes it to a new buffer, of possibly different type.
pub trait RenderPipelineInOutStage<InputT: ImageDataType, OutputT: ImageDataType> {
    /// Amount of padding required on the input side.
    const BORDER: (usize, usize);
    /// log2 of the number of rows/columns produced for each row/column of input.
    const SHIFT: (usize, usize);

    fn uses_channel(&self, c: usize) -> bool;

    /// For each channel:
    ///  - the input slice contains 1 + Self::BORDER.1 * 2 slices, each of length
    ///    xsize + Self::BORDER.0 * 2, i.e. covering one input row and up to BORDER pixels of
    ///    padding on either side.
    ///  - the output slice contains 1 << SHIFT.1 slices, each of length xsize << SHIFT.0, the
    ///    corresponding output pixels.
    ///
    /// `input` and `output` have as many entries as the number of used channels for the stage,
    /// i.e. the channels for which `uses_channel` returns true.
    fn process_row_chunk(
        &mut self,
        position: (usize, usize),
        xsize: usize,
        input: &[&[&[InputT]]],
        output: &mut [&mut [&mut [OutputT]]],
    );
}

/// Does not directly modify the current image pixels, but extends the current image with
/// additional data.
/// ` uses_channel` is assumed to be always true, i.e. such a stage must extend all channels at
/// once.
pub trait RenderPipelineExtendStage<T: ImageDataType> {
    fn new_size(&self) -> (usize, usize);
    fn original_data_origin(&self) -> (usize, usize);
    /// The given buffer must always be entirely outside of the original image.
    fn fill_padding_row_chunk(
        &mut self,
        new_position: (usize, usize),
        xsize: usize,
        row: &mut [&mut [T]],
    );
}

pub trait RenderPipelineBuilder: Sized {
    type RenderPipeline: RenderPipeline;

    fn new(num_channels: usize, size: (usize, usize), group_size: usize) -> Self;

    fn add_input_stage<T: ImageDataType, Stage: RenderPipelineInputStage<T>>(
        self,
        stage: Stage,
    ) -> Result<Self>;

    fn add_inplace_stage<T: ImageDataType, Stage: RenderPipelineInPlaceStage<T>>(
        self,
        stage: Stage,
    ) -> Result<Self>;

    fn add_inout_stage<
        InT: ImageDataType,
        OutT: ImageDataType,
        Stage: RenderPipelineInOutStage<InT, OutT>,
    >(
        self,
        stage: Stage,
    ) -> Result<Self>;

    fn add_extend_stage<T: ImageDataType, Stage: RenderPipelineExtendStage<T>>(
        self,
        stage: Stage,
    ) -> Result<Self>;

    fn build(self) -> Self::RenderPipeline;
}

#[allow(dead_code)]
pub struct GroupFillInfo<F> {
    group_id: usize,
    num_filled_passes: usize,
    fill_fn: F,
}

fn fake_fill_fn(_: &mut [ImageRectMut<u8>]) {
    panic!("can only use fill_input_same_type if the inputs are of the same type");
}

pub trait RenderPipeline {
    type Builder: RenderPipelineBuilder<RenderPipeline = Self>;

    /// Feeds input into the pipeline. Takes as input a vector specifying, for each group that will
    /// be filled in, how many passes are filled and how to fill in each channel.
    fn fill_input_same_type<T: ImageDataType, F>(&mut self, group_fill_info: Vec<GroupFillInfo<F>>)
    where
        F: FnOnce(&mut [ImageRectMut<T>]) + Send,
    {
        self.fill_input_two_types(
            group_fill_info
                .into_iter()
                .map(|x| GroupFillInfo {
                    fill_fn: (x.fill_fn, fake_fill_fn),
                    group_id: x.group_id,
                    num_filled_passes: x.num_filled_passes,
                })
                .collect(),
        );
    }

    /// Same as fill_input_same_type, but the inputs might have different types.
    /// Which type is used for which channel is determined by the stages in the render pipeline.
    fn fill_input_two_types<T1: ImageDataType, T2: ImageDataType, F1, F2>(
        &mut self,
        group_fill_info: Vec<GroupFillInfo<(F1, F2)>>,
    ) where
        F1: FnOnce(&mut [ImageRectMut<T1>]) + Send,
        F2: FnOnce(&mut [ImageRectMut<T2>]) + Send;
}
