// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::{any::Any, marker::PhantomData};

use crate::{
    error::Result,
    image::{DataTypeTag, ImageDataType, ImageRectMut},
};

mod internal;
mod simple_pipeline;
pub mod stages;
#[cfg(test)]
mod test;

use internal::RenderPipelineStageInfo;

pub use simple_pipeline::SimpleRenderPipeline;

/// Inspects channels and passes data to the following stage as is.
///
/// These are the only stages that are assumed to have observable effects, i.e. calls to
/// `process_row_chunk` for other stages may be omitted if it can be shown they can't affect any
/// Inspect stage `process_row_chunk` call that happens inside image boundaries.
/// For these stages, `process_row_chunk` receives read-only access to each row in each channel.
pub struct RenderPipelineInspectStage<InputT: ImageDataType> {
    _phantom: PhantomData<InputT>,
}

/// Modifies channels in-place.
///
/// For these stages, `process_row_chunk` receives read-write access to each row in each channel.
pub struct RenderPipelineInPlaceStage<T: ImageDataType> {
    _phantom: PhantomData<T>,
}

/// Modifies data and writes it to a new buffer, of possibly different type.
///
/// BORDER_X and BORDER_Y represent the amount of padding required on the input side.
/// SHIFT_X and SHIFT_Y represent the base 2 log of the number of rows/columns produced
/// for each row/column of input.
///
/// For each channel:
///  - the input slice contains 1 + BORDER_Y * 2 slices, each of length
///    xsize + BORDER_X * 2, i.e. covering one input row and up to BORDER pixels of
///    padding on either side.
///  - the output slice contains 1 << SHIFT_Y slices, each of length xsize << SHIFT_X, the
///    corresponding output pixels.
///
/// `process_row_chunk` is passed a pair of q(input, output)` slices.
pub struct RenderPipelineInOutStage<
    InputT: ImageDataType,
    OutputT: ImageDataType,
    const BORDER_X: u8,
    const BORDER_Y: u8,
    const SHIFT_X: u8,
    const SHIFT_Y: u8,
> {
    _phantom: PhantomData<(InputT, OutputT)>,
}

/// Does not directly modify the current image pixels, but extends the current image with
/// additional data.
///
/// `uses_channel` must always return true, and stages of this type should override
/// `new_size` and `original_data_origin`.
/// `process_row_chunk` will be called with the *new* image coordinates, and will only be called
/// on row chunks outside of the original image data.
/// After stages of this type, no stage can have a non-0 SHIFT_X or SHIFT_Y.
pub struct RenderPipelineExtendStage<T: ImageDataType> {
    _phantom: PhantomData<T>,
}

// TODO(veluca): figure out how to modify the interface for concurrent usage.
pub trait RenderPipelineStage: Any + std::fmt::Display {
    type Type: RenderPipelineStageInfo;

    /// Which channels are actually used by this stage.
    /// Must always return `true` if `Self::Type` is `RenderPipelineExtendStage`.
    fn uses_channel(&self, c: usize) -> bool;

    /// Process one chunk of row. The semantics of this function are detailed in the
    /// documentation of the various types of stages.
    fn process_row_chunk(
        &mut self,
        position: (usize, usize),
        xsize: usize,
        // one for each channel
        row: &mut [<Self::Type as RenderPipelineStageInfo>::RowType<'_>],
    );

    /// Returns the new size of the image after this stage. Should be implemented by
    /// `RenderPipelineExtendStage` stages.
    fn new_size(&self, current_size: (usize, usize)) -> (usize, usize) {
        current_size
    }

    /// Returns the origin of the original image data in the output of this stage.
    /// Should be implemented by `RenderPipelineExtendStage` stages.
    fn original_data_origin(&self) -> (usize, usize) {
        (0, 0)
    }
}

pub trait RenderPipelineBuilder: Sized {
    type RenderPipeline: RenderPipeline;
    fn new(num_channels: usize, size: (usize, usize), log_group_size: usize) -> Self;
    fn add_stage<Stage: RenderPipelineStage>(self, stage: Stage) -> Result<Self>;
    fn build(self) -> Result<Self::RenderPipeline>;
}

pub struct GroupFillInfo<F> {
    group_id: usize,
    num_filled_passes: usize,
    fill_fn: F,
}

fn fake_fill_fn(_: &mut [ImageRectMut<f64>]) -> Result<()> {
    panic!("can only use fill_input_same_type if the inputs are of the same type");
}

pub trait RenderPipeline {
    type Builder: RenderPipelineBuilder<RenderPipeline = Self>;

    /// Feeds input into the pipeline. Takes as input a vector specifying, for each group that will
    /// be filled in, how many passes are filled and how to fill in each channel.
    fn fill_input_same_type<T: ImageDataType, F>(
        &mut self,
        group_fill_info: Vec<GroupFillInfo<F>>,
    ) -> Result<()>
    where
        F: FnOnce(&mut [ImageRectMut<T>]) -> Result<()>,
    {
        const {
            #[allow(clippy::single_match)]
            match T::DATA_TYPE_ID {
                DataTypeTag::F64 => panic!("cannot use f64 with fill_input_same_type"),
                _ => (),
            };
        };
        self.fill_input_two_types(
            group_fill_info
                .into_iter()
                .map(|x| GroupFillInfo {
                    fill_fn: (x.fill_fn, fake_fill_fn),
                    group_id: x.group_id,
                    num_filled_passes: x.num_filled_passes,
                })
                .collect(),
        )
    }

    /// Same as fill_input_same_type, but the inputs might have different types.
    /// Which type is used for which channel is determined by the stages in the render pipeline.
    fn fill_input_two_types<T1: ImageDataType, T2: ImageDataType, F1, F2>(
        &mut self,
        group_fill_info: Vec<GroupFillInfo<(F1, F2)>>,
    ) -> Result<()>
    where
        F1: FnOnce(&mut [ImageRectMut<T1>]) -> Result<()>,
        F2: FnOnce(&mut [ImageRectMut<T2>]) -> Result<()>;

    fn into_stages(self) -> Vec<Box<dyn Any>>;
    fn num_groups(&self) -> usize;
}
