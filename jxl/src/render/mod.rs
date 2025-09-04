// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use internal::RenderPipelineStageInfo;
use std::{any::Any, marker::PhantomData};

use crate::{
    api::JxlOutputBuffer,
    error::Result,
    image::{Image, ImageDataType},
};

mod internal;
mod simple_pipeline;
pub mod stages;
#[cfg(test)]
mod test;

pub use simple_pipeline::{
    SimpleRenderPipeline, SimpleRenderPipelineBuilder,
    save::{SaveStage, SaveStageType},
};

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
        &self,
        position: (usize, usize),
        xsize: usize,
        // one for each channel
        row: &mut [<Self::Type as RenderPipelineStageInfo>::RowType<'_>],
        state: Option<&mut dyn Any>,
    );

    /// Returns the new size of the image after this stage. Should be implemented by
    /// `RenderPipelineExtendStage` stages.
    fn new_size(&self, current_size: (usize, usize)) -> (usize, usize) {
        current_size
    }

    /// Returns the origin of the original image data in the output of this stage.
    /// Should be implemented by `RenderPipelineExtendStage` stages.
    fn original_data_origin(&self) -> (isize, isize) {
        (0, 0)
    }

    /// Initializes thread local state for the stage. Returns `Ok(None)` if no state is needed.
    ///
    /// This method returns `Box<dyn Any>` for dyn compatibility. `process_row_chunk` should
    /// downcast the state to the desired type.
    fn init_local_state(&self) -> Result<Option<Box<dyn Any>>> {
        Ok(None)
    }
}

pub trait RenderPipelineBuilder: Sized {
    type RenderPipeline: RenderPipeline;
    fn new(
        num_channels: usize,
        size: (usize, usize),
        downsampling_shift: usize,
        log_group_size: usize,
    ) -> Self;
    fn add_stage<Stage: RenderPipelineStage>(self, stage: Stage) -> Result<Self>;
    fn add_save_stage<T: ImageDataType>(self, stage: SaveStage<T>) -> Result<Self>;
    fn build(self) -> Result<Self::RenderPipeline>;
}

pub trait RenderPipeline {
    type Builder: RenderPipelineBuilder<RenderPipeline = Self>;

    /// Obtains a buffer suitable for storing the input at  channel `channel` of group `group_id`.
    /// This *might* be a buffer that was used to store that channel for that group in a previous
    /// pass, a new buffer, or a re-used buffer from i.e. previously decoded frames.
    fn get_buffer_for_group<T: ImageDataType>(
        &mut self,
        channel: usize,
        group_id: usize,
    ) -> Result<Image<T>>;

    /// Gives back the buffer for a channel and group to the render pipeline, marking that
    /// `num_passes` additional passes (wrt. the previous call to this method for the same channel
    /// and group, or 0 if no previous call happend) were rendered into the input buffer.
    fn set_buffer_for_group<T: ImageDataType>(
        &mut self,
        channel: usize,
        group_id: usize,
        num_passes: usize,
        buf: Image<T>,
    );

    /// Renders new data that is available after the last call to `render`.
    fn do_render(&mut self, buffers: &mut [Option<JxlOutputBuffer>]) -> Result<()>;

    fn into_stages(self) -> Vec<Box<dyn Any>>;
    fn num_groups(&self) -> usize;
}
