// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::any::Any;

use crate::{image::ImageDataType, util::tracing_wrappers::*};

use super::{
    super::{
        RenderPipelineExtendStage, RenderPipelineInOutStage, RenderPipelineInPlaceStage,
        RenderPipelineStage, internal::RenderPipelineRunStage,
    },
    row_buffers::RowBuffer,
};

impl<T: ImageDataType> RenderPipelineRunStage<RowBuffer> for RenderPipelineInPlaceStage<T> {
    #[instrument(skip_all)]
    fn run_stage_on<S: RenderPipelineStage<Type = Self>>(
        stage: &S,
        chunk_size: usize,
        input_buffers: &[&RowBuffer],
        output_buffers: &mut [&mut RowBuffer],
        state: Option<&mut dyn Any>,
    ) {
        todo!()
    }
}

impl<
    InputT: ImageDataType,
    OutputT: ImageDataType,
    const BORDER_X: u8,
    const BORDER_Y: u8,
    const SHIFT_X: u8,
    const SHIFT_Y: u8,
> RenderPipelineRunStage<RowBuffer>
    for RenderPipelineInOutStage<InputT, OutputT, BORDER_X, BORDER_Y, SHIFT_X, SHIFT_Y>
{
    #[instrument(skip_all)]
    fn run_stage_on<S: RenderPipelineStage<Type = Self>>(
        stage: &S,
        chunk_size: usize,
        input_buffers: &[&RowBuffer],
        output_buffers: &mut [&mut RowBuffer],
        state: Option<&mut dyn Any>,
    ) {
        todo!()
    }
}

impl<T: ImageDataType> RenderPipelineRunStage<RowBuffer> for RenderPipelineExtendStage<T> {
    #[instrument(skip_all)]
    fn run_stage_on<S: RenderPipelineStage<Type = Self>>(
        stage: &S,
        chunk_size: usize,
        input_buffers: &[&RowBuffer],
        output_buffers: &mut [&mut RowBuffer],
        state: Option<&mut dyn Any>,
    ) {
        todo!()
    }
}
