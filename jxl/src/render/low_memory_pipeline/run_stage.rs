// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::any::Any;

use crate::{
    render::{RunInPlaceStage, internal::RunInOutStage},
    util::tracing_wrappers::*,
};

use super::{
    super::{RenderPipelineInOutStage, RenderPipelineInPlaceStage},
    row_buffers::RowBuffer,
};

impl<T: RenderPipelineInPlaceStage> RunInPlaceStage<RowBuffer> for T {
    #[instrument(skip_all)]
    fn run_stage_on(
        &self,
        chunk_size: usize,
        buffers: &mut [&mut RowBuffer],
        state: Option<&mut dyn Any>,
    ) {
        todo!()
    }
}

impl<T: RenderPipelineInOutStage> RunInOutStage<RowBuffer> for T {
    #[instrument(skip_all)]
    fn run_stage_on(
        &self,
        chunk_size: usize,
        input_buffers: &[&RowBuffer],
        output_buffers: &mut [&mut RowBuffer],
        state: Option<&mut dyn Any>,
    ) {
        todo!()
    }
}
