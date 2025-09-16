// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use row_buffers::RowBuffer;

use crate::api::JxlOutputBuffer;
use crate::error::Result;
use crate::image::{Image, ImageDataType};
use crate::simd::CACHE_LINE_BYTE_SIZE;
use crate::util::tracing_wrappers::*;

use super::internal::{RenderPipelineShared, RunStage};
use super::{RenderPipeline, RenderPipelineStage};

pub(super) mod row_buffers;
mod run_stage;
mod save;

const MAX_OVERALL_BORDER: usize = 16; // probably an overestimate.

const _: () = assert!(MAX_OVERALL_BORDER * 8 <= CACHE_LINE_BYTE_SIZE * 2);

pub struct LowMemoryRenderPipeline {
    //
}

impl RenderPipeline for LowMemoryRenderPipeline {
    type Buffer = RowBuffer;

    fn new_from_shared(shared: RenderPipelineShared<Self::Buffer>) -> Result<Self> {
        todo!()
    }

    #[instrument(skip_all, err)]
    fn get_buffer_for_group<T: ImageDataType>(
        &mut self,
        channel: usize,
        group_id: usize,
    ) -> Result<Image<T>> {
        todo!()
    }

    fn set_buffer_for_group<T: ImageDataType>(
        &mut self,
        channel: usize,
        group_id: usize,
        num_passes: usize,
        buf: Image<T>,
    ) {
        todo!()
    }

    fn do_render(&mut self, buffers: &mut [Option<JxlOutputBuffer>]) -> Result<()> {
        todo!()
    }

    fn num_groups(&self) -> usize {
        todo!()
    }

    fn box_stage<S: RenderPipelineStage>(stage: S) -> Box<dyn RunStage<Self::Buffer>> {
        todo!()
    }
}
