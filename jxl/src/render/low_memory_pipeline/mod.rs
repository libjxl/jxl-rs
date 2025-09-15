use std::fmt::Display;

use crate::api::JxlOutputBuffer;
use crate::image::{Image, ImageDataType};
use crate::util::tracing_wrappers::*;
use crate::error::Result;

use super::internal::{BoxedStage, RenderPipelineShared};
use super::{RenderPipeline, RenderPipelineStage};

mod save;

pub struct LowMemoryRenderPipeline {
    //
}

impl RenderPipeline for LowMemoryRenderPipeline {
    type BoxedStage = Box<dyn RunStage>; // TODO

    fn new_from_shared(shared: RenderPipelineShared<Self::BoxedStage>) -> Result<Self> {
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
}

pub(crate) trait RunStage: Display {
    // TODO
}

impl BoxedStage for Box<dyn RunStage> {
    fn new<S: RenderPipelineStage>(stage: S) -> Self {
        todo!()
    }
    fn input_type(&self) -> crate::image::DataTypeTag {
        todo!()
    }
    fn output_type(&self) -> crate::image::DataTypeTag {
        todo!()
    }
    fn uses_channel(&self, channel: usize) -> bool {
        todo!()
    }
}
