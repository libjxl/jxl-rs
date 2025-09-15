use crate::api::{JxlColorType, JxlDataFormat, JxlOutputBuffer};
use crate::image::{Image, ImageDataType};
use crate::util::tracing_wrappers::*;
use crate::{error::Result, headers::Orientation};

use super::{RenderPipeline, RenderPipelineBuilder, RenderPipelineStage};

mod save;
mod stage;

pub struct LowMemoryRenderPipeline {
    //
}

pub struct LowMemoryRenderPipelineBuilder {
    //
}

impl RenderPipelineBuilder for LowMemoryRenderPipelineBuilder {
    type RenderPipeline = LowMemoryRenderPipeline;

    fn new(
        num_channels: usize,
        size: (usize, usize),
        downsampling_shift: usize,
        log_group_size: usize,
        num_passes: usize,
    ) -> Self {
        todo!()
    }

    #[instrument(skip_all, err)]
    fn add_stage<S: RenderPipelineStage>(self, stage: S) -> Result<Self> {
        todo!()
    }

    #[instrument(skip_all, err)]
    fn add_save_stage(
        self,
        channels: &[usize],
        orientation: Orientation,
        output_buffer_index: usize,
        color_type: JxlColorType,
        data_format: JxlDataFormat,
    ) -> Result<Self> {
        todo!()
    }

    #[instrument(skip_all, err)]
    fn build(mut self) -> Result<Box<Self::RenderPipeline>> {
        todo!()
    }
}

impl RenderPipeline for LowMemoryRenderPipeline {
    type Builder = LowMemoryRenderPipelineBuilder;

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
