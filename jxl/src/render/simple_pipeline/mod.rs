// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    BLOCK_DIM,
    api::JxlOutputBuffer,
    error::Result,
    image::{Image, ImageDataType},
    render::internal::ChannelInfo,
    util::{ShiftRightCeil, tracing_wrappers::*},
};

use super::{
    RenderPipeline, RenderPipelineInOutStage, RenderPipelineInPlaceStage,
    internal::{RenderPipelineShared, Stage},
};

mod extend;
mod run_stage;
mod save;

/// A RenderPipeline that waits for all input of a pass to be ready before doing any rendering, and
/// prioritizes simplicity over memory usage and computational efficiency.
/// Eventually meant to be used only for verification purposes.
pub struct SimpleRenderPipeline {
    shared: RenderPipelineShared<Image<f64>>,
    input_buffers: Vec<Image<f64>>,
    completed_passes: usize,
}

fn clone_images<T: ImageDataType>(images: &[Image<T>]) -> Result<Vec<Image<T>>> {
    images.iter().map(|x| x.as_rect().to_image()).collect()
}

impl RenderPipeline for SimpleRenderPipeline {
    type Buffer = Image<f64>;

    fn new_from_shared(shared: RenderPipelineShared<Self::Buffer>) -> Result<Self> {
        let input_buffers = shared.channel_info[0]
            .iter()
            .map(|x| {
                let xsize = shared.input_size.0.shrc(x.downsample.0);
                let ysize = shared.input_size.1.shrc(x.downsample.1);
                Image::new((xsize, ysize))
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            shared,
            input_buffers,
            completed_passes: 0,
        })
    }

    #[instrument(skip_all, err)]
    fn get_buffer_for_group<T: ImageDataType>(
        &mut self,
        channel: usize,
        group_id: usize,
    ) -> Result<Image<T>> {
        let sz = self
            .shared
            .group_size_for_channel(channel, group_id, T::DATA_TYPE_ID);
        Image::<T>::new(sz)
    }

    fn set_buffer_for_group<T: ImageDataType>(
        &mut self,
        channel: usize,
        group_id: usize,
        num_passes: usize,
        buf: Image<T>,
    ) {
        debug!(
            "filling data for group {}, channel {}, using type {:?}",
            group_id,
            channel,
            T::DATA_TYPE_ID,
        );
        let sz = self
            .shared
            .group_size_for_channel(channel, group_id, T::DATA_TYPE_ID);
        let goffset = self.shared.group_offset(group_id);
        let ChannelInfo { ty, downsample } = self.shared.channel_info[0][channel];
        let off = (goffset.0 >> downsample.0, goffset.1 >> downsample.1);
        debug!(?sz, input_buffers_sz=?self.input_buffers[channel].size(), offset=?off, ?downsample, ?goffset);
        let bsz = buf.size();
        assert!(sz.0 <= bsz.0);
        assert!(sz.1 <= bsz.1);
        assert!(sz.0 + BLOCK_DIM > bsz.0);
        assert!(sz.1 + BLOCK_DIM > bsz.1);
        let ty = ty.unwrap();
        assert_eq!(ty, T::DATA_TYPE_ID);
        for y in 0..sz.1 {
            for x in 0..sz.0 {
                self.input_buffers[channel].as_rect_mut().row(y + off.1)[x + off.0] =
                    buf.as_rect().row(y)[x].to_f64();
            }
        }
        self.shared.group_chan_ready_passes[group_id][channel] += num_passes;
    }

    #[instrument(skip_all, err)]
    fn do_render(&mut self, buffers: &mut [Option<JxlOutputBuffer>]) -> Result<()> {
        let ready_passes = self
            .shared
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

        let mut current_buffers = clone_images(&self.input_buffers)?;

        let mut current_size = self.shared.input_size;

        for (i, stage) in self.shared.stages.iter().enumerate() {
            debug!("running stage {i}: {stage}");
            let mut output_buffers = clone_images(&current_buffers)?;
            if stage.shift() != (0, 0) || stage.new_size(current_size) != current_size {
                // Replace buffers of different sizes.
                current_size = stage.new_size(current_size);
                for (c, info) in self.shared.channel_info[i + 1].iter().enumerate() {
                    if stage.uses_channel(c) {
                        let xsize = current_size.0.shrc(info.downsample.0);
                        let ysize = current_size.1.shrc(info.downsample.1);
                        debug!("reallocating channel {c} to new size {xsize}x{ysize}");
                        output_buffers[c] = Image::new((xsize, ysize))?;
                    }
                }
            }
            match stage {
                Stage::InOut(stage) => {
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
                        self.shared.chunk_size,
                        &input_buf,
                        &mut output_buf,
                        state.as_deref_mut(),
                    );
                }
                Stage::InPlace(stage) => {
                    let mut output_buf: Vec<_> = output_buffers
                        .iter_mut()
                        .enumerate()
                        .filter(|x| stage.uses_channel(x.0))
                        .map(|x| x.1)
                        .collect();
                    let mut state = stage.init_local_state()?;
                    stage.run_stage_on(
                        self.shared.chunk_size,
                        &mut output_buf,
                        state.as_deref_mut(),
                    );
                }
                Stage::Extend(e) => {
                    e.extend_simple(
                        self.shared.chunk_size,
                        &current_buffers,
                        &mut output_buffers,
                    );
                }
                Stage::Save(stage) => {
                    stage.save_simple(&output_buffers, buffers)?;
                }
            }
            current_buffers = output_buffers;
        }

        self.completed_passes = ready_passes;

        Ok(())
    }

    fn num_groups(&self) -> usize {
        self.shared.num_groups()
    }

    fn box_inout_stage<S: RenderPipelineInOutStage>(
        stage: S,
    ) -> Box<dyn super::RunInOutStage<Self::Buffer>> {
        Box::new(stage)
    }

    fn box_inplace_stage<S: RenderPipelineInPlaceStage>(
        stage: S,
    ) -> Box<dyn super::RunInPlaceStage<Self::Buffer>> {
        Box::new(stage)
    }
}
