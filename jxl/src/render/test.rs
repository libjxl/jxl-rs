// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    error::Result,
    image::{Image, ImageDataType, ImageRectMut},
    util::{tracing::instrument, ShiftRightCeil},
};
use rand::SeedableRng;

use super::{
    internal::RenderPipelineStageInfo, simple_pipeline::SimpleRenderPipelineBuilder,
    stages::SaveStage, GroupFillInfo, RenderPipeline, RenderPipelineBuilder, RenderPipelineStage,
};

pub(super) fn make_and_run_simple_pipeline<
    S: RenderPipelineStage,
    InputT: ImageDataType,
    OutputT: ImageDataType,
>(
    stage: S,
    input_images: &[Image<InputT>],
    image_size: (usize, usize),
    chunk_size: usize,
) -> Result<(S, Vec<Image<OutputT>>)> {
    let final_size = stage.new_size(image_size);
    const LOG_GROUP_SIZE: usize = 8;
    let uses_channel: Vec<_> = (0..input_images.len())
        .map(|x| stage.uses_channel(x))
        .collect();
    let mut pipeline = SimpleRenderPipelineBuilder::new_with_chunk_size(
        input_images.len(),
        image_size,
        LOG_GROUP_SIZE,
        chunk_size,
    )
    .add_stage(stage)?;
    for i in 0..input_images.len() {
        pipeline = pipeline.add_stage(SaveStage::<OutputT>::new(i, final_size)?)?;
    }
    let mut pipeline = pipeline.build()?;

    let fill_info = (0..pipeline.num_groups()).map(|g| {
        let uses_channel = uses_channel.clone();
        GroupFillInfo {
            group_id: g,
            num_filled_passes: 1,
            fill_fn: move |rects: &mut [ImageRectMut<InputT>]| {
                for ((input, fill), used) in input_images
                    .iter()
                    .zip(rects.iter_mut())
                    .zip(uses_channel.iter().copied())
                {
                    let log_group_size = if used {
                        (
                            LOG_GROUP_SIZE - S::Type::SHIFT.0 as usize,
                            LOG_GROUP_SIZE - S::Type::SHIFT.1 as usize,
                        )
                    } else {
                        (LOG_GROUP_SIZE, LOG_GROUP_SIZE)
                    };
                    fill.copy_from(input.group_rect(g, log_group_size))?;
                }
                Ok(())
            },
        }
    });
    pipeline.fill_input_same_type(fill_info.collect())?;

    let mut stages = pipeline.into_stages().into_iter();
    let stage = stages
        .next()
        .unwrap()
        .downcast::<S>()
        .expect("first stage is always the tested stage");

    let outputs = stages
        .map(|s| {
            s.downcast::<SaveStage<OutputT>>()
                .expect("all later stages are always SaveStage")
                .into_buffer()
        })
        .collect();

    Ok((*stage, outputs))
}

#[instrument(skip(stage), err)]
pub(super) fn test_stage_consistency<
    S: RenderPipelineStage,
    InputT: ImageDataType,
    OutputT: ImageDataType,
>(
    stage: S,
    image_size: (usize, usize),
    num_image_channels: usize,
) -> Result<()> {
    let mut rng = rand_xorshift::XorShiftRng::seed_from_u64(0);
    let images: Result<Vec<_>> = (0..num_image_channels)
        .map(|c| {
            let size = if stage.uses_channel(c) {
                (
                    image_size.0.shrc(S::Type::SHIFT.0),
                    image_size.1.shrc(S::Type::SHIFT.1),
                )
            } else {
                image_size
            };
            Image::new_random(size, &mut rng)
        })
        .collect();
    let images = images?;

    let (stage, base_output) =
        make_and_run_simple_pipeline::<_, InputT, OutputT>(stage, &images, image_size, 256)?;

    let mut stage = Some(stage);

    arbtest::arbtest(move |p| {
        let chunk_size = p.arbitrary::<u16>()? as usize;

        let (s, output) = make_and_run_simple_pipeline::<_, InputT, OutputT>(
            stage.take().unwrap(),
            &images,
            image_size,
            chunk_size,
        )
        .unwrap_or_else(|_| panic!("error running pipeline with chunk size {}", chunk_size));
        stage = Some(s);

        for (o, bo) in output.iter().zip(base_output.iter()) {
            bo.as_rect().check_equal(o.as_rect());
        }

        Ok(())
    });
    Ok(())
}
