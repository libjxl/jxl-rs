// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    error::Result,
    headers::Orientation,
    image::{Image, ImageDataType, ImageRectMut},
    util::{ShiftRightCeil, tracing_wrappers::instrument},
};
use rand::SeedableRng;

use super::{
    RenderPipeline, RenderPipelineBuilder, RenderPipelineStage,
    internal::RenderPipelineStageInfo,
    simple_pipeline::SimpleRenderPipelineBuilder,
    stages::{SaveStage, SaveStageType},
};

pub(super) fn make_and_run_simple_pipeline<
    S: RenderPipelineStage,
    InputT: ImageDataType,
    OutputT: ImageDataType + std::ops::Mul<Output = OutputT>,
>(
    stage: S,
    input_images: &[Image<InputT>],
    image_size: (usize, usize),
    downsampling_shift: usize,
    chunk_size: usize,
) -> Result<(S, Vec<Image<OutputT>>)> {
    let final_size = stage.new_size(image_size);
    const LOG_GROUP_SIZE: usize = 8;
    let all_channels = (0..input_images.len()).collect::<Vec<_>>();
    let uses_channel: Vec<_> = all_channels
        .iter()
        .map(|x| stage.uses_channel(*x))
        .collect();
    let mut pipeline = SimpleRenderPipelineBuilder::new_with_chunk_size(
        input_images.len(),
        image_size,
        downsampling_shift,
        LOG_GROUP_SIZE,
        chunk_size,
    )
    .add_stage(stage)?;
    for i in 0..input_images.len() {
        pipeline = pipeline.add_stage(SaveStage::<OutputT>::new(
            SaveStageType::Output,
            i,
            final_size,
            OutputT::from_f64(1.0),
            Orientation::Identity,
        )?)?;
    }
    let mut pipeline = pipeline.build()?;

    for g in 0..pipeline.num_groups() {
        pipeline.fill_input_channels(
            &all_channels,
            g,
            1,
            |rects: &mut [ImageRectMut<InputT>]| {
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
        )?;
    }

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
    OutputT: ImageDataType + std::ops::Mul<Output = OutputT>,
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
        make_and_run_simple_pipeline::<_, InputT, OutputT>(stage, &images, image_size, 0, 256)?;

    let mut stage = Some(stage);

    arbtest::arbtest(move |p| {
        let chunk_size = p.arbitrary::<u16>()?.saturating_add(1) as usize;
        let (s, output) = make_and_run_simple_pipeline::<_, InputT, OutputT>(
            stage.take().unwrap(),
            &images,
            image_size,
            0,
            chunk_size,
        )
        .unwrap_or_else(|_| panic!("error running pipeline with chunk size {chunk_size}"));
        stage = Some(s);

        for (o, bo) in output.iter().zip(base_output.iter()) {
            bo.as_rect().check_equal(o.as_rect());
        }

        Ok(())
    });
    Ok(())
}
