// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    error::Result,
    headers::Orientation,
    image::{Image, ImageDataType},
    util::{ShiftRightCeil, tracing_wrappers::instrument},
};
use rand::SeedableRng;

use super::{
    RenderPipeline, RenderPipelineBuilder, RenderPipelineStage, SaveStage, SaveStageType,
    internal::RenderPipelineStageInfo, simple_pipeline::SimpleRenderPipelineBuilder,
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
        pipeline = pipeline.add_save_stage(SaveStage::<OutputT>::new(
            SaveStageType::Output,
            i,
            final_size,
            OutputT::from_f64(1.0),
            Orientation::Identity,
        )?)?;
    }
    let mut pipeline = pipeline.build()?;

    for g in 0..pipeline.num_groups() {
        for &c in all_channels.iter() {
            let log_group_size = if uses_channel[c] {
                (
                    LOG_GROUP_SIZE - S::Type::SHIFT.0 as usize,
                    LOG_GROUP_SIZE - S::Type::SHIFT.1 as usize,
                )
            } else {
                (LOG_GROUP_SIZE, LOG_GROUP_SIZE)
            };
            pipeline.set_buffer_for_group(
                c,
                g,
                1,
                input_images[c].group_rect(g, log_group_size).to_image()?,
            );
        }
    }

    // TODO(veluca): pass actual output buffers.
    pipeline.do_render(&mut [])?;

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

macro_rules! create_in_out_rows {
    ($u:expr, $border_x:expr, $border_y:expr, $rows:ident, $xsize:ident) => {
        use crate::simd::round_up_size_to_two_cache_lines;
        let $xsize: usize = 1 + $u.arbitrary::<usize>()? % 4095;
        let mut row_vecs = vec![(
            vec![
                vec![
                    0f32;
                    round_up_size_to_two_cache_lines::<f32>(
                        round_up_size_to_two_cache_lines::<f32>($xsize) + $border_x * 2
                    )
                ];
                1 + $border_y * 2
            ],
            vec![vec![0f32; round_up_size_to_two_cache_lines::<f32>($xsize)]],
        )];

        let mut row_vecs_refs: Vec<(Vec<&[f32]>, Vec<&mut [f32]>)> = row_vecs
            .iter_mut()
            .map(|(left, right)| {
                (
                    left.iter().map(|v| v.as_slice()).collect(),
                    right.iter_mut().map(|v| v.as_mut_slice()).collect(),
                )
            })
            .collect();

        let mut outer: Vec<(&[&[f32]], &mut [&mut [f32]])> = row_vecs_refs
            .iter_mut()
            .map(|(left, right)| (left.as_slice(), right.as_mut_slice()))
            .collect();

        let $rows = &mut outer[..];
    };
}
pub(crate) use create_in_out_rows;
