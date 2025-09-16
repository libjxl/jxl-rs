// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    api::{Endianness, JxlColorType, JxlDataFormat, JxlOutputBuffer},
    error::Result,
    headers::Orientation,
    image::{DataTypeTag, Image, ImageDataType},
    render::SimpleRenderPipeline,
    util::{ShiftRightCeil, tracing_wrappers::instrument},
};
use rand::SeedableRng;

use super::{
    RenderPipeline, RenderPipelineBuilder, RenderPipelineStage,
    internal::{RenderPipelineRunStage, RenderPipelineStageInfo},
};

pub(super) fn make_and_run_simple_pipeline<
    S: RenderPipelineStage,
    InputT: ImageDataType,
    OutputT: ImageDataType,
>(
    stage: S,
    input_images: &[Image<InputT>],
    image_size: (usize, usize),
    downsampling_shift: usize,
    chunk_size: usize,
) -> Result<Vec<Image<OutputT>>>
where
    S::Type: RenderPipelineRunStage<Image<f64>>,
{
    let final_size = stage.new_size(image_size);
    const LOG_GROUP_SIZE: usize = 8;
    let all_channels = (0..input_images.len()).collect::<Vec<_>>();
    let uses_channel: Vec<_> = all_channels
        .iter()
        .map(|x| stage.uses_channel(*x))
        .collect();
    let mut pipeline = RenderPipelineBuilder::<SimpleRenderPipeline>::new_with_chunk_size(
        input_images.len(),
        image_size,
        downsampling_shift,
        LOG_GROUP_SIZE,
        1,
        chunk_size,
    )
    .add_stage(stage)?;

    let jxl_data_type = match OutputT::DATA_TYPE_ID {
        DataTypeTag::U8 | DataTypeTag::I8 => JxlDataFormat::U8 { bit_depth: 8 },
        DataTypeTag::U16 | DataTypeTag::I16 => JxlDataFormat::U16 {
            bit_depth: 16,
            endianness: Endianness::native(),
        },
        DataTypeTag::F32 => JxlDataFormat::f32(),
        DataTypeTag::F16 => JxlDataFormat::F16 {
            endianness: Endianness::native(),
        },
        _ => unimplemented!("unsupported data type"),
    };

    for i in 0..input_images.len() {
        pipeline = pipeline.add_save_stage(
            &[i],
            Orientation::Identity,
            i,
            JxlColorType::Grayscale,
            jxl_data_type,
        )?;
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

    let mut outputs = (0..input_images.len())
        .map(|_| Image::<OutputT>::new(final_size))
        .collect::<Result<Vec<_>, _>>()?;

    let mut buf_ptrs: Vec<_> = outputs
        .iter_mut()
        .map(|x| Some(JxlOutputBuffer::from_image(x)))
        .collect();

    pipeline.do_render(&mut buf_ptrs)?;

    Ok(outputs)
}

#[instrument(skip(make_stage), err)]
pub(super) fn test_stage_consistency<
    S: RenderPipelineStage,
    InputT: ImageDataType,
    OutputT: ImageDataType + std::ops::Mul<Output = OutputT>,
>(
    make_stage: impl Fn() -> S,
    image_size: (usize, usize),
    num_image_channels: usize,
) -> Result<()>
where
    S::Type: RenderPipelineRunStage<Image<f64>>,
{
    let mut rng = rand_xorshift::XorShiftRng::seed_from_u64(0);
    let stage = make_stage();
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

    let base_output =
        make_and_run_simple_pipeline::<_, InputT, OutputT>(stage, &images, image_size, 0, 256)?;

    arbtest::arbtest(move |p| {
        let chunk_size = p.arbitrary::<u16>()?.saturating_add(1) as usize;
        let output = make_and_run_simple_pipeline::<_, InputT, OutputT>(
            make_stage(),
            &images,
            image_size,
            0,
            chunk_size,
        )
        .unwrap_or_else(|_| panic!("error running pipeline with chunk size {chunk_size}"));

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
