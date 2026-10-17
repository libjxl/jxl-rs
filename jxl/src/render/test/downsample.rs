// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::fmt;
use std::sync::Arc;

use crate::render::stages::GaborishStage;
use rand::SeedableRng;

use crate::api::{JxlColorType, JxlDataFormat, JxlOutputBuffer};
use crate::headers::Orientation;
use crate::image::{BufferRecycler, DataTypeTag, Image, Rect};
use crate::render::buffer_splitter::BufferSplitter;
use crate::render::{
    ErasedLocalState, LowMemoryRenderPipeline, RenderPipeline, RenderPipelineBuilder,
    RenderPipelineDownsampleStage, RowBatch, SimpleRenderPipeline,
};
use crate::util::ShiftRightCeil;

struct Average<const X: u8, const Y: u8>;

impl<const X: u8, const Y: u8> fmt::Display for Average<X, Y> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "average {X}x{Y}")
    }
}

impl<const X: u8, const Y: u8> RenderPipelineDownsampleStage for Average<X, Y> {
    type InputT = f32;
    type OutputT = f32;
    const DOWNSAMPLE: (u8, u8) = (X, Y);

    fn uses_channel(&self, _: usize) -> bool {
        true
    }

    fn process_row_batch(
        &self,
        _: (usize, usize),
        xsize: usize,
        input: &RowBatch<f32>,
        output: &mut [&mut [f32]],
        _: Option<&mut ErasedLocalState>,
        _: bool,
    ) {
        assert_eq!(input.num_channels(), output.len());
        assert!((1..=1 << Y).contains(&input.num_rows()));
        for (c, dst) in output.iter_mut().enumerate() {
            for x in 0..xsize {
                let start = x << X;
                let end = ((x + 1) << X).min(input.row(c, 0).len());
                let mut sum = 0.0;
                for y in 0..input.num_rows() {
                    for &sample in &input.row(c, y)[start..end] {
                        sum += sample;
                    }
                }
                dst[x] = sum / ((end - start) * input.num_rows()) as f32;
            }
        }
    }
}

fn run<P: RenderPipeline>(
    inputs: &[Image<f32>],
    output_size: (usize, usize),
    chunk_size: usize,
    configure: impl FnOnce(RenderPipelineBuilder<P>) -> RenderPipelineBuilder<P>,
) -> Vec<Image<f32>> {
    const LOG_GROUP: usize = 5;
    let size = inputs[0].size();
    let types = vec![(DataTypeTag::F32, None); inputs.len()];
    let mut builder = configure(RenderPipelineBuilder::<P>::new_with_chunk_size(
        &types,
        size,
        0,
        LOG_GROUP,
        chunk_size,
        Arc::new(BufferRecycler::new()),
    ));
    for c in 0..inputs.len() {
        builder = builder.add_save_stage(
            &[c],
            Orientation::Identity,
            c,
            JxlColorType::Grayscale,
            JxlDataFormat::f32(),
            false,
        );
    }
    let pipeline = builder.build().unwrap();
    let mut outputs: Vec<_> = inputs
        .iter()
        .map(|_| Image::<f32>::new(output_size).unwrap())
        .collect();
    let mut bufs: Vec<_> = outputs
        .iter_mut()
        .map(|im| {
            Some(JxlOutputBuffer::from_image_rect_mut(
                im.get_rect_mut(Rect {
                    origin: (0, 0),
                    size: output_size,
                })
                .into_raw(),
            ))
        })
        .collect();
    pipeline.check_buffer_sizes(&mut bufs).unwrap();
    let splitter = BufferSplitter::new(&mut bufs);
    let groups_x = size.0.shrc(LOG_GROUP);
    // Reverse submission exercises the neighbour-border storage as well as the core path.
    for g in (0..groups_x * size.1.shrc(LOG_GROUP)).rev() {
        for (c, image) in inputs.iter().enumerate() {
            let mut buf = pipeline.get_buffer::<f32>(c).unwrap();
            let origin = ((g % groups_x) << LOG_GROUP, (g / groups_x) << LOG_GROUP);
            for y in 0..buf.size().1.min(size.1 - origin.1) {
                let width = buf.size().0.min(size.0 - origin.0);
                buf.row_mut(y)[..width]
                    .copy_from_slice(&image.row(origin.1 + y)[origin.0..origin.0 + width]);
            }
            pipeline
                .set_buffer_for_group(c, g, true, buf, &splitter)
                .unwrap();
        }
    }
    drop(splitter);
    drop(bufs);
    outputs
}

#[test]
fn downsample_partial_batches() {
    for size in [(3, 3), (1, 1), (1, 3)] {
        let mut input = Image::<f32>::new(size).unwrap();
        for y in 0..size.1 {
            for x in 0..size.0 {
                input.row_mut(y)[x] = (y * 3 + x) as f32;
            }
        }
        let outputs = run::<SimpleRenderPipeline>(
            &[input.try_clone().unwrap()],
            (size.0.shrc(1), size.1.shrc(1)),
            256,
            |b| b.add_downsample_stage(Average::<1, 1>),
        );
        let low =
            run::<LowMemoryRenderPipeline>(&[input], (size.0.shrc(1), size.1.shrc(1)), 256, |b| {
                b.add_downsample_stage(Average::<1, 1>)
            });
        crate::tests::assert_image_eq!(&outputs[0], &low[0]);
        let expected: &[&[f32]] = match size {
            (3, 3) => &[&[2.0, 3.5], &[6.5, 8.0]],
            (1, 1) => &[&[0.0]],
            _ => &[&[1.5], &[6.0]],
        };
        for (y, row) in expected.iter().enumerate() {
            assert_eq!(outputs[0].row(y), *row);
        }
    }
}

#[test]
fn downsample_averages_only_valid_edge_samples() {
    for (size, expected) in [
        ((1, 3), &[1.0][..]),
        ((3, 1), &[1.0][..]),
        ((1, 5), &[1.5, 4.0][..]),
        ((5, 1), &[1.5, 4.0][..]),
        ((5, 5), &[9.0, 11.5, 21.5, 24.0][..]),
    ] {
        let mut input = Image::<f32>::new(size).unwrap();
        for y in 0..size.1 {
            for x in 0..size.0 {
                input.row_mut(y)[x] = (y * size.0 + x) as f32;
            }
        }
        let output_size = (size.0.shrc(2), size.1.shrc(2));
        let simple =
            run::<SimpleRenderPipeline>(&[input.try_clone().unwrap()], output_size, 256, |b| {
                b.add_downsample_stage(Average::<2, 2>)
            });
        let low = run::<LowMemoryRenderPipeline>(&[input], output_size, 256, |b| {
            b.add_downsample_stage(Average::<2, 2>)
        });
        crate::tests::assert_image_eq!(&simple[0], &low[0]);
        for y in 0..output_size.1 {
            assert_eq!(
                low[0].row(y),
                &expected[y * output_size.0..(y + 1) * output_size.0]
            );
        }
    }
}

fn check_batches<const X: u8, const Y: u8>(inputs: &[Image<f32>], chunk: usize) {
    let size = inputs[0].size();
    let size = (size.0.shrc(X), size.1.shrc(Y));
    let simple = run::<SimpleRenderPipeline>(inputs, size, chunk, |b| {
        b.add_downsample_stage(Average::<X, Y>)
    });
    let low = run::<LowMemoryRenderPipeline>(inputs, size, chunk, |b| {
        b.add_downsample_stage(Average::<X, Y>)
    });
    for (a, b) in simple.iter().zip(&low) {
        crate::tests::assert_image_eq!(a, b);
    }
}

#[test]
fn downsample_across_groups_and_chunks() {
    let mut rng = rand_xorshift::XorShiftRng::seed_from_u64(0);
    let inputs: Vec<_> = (0..3)
        .map(|_| Image::new_random((513, 515), &mut rng).unwrap())
        .collect();
    for chunk in [128, 256] {
        check_batches::<1, 1>(&inputs, chunk);
        check_batches::<2, 0>(&inputs, chunk);
        check_batches::<0, 2>(&inputs, chunk);
    }
}

#[test]
fn downsample_chained_with_border_stages() {
    let mut rng = rand_xorshift::XorShiftRng::seed_from_u64(1);
    let inputs: Vec<_> = (0..3)
        .map(|_| Image::new_random((515, 513), &mut rng).unwrap())
        .collect();
    let configure = |b: RenderPipelineBuilder<SimpleRenderPipeline>| {
        b.add_inout_stage(GaborishStage::new([0.1; 3], [0.2; 3]))
            .add_inout_stage(GaborishStage::new([0.1; 3], [0.2; 3]))
            .add_downsample_stage(Average::<1, 1>)
            .add_downsample_stage(Average::<1, 1>)
    };
    let simple = run::<SimpleRenderPipeline>(&inputs, (129, 129), 256, configure);
    let low = run::<LowMemoryRenderPipeline>(&inputs, (129, 129), 256, |b| {
        b.add_inout_stage(GaborishStage::new([0.1; 3], [0.2; 3]))
            .add_inout_stage(GaborishStage::new([0.1; 3], [0.2; 3]))
            .add_downsample_stage(Average::<1, 1>)
            .add_downsample_stage(Average::<1, 1>)
    });
    for (a, b) in simple.iter().zip(&low) {
        crate::tests::assert_image_eq!(a, b);
    }
}
