// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use super::*;
use crate::render::{LowMemoryRenderPipeline, stages::Downsample};

fn run<P: RenderPipeline>(
    image: &Image<f32>,
    level: u8,
    chunk: usize,
    reverse: bool,
    stages: impl FnOnce(RenderPipelineBuilder<P>) -> RenderPipelineBuilder<P>,
) -> Image<f32> {
    let size = image.size();
    let builder = RenderPipelineBuilder::<P>::new_with_chunk_size(
        &[(DataTypeTag::F32, None)],
        size,
        0,
        8,
        chunk,
        Arc::new(BufferRecycler::new()),
    )
    .with_output_downsample((level, level));
    let mut pipeline = stages(builder)
        .add_save_stage(
            &[0],
            Orientation::Identity,
            0,
            JxlColorType::Grayscale,
            JxlDataFormat::f32(),
            false,
        )
        .build()
        .unwrap();
    let output_size = (size.0.shrc(level), size.1.shrc(level));
    let mut output = Image::new_with_value(output_size, f32::NAN).unwrap();
    let mut buffers = [Some(JxlOutputBuffer::from_image_rect_mut(
        output
            .get_rect_mut(Rect {
                origin: (0, 0),
                size: output_size,
            })
            .into_raw(),
    ))];
    pipeline.check_buffer_sizes(&mut buffers).unwrap();
    let splitter = BufferSplitter::new(&mut buffers);
    let mut groups: Vec<_> = (0..size.0.shrc(8) * size.1.shrc(8)).collect();
    if reverse {
        groups.reverse();
    }
    // Exercise reused accumulators as well as fresh buffers. The first pass
    // deliberately differs so stale contributions cannot silently match.
    for complete in [false, true] {
        if complete {
            for &g in &groups {
                pipeline.mark_group_to_rerender(g);
            }
            groups.reverse();
        }
        for &g in &groups {
            let mut input = extract_group_rect(image, g, (8, 8)).unwrap();
            if !complete {
                for y in 0..input.size().1 {
                    input.row_mut(y).fill(-7.0);
                }
            }
            pipeline
                .set_buffer_for_group(0, g, complete, input, &splitter)
                .unwrap();
        }
    }
    drop(splitter);
    output
}

fn reference(image: &Image<f32>, shift: u8) -> Image<f32> {
    let factor = 1 << shift;
    let (w, h) = image.size();
    let mut out = Image::new((w.shrc(shift), h.shrc(shift))).unwrap();
    for y in 0..out.size().1 {
        for x in 0..out.size().0 {
            let mut sum = 0.0;
            let mut count = 0;
            for iy in y * factor..((y + 1) * factor).min(h) {
                for ix in x * factor..((x + 1) * factor).min(w) {
                    sum += image.row(iy)[ix];
                    count += 1;
                }
            }
            out.row_mut(y)[x] = sum / count as f32;
        }
    }
    out
}

fn check(actual: &Image<f32>, expected: &Image<f32>) {
    assert_eq!(actual.size(), expected.size());
    for y in 0..actual.size().1 {
        for x in 0..actual.size().0 {
            let (a, b) = (actual.row(y)[x], expected.row(y)[x]);
            assert!(
                (a - b).abs() <= 1e-5 * b.abs().max(1.0),
                "{x},{y}: {a} != {b}"
            );
        }
    }
}

fn exercise(image: &Image<f32>) {
    let size = image.size();
    let expected = reference(image, 3);
    for chunk in [1, 7, 31, 256] {
        let simple = run::<SimpleRenderPipeline>(image, 3, chunk, false, |b| {
            b.add_inout_stage(Downsample::<3, 3>::new(0, size))
        });
        check(&simple, &expected);
    }
    for reverse in [false, true] {
        let lowmem = run::<LowMemoryRenderPipeline>(image, 3, 256, reverse, |b| {
            b.add_inout_stage(Downsample::<3, 3>::new(0, size))
        });
        check(&lowmem, &expected);
    }
    // Compare two actual 8x stages, including a truncated final block, rather
    // than comparing with a different (single 64x box) edge-weighting policy.
    let small = (size.0.shrc(3), size.1.shrc(3));
    let expected = reference(&expected, 3);
    let stages = |b: RenderPipelineBuilder<LowMemoryRenderPipeline>| {
        b.add_inout_stage(Downsample::<3, 3>::new(0, size))
            .add_inout_stage(Downsample::<3, 3>::new(0, small))
    };
    check(
        &run::<LowMemoryRenderPipeline>(image, 6, 256, true, stages),
        &expected,
    );
}

#[test]
fn accumulation_downsample_2x_and_4x() {
    fn exercise_factor<const K: u8>(image: &Image<f32>) {
        let size = image.size();
        let expected = reference(image, K);
        let simple = run::<SimpleRenderPipeline>(image, K, 7, false, |b| {
            b.add_inout_stage(Downsample::<K, K>::new(0, size))
        });
        let lowmem = run::<LowMemoryRenderPipeline>(image, K, 256, true, |b| {
            b.add_inout_stage(Downsample::<K, K>::new(0, size))
        });
        check(&simple, &expected);
        check(&lowmem, &expected);
    }
    let size = (259, 517);
    let mut image = Image::new(size).unwrap();
    for y in 0..size.1 {
        for x in 0..size.0 {
            image.row_mut(y)[x] = ((x * 7 + y * 13) % 251) as f32;
        }
    }
    exercise_factor::<1>(&image);
    exercise_factor::<2>(&image);
}

#[test]
fn accumulation_downsample_edges_and_groups() {
    for size in [(1, 1), (1, 17), (7, 5), (257, 259), (513, 519)] {
        let mut image = Image::new(size).unwrap();
        for y in 0..size.1 {
            for x in 0..size.0 {
                image.row_mut(y)[x] = ((x * 7 + y * 13) % 251) as f32;
            }
        }
        exercise(&image);
    }
}

struct VerticalBlur;

impl std::fmt::Display for VerticalBlur {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("vertical blur")
    }
}

impl RenderPipelineInOutStage for VerticalBlur {
    type InputT = f32;
    type OutputT = f32;
    const BORDER: (u8, u8) = (0, 1);
    const SHIFT: (i8, i8) = (0, 0);
    fn uses_channel(&self, c: usize) -> bool {
        c == 0
    }
    fn process_row_chunk(
        &self,
        _: (usize, usize),
        xsize: usize,
        input: &crate::render::Channels<f32>,
        output: &mut crate::render::ChannelsMut<f32>,
        _: Option<&mut crate::render::ErasedLocalState>,
        _: bool,
    ) {
        let out = output.row_mut(0, 0);
        for x in 0..xsize {
            out[x] = (input.row_at(0, -1)[x] + input.row_at(0, 0)[x] + input.row_at(0, 1)[x]) / 3.0;
        }
    }
}

#[test]
fn accumulation_downsample_with_downstream_border() {
    for size in [(1, 1), (7, 5), (257, 259), (513, 519)] {
        let mut image = Image::new(size).unwrap();
        for y in 0..size.1 {
            for x in 0..size.0 {
                image.row_mut(y)[x] = ((x * 7 + y * 13) % 251) as f32;
            }
        }
        let simple = run::<SimpleRenderPipeline>(&image, 3, 32, false, |b| {
            b.add_inout_stage(Downsample::<3, 3>::new(0, size))
                .add_inout_stage(VerticalBlur)
        });
        for reverse in [false, true] {
            let lowmem = run::<LowMemoryRenderPipeline>(&image, 3, 256, reverse, |b| {
                b.add_inout_stage(Downsample::<3, 3>::new(0, size))
                    .add_inout_stage(VerticalBlur)
            });
            check(&lowmem, &simple);
        }
    }
}

#[test]
fn accumulation_downsample_mixed_with_upsampling() {
    use crate::render::stages::HorizontalChromaUpsample;
    let size = (513, 519);
    let mut image = Image::new(size).unwrap();
    for y in 0..size.1 {
        for x in 0..size.0 {
            image.row_mut(y)[x] = ((x * 7 + y * 13) % 251) as f32;
        }
    }
    let expanded = (size.0.shrc(2), size.1.shrc(3));
    let simple = run::<SimpleRenderPipeline>(&image, 3, 32, false, |b| {
        b.add_inout_stage(Downsample::<3, 3>::new(0, size))
            .add_inout_stage(HorizontalChromaUpsample::new(0))
            .add_inout_stage(Downsample::<1, 0>::new(0, expanded))
    });
    for reverse in [false, true] {
        let lowmem = run::<LowMemoryRenderPipeline>(&image, 3, 256, reverse, |b| {
            b.add_inout_stage(Downsample::<3, 3>::new(0, size))
                .add_inout_stage(HorizontalChromaUpsample::new(0))
                .add_inout_stage(Downsample::<1, 0>::new(0, expanded))
        });
        check(&lowmem, &simple);
    }
}

#[test]
fn accumulation_downsample_vertical_upsampling_and_64x_border() {
    use crate::render::stages::VerticalChromaUpsample;
    for size in [(7, 5), (257, 259), (513, 519)] {
        let mut image = Image::new(size).unwrap();
        for y in 0..size.1 {
            for x in 0..size.0 {
                image.row_mut(y)[x] = ((x * 7 + y * 13) % 251) as f32;
            }
        }
        let expanded = (size.0.shrc(3), size.1.shrc(2));
        let simple = run::<SimpleRenderPipeline>(&image, 3, 32, false, |b| {
            b.add_inout_stage(Downsample::<3, 3>::new(0, size))
                .add_inout_stage(VerticalChromaUpsample::new(0))
                .add_inout_stage(Downsample::<0, 1>::new(0, expanded))
        });
        let lowmem = run::<LowMemoryRenderPipeline>(&image, 3, 256, true, |b| {
            b.add_inout_stage(Downsample::<3, 3>::new(0, size))
                .add_inout_stage(VerticalChromaUpsample::new(0))
                .add_inout_stage(Downsample::<0, 1>::new(0, expanded))
        });
        check(&lowmem, &simple);

        let small = (size.0.shrc(3), size.1.shrc(3));
        let simple = run::<SimpleRenderPipeline>(&image, 6, 32, false, |b| {
            b.add_inout_stage(Downsample::<3, 3>::new(0, size))
                .add_inout_stage(Downsample::<3, 3>::new(0, small))
                .add_inout_stage(VerticalBlur)
        });
        let lowmem = run::<LowMemoryRenderPipeline>(&image, 6, 256, true, |b| {
            b.add_inout_stage(Downsample::<3, 3>::new(0, size))
                .add_inout_stage(Downsample::<3, 3>::new(0, small))
                .add_inout_stage(VerticalBlur)
        });
        check(&lowmem, &simple);
    }
}

#[test]
fn accumulation_downsample_64x_horizontal_border() {
    use crate::render::stages::HorizontalChromaUpsample;
    let size = (513, 519);
    let small = (size.0.shrc(3), size.1.shrc(3));
    let expanded = (size.0.shrc(5), size.1.shrc(6));
    // This combination used to underflow BufferFiller's left-padding offset.
    // Until row buffers support dynamic left padding, reject it at build time.
    let result = RenderPipelineBuilder::<LowMemoryRenderPipeline>::new(
        &[(DataTypeTag::F32, None)],
        size,
        0,
        8,
        Arc::new(BufferRecycler::new()),
    )
    .with_output_downsample((6, 6))
    .add_inout_stage(Downsample::<3, 3>::new(0, size))
    .add_inout_stage(Downsample::<3, 3>::new(0, small))
    .add_inout_stage(HorizontalChromaUpsample::new(0))
    .add_inout_stage(Downsample::<1, 0>::new(0, expanded))
    .add_save_stage(
        &[0],
        Orientation::Identity,
        0,
        JxlColorType::Grayscale,
        JxlDataFormat::f32(),
        false,
    )
    .build();
    assert!(matches!(
        result,
        Err(crate::error::Error::PipelineDownsampleBorderTooLarge)
    ));
}

#[test]
fn accumulation_downsample_decoded_samples() {
    use crate::api::JxlPixelFormat;
    use crate::tests::decode::{DecodeParams, decode};
    for name in ["basic.jxl", "lossy_with_icc.jxl"] {
        let bytes = std::fs::read(format!("resources/test/{name}")).unwrap();
        let frames = decode::<f32>(
            &bytes,
            DecodeParams {
                pixel_format: Some(JxlPixelFormat {
                    color_type: JxlColorType::Rgb,
                    color_data_format: Some(JxlDataFormat::f32()),
                    extra_channel_format: vec![],
                }),
                ..Default::default()
            },
        )
        .unwrap();
        let rgb = &frames[0][0];
        let size = (rgb.size().0 / 3, rgb.size().1);
        let mut red = Image::new(size).unwrap();
        for y in 0..size.1 {
            for x in 0..size.0 {
                red.row_mut(y)[x] = rgb.row(y)[3 * x];
            }
        }
        exercise(&red);
    }
}
