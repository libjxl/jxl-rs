// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use super::*;
use crate::api::{JxlDecoderOptions, TestOptions};
use crate::image::BufferRecycler;
use crate::tests::decode::read_headers_and_toc;

fn render_with_input_preference<P: RenderPipeline + Send + Sync + 'static>(
    simple: bool,
    explicit_downsample: bool,
) -> Image<f32> {
    let data = std::fs::read("resources/test/narrow_edge_group.jxl").unwrap();
    let (file_header, mut header, toc) = read_headers_and_toc(&data).unwrap();
    assert_eq!(header.upsampling, 1);
    assert!(header.is444());
    // Reference processing keeps the virtual grid fixed across both stage layouts.
    header.can_be_referenced = true;
    header.save_before_ct = true;
    let size = header.size_upsampled();
    let options = JxlDecoderOptions {
        downsample_factor: 1,
        test_options: TestOptions {
            use_simple_pipeline: simple,
            ..Default::default()
        },
        ..Default::default()
    };
    let recycler = Arc::new(BufferRecycler::new());
    let mut builder = RenderPipelineBuilder::<P>::new(
        &[(DataTypeTag::F32, None)],
        size,
        0,
        header.log_group_dim(),
        recycler.clone(),
    )
    .with_channel_output_downsamples(vec![1]);
    if explicit_downsample {
        builder = builder.add_inout_stage(DownsampleFrame::<1>::new(size, vec![None]));
    }
    let pipeline = builder
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
    let output_size = (size.0.shrc(1), size.1.shrc(1));
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
    let state = DecoderState::new(file_header, &options, false);
    let mut frame = Frame::from_header_and_toc(header, toc, state, (0, 0), recycler).unwrap();
    frame.render_pipeline = Some(pipeline);
    let level = usize::from(!explicit_downsample);
    assert_eq!(frame.preferred_render_input_shift(0), (level, level));

    let splitter = BufferSplitter::new(&mut buffers);
    for group in (0..frame.header.num_groups()).rev() {
        let rect = frame.header.group_rect(group);
        let padded_size = (
            rect.size.0.next_multiple_of(16),
            rect.size.1.next_multiple_of(16),
        );
        let mut decoded = Image::new_with_value(padded_size, -1000.0).unwrap();
        for y in 0..rect.size.1 {
            for x in 0..rect.size.0 {
                decoded.row_mut(y)[x] = (rect.origin.0 + x + 10 * (rect.origin.1 + y)) as f32;
            }
        }
        frame
            .set_render_input(0, group, true, (0, 0), decoded, &splitter)
            .unwrap();
    }
    drop(splitter);
    output
}

#[test]
fn pipeline_input_preference_follows_stage_layout() {
    let expected = render_with_input_preference::<SimpleRenderPipeline>(true, true);
    for explicit_downsample in [false, true] {
        let lowmem =
            render_with_input_preference::<LowMemoryRenderPipeline>(false, explicit_downsample);
        let simple =
            render_with_input_preference::<SimpleRenderPipeline>(true, explicit_downsample);
        assert_eq!(lowmem.size(), expected.size());
        assert_eq!(simple.size(), expected.size());
        for y in 0..expected.size().1 {
            assert_eq!(lowmem.row(y), expected.row(y));
            assert_eq!(simple.row(y), expected.row(y));
        }
    }
}
