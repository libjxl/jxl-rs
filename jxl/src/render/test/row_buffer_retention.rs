// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use super::*;
use crate::render::for_each_chunk;
use crate::render::{Channels, ChannelsMut, ErasedLocalState, LowMemoryRenderPipeline};
use jxl_simd::{F32SimdVec, ScalarDescriptor, SimdDescriptor};

struct RowNumberStage;

impl std::fmt::Display for RowNumberStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("RowNumber")
    }
}

impl RenderPipelineInOutStage for RowNumberStage {
    type InputT = f32;
    type OutputT = f32;
    const BORDER: (u8, u8) = (0, 0);
    const SHIFT: (u8, u8) = (0, 0);

    fn uses_channel(&self, c: usize) -> bool {
        c < 2
    }

    fn process_row_chunk(
        &self,
        position: (usize, usize),
        xsize: usize,
        input: &Channels<f32>,
        output: &mut ChannelsMut<f32>,
        _state: Option<&mut ErasedLocalState>,
        _previous: bool,
    ) {
        let d = ScalarDescriptor::new().unwrap();
        let value = <ScalarDescriptor as SimdDescriptor>::F32Vec::splat(d, position.1 as f32);
        for_each_chunk(
            d,
            xsize,
            input.view::<2, 1, 0>(),
            output.view::<2, 1, 1>(),
            |_x, _input, out| {
                out.store::<_, 0>(d, 0, value);
                out.store::<_, 1>(d, 0, value);
            },
        );
    }
}

struct VerticalFilter;

impl std::fmt::Display for VerticalFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("VerticalFilter")
    }
}

impl RenderPipelineInOutStage for VerticalFilter {
    type InputT = f32;
    type OutputT = f32;
    const BORDER: (u8, u8) = (0, 1);
    const SHIFT: (u8, u8) = (0, 0);

    fn uses_channel(&self, c: usize) -> bool {
        c == 1
    }

    fn process_row_chunk(
        &self,
        _position: (usize, usize),
        xsize: usize,
        input: &Channels<f32>,
        output: &mut ChannelsMut<f32>,
        _state: Option<&mut ErasedLocalState>,
        _previous: bool,
    ) {
        let d = ScalarDescriptor::new().unwrap();
        let quarter = <ScalarDescriptor as SimdDescriptor>::F32Vec::splat(d, 0.25);
        for_each_chunk(
            d,
            xsize,
            input.view::<1, 3, 0>(),
            output.view::<1, 1, 1>(),
            |_x, input, out| {
                let above = input.load::<_, 0>(d, -1, 0);
                let center = input.load::<_, 0>(d, 0, 0);
                let below = input.load::<_, 0>(d, 1, 0);
                out.store::<_, 0>(d, 0, (above + center + center + below) * quarter);
            },
        );
    }
}

fn render_branch<P: RenderPipeline>() -> Result<Vec<Image<f32>>> {
    let size = (8, 32);
    let mut builder = RenderPipelineBuilder::<P>::new(
        &[(DataTypeTag::F32, None); 2],
        size,
        0,
        8,
        Arc::new(BufferRecycler::new()),
    );

    // Channel 0, 1
    // |
    // RowNumberStage
    // +-> Channel 0 -> Save
    // `-> Channel 1 -> VerticalFilter x4 -> Save

    builder = builder.add_inout_stage(RowNumberStage);
    builder = builder.add_save_stage(
        &[0],
        Orientation::Identity,
        0,
        JxlColorType::Grayscale,
        JxlDataFormat::f32(),
        false,
    );

    for _ in 0..4 {
        builder = builder.add_inout_stage(VerticalFilter);
    }
    let pipeline = builder
        .add_save_stage(
            &[1],
            Orientation::Identity,
            1,
            JxlColorType::Grayscale,
            JxlDataFormat::f32(),
            false,
        )
        .build()?;

    let mut outputs = vec![Image::<f32>::new(size)?, Image::<f32>::new(size)?];
    let mut buffers: Vec<_> = outputs
        .iter_mut()
        .map(|image| {
            Some(JxlOutputBuffer::from_image_rect_mut(
                image
                    .get_rect_mut(Rect {
                        origin: (0, 0),
                        size,
                    })
                    .into_raw(),
            ))
        })
        .collect();
    let splitter = BufferSplitter::new(&mut buffers);
    for c in 0..2 {
        pipeline.set_buffer_for_group(c, 0, true, Image::<f32>::new(size)?, &splitter)?;
    }
    drop(splitter);
    Ok(outputs)
}

#[test]
fn branched_pipeline_four_filters() {
    let simple = render_branch::<SimpleRenderPipeline>().unwrap();
    let lowmem = render_branch::<LowMemoryRenderPipeline>().unwrap();
    for y in 0..32 {
        assert_eq!(simple[0].row(y), &[y as f32; 8]);
        assert_eq!(lowmem[0].row(y), simple[0].row(y), "channel A, row {y}");
    }
    for y in 0..32 {
        assert_eq!(lowmem[1].row(y), simple[1].row(y), "channel B, row {y}");
    }
}
