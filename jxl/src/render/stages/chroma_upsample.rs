// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::render::RenderPipelineInOutStage;
use jxl_simd::{F32SimdVec, simd_function};

pub struct HorizontalChromaUpsample {
    channel: usize,
}

impl HorizontalChromaUpsample {
    pub fn new(channel: usize) -> HorizontalChromaUpsample {
        HorizontalChromaUpsample { channel }
    }
}

impl std::fmt::Display for HorizontalChromaUpsample {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "chroma upsample of channel {}, horizontally",
            self.channel
        )
    }
}

// SIMD horizontal chroma upsampling matching C++ Highway implementation
simd_function!(
    hchroma_upsample_simd_dispatch,
    d: D,
    fn hchroma_upsample_simd(
        input: &[f32],
        output: &mut [f32],
        xsize: usize,
    ) {
        let simd_width = D::F32Vec::LEN;

        // Precompute constants
        let c025 = D::F32Vec::splat(d, 0.25);
        let c075 = D::F32Vec::splat(d, 0.75);

        // SIMD loop
        let mut x = 0;
        while x + simd_width <= xsize {
            // Load: prev, cur, next
            let prev_vec = D::F32Vec::load(d, &input[x..]);
            let cur_vec = D::F32Vec::load(d, &input[x+1..]);
            let next_vec = D::F32Vec::load(d, &input[x+2..]);

            // Compute: left = 0.25 * prev + 0.75 * cur
            let left = prev_vec.mul_add(c025, cur_vec * c075);

            // Compute: right = 0.25 * next + 0.75 * cur
            let right = next_vec.mul_add(c025, cur_vec * c075);

            // Interleaved store: [left0, right0, left1, right1, ...]
            D::F32Vec::store_interleaved_2(left, right, output, x*2);

            x += simd_width;
        }

        // Scalar remainder
        for i in x..xsize {
            let scaled_cur = input[i + 1] * 0.75;
            let prev = input[i];
            let next = input[i + 2];
            let left = 0.25 * prev + scaled_cur;
            let right = 0.25 * next + scaled_cur;
            output[2 * i] = left;
            output[2 * i + 1] = right;
        }
    }
);

impl RenderPipelineInOutStage for HorizontalChromaUpsample {
    type InputT = f32;
    type OutputT = f32;
    const SHIFT: (u8, u8) = (1, 0);
    const BORDER: (u8, u8) = (1, 0);

    fn uses_channel(&self, c: usize) -> bool {
        c == self.channel
    }

    fn process_row_chunk(
        &self,
        _position: (usize, usize),
        xsize: usize,
        input_rows: &[&[&[f32]]],
        output_rows: &mut [&mut [&mut [f32]]],
        _state: Option<&mut dyn std::any::Any>,
    ) {
        let input = input_rows[0][0];
        let output = &mut output_rows[0][0];
        hchroma_upsample_simd_dispatch(input, output, xsize);
    }
}

pub struct VerticalChromaUpsample {
    channel: usize,
}

impl VerticalChromaUpsample {
    pub fn new(channel: usize) -> VerticalChromaUpsample {
        VerticalChromaUpsample { channel }
    }
}

impl std::fmt::Display for VerticalChromaUpsample {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "chroma upsample of channel {}, vertically", self.channel)
    }
}

// SIMD vertical chroma upsampling matching C++ Highway implementation
simd_function!(
    vchroma_upsample_simd_dispatch,
    d: D,
    fn vchroma_upsample_simd(
        input_prev: &[f32],
        input_cur: &[f32],
        input_next: &[f32],
        output_up: &mut [f32],
        output_down: &mut [f32],
        xsize: usize,
    ) {
        let simd_width = D::F32Vec::LEN;

        // Precompute constants
        let c025 = D::F32Vec::splat(d, 0.25);
        let c075 = D::F32Vec::splat(d, 0.75);

        // SIMD loop
        let mut x = 0;
        while x + simd_width <= xsize {
            let prev_vec = D::F32Vec::load(d, &input_prev[x..]);
            let cur_vec = D::F32Vec::load(d, &input_cur[x..]);
            let next_vec = D::F32Vec::load(d, &input_next[x..]);

            // Compute: up = 0.25 * prev + 0.75 * cur
            let up = prev_vec.mul_add(c025, cur_vec * c075);

            // Compute: down = 0.25 * next + 0.75 * cur
            let down = next_vec.mul_add(c025, cur_vec * c075);

            // Store results
            up.store(&mut output_up[x..]);
            down.store(&mut output_down[x..]);

            x += simd_width;
        }

        // Scalar remainder
        for i in x..xsize {
            let scaled_cur = input_cur[i] * 0.75;
            let prev = input_prev[i];
            let next = input_next[i];
            output_up[i] = 0.25 * prev + scaled_cur;
            output_down[i] = 0.25 * next + scaled_cur;
        }
    }
);

impl RenderPipelineInOutStage for VerticalChromaUpsample {
    type InputT = f32;
    type OutputT = f32;
    const SHIFT: (u8, u8) = (0, 1);
    const BORDER: (u8, u8) = (0, 1);

    fn uses_channel(&self, c: usize) -> bool {
        c == self.channel
    }

    fn process_row_chunk(
        &self,
        _position: (usize, usize),
        xsize: usize,
        input_rows: &[&[&[f32]]],
        output_rows: &mut [&mut [&mut [f32]]],
        _state: Option<&mut dyn std::any::Any>,
    ) {
        let input = input_rows[0];
        let [output_up, output_down] = output_rows[0] else {
            panic!("Expected 2 output rows for vertical chroma upsampling");
        };
        vchroma_upsample_simd_dispatch(input[0], input[1], input[2], output_up, output_down, xsize);
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{error::Result, image::Image, render::test::make_and_run_simple_pipeline};
    use test_log::test;

    #[test]
    fn hchr_consistency() -> Result<()> {
        crate::render::test::test_stage_consistency(
            || HorizontalChromaUpsample::new(0),
            (500, 500),
            1,
        )
    }

    #[test]
    fn test_hchr() -> Result<()> {
        let mut input = Image::new((3, 1))?;
        input.row_mut(0).copy_from_slice(&[1.0f32, 2.0, 4.0]);
        let stage = HorizontalChromaUpsample::new(0);
        let output: Vec<Image<f32>> =
            make_and_run_simple_pipeline(stage, &[input], (6, 1), 0, 256)?;
        assert_eq!(output[0].row(0), [1.0, 1.25, 1.75, 2.5, 3.5, 4.0]);
        Ok(())
    }

    #[test]
    fn vchr_consistency() -> Result<()> {
        crate::render::test::test_stage_consistency(
            || VerticalChromaUpsample::new(0),
            (500, 500),
            1,
        )
    }

    #[test]
    fn test_vchr() -> Result<()> {
        let mut input = Image::new((1, 3))?;
        input.row_mut(0)[0] = 1.0f32;
        input.row_mut(1)[0] = 2.0f32;
        input.row_mut(2)[0] = 4.0f32;
        let stage = VerticalChromaUpsample::new(0);
        let output: Vec<Image<f32>> =
            make_and_run_simple_pipeline(stage, &[input], (1, 6), 0, 256)?;
        assert_eq!(output[0].row(0)[0], 1.0);
        assert_eq!(output[0].row(1)[0], 1.25);
        assert_eq!(output[0].row(2)[0], 1.75);
        assert_eq!(output[0].row(3)[0], 2.5);
        assert_eq!(output[0].row(4)[0], 3.5);
        assert_eq!(output[0].row(5)[0], 4.0);
        Ok(())
    }
}
