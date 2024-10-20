// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::render::{RenderPipelineInOutStage, RenderPipelineStage};

pub struct NearestNeighbourUpsample {
    channel: usize,
}

impl NearestNeighbourUpsample {
    pub fn new(channel: usize) -> NearestNeighbourUpsample {
        NearestNeighbourUpsample { channel }
    }
}

impl std::fmt::Display for NearestNeighbourUpsample {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "2x2 nearest neighbour upsample of channel {}",
            self.channel
        )
    }
}

impl RenderPipelineStage for NearestNeighbourUpsample {
    type Type = RenderPipelineInOutStage<u8, u8, 0, 0, 1, 1>;

    fn uses_channel(&self, c: usize) -> bool {
        c == self.channel
    }

    fn process_row_chunk(
        &mut self,
        _position: (usize, usize),
        xsize: usize,
        row: &mut [(&[&[u8]], &mut [&mut [u8]])],
    ) {
        let (input, output) = &mut row[0];
        for i in 0..xsize {
            output[0][i * 2] = input[0][i];
            output[0][i * 2 + 1] = input[0][i];
            output[1][i * 2] = input[0][i];
            output[1][i * 2 + 1] = input[0][i];
        }
    }
}

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

impl RenderPipelineStage for HorizontalChromaUpsample {
    type Type = RenderPipelineInOutStage<f32, f32, 1, 0, 1, 0>;

    fn uses_channel(&self, c: usize) -> bool {
        c == self.channel
    }

    fn process_row_chunk(
        &mut self,
        _position: (usize, usize),
        xsize: usize,
        row: &mut [(&[&[f32]], &mut [&mut [f32]])],
    ) {
        let (input, output) = &mut row[0];
        for i in 0..xsize {
            let scaled_cur = input[0][i + 1] * 0.75;
            let prev = input[0][i];
            let next = input[0][i + 2];
            let left = 0.25 * prev + scaled_cur;
            let right = 0.25 * next + scaled_cur;
            output[0][2 * i] = left;
            output[0][2 * i + 1] = right;
        }
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

impl RenderPipelineStage for VerticalChromaUpsample {
    type Type = RenderPipelineInOutStage<f32, f32, 0, 1, 0, 1>;

    fn uses_channel(&self, c: usize) -> bool {
        c == self.channel
    }

    fn process_row_chunk(
        &mut self,
        _position: (usize, usize),
        xsize: usize,
        row: &mut [(&[&[f32]], &mut [&mut [f32]])],
    ) {
        let (input, output) = &mut row[0];
        for i in 0..xsize {
            let scaled_cur = input[1][i] * 0.75;
            let prev = input[0][i];
            let next = input[2][i];
            let up = 0.25 * prev + scaled_cur;
            let down = 0.25 * next + scaled_cur;
            output[0][i] = up;
            output[1][i] = down;
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{error::Result, image::Image, render::test::make_and_run_simple_pipeline};
    use rand::SeedableRng;
    use test_log::test;

    #[test]
    fn nn_consistency() -> Result<()> {
        crate::render::test::test_stage_consistency::<_, u8, u8>(
            NearestNeighbourUpsample::new(0),
            (500, 500),
            1,
        )
    }

    #[test]
    fn test_nn() -> Result<()> {
        let image_size = (500, 400);
        let input_size = (image_size.0 / 2, image_size.1 / 2);
        let mut rng = rand_xorshift::XorShiftRng::seed_from_u64(0);
        let input = vec![Image::<u8>::new_random(input_size, &mut rng)?];
        let stage = NearestNeighbourUpsample::new(0);
        let output: Vec<Image<u8>> =
            make_and_run_simple_pipeline(stage, &input, image_size, 256)?.1;
        assert_eq!(image_size, output[0].size());
        for y in 0..image_size.1 {
            for x in 0..image_size.0 {
                let ix = x / 2;
                let iy = y / 2;
                let i = input[0].as_rect().row(iy)[ix];
                let o = output[0].as_rect().row(y)[x];
                if i != o {
                    panic!("Mismatch at output position {x}x{y}: {i} vs output {o}");
                }
            }
        }

        Ok(())
    }

    #[test]
    fn hchr_consistency() -> Result<()> {
        crate::render::test::test_stage_consistency::<_, f32, f32>(
            HorizontalChromaUpsample::new(0),
            (500, 500),
            1,
        )
    }

    #[test]
    fn test_hchr() -> Result<()> {
        let mut input = Image::new((3, 1))?;
        input
            .as_rect_mut()
            .row(0)
            .copy_from_slice(&[1.0f32, 2.0, 4.0]);
        let stage = HorizontalChromaUpsample::new(0);
        let output: Vec<Image<f32>> = make_and_run_simple_pipeline(stage, &[input], (6, 1), 256)?.1;
        assert_eq!(output[0].as_rect().row(0), [1.0, 1.25, 1.75, 2.5, 3.5, 4.0]);
        Ok(())
    }

    #[test]
    fn vchr_consistency() -> Result<()> {
        crate::render::test::test_stage_consistency::<_, f32, f32>(
            VerticalChromaUpsample::new(0),
            (500, 500),
            1,
        )
    }

    #[test]
    fn test_vchr() -> Result<()> {
        let mut input = Image::new((1, 3))?;
        input.as_rect_mut().row(0)[0] = 1.0f32;
        input.as_rect_mut().row(1)[0] = 2.0f32;
        input.as_rect_mut().row(2)[0] = 4.0f32;
        let stage = VerticalChromaUpsample::new(0);
        let output: Vec<Image<f32>> = make_and_run_simple_pipeline(stage, &[input], (1, 6), 256)?.1;
        assert_eq!(output[0].as_rect().row(0)[0], 1.0);
        assert_eq!(output[0].as_rect().row(1)[0], 1.25);
        assert_eq!(output[0].as_rect().row(2)[0], 1.75);
        assert_eq!(output[0].as_rect().row(3)[0], 2.5);
        assert_eq!(output[0].as_rect().row(4)[0], 3.5);
        assert_eq!(output[0].as_rect().row(5)[0], 4.0);
        Ok(())
    }
}