// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//TODO(firsching): implement Upsample4x and Upsample8x -- use templates

use crate::{
    headers::CustomTransformData,
    render::{RenderPipelineInOutStage, RenderPipelineStage},
};

pub struct Upsample2x {
    kernel: [[[[f32; 5]; 5]; 2]; 2],
    channel: usize,
}

impl Upsample2x {
    #[allow(dead_code)]
    pub fn new(ups_factors: &CustomTransformData, channel: usize) -> Upsample2x {
        let weights = ups_factors.weights2;
        let n = 1;
        let mut kernel = [[[[0.0; 5]; 5]; 2]; 2];
        for i in 0..5 * n {
            for j in 0..5 * n {
                let y = i.min(j);
                let x = i.max(j);
                let y = y as isize;
                let x = x as isize;
                let n = n as isize;
                let index = (5 * n * y - y * (y - 1) / 2 + x - y) as usize;
                // Filling in the top left corner from the weights
                kernel[j / 5][i / 5][j % 5][i % 5] = weights[index];
                // Mirroring to get the rest of the kernel.
                kernel[(n as usize) + j / 5][i / 5][4 - (j % 5)][i % 5] = weights[index];
                kernel[j / 5][(n as usize) + i / 5][j % 5][4 - (i % 5)] = weights[index];
                kernel[(n as usize) + j / 5][(n as usize) + i / 5][4 - (j % 5)][4 - (i % 5)] =
                    weights[index];
            }
        }
        Upsample2x { kernel, channel }
    }
}

impl std::fmt::Display for Upsample2x {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "2x2 upsampling of channel {}", self.channel)
    }
}

impl RenderPipelineStage for Upsample2x {
    type Type = RenderPipelineInOutStage<f32, f32, 2, 2, 1, 1>;

    fn uses_channel(&self, c: usize) -> bool {
        c == self.channel
    }
    /// Processes a chunk of a row, applying 2x2 upsampling using a 5x5 kernel.
    /// Each input value expands into a 2x2 region in the output, based on neighboring inputs.
    fn process_row_chunk(
        &mut self,
        _position: (usize, usize),
        xsize: usize,
        row: &mut [(&[&[f32]], &mut [&mut [f32]])],
    ) {
        let n = 2;
        let (input, output) = &mut row[0];

        for x in 0..xsize {
            // Upsample this input value into a 2x2 region in the output
            for di in 0..n {
                for dj in 0..n {
                    let mut output_val = 0.0f32;
                    // Iterate over the input rows and columns
                    for i in 0..5 {
                        for j in 0..5 {
                            let input_value = input[i][j + x];
                            output_val += input_value * self.kernel[di][dj][i % 5][j % 5];
                        }
                    }
                    output[di][dj + 2 * x] = output_val;
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        error::Result, image::Image, render::test::make_and_run_simple_pipeline,
        util::test::assert_almost_eq,
    };
    use test_log::test;

    #[test]
    fn upsample2x_consistency() -> Result<()> {
        let ups_factors = CustomTransformData::default();
        crate::render::test::test_stage_consistency::<_, f32, f32>(
            Upsample2x::new(&ups_factors, 0),
            (500, 500),
            1,
        )
    }

    #[test]
    fn upsample2x_constant() -> Result<()> {
        let ups_factors = CustomTransformData::default();
        let image_size = (238, 412);
        let input_size = (image_size.0 / 2, image_size.1 / 2);
        let val = 0.777f32;
        let input = Image::new_constant(input_size, val)?;
        let stage = Upsample2x::new(&ups_factors, 0);
        let output: Vec<Image<f32>> =
            make_and_run_simple_pipeline(stage, &[input], image_size, 123)?.1;
        for x in 0..image_size.0 {
            for y in 0..image_size.1 {
                assert_almost_eq!(output[0].as_rect().row(y)[x], val, 0.0000001);
            }
        }
        Ok(())
    }

    #[test]
    fn test_upsample2() -> Result<()> {
        let eps = 0.0000001;
        let mut input = Image::new((7, 7))?;
        // Put a single "1.0" in the middle of the image.
        input.as_rect_mut().row(3)[3] = 1.0f32;
        let ups_factors = CustomTransformData::default();
        let stage = Upsample2x::new(&ups_factors, 0);
        let output: Vec<Image<f32>> =
            make_and_run_simple_pipeline(stage, &[input], (14, 14), 77)?.1;
        assert_eq!(output[0].as_rect().size(), (14, 14));
        // Check we have a border with zeros
        for i in 0..14 {
            for j in 0..2 {
                assert_almost_eq!(output[0].as_rect().row(j)[i], 0.0, eps);
                assert_almost_eq!(output[0].as_rect().row(i)[j], 0.0, eps);
                assert_almost_eq!(output[0].as_rect().row(13 - j)[i], 0.0, eps);
                assert_almost_eq!(output[0].as_rect().row(i)[13 - j], 0.0, eps);
            }
        }
        // Define the mapping for the symmetric top-left kernel
        let index_map = [
            [0, 1, 2, 3, 4],
            [1, 5, 6, 7, 8],
            [2, 6, 9, 10, 11],
            [3, 7, 10, 12, 13],
            [4, 8, 11, 13, 14],
        ];

        // Validate weights from the kernel
        let kernel_size = 5;
        let kernel_offset = 2;
        let weights = &ups_factors.weights2;
        for di in 0..2 {
            for dj in 0..2 {
                for i in 0..kernel_size {
                    for j in 0..kernel_size {
                        let output_value = output[0].as_rect().row(kernel_offset + di + 2 * i)
                            [kernel_offset + dj + 2 * j];
                        let mapped_i = if di == 0 { kernel_size - 1 - i } else { i };
                        let mapped_j = if dj == 0 { kernel_size - 1 - j } else { j };
                        let weight_index = index_map[mapped_i][mapped_j];
                        assert_almost_eq!(output_value, weights[weight_index], eps);
                    }
                }
            }
        }

        Ok(())
    }
}
