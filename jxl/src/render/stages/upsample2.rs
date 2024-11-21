// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    headers::CustomTransformData,
    render::{RenderPipelineInOutStage, RenderPipelineStage},
};

pub struct Upsample2x {
    kernel: [[[[f32; 5]; 5]; 1]; 1],
    channel: usize,
}

impl Upsample2x {
    pub fn new(ups_factors: CustomTransformData, channel: usize) -> Upsample2x {
        // don't copy
        let cloned_ups_factors = ups_factors.clone();
        let weights = cloned_ups_factors.weights2;
        let n = 1;
        let mut kernel = [[[[0.0; 5]; 5]; 1]; 1];
        for i in 0..5 * n {
            for j in 0..5 * n {
                let y = i.min(j);
                let x = i.max(j);
                //println!("n:{n}, x:{x}, y:{y}");
                let y = y as isize;
                let x = x as isize;
                let n = n as isize;
                let index = (5 * n * y - y * (y - 1) / 2 + x - y) as usize;
                //println!(
                //    "index: {index} n: {}, x: {}, y: {}, i:{}, j:{}",
                //    n, x, y, i, j
                //);
                kernel[j / 5][i / 5][j % 5][i % 5] = weights[index];
                //println!("weight: {:}", weights[index]);
            }
        }
        Upsample2x { kernel, channel }
    }
    pub fn kernel(&self, x: usize, y: usize, ix: usize, iy: usize) -> f32 {
        self.kernel[0][0][if y % 2 == 0 { iy } else { 4 - iy }]
            [if x % 2 == 0 { ix } else { 4 - ix }]
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
    // Takes a 5x5 area and upsamples it to 10x10 using the kernel.
    fn process_row_chunk(
        &mut self,
        _position: (usize, usize),
        xsize: usize,
        row: &mut [(&[&[f32]], &mut [&mut [f32]])],
    ) {
        let n = 2;
        let (input, output) = &mut row[0];
        println!("output size: {}, {}", output.len(), output[0].len());

        // Iterate over the input rows and columns
        for i in 0..5 {
            for j in 0..5 {
                let input_value = input[i][j];
                if input_value != 0.0 {
                    println!("saw non_zero input!");
                }

                // Upsample this input value into a 2x2 region in the output
                for di in 0..n {
                    for dj in 0..n {
                        let oi = n * i + di;
                        let oj = n * j + dj;

                        if oi < output.len() && oj < output[oi].len() {
                            output[oi][oj] += input_value * self.kernel(di, dj, i % 5, j % 5);
                        } else {
                            println!("shouldn't happen!?");
                            println!("oi{oi}, oj{oj}, n{n}, i{i}, ,j{j} di{di},.dj{dj}");
                        }
                    }
                }
            }
        }
    }
    fn new_size(&self, current_size: (usize, usize)) -> (usize, usize) {
        (2 * current_size.0, 2 * current_size.1)
    }

    fn original_data_origin(&self) -> (usize, usize) {
        (4, 4)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{error::Result, image::Image, render::test::make_and_run_simple_pipeline};
    use rand::SeedableRng;
    use test_log::test;

    #[test]
    fn test_upsample2() -> Result<()> {
        let mut input = Image::new((5, 5))?;
        input
            .as_rect_mut()
            .row(2)
            .copy_from_slice(&[0.0f32, 0.0, 10.0, 0.0, 0.0]);
        for i in 0..5 {
            println!("row{i}: {:?}", input.as_rect().row(i));
        }
        let ups_factors = CustomTransformData::default();
        let stage = Upsample2x::new(ups_factors, 0);
        let output: Vec<Image<f32>> =
            make_and_run_simple_pipeline(stage, &[input], (5, 5), 512)?.1;
        assert_eq!(output[0].as_rect().size(), (10, 10));
        println!("{:?}", output[0].as_rect());

        for i in 0..10 {
            println!("row{i}: {:?}", output[0].as_rect().row(i));
        }
        // TODO: compare to original weights
        //assert_eq!(
        //    output[0].as_rect().row(6),
        //    [1.0, 1.25, 1.75, 2.5, 3.5, 1.0, 1.25, 1.75, 2.5, 3.5]
        //);
        Ok(())
    }
}
