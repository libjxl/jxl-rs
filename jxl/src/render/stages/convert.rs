// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    frame::quantizer::LfQuantFactors,
    headers::bit_depth::BitDepth,
    render::{RenderPipelineInOutStage, RenderPipelineStage},
};

pub struct ConvertU8F32Stage {
    channel: usize,
}

impl ConvertU8F32Stage {
    pub fn new(channel: usize) -> ConvertU8F32Stage {
        ConvertU8F32Stage { channel }
    }
}

impl std::fmt::Display for ConvertU8F32Stage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "convert U8 data to F32 in channel {}", self.channel)
    }
}

impl RenderPipelineStage for ConvertU8F32Stage {
    type Type = RenderPipelineInOutStage<u8, f32, 0, 0, 0, 0>;

    fn uses_channel(&self, c: usize) -> bool {
        c == self.channel
    }

    fn process_row_chunk(
        &self,
        _position: (usize, usize),
        xsize: usize,
        row: &mut [(&[&[u8]], &mut [&mut [f32]])],
        _state: Option<&mut dyn std::any::Any>,
    ) {
        let (input, output) = &mut row[0];
        for i in 0..xsize {
            output[0][i] = input[0][i] as f32 * (1.0 / 255.0);
        }
    }
}

pub struct ConvertModularXYBToF32Stage {
    first_channel: usize,
    scale: [f32; 3],
}

impl ConvertModularXYBToF32Stage {
    pub fn new(first_channel: usize, lf_quant: &LfQuantFactors) -> ConvertModularXYBToF32Stage {
        ConvertModularXYBToF32Stage {
            first_channel,
            scale: lf_quant.quant_factors,
        }
    }
}

impl std::fmt::Display for ConvertModularXYBToF32Stage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "convert modular xyb data to F32 in channels {}..{} with scales {:?}",
            self.first_channel,
            self.first_channel + 2,
            self.scale
        )
    }
}

impl RenderPipelineStage for ConvertModularXYBToF32Stage {
    type Type = RenderPipelineInOutStage<i32, f32, 0, 0, 0, 0>;

    fn uses_channel(&self, c: usize) -> bool {
        (self.first_channel..self.first_channel + 3).contains(&c)
    }

    fn process_row_chunk(
        &self,
        _position: (usize, usize),
        xsize: usize,
        row: &mut [(&[&[i32]], &mut [&mut [f32]])],
        _state: Option<&mut dyn std::any::Any>,
    ) {
        let [scale_x, scale_y, scale_b] = self.scale;
        let [
            (input_y, output_x),
            (input_x, output_y),
            (input_b, output_b),
        ] = row
        else {
            panic!(
                "incorrect number of channels; expected 3, found {}",
                row.len()
            );
        };
        for i in 0..xsize {
            output_x[0][i] = input_x[0][i] as f32 * scale_x;
            output_y[0][i] = input_y[0][i] as f32 * scale_y;
            output_b[0][i] = (input_b[0][i] + input_y[0][i]) as f32 * scale_b;
        }
    }
}

pub struct ConvertModularToF32Stage {
    channel: usize,
    scale: f32,
}

impl ConvertModularToF32Stage {
    pub fn new(channel: usize, bit_depth: BitDepth) -> ConvertModularToF32Stage {
        // TODO(szabadka): Support floating point samples.
        let bits_per_sample = bit_depth.bits_per_sample();
        let scale = 1.0 / ((1u64 << bits_per_sample) - 1) as f32;
        ConvertModularToF32Stage { channel, scale }
    }
}

impl std::fmt::Display for ConvertModularToF32Stage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "convert modular data to F32 in channel {} with scale {}",
            self.channel, self.scale
        )
    }
}

impl RenderPipelineStage for ConvertModularToF32Stage {
    type Type = RenderPipelineInOutStage<i32, f32, 0, 0, 0, 0>;

    fn uses_channel(&self, c: usize) -> bool {
        c == self.channel
    }

    fn process_row_chunk(
        &self,
        _position: (usize, usize),
        xsize: usize,
        row: &mut [(&[&[i32]], &mut [&mut [f32]])],
        _state: Option<&mut dyn std::any::Any>,
    ) {
        let (input, output) = &mut row[0];
        for i in 0..xsize {
            output[0][i] = input[0][i] as f32 * self.scale;
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::error::Result;
    use test_log::test;

    #[test]
    fn u8_consistency() -> Result<()> {
        crate::render::test::test_stage_consistency::<_, u8, f32>(
            ConvertU8F32Stage::new(0),
            (500, 500),
            1,
        )
    }
}
