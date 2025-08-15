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
    bit_depth: BitDepth,
}

impl ConvertModularToF32Stage {
    pub fn new(channel: usize, bit_depth: BitDepth) -> ConvertModularToF32Stage {
        ConvertModularToF32Stage { channel, bit_depth }
    }
}

impl std::fmt::Display for ConvertModularToF32Stage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "convert modular data to F32 in channel {} with bit depth {:?}",
            self.channel, self.bit_depth
        )
    }
}

// Converts custom [bits]-bit float (with [exp_bits] exponent bits) stored as
// int back to binary32 float.
// TODO(sboukortt): SIMD
fn int_to_float(input: &[i32], output: &mut [f32], bit_depth: &BitDepth) {
    assert_eq!(input.len(), output.len());
    let bits = bit_depth.bits_per_sample();
    let exp_bits = bit_depth.exponent_bits_per_sample();
    if bits == 32 {
        assert!(exp_bits == 8);
        for (&in_val, out_val) in input.iter().zip(output) {
            *out_val = f32::from_bits(in_val as u32);
        }
        return;
    }
    let exp_bias = (1 << (exp_bits - 1)) - 1;
    let sign_shift = bits - 1;
    let mant_bits = bits - exp_bits - 1;
    let mant_shift = 23 - mant_bits;
    for (&in_val, out_val) in input.iter().zip(output) {
        let mut f = in_val as u32;
        let signbit = (f >> sign_shift) != 0;
        f &= (1 << sign_shift) - 1;
        if f == 0 {
            *out_val = if signbit { -0.0 } else { 0.0 };
            continue;
        }
        let mut exp = (f >> mant_bits) as i32;
        let mut mantissa = f & ((1 << mant_bits) - 1);
        mantissa <<= mant_shift;
        // Try to normalize only if there is space for maneuver.
        if exp == 0 && exp_bits < 8 {
            // subnormal number
            while (mantissa & 0x800000) == 0 {
                mantissa <<= 1;
                exp -= 1;
            }
            exp += 1;
            // remove leading 1 because it is implicit now
            mantissa &= 0x7fffff;
        }
        exp -= exp_bias;
        // broke up the arbitrary float into its parts, now reassemble into
        // binary32
        exp += 127;
        assert!(exp >= 0);
        f = if signbit { 0x80000000 } else { 0 };
        f |= (exp as u32) << 23;
        f |= mantissa;
        *out_val = f32::from_bits(f);
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
        if self.bit_depth.floating_point_sample() {
            int_to_float(&input[0][..xsize], &mut output[0][..xsize], &self.bit_depth);
        } else {
            let scale = 1.0 / ((1u64 << self.bit_depth.bits_per_sample()) - 1) as f32;
            for i in 0..xsize {
                output[0][i] = input[0][i] as f32 * scale;
            }
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
