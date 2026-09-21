// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use jxl_simd::{F32SimdVec, I32SimdVec, simd_function};

use crate::frame::quantizer::LfQuantFactors;
use crate::headers::bit_depth::BitDepth;
use crate::render::{
    Channels, ChannelsMut, ErasedLocalState, ForEachChunk, RenderPipelineInOutStage,
    StageSpecialCase,
};
use crate::util::sync::{Arc, RwLock};

pub struct ConvertModularXYBToF32Stage {
    first_channel: usize,
    lf_quant: Arc<RwLock<LfQuantFactors>>,
}

impl ConvertModularXYBToF32Stage {
    pub fn new(
        first_channel: usize,
        lf_quant: Arc<RwLock<LfQuantFactors>>,
    ) -> ConvertModularXYBToF32Stage {
        ConvertModularXYBToF32Stage {
            first_channel,
            lf_quant,
        }
    }
}

impl std::fmt::Display for ConvertModularXYBToF32Stage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "convert modular xyb data to F32 in channels {}..{}",
            self.first_channel,
            self.first_channel + 2,
        )
    }
}

pub struct ConvertModular16XYBToF32Stage(pub ConvertModularXYBToF32Stage);

impl ConvertModular16XYBToF32Stage {
    pub fn new(
        first_channel: usize,
        lf_quant: Arc<RwLock<LfQuantFactors>>,
    ) -> ConvertModular16XYBToF32Stage {
        ConvertModular16XYBToF32Stage(ConvertModularXYBToF32Stage::new(first_channel, lf_quant))
    }
}

impl std::fmt::Display for ConvertModular16XYBToF32Stage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

macro_rules! define_modular_xyb_to_float_simd {
    ($dispatch_name:ident, $fn_name:ident, $in_ty:ty) => {
        simd_function!(
            $dispatch_name,
            d: D,
            fn $fn_name(
                xsize: usize,
                input_rows: &Channels<$in_ty>,
                output_rows: &mut ChannelsMut<f32>,
                scale_x: f32,
                scale_y: f32,
                scale_b: f32,
            ) {
                let scale_x = D::F32Vec::splat(d, scale_x);
                let scale_y = D::F32Vec::splat(d, scale_y);
                let scale_b = D::F32Vec::splat(d, scale_b);

                ForEachChunk::<3, 1, 0, 3, 1>::run(
                    d,
                    xsize,
                    input_rows,
                    output_rows,
                    #[inline(always)]
                    |_x, in_view, out_view| {
                        // Input channels: [Y, X, B] (modular XYB order)
                        let in_y = in_view.load::<0, 0, 0>();
                        let in_x = in_view.load::<1, 0, 0>();
                        let in_b = in_view.load::<2, 0, 0>();

                        let vy = in_y.as_f32();
                        let vx = in_x.as_f32();
                        let vb = in_b.as_f32();

                        // Output channels: [X, Y, B] (standard XYB order)
                        out_view.store::<0, 0>(vx * scale_x);
                        out_view.store::<1, 0>(vy * scale_y);
                        out_view.store::<2, 0>((vb + vy) * scale_b);
                    },
                );
            }
        );
    };
}

define_modular_xyb_to_float_simd!(
    modular_xyb_to_float_simd_dispatch,
    modular_xyb_to_float_simd,
    i32
);

define_modular_xyb_to_float_simd!(
    modular16_xyb_to_float_simd_dispatch,
    modular16_xyb_to_float_simd,
    i16
);

impl RenderPipelineInOutStage for ConvertModularXYBToF32Stage {
    type InputT = i32;
    type OutputT = f32;
    const SHIFT: (u8, u8) = (0, 0);
    const BORDER: (u8, u8) = (0, 0);

    fn uses_channel(&self, c: usize) -> bool {
        (self.first_channel..self.first_channel + 3).contains(&c)
    }

    fn process_row_chunk(
        &self,
        _position: (usize, usize),
        xsize: usize,
        input_rows: &Channels<i32>,
        output_rows: &mut ChannelsMut<f32>,
        _state: Option<&mut ErasedLocalState>,
        _previous_call_was_previous_row: bool,
    ) {
        let lf_quant = self.lf_quant.try_read().unwrap();
        let [scale_x, scale_y, scale_b] = lf_quant.quant_factors;
        modular_xyb_to_float_simd_dispatch(
            xsize,
            input_rows,
            output_rows,
            scale_x,
            scale_y,
            scale_b,
        );
    }
}


impl RenderPipelineInOutStage for ConvertModular16XYBToF32Stage {
    type InputT = i16;
    type OutputT = f32;
    const SHIFT: (u8, u8) = (0, 0);
    const BORDER: (u8, u8) = (0, 0);

    fn uses_channel(&self, c: usize) -> bool {
        self.0.uses_channel(c)
    }

    fn process_row_chunk(
        &self,
        _position: (usize, usize),
        xsize: usize,
        input_rows: &Channels<i16>,
        output_rows: &mut ChannelsMut<f32>,
        _state: Option<&mut ErasedLocalState>,
        _previous_call_was_previous_row: bool,
    ) {
        let lf_quant = self.0.lf_quant.try_read().unwrap();
        let [scale_x, scale_y, scale_b] = lf_quant.quant_factors;
        modular16_xyb_to_float_simd_dispatch(
            xsize,
            input_rows,
            output_rows,
            scale_x,
            scale_y,
            scale_b,
        );
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

pub struct ConvertModular16ToF32Stage(pub ConvertModularToF32Stage);

impl ConvertModular16ToF32Stage {
    pub fn new(channel: usize, bit_depth: BitDepth) -> ConvertModular16ToF32Stage {
        ConvertModular16ToF32Stage(ConvertModularToF32Stage::new(channel, bit_depth))
    }
}

impl std::fmt::Display for ConvertModular16ToF32Stage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

// SIMD 32-bit float passthrough (bitcast i32 to f32)
simd_function!(
    int_to_float_32bit_simd_dispatch,
    d: D,
    fn int_to_float_32bit_simd(input: &[i32], output: &mut [f32], xsize: usize) {
        let simd_width = D::I32Vec::LEN;

        // Process complete SIMD vectors
        for (in_chunk, out_chunk) in input
            .chunks_exact(simd_width)
            .zip(output.chunks_exact_mut(simd_width))
            .take(xsize.div_ceil(simd_width))
        {
            let val = D::I32Vec::load(d, in_chunk);
            val.bitcast_to_f32().store(out_chunk);
        }
    }
);

// SIMD 16-bit float (half-precision) to 32-bit float conversion
// Uses hardware F16C/NEON instructions when available via F32Vec::load_f16_bits()
simd_function!(
    int_to_float_16bit_simd_dispatch,
    d: D,
    fn int_to_float_16bit_simd(input: &[i32], output: &mut [f32], xsize: usize) {
        let simd_width = D::F32Vec::LEN;

        // Temporary buffer for i32->u16 conversion via SIMD
        // Note: Using constant 16 (max AVX-512 width) because D::F32Vec::LEN
        // cannot be used as array size in Rust (const generics limitation)
        const { assert!(D::F32Vec::LEN <= 16) }
        let mut u16_buf = [0u16; 16];

        // Process complete SIMD vectors
        for (in_chunk, out_chunk) in input
            .chunks_exact(simd_width)
            .zip(output.chunks_exact_mut(simd_width))
            .take(xsize.div_ceil(simd_width))
        {
            // Use SIMD to extract lower 16 bits from each i32 lane
            let i32_vec = D::I32Vec::load(d, in_chunk);
            i32_vec.store_u16(&mut u16_buf[..simd_width]);
            // Use hardware f16->f32 conversion
            let result = D::F32Vec::load_f16_bits(d, &u16_buf[..simd_width]);
            result.store(out_chunk);
        }
    }
);

// Converts custom [bits]-bit float (with [exp_bits] exponent bits) stored as
// int back to binary32 float.
fn int_to_float(input: &[i32], output: &mut [f32], bit_depth: &BitDepth, xsize: usize) {
    assert!(input.len() >= xsize && output.len() >= xsize);
    let bits = bit_depth.bits_per_sample();
    let exp_bits = bit_depth.exponent_bits_per_sample();

    // Use SIMD fast paths for common formats
    if bits == 32 && exp_bits == 8 {
        // 32-bit float passthrough
        int_to_float_32bit_simd_dispatch(input, output, xsize);
        return;
    }

    if bits == 16 && exp_bits == 5 {
        // IEEE 754 half-precision (f16) - common HDR format
        int_to_float_16bit_simd_dispatch(input, output, xsize);
        return;
    }

    // Generic scalar path for other custom float formats
    int_to_float_generic(&input[..xsize], &mut output[..xsize], bits, exp_bits);
}

fn custom_float_sample_to_f32(mut f: u32, bits: u32, exp_bits: u32) -> f32 {
    let exp_bias = (1 << (exp_bits - 1)) - 1;
    let sign_shift = bits - 1;
    let mant_bits = bits - exp_bits - 1;
    let mant_shift = 23 - mant_bits;
    let signbit = (f >> sign_shift) != 0;
    f &= (1 << sign_shift) - 1;
    if f == 0 {
        return if signbit { -0.0 } else { 0.0 };
    }
    let mut exp = (f >> mant_bits) as i32;
    let mut mantissa = f & ((1 << mant_bits) - 1);
    if exp == (1 << exp_bits) - 1 {
        // NaN or infinity
        f = if signbit { 0x80000000 } else { 0 };
        f |= 0b11111111 << 23;
        f |= mantissa << mant_shift;
        return f32::from_bits(f);
    }
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
    f32::from_bits(f)
}

// Generic scalar conversion for arbitrary bit-depth floats
// TODO: SIMD optimization for custom float formats
fn int_to_float_generic(input: &[i32], output: &mut [f32], bits: u32, exp_bits: u32) {
    for (&in_val, out_val) in input.iter().zip(output) {
        *out_val = custom_float_sample_to_f32(in_val as u32, bits, exp_bits);
    }
}

macro_rules! define_modular_to_float_simd {
    ($dispatch_name:ident, $fn_name:ident, $in_ty:ty) => {
        simd_function!(
            $dispatch_name,
            d: D,
            fn $fn_name(
                xsize: usize,
                input_rows: &Channels<$in_ty>,
                output_rows: &mut ChannelsMut<f32>,
                scale: f32,
            ) {
                let scale_vec = D::F32Vec::splat(d, scale);

                ForEachChunk::<1, 1, 0, 1, 1>::run(
                    d,
                    xsize,
                    input_rows,
                    output_rows,
                    #[inline(always)]
                    |_x, in_view, out_view| {
                        let val = in_view.load::<0, 0, 0>();
                        out_view.store::<0, 0>(val.as_f32() * scale_vec);
                    },
                );
            }
        );
    };
}

define_modular_to_float_simd!(
    modular_to_float_32bit_simd_dispatch,
    modular_to_float_32bit_simd,
    i32
);

define_modular_to_float_simd!(
    modular16_to_float_simd_dispatch,
    modular16_to_float_simd,
    i16
);

impl RenderPipelineInOutStage for ConvertModularToF32Stage {
    type InputT = i32;
    type OutputT = f32;
    const SHIFT: (u8, u8) = (0, 0);
    const BORDER: (u8, u8) = (0, 0);

    fn uses_channel(&self, c: usize) -> bool {
        c == self.channel
    }

    fn process_row_chunk(
        &self,
        _position: (usize, usize),
        xsize: usize,
        input_rows: &Channels<i32>,
        output_rows: &mut ChannelsMut<f32>,
        _state: Option<&mut ErasedLocalState>,
        _previous_call_was_previous_row: bool,
    ) {
        if self.bit_depth.floating_point_sample() {
            int_to_float(
                input_rows.get_row(0, 0),
                output_rows.get_row_mut(0, 0),
                &self.bit_depth,
                xsize,
            );
        } else {
            let scale = 1.0 / ((1u64 << self.bit_depth.bits_per_sample()) - 1) as f32;
            modular_to_float_32bit_simd_dispatch(xsize, input_rows, output_rows, scale);
        }
    }

    fn is_special_case(&self) -> Option<StageSpecialCase> {
        if self.bit_depth.floating_point_sample() {
            None
        } else {
            Some(StageSpecialCase::ModularToF32 {
                channel: self.channel,
                bit_depth: self.bit_depth.bits_per_sample() as u8,
            })
        }
    }
}

// SIMD 16-bit float (half-precision) to 32-bit float conversion for i16 inputs
simd_function!(
    int16_to_float_16bit_simd_dispatch,
    d: D,
    #[allow(unsafe_code)]
    fn int16_to_float_16bit_simd(input: &[i16], output: &mut [f32], xsize: usize) {
        let simd_width = D::F32Vec::LEN;

        for (in_chunk, out_chunk) in input
            .chunks_exact(simd_width)
            .zip(output.chunks_exact_mut(simd_width))
            .take(xsize.div_ceil(simd_width))
        {
            // SAFETY: `in_chunk` is a valid, aligned slice of `simd_width` elements of `i16`.
            // Reinterpreting `i16` as `u16` of identical size/alignment is sound since all bit
            // patterns are valid for `u16`.
            let u16_chunk =
                unsafe { std::slice::from_raw_parts(in_chunk.as_ptr().cast::<u16>(), simd_width) };
            let result = D::F32Vec::load_f16_bits(d, u16_chunk);
            result.store(out_chunk);
        }
    }
);

fn int16_to_float(input: &[i16], output: &mut [f32], bit_depth: &BitDepth, xsize: usize) {
    assert!(input.len() >= xsize && output.len() >= xsize);
    let bits = bit_depth.bits_per_sample();
    let exp_bits = bit_depth.exponent_bits_per_sample();

    if bits == 16 && exp_bits == 5 {
        int16_to_float_16bit_simd_dispatch(input, output, xsize);
        return;
    }

    for (&in_val, out_val) in input[..xsize].iter().zip(&mut output[..xsize]) {
        *out_val = custom_float_sample_to_f32((in_val as u16) as u32, bits, exp_bits);
    }
}


impl RenderPipelineInOutStage for ConvertModular16ToF32Stage {
    type InputT = i16;
    type OutputT = f32;
    const SHIFT: (u8, u8) = (0, 0);
    const BORDER: (u8, u8) = (0, 0);

    fn uses_channel(&self, c: usize) -> bool {
        self.0.uses_channel(c)
    }

    fn process_row_chunk(
        &self,
        _position: (usize, usize),
        xsize: usize,
        input_rows: &Channels<i16>,
        output_rows: &mut ChannelsMut<f32>,
        _state: Option<&mut ErasedLocalState>,
        _previous_call_was_previous_row: bool,
    ) {
        if self.0.bit_depth.floating_point_sample() {
            int16_to_float(
                input_rows.get_row(0, 0),
                output_rows.get_row_mut(0, 0),
                &self.0.bit_depth,
                xsize,
            );
        } else {
            let scale = 1.0 / ((1u64 << self.0.bit_depth.bits_per_sample()) - 1) as f32;
            modular16_to_float_simd_dispatch(xsize, input_rows, output_rows, scale);
        }
    }

    fn is_special_case(&self) -> Option<StageSpecialCase> {
        if self.0.bit_depth.floating_point_sample() {
            None
        } else {
            Some(StageSpecialCase::Modular16ToF32 {
                channel: self.0.channel,
                bit_depth: self.0.bit_depth.bits_per_sample() as u8,
            })
        }
    }
}

#[cfg(test)]
mod test {
    use test_log::test;

    use super::*;
    use crate::error::Result;
    use crate::headers::bit_depth::BitDepth;



    /// Test ConvertModularToF32Stage consistency with different bit depths.
    #[test]
    fn modular_to_f32_8bit_consistency() -> Result<()> {
        crate::render::test::test_stage_consistency(
            || ConvertModularToF32Stage::new(0, BitDepth::integer_samples(8)),
            (256, 256),
            1,
        )
    }

    #[test]
    fn modular_to_f32_16bit_consistency() -> Result<()> {
        crate::render::test::test_stage_consistency(
            || ConvertModularToF32Stage::new(0, BitDepth::integer_samples(16)),
            (256, 256),
            1,
        )
    }

    #[test]
    fn modular16_to_f32_8bit_consistency() -> Result<()> {
        crate::render::test::test_stage_consistency(
            || ConvertModular16ToF32Stage::new(0, BitDepth::integer_samples(8)),
            (256, 256),
            1,
        )
    }

    #[test]
    fn modular16_to_f32_16bit_consistency() -> Result<()> {
        crate::render::test::test_stage_consistency(
            || ConvertModular16ToF32Stage::new(0, BitDepth::integer_samples(16)),
            (256, 256),
            1,
        )
    }

    #[test]
    fn modular16_xyb_to_f32_consistency() -> Result<()> {
        crate::render::test::test_stage_consistency(
            || {
                ConvertModular16XYBToF32Stage::new(
                    0,
                    Arc::new(RwLock::new(LfQuantFactors::default())),
                )
            },
            (256, 256),
            3,
        )
    }

    #[test]
    fn modular_xyb_to_f32_consistency() -> Result<()> {
        crate::render::test::test_stage_consistency(
            || {
                ConvertModularXYBToF32Stage::new(
                    0,
                    Arc::new(RwLock::new(LfQuantFactors::default())),
                )
            },
            (256, 256),
            3,
        )
    }

    #[test]
    fn test_int_to_float_32bit() {
        // Test 32-bit float passthrough
        let bit_depth = BitDepth::f32();
        let test_values: Vec<f32> = vec![
            0.0,
            1.0,
            -1.0,
            0.5,
            -0.5,
            f32::INFINITY,
            f32::NEG_INFINITY,
            1e-30,
            1e30,
        ];
        let input: Vec<i32> = test_values
            .iter()
            .map(|&f| f.to_bits() as i32)
            .chain(std::iter::repeat(0))
            .take(16)
            .collect();
        let mut output = vec![0.0f32; 16];

        int_to_float(&input, &mut output, &bit_depth, test_values.len());

        for (i, (&expected, &actual)) in test_values.iter().zip(output.iter()).enumerate() {
            if expected.is_nan() {
                assert!(actual.is_nan(), "index {}: expected NaN, got {}", i, actual);
            } else {
                assert_eq!(expected, actual, "index {}: mismatch", i);
            }
        }
    }

    // f16 format: 1 sign, 5 exp, 10 mantissa
    // Test cases: (f16_bits, expected_f32)
    const F16_TEST_CASES: &[(u16, f32)] = &[
        (0x0000, 0.0),               // +0
        (0x8000, -0.0),              // -0
        (0x3C00, 1.0),               // 1.0
        (0xBC00, -1.0),              // -1.0
        (0x3800, 0.5),               // 0.5
        (0x4000, 2.0),               // 2.0
        (0x4400, 4.0),               // 4.0
        (0x7BFF, 65504.0),           // max normal f16
        (0x7C00, f32::INFINITY),     // +inf
        (0xFC00, f32::NEG_INFINITY), // -inf
        (0x0001, 5.960_464_5e-8),    // smallest positive subnormal
        (0x03FF, 6.097_555e-5),      // largest positive subnormal
        (0x8001, -5.960_464_5e-8),   // smallest negative subnormal
    ];

    #[test]
    fn test_int_to_float_16bit() {
        let bit_depth = BitDepth::f16();

        let input: Vec<i32> = F16_TEST_CASES
            .iter()
            .map(|(bits, _)| *bits as i32)
            .chain(std::iter::repeat(0))
            .take(16)
            .collect();
        let mut output = vec![0.0f32; 16];

        int_to_float(&input, &mut output, &bit_depth, F16_TEST_CASES.len());

        for (i, (&(_, expected), &actual)) in F16_TEST_CASES.iter().zip(output.iter()).enumerate() {
            assert!(
                (expected - actual).abs() < 1e-6
                    || expected == actual
                    || (expected.is_sign_negative() == actual.is_sign_negative()
                        && expected == 0.0
                        && actual == 0.0),
                "index {}: expected {}, got {}",
                i,
                expected,
                actual
            );
        }
    }

    #[test]
    fn test_int16_to_float_16bit() {
        let bit_depth = BitDepth::f16();

        let input: Vec<i16> = F16_TEST_CASES
            .iter()
            .map(|(bits, _)| *bits as i16)
            .chain(std::iter::repeat(0))
            .take(16)
            .collect();
        let mut output = vec![0.0f32; 16];

        int16_to_float(&input, &mut output, &bit_depth, F16_TEST_CASES.len());

        for (i, (&(_, expected), &actual)) in F16_TEST_CASES.iter().zip(output.iter()).enumerate() {
            assert!(
                (expected - actual).abs() < 1e-6
                    || expected == actual
                    || (expected.is_sign_negative() == actual.is_sign_negative()
                        && expected == 0.0
                        && actual == 0.0),
                "index {}: expected {}, got {}",
                i,
                expected,
                actual
            );
        }
    }
}
