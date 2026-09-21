// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(clippy::too_many_arguments)]

use jxl_simd::{
    F32SimdVec, I16SimdVec, I32SimdVec, SimdDescriptor, SimdMask, SimdMask16, U8SimdVec,
    U16SimdVec, simd_function,
};

use crate::image::ImageDataType;

#[inline(always)]
fn convert_f32_vec<D: SimdDescriptor>(
    d: D,
    val: D::F32Vec,
    scale: D::F32Vec,
    zero: D::F32Vec,
    dither_row: &[f32; 64],
    dither_x: usize,
) -> D::F32Vec {
    let dither = D::F32Vec::load(d, &dither_row[dither_x..]);
    let scaled = val * scale;
    let dithered = scaled + dither;
    dithered.max(zero).min(scale)
}

#[inline(always)]
fn scalar_f32_to_u8(val: f32, max: f32, dither: f32) -> u8 {
    (val * max + dither).clamp(0.0, max).round() as u8
}

#[inline(always)]
fn scalar_i16_to_u8(val: i16, mult: i32, max: i32) -> u8 {
    ((val as i32) * mult).clamp(0, max) as u8
}

#[inline(always)]
fn scalar_f32_to_f16(val: f32, clamp_range: Option<(f32, f32)>) -> u16 {
    let val = match clamp_range {
        Some((min, max)) => val.clamp(min, max),
        None => val,
    };
    crate::util::f16::from_f32(val).to_bits()
}

macro_rules! define_run_interleaved {
    ($fn_name:ident, $ty:ty, $vec_trait:ident, $store_fn:ident, $cnt:expr, $($arg:ident),+) => {
        #[inline(always)]
        fn $fn_name<D: SimdDescriptor>(
            d: D,
            $($arg: &[$ty]),+,
            out: &mut [$ty],
        ) -> usize {
            let len = D::$vec_trait::LEN;
            let mut n = 0;
            let limit = [$($arg.len()),+][0];

            {
                let out_chunks = out[..limit * $cnt].chunks_exact_mut(len * $cnt);
                $(let mut $arg = $arg.chunks_exact(len);)+
                for out_chunk in out_chunks {
                    $(let $arg = D::$vec_trait::load(d, $arg.next().unwrap());)+
                    D::$vec_trait::$store_fn($($arg),+, out_chunk);
                    n += len;
                }
            }

            let d256 = d.maybe_downgrade_256bit();
            let len256 = <D::Descriptor256 as SimdDescriptor>::$vec_trait::LEN;
            if len256 < len {
                let out_chunks = out[n * $cnt..limit * $cnt].chunks_exact_mut(len256 * $cnt);
                $(let mut $arg = $arg[n..limit].chunks_exact(len256);)+
                for out_chunk in out_chunks {
                    $(let $arg = <D::Descriptor256 as SimdDescriptor>::$vec_trait::load(d256, $arg.next().unwrap());)+
                    <D::Descriptor256 as SimdDescriptor>::$vec_trait::$store_fn($($arg),+, out_chunk);
                    n += len256;
                }
            }

            let d128 = d.maybe_downgrade_128bit();
            let len128 = <D::Descriptor128 as SimdDescriptor>::$vec_trait::LEN;
            if len128 < len {
                let out_chunks = out[n * $cnt..limit * $cnt].chunks_exact_mut(len128 * $cnt);
                $(let mut $arg = $arg[n..limit].chunks_exact(len128);)+
                for out_chunk in out_chunks {
                    $(let $arg = <D::Descriptor128 as SimdDescriptor>::$vec_trait::load(d128, $arg.next().unwrap());)+
                    <D::Descriptor128 as SimdDescriptor>::$vec_trait::$store_fn($($arg),+, out_chunk);
                    n += len128;
                }
            }

            n
        }
    };
}

define_run_interleaved!(
    run_interleaved_2_f32,
    f32,
    F32Vec,
    store_interleaved_2,
    2,
    a,
    b
);
define_run_interleaved!(
    run_interleaved_3_f32,
    f32,
    F32Vec,
    store_interleaved_3,
    3,
    a,
    b,
    c
);
define_run_interleaved!(
    run_interleaved_4_f32,
    f32,
    F32Vec,
    store_interleaved_4,
    4,
    a,
    b,
    c,
    e
);

simd_function!(
    store_interleaved_f32,
    d: D,
    fn store_interleaved_impl_f32(inputs: &[&[f32]], output: &mut [f32]) -> usize {
        match inputs.len() {
            2 => run_interleaved_2_f32(d, inputs[0], inputs[1], output),
            3 => run_interleaved_3_f32(d, inputs[0], inputs[1], inputs[2], output),
            4 => run_interleaved_4_f32(d, inputs[0], inputs[1], inputs[2], inputs[3], output),
            _ => 0,
        }
    }
);

define_run_interleaved!(
    run_interleaved_2_u8,
    u8,
    U8Vec,
    store_interleaved_2,
    2,
    a,
    b
);
define_run_interleaved!(
    run_interleaved_3_u8,
    u8,
    U8Vec,
    store_interleaved_3,
    3,
    a,
    b,
    c
);
define_run_interleaved!(
    run_interleaved_4_u8,
    u8,
    U8Vec,
    store_interleaved_4,
    4,
    a,
    b,
    c,
    e
);

simd_function!(
    store_interleaved_u8,
    d: D,
    fn store_interleaved_impl_u8(inputs: &[&[u8]], output: &mut [u8]) -> usize {
        match inputs.len() {
            2 => run_interleaved_2_u8(d, inputs[0], inputs[1], output),
            3 => run_interleaved_3_u8(d, inputs[0], inputs[1], inputs[2], output),
            4 => run_interleaved_4_u8(d, inputs[0], inputs[1], inputs[2], inputs[3], output),
            _ => 0,
        }
    }
);

define_run_interleaved!(
    run_interleaved_2_u16,
    u16,
    U16Vec,
    store_interleaved_2,
    2,
    a,
    b
);
define_run_interleaved!(
    run_interleaved_3_u16,
    u16,
    U16Vec,
    store_interleaved_3,
    3,
    a,
    b,
    c
);
define_run_interleaved!(
    run_interleaved_4_u16,
    u16,
    U16Vec,
    store_interleaved_4,
    4,
    a,
    b,
    c,
    e
);

simd_function!(
    store_interleaved_u16,
    d: D,
    fn store_interleaved_impl_u16(inputs: &[&[u16]], output: &mut [u16]) -> usize {
        match inputs.len() {
            2 => run_interleaved_2_u16(d, inputs[0], inputs[1], output),
            3 => run_interleaved_3_u16(d, inputs[0], inputs[1], inputs[2], output),
            4 => run_interleaved_4_u16(d, inputs[0], inputs[1], inputs[2], inputs[3], output),
            _ => 0,
        }
    }
);

pub(super) fn store_u8(slices: &[&[u8]], output_buf: &mut [u8]) -> usize {
    let channels = slices.len();
    if channels == 0 {
        return 0;
    }
    let xsize = slices[0].len();
    let out = &mut output_buf[..xsize * channels];
    if channels == 1 {
        out.copy_from_slice(slices[0]);
        xsize
    } else if (2..=4).contains(&channels) {
        let n = store_interleaved_u8(slices, out);
        for i in n..xsize {
            for c in 0..channels {
                out[i * channels + c] = slices[c][i];
            }
        }
        xsize
    } else {
        0
    }
}

pub(super) fn store_u16(slices: &[&[u16]], output_buf: &mut [u8]) -> usize {
    let channels = slices.len();
    if channels == 0 {
        return 0;
    }
    let xsize = slices[0].len();
    let out_bytes = &mut output_buf[..xsize * channels * 2];
    let ptr = out_bytes.as_mut_ptr();
    if ptr.align_offset(std::mem::align_of::<u16>()) == 0 {
        let out_u16 = u16::cast_slice_mut(out_bytes);
        if channels == 1 {
            out_u16.copy_from_slice(slices[0]);
            xsize
        } else if (2..=4).contains(&channels) {
            let n = store_interleaved_u16(slices, out_u16);
            for i in n..xsize {
                for c in 0..channels {
                    out_u16[i * channels + c] = slices[c][i];
                }
            }
            xsize
        } else {
            0
        }
    } else {
        0
    }
}

pub(super) fn store_f32(slices: &[&[f32]], output_buf: &mut [u8]) -> usize {
    let channels = slices.len();
    if channels == 0 {
        return 0;
    }
    let xsize = slices[0].len();
    let out_bytes = &mut output_buf[..xsize * channels * 4];
    let ptr = out_bytes.as_mut_ptr();
    if ptr.align_offset(std::mem::align_of::<f32>()) == 0 {
        let out_f32 = f32::cast_slice_mut(out_bytes);
        if channels == 1 {
            out_f32.copy_from_slice(slices[0]);
            xsize
        } else if (2..=4).contains(&channels) {
            let n = store_interleaved_f32(slices, out_f32);
            for i in n..xsize {
                for c in 0..channels {
                    out_f32[i * channels + c] = slices[c][i];
                }
            }
            xsize
        } else {
            0
        }
    } else {
        0
    }
}

simd_function!(
    f32_to_u8_simd,
    d: D,
    pub(super) fn f32_to_u8_simd_impl(
        input: &[f32],
        output: &mut [u8],
        max: f32,
        position: (usize, usize),
        channel: usize,
    ) {
        let (x0, y0) = position;
        let simd_width = D::F32Vec::LEN;
        let zero = D::F32Vec::splat(d, 0.0);
        let scale = D::F32Vec::splat(d, max);
        let dither_y = (y0 + channel * 13) % 32;
        let dither_row = &crate::util::DITHER_TABLE[dither_y];

        let xsize = output.len();
        let num_full_vecs = xsize / simd_width;

        for v in 0..num_full_vecs {
            let x = v * simd_width;
            let val = D::F32Vec::load(d, &input[x..]);
            let dither_x = (x0 + x + channel * 23) % 32;
            let clamped = convert_f32_vec(d, val, scale, zero, dither_row, dither_x);
            clamped.round_store_u8(&mut output[x..x + simd_width]);
        }

        for x in (num_full_vecs * simd_width)..xsize {
            let dither_x = (x0 + x + channel * 23) % 32;
            output[x] = scalar_f32_to_u8(input[x], max, dither_row[dither_x]);
        }
    }
);

#[inline(always)]
fn convert_i16_vec<D: SimdDescriptor>(
    _d: D,
    val: D::I16Vec,
    scale: D::I16Vec,
    zero: D::I16Vec,
    max_vec: D::I16Vec,
) -> D::I16Vec {
    let scaled = val * scale;
    let zeroclip = scaled.lt_zero().if_then_else_i16(zero, scaled);
    scaled.gt(max_vec).if_then_else_i16(max_vec, zeroclip)
}

simd_function!(
    i16_to_u8_simd,
    d: D,
    pub(super) fn i16_to_u8_simd_impl(
        input: &[i16],
        output: &mut [u8],
        mult: i16,
        max: i16,
    ) {
        let simd_width = D::I16Vec::LEN;
        let scale = D::I16Vec::splat(d, mult);
        let max_vec = D::I16Vec::splat(d, max);
        let zero = D::I16Vec::splat(d, 0);

        let xsize = output.len();
        let num_full_vecs = xsize / simd_width;

        for v in 0..num_full_vecs {
            let x = v * simd_width;
            let val = D::I16Vec::load(d, &input[x..]);
            let clip = convert_i16_vec(d, val, scale, zero, max_vec);
            clip.store_u8(&mut output[x..x + simd_width]);
        }

        for x in (num_full_vecs * simd_width)..xsize {
            output[x] = scalar_i16_to_u8(input[x], mult as i32, max as i32);
        }
    }
);

// SIMD I32 to U8 conversion used by SaveStage
simd_function!(
    i32_to_u8_simd_dispatch,
    d: D,
    pub(crate) fn i32_to_u8_simd(
        input: &[i32],
        output: &mut [u8],
        scale: i32,
        max: i32,
        xsize: usize,
    ) {
        let simd_width = D::F32Vec::LEN;
        let scale = D::I32Vec::splat(d, scale);
        let max = D::I32Vec::splat(d, max);
        let zero = D::I32Vec::splat(d, 0);

        for (input_chunk, output_chunk) in input
            .chunks_exact(simd_width)
            .zip(output.chunks_exact_mut(simd_width))
            .take(xsize.div_ceil(simd_width))
        {
            let val = D::I32Vec::load(d, input_chunk);
            let scaled = val * scale;
            let zeroclip = scaled.lt_zero().if_then_else_i32(zero, scaled);
            let clip = scaled.gt(max).if_then_else_i32(max, zeroclip);
            clip.store_u8(output_chunk);
        }
    }
);

// SIMD F32 to U16 conversion used by SaveStage
simd_function!(
    f32_to_u16_simd_dispatch,
    d: D,
    pub(crate) fn f32_to_u16_simd(input: &[f32], output: &mut [u16], max: f32, xsize: usize) {
        let simd_width = D::F32Vec::LEN;
        let zero = D::F32Vec::splat(d, 0.0);
        let scale = D::F32Vec::splat(d, max);

        for (input_chunk, output_chunk) in input
            .chunks_exact(simd_width)
            .zip(output.chunks_exact_mut(simd_width))
            .take(xsize.div_ceil(simd_width))
        {
            let val = D::F32Vec::load(d, input_chunk);
            let scaled = val * scale;
            let clamped = scaled.max(zero).min(scale);
            clamped.round_store_u16(output_chunk);
        }
    }
);

// SIMD F32 to F16 conversion used by SaveStage
simd_function!(
    f32_to_f16_simd_dispatch,
    d: D,
    pub(crate) fn f32_to_f16_simd(
        input: &[f32],
        output: &mut [u16],
        clamp_range: Option<(f32, f32)>,
        xsize: usize,
    ) {
        let simd_width = D::F32Vec::LEN;
        let mut x = 0;
        let clamp_vecs =
            clamp_range.map(|(min, max)| (D::F32Vec::splat(d, min), D::F32Vec::splat(d, max)));

        while x + simd_width <= input.len() && x + simd_width <= output.len() && x < xsize {
            let mut val = D::F32Vec::load(d, &input[x..x + simd_width]);
            if let Some((min_vec, max_vec)) = clamp_vecs {
                val = val.max(min_vec).min(max_vec);
            }
            val.store_f16_bits(&mut output[x..x + simd_width]);
            x += simd_width;
        }

        for i in x..xsize {
            output[i] = scalar_f32_to_f16(input[i], clamp_range);
        }
    }
);

