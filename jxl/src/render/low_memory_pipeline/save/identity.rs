// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::ops::Range;

use jxl_simd::{
    F32SimdVec, I16SimdVec, I32SimdVec, SimdDescriptor, SimdMask, SimdMask16, U8SimdVec,
    U16SimdVec, simd_function,
};

use crate::api::{Endianness, JxlDataFormat, JxlOutputBuffer};
use crate::image::ImageDataType;
use crate::render::low_memory_pipeline::row_buffers::RowBuffer;
use crate::util::DITHER_TABLE;

#[inline(always)]
fn f32_to_u8_simd<D: SimdDescriptor>(
    d: D,
    input: &[f32],
    output: &mut [u8],
    max: f32,
    position: (usize, usize),
    channel: usize,
    xsize: usize,
) {
    let (x0, y0) = position;
    let simd_width = D::F32Vec::LEN;
    let zero = D::F32Vec::splat(d, 0.0);
    let scale = D::F32Vec::splat(d, max);
    let dither_y = (y0 + channel * 13) % 32;

    for (block, (input_chunk, output_chunk)) in input
        .chunks_exact(simd_width)
        .zip(output.chunks_exact_mut(simd_width))
        .take(xsize.div_ceil(simd_width))
        .enumerate()
    {
        let x = block * simd_width;
        let val = D::F32Vec::load(d, input_chunk);
        let dither_x = (x0 + x + channel * 23) % 32;
        let dither = D::F32Vec::load(d, &DITHER_TABLE[dither_y][dither_x..]);
        let scaled = val * scale;
        let dithered = scaled + dither;
        let clamped = dithered.max(zero).min(scale);
        clamped.round_store_u8(output_chunk);
    }
}

simd_function!(
    f32_to_u8,
    d: D,
    pub(super) fn f32_to_u8_impl(
        input: &[f32],
        output: &mut [u8],
        max: f32,
        position: (usize, usize),
        channel: usize,
        xsize: usize,
    ) {
        f32_to_u8_simd(d, input, output, max, position, channel, xsize);
    }
);

#[inline(always)]
fn f32_to_u16_simd<D: SimdDescriptor>(
    d: D,
    input: &[f32],
    output: &mut [u16],
    max: f32,
    xsize: usize,
) {
    let simd_width = D::F32Vec::LEN;
    let zero = D::F32Vec::splat(d, 0.0);
    let one = D::F32Vec::splat(d, 1.0);
    let scale = D::F32Vec::splat(d, max);

    for (input_chunk, output_chunk) in input
        .chunks_exact(simd_width)
        .zip(output.chunks_exact_mut(simd_width))
        .take(xsize.div_ceil(simd_width))
    {
        let val = D::F32Vec::load(d, input_chunk);
        let clamped = val.max(zero).min(one);
        let scaled = clamped * scale;
        scaled.round_store_u16(output_chunk);
    }
}

simd_function!(
    f32_to_u16,
    d: D,
    pub(super) fn f32_to_u16_impl(input: &[f32], output: &mut [u16], max: f32, xsize: usize) {
        f32_to_u16_simd(d, input, output, max, xsize);
    }
);

#[inline(always)]
fn f32_to_f16_simd<D: SimdDescriptor>(d: D, input: &[f32], output: &mut [u16], xsize: usize) {
    let simd_width = D::F32Vec::LEN;
    for (input_chunk, output_chunk) in input
        .chunks_exact(simd_width)
        .zip(output.chunks_exact_mut(simd_width))
        .take(xsize.div_ceil(simd_width))
    {
        D::F32Vec::load(d, input_chunk).store_f16_bits(output_chunk);
    }
}

simd_function!(
    f32_to_f16,
    d: D,
    pub(super) fn f32_to_f16_impl(input: &[f32], output: &mut [u16], xsize: usize) {
        f32_to_f16_simd(d, input, output, xsize);
    }
);

#[inline(always)]
fn i16_to_u8_simd<D: SimdDescriptor>(
    d: D,
    input: &[i16],
    output: &mut [u8],
    scale: i32,
    max: i32,
    xsize: usize,
) {
    let simd_width = D::I16Vec::LEN;
    let scale = D::I16Vec::splat(d, scale as i16);
    let max = D::I16Vec::splat(d, max as i16);
    let zero = D::I16Vec::splat(d, 0);

    for (input_chunk, output_chunk) in input
        .chunks_exact(simd_width)
        .zip(output.chunks_exact_mut(simd_width))
        .take(xsize.div_ceil(simd_width))
    {
        let val = D::I16Vec::load(d, input_chunk);
        let scaled = val * scale;
        let zeroclip = scaled.lt_zero().if_then_else_i16(zero, scaled);
        let clip = scaled.gt(max).if_then_else_i16(max, zeroclip);
        clip.store_u8(output_chunk);
    }
}

simd_function!(
    i16_to_u8,
    d: D,
    pub(super) fn i16_to_u8_impl(
        input: &[i16],
        output: &mut [u8],
        scale: i32,
        max: i32,
        xsize: usize,
    ) {
        i16_to_u8_simd(d, input, output, scale, max, xsize);
    }
);

#[inline(always)]
fn i16_to_f16_simd<D: SimdDescriptor>(
    d: D,
    input: &[i16],
    output: &mut [u16],
    scale: f32,
    xsize: usize,
) {
    let simd_width = D::F32Vec::LEN;
    let scale = D::F32Vec::splat(d, scale);
    for (input_chunk, output_chunk) in input
        .chunks_exact(simd_width)
        .zip(output.chunks_exact_mut(simd_width))
        .take(xsize.div_ceil(simd_width))
    {
        let val = D::I32Vec::load_from_i16(d, input_chunk).as_f32() * scale;
        val.store_f16_bits(output_chunk);
    }
}

simd_function!(
    i16_to_f16,
    d: D,
    pub(super) fn i16_to_f16_impl(input: &[i16], output: &mut [u16], scale: f32, xsize: usize) {
        i16_to_f16_simd(d, input, output, scale, xsize);
    }
);

#[inline(always)]
fn i32_to_u8_simd<D: SimdDescriptor>(
    d: D,
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

simd_function!(
    i32_to_u8,
    d: D,
    pub(super) fn i32_to_u8_impl(
        input: &[i32],
        output: &mut [u8],
        scale: i32,
        max: i32,
        xsize: usize,
    ) {
        i32_to_u8_simd(d, input, output, scale, max, xsize);
    }
);

macro_rules! run_simd_chunks {
    ($out:expr, $ty:ty, $vec_trait:ident, $cnt:expr, |$n_var:ident, $out_chunk:ident| $body:block) => {{
        let len = D::$vec_trait::LEN;
        let step = {
            #[inline(always)]
            |$n_var: usize, $out_chunk: &mut [$ty]| $body
        };
        let mut n = 0;

        let mut chunks = $out.chunks_exact_mut(len * $cnt);
        for chunk in &mut chunks {
            step(n, chunk);
            n += len;
        }

        let rem = chunks.into_remainder();
        if !rem.is_empty() {
            let mut tail = [<$ty>::default(); 64 / std::mem::size_of::<$ty>() * $cnt];
            step(n, &mut tail[..len * $cnt]);
            rem.copy_from_slice(&tail[..rem.len()]);
        }
    }};
}

macro_rules! define_store_interleaved {
    ($fn_name:ident, $impl_name:ident, $ty:ty, $vec_trait:ident, $simd_trait:ident) => {
        simd_function!(
            $fn_name,
            d: D,
            fn $impl_name(inputs: &[&[$ty]], out: &mut [$ty]) {
                match inputs.len() {
                    2 => run_simd_chunks!(out, $ty, $vec_trait, 2, |n, out_chunk| {
                        let a = D::$vec_trait::load(d, &inputs[0][n..]);
                        let b = D::$vec_trait::load(d, &inputs[1][n..]);
                        $simd_trait::store_interleaved_2(a, b, out_chunk);
                    }),
                    3 => run_simd_chunks!(out, $ty, $vec_trait, 3, |n, out_chunk| {
                        let a = D::$vec_trait::load(d, &inputs[0][n..]);
                        let b = D::$vec_trait::load(d, &inputs[1][n..]);
                        let c = D::$vec_trait::load(d, &inputs[2][n..]);
                        $simd_trait::store_interleaved_3(a, b, c, out_chunk);
                    }),
                    4 => run_simd_chunks!(out, $ty, $vec_trait, 4, |n, out_chunk| {
                        let a = D::$vec_trait::load(d, &inputs[0][n..]);
                        let b = D::$vec_trait::load(d, &inputs[1][n..]);
                        let c = D::$vec_trait::load(d, &inputs[2][n..]);
                        let e = D::$vec_trait::load(d, &inputs[3][n..]);
                        $simd_trait::store_interleaved_4(a, b, c, e, out_chunk);
                    }),
                    _ => unreachable!(),
                }
            }
        );
    };
}

define_store_interleaved!(
    store_interleaved_f32,
    store_interleaved_impl_f32,
    f32,
    F32Vec,
    F32SimdVec
);
define_store_interleaved!(
    store_interleaved_u8,
    store_interleaved_impl_u8,
    u8,
    U8Vec,
    U8SimdVec
);
define_store_interleaved!(
    store_interleaved_u16,
    store_interleaved_impl_u16,
    u16,
    U16Vec,
    U16SimdVec
);

pub(super) fn store(
    input_buf: &[&RowBuffer],
    input_y: usize,
    xrange: Range<usize>,
    output_buf: &mut JxlOutputBuffer,
    output_y: usize,
    data_format: JxlDataFormat,
) -> bool {
    let byte_start = xrange.start * data_format.bytes_per_sample() + RowBuffer::x0_byte_offset();
    let byte_end = xrange.end * data_format.bytes_per_sample() + RowBuffer::x0_byte_offset();
    let is_native_endian = match data_format {
        JxlDataFormat::U8 { .. } => true,
        JxlDataFormat::F16 { endianness, .. }
        | JxlDataFormat::U16 { endianness, .. }
        | JxlDataFormat::F32 { endianness, .. } => endianness == Endianness::native(),
    };
    let output_buf = output_buf.row_mut(output_y);
    let output_buf = &mut output_buf[0..(byte_end - byte_start) * input_buf.len()];
    match (
        input_buf.len(),
        data_format.bytes_per_sample(),
        is_native_endian,
    ) {
        (1, _, true) => {
            // We can just do a memcpy.
            let input_buf = &input_buf[0].get_row::<u8>(input_y)[byte_start..byte_end];
            output_buf.copy_from_slice(input_buf);
            true
        }
        (channels, 1, true) if (2..=4).contains(&channels) => {
            let start_u8 = byte_start;
            let mut slices = [&[] as &[u8]; 4];
            for (i, buf) in input_buf.iter().enumerate() {
                slices[i] = &buf.get_row::<u8>(input_y)[start_u8..];
            }
            store_interleaved_u8(&slices[..channels], output_buf);
            true
        }
        (channels, 2, true) if (2..=4).contains(&channels) => {
            let ptr = output_buf.as_mut_ptr();
            if ptr.align_offset(std::mem::align_of::<u16>()) == 0 {
                let output_u16 = u16::cast_slice_mut(output_buf);
                let start_u16 = byte_start / 2;
                let mut slices = [&[] as &[u16]; 4];
                for (i, buf) in input_buf.iter().enumerate() {
                    slices[i] = &buf.get_row::<u16>(input_y)[start_u16..];
                }
                store_interleaved_u16(&slices[..channels], output_u16);
                true
            } else {
                false
            }
        }
        (channels, 4, true) if (2..=4).contains(&channels) => {
            let ptr = output_buf.as_mut_ptr();
            if ptr.align_offset(std::mem::align_of::<f32>()) == 0 {
                let output_f32 = f32::cast_slice_mut(output_buf);
                let start_f32 = byte_start / 4;
                let mut slices = [&[] as &[f32]; 4];
                for (i, buf) in input_buf.iter().enumerate() {
                    slices[i] = &buf.get_row::<f32>(input_y)[start_f32..];
                }
                store_interleaved_f32(&slices[..channels], output_f32);
                true
            } else {
                false
            }
        }
        _ => false,
    }
}
