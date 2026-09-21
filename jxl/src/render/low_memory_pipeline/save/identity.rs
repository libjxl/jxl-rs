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

// --- Generic Channel Readers and Fused Loops ---

pub(crate) trait ChannelReaderU8<D: SimdDescriptor> {
    fn read_u8_vec(&self, d: D, base_x: usize, stack: &mut [u8; 64]) -> D::U8Vec;
    fn read_scalar(&self, x: usize) -> u8;
}

pub(crate) trait ChannelReaderU16<D: SimdDescriptor> {
    fn read_u16_vec(&self, d: D, base_x: usize, stack: &mut [u16; 64]) -> D::U16Vec;
    fn read_scalar(&self, x: usize) -> u16;
}

pub(crate) struct F32ToU8Reader<'a, D: SimdDescriptor> {
    slice: &'a [f32],
    scale_vec: D::F32Vec,
    zero: D::F32Vec,
    dither_row: &'static [f32; 64],
    x0: usize,
    dither_channel: usize,
    max: f32,
}

impl<'a, D: SimdDescriptor> F32ToU8Reader<'a, D> {
    #[inline(always)]
    pub(crate) fn new(
        d: D,
        slice: &'a [f32],
        max: f32,
        position: (usize, usize),
        dither_channel: usize,
    ) -> Self {
        let (x0, y0) = position;
        let dither_y = (y0 + dither_channel * 13) % 32;
        Self {
            slice,
            scale_vec: D::F32Vec::splat(d, max),
            zero: D::F32Vec::splat(d, 0.0),
            dither_row: &crate::util::DITHER_TABLE[dither_y],
            x0,
            dither_channel,
            max,
        }
    }
}

impl<'a, D: SimdDescriptor> ChannelReaderU8<D> for F32ToU8Reader<'a, D> {
    #[inline(always)]
    fn read_u8_vec(&self, d: D, base_x: usize, stack: &mut [u8; 64]) -> D::U8Vec {
        let u8_len = D::U8Vec::LEN;
        let f32_len = D::F32Vec::LEN;
        let ratio = u8_len.checked_div(f32_len).unwrap_or(1);
        for k in 0..ratio {
            let x = base_x + k * f32_len;
            let val = D::F32Vec::load(d, &self.slice[x..]);
            let dx = (self.x0 + x + self.dither_channel * 23) % 32;
            let clamped = convert_f32_vec(d, val, self.scale_vec, self.zero, self.dither_row, dx);
            clamped.round_store_u8(&mut stack[k * f32_len..(k + 1) * f32_len]);
        }
        D::U8Vec::load(d, &stack[..u8_len])
    }

    #[inline(always)]
    fn read_scalar(&self, x: usize) -> u8 {
        let dx = (self.x0 + x + self.dither_channel * 23) % 32;
        scalar_f32_to_u8(self.slice[x], self.max, self.dither_row[dx])
    }
}

pub(crate) struct I16ToU8Reader<'a, D: SimdDescriptor> {
    slice: &'a [i16],
    scale: D::I16Vec,
    zero: D::I16Vec,
    max_vec: D::I16Vec,
    mult: i16,
    max: i16,
}

impl<'a, D: SimdDescriptor> I16ToU8Reader<'a, D> {
    #[inline(always)]
    pub(crate) fn new(d: D, slice: &'a [i16], mult: i16, max: i16) -> Self {
        Self {
            slice,
            scale: D::I16Vec::splat(d, mult),
            zero: D::I16Vec::splat(d, 0),
            max_vec: D::I16Vec::splat(d, max),
            mult,
            max,
        }
    }
}

impl<'a, D: SimdDescriptor> ChannelReaderU8<D> for I16ToU8Reader<'a, D> {
    #[inline(always)]
    fn read_u8_vec(&self, d: D, base_x: usize, stack: &mut [u8; 64]) -> D::U8Vec {
        let u8_len = D::U8Vec::LEN;
        let i16_len = D::I16Vec::LEN;
        let ratio = u8_len.checked_div(i16_len).unwrap_or(1);
        for k in 0..ratio {
            let x = base_x + k * i16_len;
            let val = D::I16Vec::load(d, &self.slice[x..]);
            let clip = convert_i16_vec(d, val, self.scale, self.zero, self.max_vec);
            clip.store_u8(&mut stack[k * i16_len..(k + 1) * i16_len]);
        }
        D::U8Vec::load(d, &stack[..u8_len])
    }

    #[inline(always)]
    fn read_scalar(&self, x: usize) -> u8 {
        scalar_i16_to_u8(self.slice[x], self.mult as i32, self.max as i32)
    }
}

pub(crate) struct OpaqueU8AlphaReader<D: SimdDescriptor> {
    alpha_vec: D::U8Vec,
}

impl<D: SimdDescriptor> OpaqueU8AlphaReader<D> {
    #[inline(always)]
    pub(crate) fn new(d: D) -> Self {
        Self {
            alpha_vec: D::U8Vec::splat(d, 255),
        }
    }
}

impl<D: SimdDescriptor> ChannelReaderU8<D> for OpaqueU8AlphaReader<D> {
    #[inline(always)]
    fn read_u8_vec(&self, _d: D, _base_x: usize, _stack: &mut [u8; 64]) -> D::U8Vec {
        self.alpha_vec
    }

    #[inline(always)]
    fn read_scalar(&self, _x: usize) -> u8 {
        255
    }
}

pub(crate) struct F32ToF16Reader<'a, D: SimdDescriptor> {
    slice: &'a [f32],
    clamp_vecs: Option<(D::F32Vec, D::F32Vec)>,
    clamp_range: Option<(f32, f32)>,
}

impl<'a, D: SimdDescriptor> F32ToF16Reader<'a, D> {
    #[inline(always)]
    pub(crate) fn new(d: D, slice: &'a [f32], clamp_range: Option<(f32, f32)>) -> Self {
        let clamp_vecs = clamp_range
            .map(|(min, max)| (D::F32Vec::splat(d, min), D::F32Vec::splat(d, max)));
        Self {
            slice,
            clamp_vecs,
            clamp_range,
        }
    }
}

impl<'a, D: SimdDescriptor> ChannelReaderU16<D> for F32ToF16Reader<'a, D> {
    #[inline(always)]
    fn read_u16_vec(&self, d: D, base_x: usize, stack: &mut [u16; 64]) -> D::U16Vec {
        let u16_len = D::U16Vec::LEN;
        let f32_len = D::F32Vec::LEN;
        let ratio = u16_len.checked_div(f32_len).unwrap_or(1);
        for k in 0..ratio {
            let x = base_x + k * f32_len;
            let mut val = D::F32Vec::load(d, &self.slice[x..]);
            if let Some((min_vec, max_vec)) = self.clamp_vecs {
                val = val.max(min_vec).min(max_vec);
            }
            val.store_f16_bits(&mut stack[k * f32_len..(k + 1) * f32_len]);
        }
        D::U16Vec::load(d, &stack[..u16_len])
    }

    #[inline(always)]
    fn read_scalar(&self, x: usize) -> u16 {
        scalar_f32_to_f16(self.slice[x], self.clamp_range)
    }
}

pub(crate) struct OpaqueF16AlphaReader<D: SimdDescriptor> {
    alpha_vec: D::U16Vec,
}

impl<D: SimdDescriptor> OpaqueF16AlphaReader<D> {
    #[inline(always)]
    pub(crate) fn new(d: D) -> Self {
        Self {
            alpha_vec: D::U16Vec::splat(d, 0x3C00),
        }
    }
}

impl<D: SimdDescriptor> ChannelReaderU16<D> for OpaqueF16AlphaReader<D> {
    #[inline(always)]
    fn read_u16_vec(&self, _d: D, _base_x: usize, _stack: &mut [u16; 64]) -> D::U16Vec {
        self.alpha_vec
    }

    #[inline(always)]
    fn read_scalar(&self, _x: usize) -> u16 {
        0x3C00
    }
}

#[derive(Clone, Copy)]
pub(super) enum ChannelSourceU8<'a> {
    F32 {
        slice: &'a [f32],
        dither_channel: usize,
    },
    I16 {
        slice: &'a [i16],
    },
    OpaqueAlpha,
}

#[derive(Clone, Copy)]
pub(super) enum ChannelSourceU16<'a> {
    F32 {
        slice: &'a [f32],
        clamp_range: Option<(f32, f32)>,
    },
    OpaqueAlpha,
}

macro_rules! define_run_fused {
    (
        $name:ident,
        $trait_name:ident,
        $ty:ty,
        $vec_mod:ident,
        $store_fn:ident,
        $read_fn:ident,
        $cnt:expr,
        $(($r:ident, $R:ident, $idx:expr)),+
    ) => {
        #[inline(always)]
        fn $name<
            D: SimdDescriptor,
            $($R: $trait_name<D>,)+
        >(
            d: D,
            $($r: &$R,)+
            output: &mut [$ty],
        ) {
            let vec_len = D::$vec_mod::LEN;
            let limit = output.len() / $cnt;
            let num_blocks = limit / vec_len;
            let mut stack = [0 as $ty; 64];

            let mut n = 0;
            for block in 0..num_blocks {
                let base_x = block * vec_len;
                $(
                    let $r = $r.$read_fn(d, base_x, &mut stack);
                )+
                D::$vec_mod::$store_fn(
                    $($r,)+
                    &mut output[base_x * $cnt..(base_x + vec_len) * $cnt],
                );
                n += vec_len;
            }

            for x in n..limit {
                $(
                    output[x * $cnt + $idx] = $r.read_scalar(x);
                )+
            }
        }
    };
}

define_run_fused!(
    run_fused_3_u8,
    ChannelReaderU8,
    u8,
    U8Vec,
    store_interleaved_3,
    read_u8_vec,
    3,
    (r0, R0, 0),
    (r1, R1, 1),
    (r2, R2, 2)
);
define_run_fused!(
    run_fused_3_u16,
    ChannelReaderU16,
    u16,
    U16Vec,
    store_interleaved_3,
    read_u16_vec,
    3,
    (r0, R0, 0),
    (r1, R1, 1),
    (r2, R2, 2)
);
define_run_fused!(
    run_fused_4_u8,
    ChannelReaderU8,
    u8,
    U8Vec,
    store_interleaved_4,
    read_u8_vec,
    4,
    (r0, R0, 0),
    (r1, R1, 1),
    (r2, R2, 2),
    (r3, R3, 3)
);
define_run_fused!(
    run_fused_4_u16,
    ChannelReaderU16,
    u16,
    U16Vec,
    store_interleaved_4,
    read_u16_vec,
    4,
    (r0, R0, 0),
    (r1, R1, 1),
    (r2, R2, 2),
    (r3, R3, 3)
);

simd_function!(
    store_fused_3_u8,
    d: D,
    pub(super) fn store_fused_3_u8_impl(
        c0: ChannelSourceU8<'_>,
        c1: ChannelSourceU8<'_>,
        c2: ChannelSourceU8<'_>,
        output: &mut [u8],
        max: f32,
        mult: i16,
        max_i16: i16,
        position: (usize, usize),
    ) {
        match (c0, c1, c2) {
            (
                ChannelSourceU8::F32 {
                    slice: s0,
                    dither_channel: dc0,
                },
                ChannelSourceU8::F32 {
                    slice: s1,
                    dither_channel: dc1,
                },
                ChannelSourceU8::F32 {
                    slice: s2,
                    dither_channel: dc2,
                },
            ) => {
                let r0 = F32ToU8Reader::new(d, s0, max, position, dc0);
                let r1 = F32ToU8Reader::new(d, s1, max, position, dc1);
                let r2 = F32ToU8Reader::new(d, s2, max, position, dc2);
                run_fused_3_u8(d, &r0, &r1, &r2, output);
            }
            (
                ChannelSourceU8::I16 { slice: s0 },
                ChannelSourceU8::I16 { slice: s1 },
                ChannelSourceU8::I16 { slice: s2 },
            ) => {
                let r0 = I16ToU8Reader::new(d, s0, mult, max_i16);
                let r1 = I16ToU8Reader::new(d, s1, mult, max_i16);
                let r2 = I16ToU8Reader::new(d, s2, mult, max_i16);
                run_fused_3_u8(d, &r0, &r1, &r2, output);
            }
            _ => unreachable!(),
        }
    }
);

simd_function!(
    store_fused_4_u8,
    d: D,
    pub(super) fn store_fused_4_u8_impl(
        c0: ChannelSourceU8<'_>,
        c1: ChannelSourceU8<'_>,
        c2: ChannelSourceU8<'_>,
        c3: ChannelSourceU8<'_>,
        output: &mut [u8],
        max: f32,
        mult: i16,
        max_i16: i16,
        position: (usize, usize),
    ) {
        match (c0, c1, c2, c3) {
            (
                ChannelSourceU8::F32 {
                    slice: s0,
                    dither_channel: dc0,
                },
                ChannelSourceU8::F32 {
                    slice: s1,
                    dither_channel: dc1,
                },
                ChannelSourceU8::F32 {
                    slice: s2,
                    dither_channel: dc2,
                },
                ChannelSourceU8::F32 {
                    slice: s3,
                    dither_channel: dc3,
                },
            ) => {
                let r0 = F32ToU8Reader::new(d, s0, max, position, dc0);
                let r1 = F32ToU8Reader::new(d, s1, max, position, dc1);
                let r2 = F32ToU8Reader::new(d, s2, max, position, dc2);
                let r3 = F32ToU8Reader::new(d, s3, max, position, dc3);
                run_fused_4_u8(d, &r0, &r1, &r2, &r3, output);
            }
            (
                ChannelSourceU8::F32 {
                    slice: s0,
                    dither_channel: dc0,
                },
                ChannelSourceU8::F32 {
                    slice: s1,
                    dither_channel: dc1,
                },
                ChannelSourceU8::F32 {
                    slice: s2,
                    dither_channel: dc2,
                },
                ChannelSourceU8::OpaqueAlpha,
            ) => {
                let r0 = F32ToU8Reader::new(d, s0, max, position, dc0);
                let r1 = F32ToU8Reader::new(d, s1, max, position, dc1);
                let r2 = F32ToU8Reader::new(d, s2, max, position, dc2);
                let r3 = OpaqueU8AlphaReader::new(d);
                run_fused_4_u8(d, &r0, &r1, &r2, &r3, output);
            }
            (
                ChannelSourceU8::F32 {
                    slice: s0,
                    dither_channel: dc0,
                },
                ChannelSourceU8::F32 {
                    slice: s1,
                    dither_channel: dc1,
                },
                ChannelSourceU8::F32 {
                    slice: s2,
                    dither_channel: dc2,
                },
                ChannelSourceU8::I16 { slice: s3 },
            ) => {
                let r0 = F32ToU8Reader::new(d, s0, max, position, dc0);
                let r1 = F32ToU8Reader::new(d, s1, max, position, dc1);
                let r2 = F32ToU8Reader::new(d, s2, max, position, dc2);
                let r3 = I16ToU8Reader::new(d, s3, mult, max_i16);
                run_fused_4_u8(d, &r0, &r1, &r2, &r3, output);
            }
            (
                ChannelSourceU8::I16 { slice: s0 },
                ChannelSourceU8::I16 { slice: s1 },
                ChannelSourceU8::I16 { slice: s2 },
                ChannelSourceU8::I16 { slice: s3 },
            ) => {
                let r0 = I16ToU8Reader::new(d, s0, mult, max_i16);
                let r1 = I16ToU8Reader::new(d, s1, mult, max_i16);
                let r2 = I16ToU8Reader::new(d, s2, mult, max_i16);
                let r3 = I16ToU8Reader::new(d, s3, mult, max_i16);
                run_fused_4_u8(d, &r0, &r1, &r2, &r3, output);
            }
            (
                ChannelSourceU8::I16 { slice: s0 },
                ChannelSourceU8::I16 { slice: s1 },
                ChannelSourceU8::I16 { slice: s2 },
                ChannelSourceU8::OpaqueAlpha,
            ) => {
                let r0 = I16ToU8Reader::new(d, s0, mult, max_i16);
                let r1 = I16ToU8Reader::new(d, s1, mult, max_i16);
                let r2 = I16ToU8Reader::new(d, s2, mult, max_i16);
                let r3 = OpaqueU8AlphaReader::new(d);
                run_fused_4_u8(d, &r0, &r1, &r2, &r3, output);
            }
            _ => unreachable!(),
        }
    }
);

simd_function!(
    store_fused_3_f16,
    d: D,
    pub(super) fn store_fused_3_f16_impl(
        c0: ChannelSourceU16<'_>,
        c1: ChannelSourceU16<'_>,
        c2: ChannelSourceU16<'_>,
        output: &mut [u16],
    ) {
        match (c0, c1, c2) {
            (
                ChannelSourceU16::F32 {
                    slice: s0,
                    clamp_range,
                },
                ChannelSourceU16::F32 { slice: s1, .. },
                ChannelSourceU16::F32 { slice: s2, .. },
            ) => {
                let r0 = F32ToF16Reader::new(d, s0, clamp_range);
                let r1 = F32ToF16Reader::new(d, s1, clamp_range);
                let r2 = F32ToF16Reader::new(d, s2, clamp_range);
                run_fused_3_u16(d, &r0, &r1, &r2, output);
            }
            _ => unreachable!(),
        }
    }
);

simd_function!(
    store_fused_4_f16,
    d: D,
    pub(super) fn store_fused_4_f16_impl(
        c0: ChannelSourceU16<'_>,
        c1: ChannelSourceU16<'_>,
        c2: ChannelSourceU16<'_>,
        c3: ChannelSourceU16<'_>,
        output: &mut [u16],
    ) {
        match (c0, c1, c2, c3) {
            (
                ChannelSourceU16::F32 {
                    slice: s0,
                    clamp_range: cr0,
                },
                ChannelSourceU16::F32 { slice: s1, .. },
                ChannelSourceU16::F32 { slice: s2, .. },
                ChannelSourceU16::F32 {
                    slice: s3,
                    clamp_range: cr3,
                },
            ) => {
                let r0 = F32ToF16Reader::new(d, s0, cr0);
                let r1 = F32ToF16Reader::new(d, s1, cr0);
                let r2 = F32ToF16Reader::new(d, s2, cr0);
                let r3 = F32ToF16Reader::new(d, s3, cr3);
                run_fused_4_u16(d, &r0, &r1, &r2, &r3, output);
            }
            (
                ChannelSourceU16::F32 {
                    slice: s0,
                    clamp_range: cr0,
                },
                ChannelSourceU16::F32 { slice: s1, .. },
                ChannelSourceU16::F32 { slice: s2, .. },
                ChannelSourceU16::OpaqueAlpha,
            ) => {
                let r0 = F32ToF16Reader::new(d, s0, cr0);
                let r1 = F32ToF16Reader::new(d, s1, cr0);
                let r2 = F32ToF16Reader::new(d, s2, cr0);
                let r3 = OpaqueF16AlphaReader::new(d);
                run_fused_4_u16(d, &r0, &r1, &r2, &r3, output);
            }
            _ => unreachable!(),
        }
    }
);

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_store_fused_f16() {
        let r: Vec<f32> = (0..128).map(|i| i as f32 * 0.1 - 5.0).collect();
        let g: Vec<f32> = (0..128).map(|i| i as f32 * 0.2 - 10.0).collect();
        let b: Vec<f32> = (0..128).map(|i| i as f32 * 0.15 - 8.0).collect();
        let a: Vec<f32> = (0..128).map(|i| (i as f32) / 128.0).collect();

        let check_u16 = |out: &[u16], expected: &[&[u16]]| {
            let n_chans = expected.len();
            for i in 0..out.len() / n_chans {
                for c in 0..n_chans {
                    assert_eq!(out[i * n_chans + c], expected[c][i]);
                }
            }
        };

        for len in [1, 7, 16, 33, 100] {
            for clamp in [None, Some((-2.0f32, 2.0f32))] {
                let exp_r: Vec<u16> = r[..len].iter().map(|&x| scalar_f32_to_f16(x, clamp)).collect();
                let exp_g: Vec<u16> = g[..len].iter().map(|&x| scalar_f32_to_f16(x, clamp)).collect();
                let exp_b: Vec<u16> = b[..len].iter().map(|&x| scalar_f32_to_f16(x, clamp)).collect();
                let exp_a: Vec<u16> = a[..len].iter().map(|&x| scalar_f32_to_f16(x, clamp)).collect();
                let exp_alpha_opaque = vec![0x3C00u16; len];

                // 3-channel RGB
                let mut out3 = vec![0u16; len * 3];
                store_fused_3_f16(
                    ChannelSourceU16::F32 { slice: &r[..len], clamp_range: clamp },
                    ChannelSourceU16::F32 { slice: &g[..len], clamp_range: clamp },
                    ChannelSourceU16::F32 { slice: &b[..len], clamp_range: clamp },
                    &mut out3,
                );
                check_u16(&out3, &[&exp_r, &exp_g, &exp_b]);

                // 4-channel RGBA
                let mut out4 = vec![0u16; len * 4];
                store_fused_4_f16(
                    ChannelSourceU16::F32 { slice: &r[..len], clamp_range: clamp },
                    ChannelSourceU16::F32 { slice: &g[..len], clamp_range: clamp },
                    ChannelSourceU16::F32 { slice: &b[..len], clamp_range: clamp },
                    ChannelSourceU16::F32 { slice: &a[..len], clamp_range: clamp },
                    &mut out4,
                );
                check_u16(&out4, &[&exp_r, &exp_g, &exp_b, &exp_a]);

                // 4-channel RGB + OpaqueAlpha
                let mut out4_alpha = vec![0u16; len * 4];
                store_fused_4_f16(
                    ChannelSourceU16::F32 { slice: &r[..len], clamp_range: clamp },
                    ChannelSourceU16::F32 { slice: &g[..len], clamp_range: clamp },
                    ChannelSourceU16::F32 { slice: &b[..len], clamp_range: clamp },
                    ChannelSourceU16::OpaqueAlpha,
                    &mut out4_alpha,
                );
                check_u16(&out4_alpha, &[&exp_r, &exp_g, &exp_b, &exp_alpha_opaque]);
            }
        }
    }

    #[test]
    fn test_store_fused_u8() {
        let r: Vec<f32> = (0..128).map(|i| (i as f32) / 128.0).collect();
        let g: Vec<f32> = (0..128).map(|i| ((i + 10) as f32) / 128.0).collect();
        let b: Vec<f32> = (0..128).map(|i| ((i + 20) as f32) / 128.0).collect();
        let a: Vec<f32> = (0..128).map(|i| ((i + 30) as f32) / 128.0).collect();

        let r_i16: Vec<i16> = (0..128).map(|i| (i * 200) as i16).collect();
        let g_i16: Vec<i16> = (0..128).map(|i| ((i + 10) * 200) as i16).collect();
        let b_i16: Vec<i16> = (0..128).map(|i| ((i + 20) * 200) as i16).collect();
        let a_i16: Vec<i16> = (0..128).map(|i| ((i + 30) * 200) as i16).collect();

        let pos = (5, 7);
        let dc = [0, 1, 2, 3];
        let dither_r = &crate::util::DITHER_TABLE[(pos.1 + dc[0] * 13) % 32];
        let dither_g = &crate::util::DITHER_TABLE[(pos.1 + dc[1] * 13) % 32];
        let dither_b = &crate::util::DITHER_TABLE[(pos.1 + dc[2] * 13) % 32];
        let dither_a = &crate::util::DITHER_TABLE[(pos.1 + dc[3] * 13) % 32];

        let exp_r: Vec<u8> = (0..128).map(|i| scalar_f32_to_u8(r[i], 255.0, dither_r[(pos.0 + i + dc[0] * 23) % 32])).collect();
        let exp_g: Vec<u8> = (0..128).map(|i| scalar_f32_to_u8(g[i], 255.0, dither_g[(pos.0 + i + dc[1] * 23) % 32])).collect();
        let exp_b: Vec<u8> = (0..128).map(|i| scalar_f32_to_u8(b[i], 255.0, dither_b[(pos.0 + i + dc[2] * 23) % 32])).collect();
        let exp_a: Vec<u8> = (0..128).map(|i| scalar_f32_to_u8(a[i], 255.0, dither_a[(pos.0 + i + dc[3] * 23) % 32])).collect();
        let exp_alpha_opaque = [255u8; 128];

        let exp_r_i16: Vec<u8> = (0..128).map(|i| scalar_i16_to_u8(r_i16[i], 1, 255)).collect();
        let exp_g_i16: Vec<u8> = (0..128).map(|i| scalar_i16_to_u8(g_i16[i], 1, 255)).collect();
        let exp_b_i16: Vec<u8> = (0..128).map(|i| scalar_i16_to_u8(b_i16[i], 1, 255)).collect();
        let exp_a_i16: Vec<u8> = (0..128).map(|i| scalar_i16_to_u8(a_i16[i], 1, 255)).collect();

        let check_u8 = |out: &[u8], expected: &[&[u8]]| {
            let n_chans = expected.len();
            for i in 0..out.len() / n_chans {
                for c in 0..n_chans {
                    assert_eq!(out[i * n_chans + c], expected[c][i]);
                }
            }
        };

        for len in [1, 7, 16, 33, 100] {
            // 3-channel RGB F32
            let mut out3 = vec![0u8; len * 3];
            store_fused_3_u8(
                ChannelSourceU8::F32 { slice: &r[..len], dither_channel: dc[0] },
                ChannelSourceU8::F32 { slice: &g[..len], dither_channel: dc[1] },
                ChannelSourceU8::F32 { slice: &b[..len], dither_channel: dc[2] },
                &mut out3,
                255.0, 0, 0, pos,
            );
            check_u8(&out3, &[&exp_r[..len], &exp_g[..len], &exp_b[..len]]);

            // 4-channel RGBA F32
            let mut out4 = vec![0u8; len * 4];
            store_fused_4_u8(
                ChannelSourceU8::F32 { slice: &r[..len], dither_channel: dc[0] },
                ChannelSourceU8::F32 { slice: &g[..len], dither_channel: dc[1] },
                ChannelSourceU8::F32 { slice: &b[..len], dither_channel: dc[2] },
                ChannelSourceU8::F32 { slice: &a[..len], dither_channel: dc[3] },
                &mut out4,
                255.0, 0, 0, pos,
            );
            check_u8(&out4, &[&exp_r[..len], &exp_g[..len], &exp_b[..len], &exp_a[..len]]);

            // 4-channel RGB F32 + OpaqueAlpha
            let mut out4_alpha = vec![0u8; len * 4];
            store_fused_4_u8(
                ChannelSourceU8::F32 { slice: &r[..len], dither_channel: dc[0] },
                ChannelSourceU8::F32 { slice: &g[..len], dither_channel: dc[1] },
                ChannelSourceU8::F32 { slice: &b[..len], dither_channel: dc[2] },
                ChannelSourceU8::OpaqueAlpha,
                &mut out4_alpha,
                255.0, 0, 0, pos,
            );
            check_u8(&out4_alpha, &[&exp_r[..len], &exp_g[..len], &exp_b[..len], &exp_alpha_opaque[..len]]);

            // 3-channel RGB I16
            let mut out3_i16 = vec![0u8; len * 3];
            store_fused_3_u8(
                ChannelSourceU8::I16 { slice: &r_i16[..len] },
                ChannelSourceU8::I16 { slice: &g_i16[..len] },
                ChannelSourceU8::I16 { slice: &b_i16[..len] },
                &mut out3_i16,
                0.0, 1, 255, pos,
            );
            check_u8(&out3_i16, &[&exp_r_i16[..len], &exp_g_i16[..len], &exp_b_i16[..len]]);

            // 4-channel RGBA I16
            let mut out4_i16 = vec![0u8; len * 4];
            store_fused_4_u8(
                ChannelSourceU8::I16 { slice: &r_i16[..len] },
                ChannelSourceU8::I16 { slice: &g_i16[..len] },
                ChannelSourceU8::I16 { slice: &b_i16[..len] },
                ChannelSourceU8::I16 { slice: &a_i16[..len] },
                &mut out4_i16,
                0.0, 1, 255, pos,
            );
            check_u8(&out4_i16, &[&exp_r_i16[..len], &exp_g_i16[..len], &exp_b_i16[..len], &exp_a_i16[..len]]);

            // 4-channel RGB F32 + I16 Alpha
            let mut out4_mixed = vec![0u8; len * 4];
            store_fused_4_u8(
                ChannelSourceU8::F32 { slice: &r[..len], dither_channel: dc[0] },
                ChannelSourceU8::F32 { slice: &g[..len], dither_channel: dc[1] },
                ChannelSourceU8::F32 { slice: &b[..len], dither_channel: dc[2] },
                ChannelSourceU8::I16 { slice: &a_i16[..len] },
                &mut out4_mixed,
                255.0, 1, 255, pos,
            );
            check_u8(&out4_mixed, &[&exp_r[..len], &exp_g[..len], &exp_b[..len], &exp_a_i16[..len]]);
        }
    }

    #[test]
    fn test_f32_to_f16_simd_against_scalar() {
        let values: Vec<f32> = (0..256)
            .map(|i| {
                let v = (i as f32 - 128.0) * 0.25;
                if i % 7 == 0 {
                    v * 100.0
                } else if i % 11 == 0 {
                    0.0
                } else {
                    v
                }
            })
            .collect();

        for len in [1, 7, 16, 33, 100, 256] {
            let input = &values[..len];
            let mut simd_output = vec![0u16; len.next_multiple_of(16)];
            let mut scalar_output = vec![0u16; len];

            // Without clamp
            f32_to_f16_simd_dispatch(input, &mut simd_output, None, len);
            for (i, &val) in input.iter().enumerate() {
                scalar_output[i] = crate::util::f16::from_f32(val).to_bits();
            }
            assert_eq!(
                &simd_output[..len],
                &scalar_output[..],
                "Mismatch for len {len} without clamp"
            );

            // With clamp
            let clamp = Some((-5.0, 5.0));
            f32_to_f16_simd_dispatch(input, &mut simd_output, clamp, len);
            for (i, &val) in input.iter().enumerate() {
                scalar_output[i] = crate::util::f16::from_f32(val.clamp(-5.0, 5.0)).to_bits();
            }
            assert_eq!(
                &simd_output[..len],
                &scalar_output[..],
                "Mismatch for len {len} with clamp"
            );
        }
    }
}
