// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(unsafe_code)]

use std::ops::Range;

use crate::{
    api::{Endianness, JxlDataFormat, JxlOutputBuffer},
    render::low_memory_pipeline::row_buffers::RowBuffer,
};

/// Store pixels with identity orientation, optionally filling opaque alpha.
/// Returns the number of pixels processed by the fast path.
pub(super) fn store(
    input_buf: &[&RowBuffer],
    input_y: usize,
    xrange: Range<usize>,
    output_buf: &mut JxlOutputBuffer,
    output_y: usize,
    data_format: JxlDataFormat,
) -> usize {
    store_impl(
        input_buf,
        input_y,
        xrange,
        output_buf,
        output_y,
        data_format,
        false,
    )
}

/// Store RGB pixels as RGBA with opaque alpha (255 for U8, 1.0 for float).
/// Returns the number of pixels processed by the fast path.
pub(super) fn store_rgb_as_rgba(
    input_buf: &[&RowBuffer],
    input_y: usize,
    xrange: Range<usize>,
    output_buf: &mut JxlOutputBuffer,
    output_y: usize,
    data_format: JxlDataFormat,
) -> usize {
    store_impl(
        input_buf,
        input_y,
        xrange,
        output_buf,
        output_y,
        data_format,
        true,
    )
}

fn store_impl(
    input_buf: &[&RowBuffer],
    input_y: usize,
    xrange: Range<usize>,
    output_buf: &mut JxlOutputBuffer,
    output_y: usize,
    data_format: JxlDataFormat,
    fill_opaque_alpha: bool,
) -> usize {
    let byte_start = xrange.start * data_format.bytes_per_sample() + RowBuffer::x0_byte_offset();
    let byte_end = xrange.end * data_format.bytes_per_sample() + RowBuffer::x0_byte_offset();
    let is_native_endian = match data_format {
        JxlDataFormat::U8 { .. } => true,
        JxlDataFormat::F16 { endianness, .. }
        | JxlDataFormat::U16 { endianness, .. }
        | JxlDataFormat::F32 { endianness, .. } => endianness == Endianness::native(),
    };

    let num_pixels = xrange.end - xrange.start;
    let output_channels = if fill_opaque_alpha {
        input_buf.len() + 1
    } else {
        input_buf.len()
    };
    let bytes_per_sample = data_format.bytes_per_sample();

    // SAFETY: we never write uninit memory to the `output_row`.
    let output_row = unsafe { output_buf.row_mut(output_y) };
    let output_row = &mut output_row[0..num_pixels * output_channels * bytes_per_sample];

    match (
        input_buf.len(),
        bytes_per_sample,
        is_native_endian,
        fill_opaque_alpha,
    ) {
        // Single channel, no alpha fill - just memcpy
        (1, _, true, false) => {
            let input_buf = &input_buf[0].get_row::<u8>(input_y)[byte_start..byte_end];
            assert_eq!(input_buf.len(), output_row.len());
            // SAFETY: we are copying `u8`s, which have an alignment of 1, from a slice of [u8] to
            // a slice of [MaybeUninit<u8>] of the same length (as we checked just above). u8 and
            // MaybeUninit<u8> have the same layout, and aliasing rules guarantee that the two
            // slices are non-overlapping.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    input_buf.as_ptr(),
                    output_row.as_mut_ptr() as *mut u8,
                    output_row.len(),
                );
            }
            num_pixels
        }
        // RGB U8 -> RGBA U8 with opaque alpha (most common case for web images)
        (3, 1, true, true) => {
            let [r, g, b] = input_buf else { unreachable!() };
            let r_row = &r.get_row::<u8>(input_y)[byte_start..byte_end];
            let g_row = &g.get_row::<u8>(input_y)[byte_start..byte_end];
            let b_row = &b.get_row::<u8>(input_y)[byte_start..byte_end];

            // SAFETY: output_row has exactly num_pixels * 4 bytes
            let out_ptr = output_row.as_mut_ptr() as *mut u8;
            for i in 0..num_pixels {
                unsafe {
                    *out_ptr.add(i * 4) = r_row[i];
                    *out_ptr.add(i * 4 + 1) = g_row[i];
                    *out_ptr.add(i * 4 + 2) = b_row[i];
                    *out_ptr.add(i * 4 + 3) = 255;
                }
            }
            num_pixels
        }
        // 3 channels, 4 bytes per sample (F32), native endian, no alpha
        (3, 4, true, false) => {
            #[cfg(target_arch = "x86_64")]
            {
                let [a, b, c] = input_buf else { unreachable!() };
                super::x86_64::interleave3_32b(
                    &[
                        &a.get_row(input_y)[byte_start..byte_end],
                        &b.get_row(input_y)[byte_start..byte_end],
                        &c.get_row(input_y)[byte_start..byte_end],
                    ],
                    output_row,
                )
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                0
            }
        }
        // RGB F32 -> RGBA F32 with opaque alpha
        (3, 4, true, true) => {
            let [r, g, b] = input_buf else { unreachable!() };
            let r_row = r.get_row::<f32>(input_y);
            let g_row = g.get_row::<f32>(input_y);
            let b_row = b.get_row::<f32>(input_y);
            let start = byte_start / 4;

            // SAFETY: output_row has exactly num_pixels * 16 bytes (4 f32s per pixel)
            let out_ptr = output_row.as_mut_ptr() as *mut f32;
            for i in 0..num_pixels {
                unsafe {
                    *out_ptr.add(i * 4) = r_row[start + i];
                    *out_ptr.add(i * 4 + 1) = g_row[start + i];
                    *out_ptr.add(i * 4 + 2) = b_row[start + i];
                    *out_ptr.add(i * 4 + 3) = 1.0;
                }
            }
            num_pixels
        }
        _ => 0,
    }
}
