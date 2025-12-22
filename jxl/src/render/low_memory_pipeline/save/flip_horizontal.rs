// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Fast path for horizontal flip (and Rotate180 which combines horizontal + vertical flip).

#![allow(unsafe_code)]

use std::mem::MaybeUninit;
use std::ops::Range;

use crate::{
    api::{Endianness, JxlDataFormat, JxlOutputBuffer},
    render::low_memory_pipeline::row_buffers::RowBuffer,
};

/// Store pixels with horizontal flip (reverse x order).
/// Returns the number of pixels processed by the fast path.
pub(super) fn store(
    input_buf: &[&RowBuffer],
    input_y: usize,
    xrange: Range<usize>,
    output_buf: &mut JxlOutputBuffer,
    output_y: usize,
    data_format: JxlDataFormat,
) -> usize {
    let num_pixels = xrange.end - xrange.start;
    let is_native_endian = match data_format {
        JxlDataFormat::U8 { .. } => true,
        JxlDataFormat::F16 { endianness, .. }
        | JxlDataFormat::U16 { endianness, .. }
        | JxlDataFormat::F32 { endianness, .. } => endianness == Endianness::native(),
    };

    if !is_native_endian {
        return 0;
    }

    // SAFETY: we never write uninit memory to the `output_row`.
    let output_row = unsafe { output_buf.row_mut(output_y) };
    let num_channels = input_buf.len();
    let bytes_per_sample = data_format.bytes_per_sample();
    let output_row = &mut output_row[0..num_pixels * num_channels * bytes_per_sample];

    match (num_channels, bytes_per_sample) {
        (1, 1) => {
            // Single channel U8 - simple byte reversal
            let input_row = input_buf[0].get_row::<u8>(input_y);
            let start = RowBuffer::x0_offset::<u8>() + xrange.start;
            let end = RowBuffer::x0_offset::<u8>() + xrange.end;
            let input_slice = &input_row[start..end];

            for (i, &px) in input_slice.iter().enumerate() {
                let out_idx = num_pixels - 1 - i;
                output_row[out_idx].write(px);
            }
            num_pixels
        }
        (nc, 1) if nc <= 4 => {
            // Multi-channel U8 - reverse pixel order, keep channel order
            for (c, buf) in input_buf.iter().enumerate() {
                let input_row = buf.get_row::<u8>(input_y);
                let start = RowBuffer::x0_offset::<u8>() + xrange.start;
                let end = RowBuffer::x0_offset::<u8>() + xrange.end;
                let input_slice = &input_row[start..end];

                for (i, &px) in input_slice.iter().enumerate() {
                    let out_idx = (num_pixels - 1 - i) * nc + c;
                    output_row[out_idx].write(px);
                }
            }
            num_pixels
        }
        (1, 4) => {
            // Single channel F32 - reverse order
            let input_row = input_buf[0].get_row::<f32>(input_y);
            let start = RowBuffer::x0_offset::<f32>() + xrange.start;
            let end = RowBuffer::x0_offset::<f32>() + xrange.end;
            let input_slice = &input_row[start..end];

            let ptr = output_row.as_mut_ptr();
            if ptr.align_offset(std::mem::align_of::<f32>()) != 0 {
                return 0;
            }

            // SAFETY: we checked alignment above
            let output_f32 =
                unsafe { std::slice::from_raw_parts_mut(ptr as *mut MaybeUninit<f32>, num_pixels) };

            for (i, &px) in input_slice.iter().enumerate() {
                let out_idx = num_pixels - 1 - i;
                output_f32[out_idx].write(px);
            }
            num_pixels
        }
        (nc, 4) if nc <= 4 => {
            // Multi-channel F32 - reverse pixel order, keep channel order
            let ptr = output_row.as_mut_ptr();
            if ptr.align_offset(std::mem::align_of::<f32>()) != 0 {
                return 0;
            }

            // SAFETY: we checked alignment above
            let output_f32 = unsafe {
                std::slice::from_raw_parts_mut(ptr as *mut MaybeUninit<f32>, num_pixels * nc)
            };

            for (c, buf) in input_buf.iter().enumerate() {
                let input_row = buf.get_row::<f32>(input_y);
                let start = RowBuffer::x0_offset::<f32>() + xrange.start;
                let end = RowBuffer::x0_offset::<f32>() + xrange.end;
                let input_slice = &input_row[start..end];

                for (i, &px) in input_slice.iter().enumerate() {
                    let out_idx = (num_pixels - 1 - i) * nc + c;
                    output_f32[out_idx].write(px);
                }
            }
            num_pixels
        }
        (1, 2) => {
            // Single channel U16/F16 - reverse order
            let input_row = input_buf[0].get_row::<u16>(input_y);
            let start = RowBuffer::x0_offset::<u16>() + xrange.start;
            let end = RowBuffer::x0_offset::<u16>() + xrange.end;
            let input_slice = &input_row[start..end];

            let ptr = output_row.as_mut_ptr();
            if ptr.align_offset(std::mem::align_of::<u16>()) != 0 {
                return 0;
            }

            // SAFETY: we checked alignment above
            let output_u16 =
                unsafe { std::slice::from_raw_parts_mut(ptr as *mut MaybeUninit<u16>, num_pixels) };

            for (i, &px) in input_slice.iter().enumerate() {
                let out_idx = num_pixels - 1 - i;
                output_u16[out_idx].write(px);
            }
            num_pixels
        }
        (nc, 2) if nc <= 4 => {
            // Multi-channel U16/F16 - reverse pixel order, keep channel order
            let ptr = output_row.as_mut_ptr();
            if ptr.align_offset(std::mem::align_of::<u16>()) != 0 {
                return 0;
            }

            // SAFETY: we checked alignment above
            let output_u16 = unsafe {
                std::slice::from_raw_parts_mut(ptr as *mut MaybeUninit<u16>, num_pixels * nc)
            };

            for (c, buf) in input_buf.iter().enumerate() {
                let input_row = buf.get_row::<u16>(input_y);
                let start = RowBuffer::x0_offset::<u16>() + xrange.start;
                let end = RowBuffer::x0_offset::<u16>() + xrange.end;
                let input_slice = &input_row[start..end];

                for (i, &px) in input_slice.iter().enumerate() {
                    let out_idx = (num_pixels - 1 - i) * nc + c;
                    output_u16[out_idx].write(px);
                }
            }
            num_pixels
        }
        _ => 0,
    }
}
