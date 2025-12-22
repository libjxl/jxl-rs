// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Fast paths for transposing orientations (Transpose, Rotate90Cw, Rotate90Ccw, AntiTranspose).
//!
//! For transposing orientations, each input row maps to an output column.
//! This means we write one pixel per output row, which is inherently more
//! scattered than row-based operations, but we can still optimize by
//! avoiding per-pixel coordinate calculations.

#![allow(unsafe_code)]

use std::ops::Range;

use crate::{
    api::{Endianness, JxlDataFormat, JxlOutputBuffer},
    render::low_memory_pipeline::row_buffers::RowBuffer,
};

/// Store pixels with transposition (input row becomes output column).
///
/// # Arguments
/// * `output_x` - The x coordinate in the output buffer where this column goes
/// * `y_forward` - If true, write to output rows 0..n; if false, write to rows n-1..0
///
/// Returns the number of pixels processed.
pub(super) fn store(
    input_buf: &[&RowBuffer],
    input_y: usize,
    xrange: Range<usize>,
    output_buf: &mut JxlOutputBuffer,
    output_x: usize,
    y_forward: bool,
    data_format: JxlDataFormat,
) -> usize {
    let num_pixels = xrange.end - xrange.start;
    let num_channels = input_buf.len();
    let bytes_per_sample = data_format.bytes_per_sample();

    let is_native_endian = match data_format {
        JxlDataFormat::U8 { .. } => true,
        JxlDataFormat::F16 { endianness, .. }
        | JxlDataFormat::U16 { endianness, .. }
        | JxlDataFormat::F32 { endianness, .. } => endianness == Endianness::native(),
    };

    if !is_native_endian {
        return 0;
    }

    // For transposing, input pixel at x goes to output row x (or num_pixels-1-x if reversed)
    // The output column is output_x for all pixels

    match (num_channels, bytes_per_sample) {
        (nc, 1) if nc <= 4 => {
            // U8 format
            for (c, buf) in input_buf.iter().enumerate() {
                let input_row = buf.get_row::<u8>(input_y);
                let start = RowBuffer::x0_offset::<u8>() + xrange.start;

                for (i, &px) in input_row[start..start + num_pixels].iter().enumerate() {
                    let out_y = if y_forward { i } else { num_pixels - 1 - i };
                    // SAFETY: we write initialized u8 values
                    let out_row = unsafe { output_buf.row_mut(out_y) };
                    let out_idx = output_x * nc + c;
                    out_row[out_idx].write(px);
                }
            }
            num_pixels
        }
        (nc, 2) if nc <= 4 => {
            // U16/F16 format
            for (c, buf) in input_buf.iter().enumerate() {
                let input_row = buf.get_row::<u16>(input_y);
                let start = RowBuffer::x0_offset::<u16>() + xrange.start;

                for (i, &px) in input_row[start..start + num_pixels].iter().enumerate() {
                    let out_y = if y_forward { i } else { num_pixels - 1 - i };
                    // SAFETY: we write initialized values
                    let out_row = unsafe { output_buf.row_mut(out_y) };
                    let out_idx = (output_x * nc + c) * 2;
                    let bytes = px.to_ne_bytes();
                    out_row[out_idx].write(bytes[0]);
                    out_row[out_idx + 1].write(bytes[1]);
                }
            }
            num_pixels
        }
        (nc, 4) if nc <= 4 => {
            // F32 format
            for (c, buf) in input_buf.iter().enumerate() {
                let input_row = buf.get_row::<f32>(input_y);
                let start = RowBuffer::x0_offset::<f32>() + xrange.start;

                for (i, &px) in input_row[start..start + num_pixels].iter().enumerate() {
                    let out_y = if y_forward { i } else { num_pixels - 1 - i };
                    // SAFETY: we write initialized values
                    let out_row = unsafe { output_buf.row_mut(out_y) };
                    let out_idx = (output_x * nc + c) * 4;
                    let bytes = px.to_ne_bytes();
                    out_row[out_idx].write(bytes[0]);
                    out_row[out_idx + 1].write(bytes[1]);
                    out_row[out_idx + 2].write(bytes[2]);
                    out_row[out_idx + 3].write(bytes[3]);
                }
            }
            num_pixels
        }
        _ => 0,
    }
}
