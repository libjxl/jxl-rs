// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use super::{DataTypeTag, ImageDataType};

/// Nearest-grid alignment, with ties towards positive infinity.
pub(crate) fn align_origin(origin: isize, shift: usize) -> isize {
    if shift == 0 {
        origin
    } else {
        (origin + (1isize << (shift - 1))) >> shift
    }
}

pub(crate) fn box_average(
    data_type: DataTypeTag,
    src_rows: &[&[u8]],
    shift: usize,
    alpha: Option<&[&[u8]]>,
    alpha_clamp: bool,
    dst_row: &mut [u8],
) {
    assert!(matches!(
        data_type,
        DataTypeTag::F32 | DataTypeTag::I32 | DataTypeTag::I16 | DataTypeTag::F64
    ));
    assert!((1..=3).contains(&shift));
    assert!(src_rows.len() <= (1 << shift));
    let src_width = src_rows[0].len() / data_type.size();
    let dst_width = dst_row.len() / data_type.size();

    let mut scratch_buffer;
    let src_scratch;
    let alpha_scratch;
    let sum_scratch;
    let weight_scratch;

    // TODO(tirr-c): Consider reusing scratch buffers.
    if data_type == DataTypeTag::F32 {
        let dst_row = f32::cast_slice_mut(dst_row);
        dst_row.fill(0.0);

        scratch_buffer = vec![0f32; dst_width];
        src_scratch = &mut [][..];
        alpha_scratch = &mut [][..];
        sum_scratch = dst_row;
        weight_scratch = &mut *scratch_buffer;
    } else {
        scratch_buffer = vec![0f32; src_width * 2 + dst_width * 2];
        let (scratch_a, scratch_b) = scratch_buffer.split_at_mut(src_width * 2);
        (src_scratch, alpha_scratch) = scratch_a.split_at_mut(src_width);
        (sum_scratch, weight_scratch) = scratch_b.split_at_mut(dst_width);
    }

    for (y, src_row) in src_rows.iter().enumerate() {
        let src_row = convert_to_f32(data_type, src_row, src_scratch);
        let alpha_row = alpha.map(|alpha| convert_to_f32(data_type, alpha[y], alpha_scratch));
        box_average_acc_f32(
            src_row,
            shift,
            alpha_row,
            alpha_clamp,
            sum_scratch,
            weight_scratch,
        );
    }

    for (sum, &weight) in sum_scratch.iter_mut().zip(&*weight_scratch) {
        *sum = if weight > 0.0 { *sum / weight } else { 0.0 };
    }

    if data_type == DataTypeTag::F32 {
        return;
    }

    let sum_scratch = &scratch_buffer[src_width * 2..][..dst_width];
    match data_type {
        DataTypeTag::I16 => {
            let dst_row = i16::cast_slice_mut(dst_row);
            for (dst, &v) in dst_row.iter_mut().zip(sum_scratch) {
                *dst = v as i16;
            }
        }
        DataTypeTag::I32 => {
            let dst_row = i32::cast_slice_mut(dst_row);
            for (dst, &v) in dst_row.iter_mut().zip(sum_scratch) {
                *dst = v as i32;
            }
        }
        DataTypeTag::F64 => {
            let dst_row = f64::cast_slice_mut(dst_row);
            for (dst, &v) in dst_row.iter_mut().zip(sum_scratch) {
                *dst = v as f64;
            }
        }
        _ => {}
    }
}

fn box_average_acc_f32(
    src_row: &[f32],
    shift: usize,
    alpha_row: Option<&[f32]>,
    alpha_clamp: bool,
    sum_row: &mut [f32],
    weight_row: &mut [f32],
) {
    if let Some(alpha_row) = alpha_row {
        assert_eq!(src_row.len(), alpha_row.len());
    }
    assert_eq!(sum_row.len(), weight_row.len());

    let src_width = src_row.len();
    if src_width == 0 {
        return;
    }

    // TODO(tirr-c): Optimize this loop.
    let src_chunks = src_row.chunks(1 << shift);
    if let Some(alpha_row) = alpha_row {
        let alpha_chunks = alpha_row.chunks(1 << shift);
        let it = sum_row
            .iter_mut()
            .zip(weight_row)
            .zip(src_chunks)
            .zip(alpha_chunks);
        for (((sum, weight), src_chunk), alpha_chunk) in it {
            for (sample, alpha) in src_chunk.iter().zip(alpha_chunk) {
                let a = if alpha_clamp {
                    alpha.clamp(0.0, 1.0)
                } else {
                    alpha.max(0.0)
                };
                *sum += sample * a;
                *weight += a;
            }
        }
    } else {
        let it = sum_row.iter_mut().zip(weight_row).zip(src_chunks);
        for ((sum, weight), src_chunk) in it {
            for sample in src_chunk {
                *sum += sample;
                *weight += 1.0;
            }
        }
    }
}

fn convert_to_f32<'a>(
    data_type: DataTypeTag,
    src_row: &'a [u8],
    scratch: &'a mut [f32],
) -> &'a [f32] {
    match data_type {
        DataTypeTag::F32 => f32::cast_slice(src_row),
        DataTypeTag::I16 => {
            let src_row = i16::cast_slice(src_row);
            for (scratch, &v) in scratch.iter_mut().zip(src_row) {
                *scratch = v as f32;
            }
            scratch
        }
        DataTypeTag::I32 => {
            let src_row = i32::cast_slice(src_row);
            for (scratch, &v) in scratch.iter_mut().zip(src_row) {
                *scratch = v as f32;
            }
            scratch
        }
        DataTypeTag::F64 => {
            let src_row = f64::cast_slice(src_row);
            for (scratch, &v) in scratch.iter_mut().zip(src_row) {
                *scratch = v as f32;
            }
            scratch
        }
        _ => unreachable!(),
    }
}
