// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::*;
use jxl_simd::{f16, F16SimdVec, SimdDescriptor};

#[inline(always)]
fn reinterpreting_dct2d_1_2_impl<D: SimdDescriptor>(d: D, data: &mut [f16], output: &mut [f16]) {
    assert_eq!(data.len(), 2, "Data length mismatch");
    assert!(output.len() > 1);
    if D::F16Vec::LEN > 1 {
        assert!(D::F16Vec::LEN >= 2 && D::F16Vec::LEN <= 8);
        let stride = D::F16Vec::LEN;
        let mut buf = [f16::ZERO; 64];
        for r in 0..2 {
            buf[r * stride] = data[r];
        }
        {
            let buf = D::F16Vec::make_array_slice_mut(&mut buf);
            do_reinterpreting_dct_2(d, buf, 1);
        }
        for x in 0..2 {
            output[x] = buf[x * stride];
        }
        return;
    }
    assert!(1usize.is_multiple_of(D::F16Vec::LEN));
    assert!(2usize.is_multiple_of(D::F16Vec::LEN));
    {
        let data = D::F16Vec::make_array_slice_mut(data);
        do_reinterpreting_dct_2(d, data, 1);
    }
    for y in 0..1 {
        output[y * 16..y * 16 + 2].copy_from_slice(&data[y * 2..(y + 1) * 2]);
    }
}

#[inline(always)]
fn reinterpreting_dct2d_2_1_impl<D: SimdDescriptor>(d: D, data: &mut [f16], output: &mut [f16]) {
    assert_eq!(data.len(), 2, "Data length mismatch");
    assert!(output.len() > 1);
    if D::F16Vec::LEN > 1 {
        assert!(D::F16Vec::LEN >= 2 && D::F16Vec::LEN <= 8);
        let stride = D::F16Vec::LEN;
        let mut buf = [f16::ZERO; 64];
        for r in 0..2 {
            buf[r * stride] = data[r];
        }
        {
            let buf = D::F16Vec::make_array_slice_mut(&mut buf);
            do_reinterpreting_dct_2(d, buf, 1);
        }
        for x in 0..2 {
            output[x] = buf[x * stride];
        }
        return;
    }
    assert!(2usize.is_multiple_of(D::F16Vec::LEN));
    assert!(1usize.is_multiple_of(D::F16Vec::LEN));
    {
        let data = D::F16Vec::make_array_slice_mut(data);
        do_reinterpreting_dct_2(d, data, 1);
    }
    for y in 0..1 {
        output[y * 16..y * 16 + 2].copy_from_slice(&data[y * 2..(y + 1) * 2]);
    }
}

#[inline(always)]
fn reinterpreting_dct2d_1_4_impl<D: SimdDescriptor>(d: D, data: &mut [f16], output: &mut [f16]) {
    assert_eq!(data.len(), 4, "Data length mismatch");
    assert!(output.len() > 3);
    if D::F16Vec::LEN > 1 {
        assert!(D::F16Vec::LEN >= 4 && D::F16Vec::LEN <= 8);
        let stride = D::F16Vec::LEN;
        let mut buf = [f16::ZERO; 64];
        for r in 0..4 {
            buf[r * stride] = data[r];
        }
        {
            let buf = D::F16Vec::make_array_slice_mut(&mut buf);
            do_reinterpreting_dct_4(d, buf, 1);
        }
        for x in 0..4 {
            output[x] = buf[x * stride];
        }
        return;
    }
    assert!(1usize.is_multiple_of(D::F16Vec::LEN));
    assert!(4usize.is_multiple_of(D::F16Vec::LEN));
    {
        let data = D::F16Vec::make_array_slice_mut(data);
        do_reinterpreting_dct_4(d, data, 1);
    }
    for y in 0..1 {
        output[y * 32..y * 32 + 4].copy_from_slice(&data[y * 4..(y + 1) * 4]);
    }
}

#[inline(always)]
fn reinterpreting_dct2d_4_1_impl<D: SimdDescriptor>(d: D, data: &mut [f16], output: &mut [f16]) {
    assert_eq!(data.len(), 4, "Data length mismatch");
    assert!(output.len() > 3);
    if D::F16Vec::LEN > 1 {
        assert!(D::F16Vec::LEN >= 4 && D::F16Vec::LEN <= 8);
        let stride = D::F16Vec::LEN;
        let mut buf = [f16::ZERO; 64];
        for r in 0..4 {
            buf[r * stride] = data[r];
        }
        {
            let buf = D::F16Vec::make_array_slice_mut(&mut buf);
            do_reinterpreting_dct_4(d, buf, 1);
        }
        for x in 0..4 {
            output[x] = buf[x * stride];
        }
        return;
    }
    assert!(4usize.is_multiple_of(D::F16Vec::LEN));
    assert!(1usize.is_multiple_of(D::F16Vec::LEN));
    {
        let data = D::F16Vec::make_array_slice_mut(data);
        do_reinterpreting_dct_4(d, data, 1);
    }
    for y in 0..1 {
        output[y * 32..y * 32 + 4].copy_from_slice(&data[y * 4..(y + 1) * 4]);
    }
}

#[inline(always)]
fn reinterpreting_dct2d_2_2_impl<D: SimdDescriptor>(d: D, data: &mut [f16], output: &mut [f16]) {
    assert_eq!(data.len(), 4, "Data length mismatch");
    assert!(output.len() > 17);
    if D::F16Vec::LEN > 2 {
        assert!(D::F16Vec::LEN >= 2 && D::F16Vec::LEN <= 8);
        let stride = D::F16Vec::LEN;
        let mut buf = [f16::ZERO; 64];
        for r in 0..2 {
            buf[r * stride..r * stride + 2].copy_from_slice(&data[r * 2..(r + 1) * 2]);
        }
        {
            let buf = D::F16Vec::make_array_slice_mut(&mut buf);
            do_reinterpreting_dct_2(d, buf, 1);
            D::F16Vec::transpose_square(d, buf, 1);
            do_reinterpreting_dct_2(d, buf, 1);
        }
        for y in 0..2 {
            output[y * 16..y * 16 + 2].copy_from_slice(&buf[y * stride..y * stride + 2]);
        }
        return;
    }
    assert!(2usize.is_multiple_of(D::F16Vec::LEN));
    assert!(2usize.is_multiple_of(D::F16Vec::LEN));
    {
        let data = D::F16Vec::make_array_slice_mut(data);
        let chunks = 2 / D::F16Vec::LEN;
        for i in 0..chunks {
            do_reinterpreting_dct_2(d, &mut data[i..], chunks);
        }
        for i in 0..chunks {
            D::F16Vec::transpose_square(d, &mut data[i * 2 + i..], chunks);
            for j in i + 1..chunks {
                D::F16Vec::transpose_square(d, &mut data[j * 2 + i..], chunks);
                D::F16Vec::transpose_square(d, &mut data[i * 2 + j..], chunks);
                for k in 0..D::F16Vec::LEN {
                    data.swap(i * 2 + j + k * chunks, j * 2 + i + k * chunks);
                }
            }
            do_reinterpreting_dct_2(d, &mut data[i..], chunks);
        }
    }
    for y in 0..2 {
        output[y * 16..y * 16 + 2].copy_from_slice(&data[y * 2..(y + 1) * 2]);
    }
}

#[inline(always)]
fn reinterpreting_dct2d_2_4_impl<D: SimdDescriptor>(d: D, data: &mut [f16], output: &mut [f16]) {
    assert_eq!(data.len(), 8, "Data length mismatch");
    assert!(output.len() > 35);
    if D::F16Vec::LEN > 2 {
        assert!(D::F16Vec::LEN >= 4 && D::F16Vec::LEN <= 8);
        let stride = D::F16Vec::LEN;
        let mut buf = [f16::ZERO; 64];
        for r in 0..2 {
            buf[r * stride..r * stride + 4].copy_from_slice(&data[r * 4..(r + 1) * 4]);
        }
        {
            let buf = D::F16Vec::make_array_slice_mut(&mut buf);
            D::F16Vec::transpose_square(d, buf, 1);
            do_reinterpreting_dct_4(d, buf, 1);
            D::F16Vec::transpose_square(d, buf, 1);
            do_reinterpreting_dct_2(d, buf, 1);
        }
        for y in 0..2 {
            output[y * 32..y * 32 + 4].copy_from_slice(&buf[y * stride..y * stride + 4]);
        }
        return;
    }
    assert!(2usize.is_multiple_of(D::F16Vec::LEN));
    assert!(4usize.is_multiple_of(D::F16Vec::LEN));
    {
        let data = D::F16Vec::make_array_slice_mut(data);
        let column_chunks = 4 / D::F16Vec::LEN;
        let row_chunks = 2 / D::F16Vec::LEN;
        for i in 0..row_chunks {
            for j in 0..column_chunks {
                D::F16Vec::transpose_square(d, &mut data[i * 4 + j..], column_chunks);
            }
            do_reinterpreting_dct_4_rowblock(d, &mut data[i * 4..]);
        }
        for i in 0..column_chunks {
            for j in 0..row_chunks {
                D::F16Vec::transpose_square(d, &mut data[j * 4 + i..], column_chunks);
            }
            do_reinterpreting_dct_2(d, &mut data[i..], column_chunks)
        }
    }
    for y in 0..2 {
        output[y * 32..y * 32 + 4].copy_from_slice(&data[y * 4..(y + 1) * 4]);
    }
}

#[inline(always)]
fn reinterpreting_dct2d_4_2_impl<D: SimdDescriptor>(d: D, data: &mut [f16], output: &mut [f16]) {
    assert_eq!(data.len(), 8, "Data length mismatch");
    assert!(output.len() > 35);
    if D::F16Vec::LEN > 2 {
        assert!(D::F16Vec::LEN >= 4 && D::F16Vec::LEN <= 8);
        let stride = D::F16Vec::LEN;
        let mut buf = [f16::ZERO; 64];
        for r in 0..4 {
            buf[r * stride..r * stride + 2].copy_from_slice(&data[r * 2..(r + 1) * 2]);
        }
        {
            let buf = D::F16Vec::make_array_slice_mut(&mut buf);
            do_reinterpreting_dct_4(d, buf, 1);
            D::F16Vec::transpose_square(d, buf, 1);
            do_reinterpreting_dct_2(d, buf, 1);
        }
        for y in 0..2 {
            output[y * 32..y * 32 + 4].copy_from_slice(&buf[y * stride..y * stride + 4]);
        }
        return;
    }
    assert!(4usize.is_multiple_of(D::F16Vec::LEN));
    assert!(2usize.is_multiple_of(D::F16Vec::LEN));
    {
        let data = D::F16Vec::make_array_slice_mut(data);
        let column_chunks = 2 / D::F16Vec::LEN;
        let row_chunks = 4 / D::F16Vec::LEN;
        for i in 0..column_chunks {
            do_reinterpreting_dct_4_trh(d, &mut data[i..]);
        }
        for l in 0..2 {
            for i in 0..column_chunks {
                let tr_block =
                    |data: &mut [<D::F16Vec as F16SimdVec>::UnderlyingArray], i, j, l| {
                        D::F16Vec::transpose_square(
                            d,
                            &mut data[i * 4 + j + l * column_chunks..],
                            row_chunks,
                        )
                    };
                tr_block(data, i, i, l);
                for j in i + 1..column_chunks {
                    tr_block(data, i, j, l);
                    tr_block(data, j, i, l);
                    for k in 0..D::F16Vec::LEN {
                        data.swap(
                            i * 4 + j + k * row_chunks + l * column_chunks,
                            j * 4 + i + k * row_chunks + l * column_chunks,
                        );
                    }
                }
                do_reinterpreting_dct_2(d, &mut data[i + l * column_chunks..], row_chunks);
            }
        }
    }
    for y in 0..2 {
        output[y * 32..y * 32 + 4].copy_from_slice(&data[y * 4..(y + 1) * 4]);
    }
}

#[inline(always)]
fn reinterpreting_dct2d_4_4_impl<D: SimdDescriptor>(d: D, data: &mut [f16], output: &mut [f16]) {
    assert_eq!(data.len(), 16, "Data length mismatch");
    assert!(output.len() > 99);
    if D::F16Vec::LEN > 4 {
        assert!(D::F16Vec::LEN >= 4 && D::F16Vec::LEN <= 8);
        let stride = D::F16Vec::LEN;
        let mut buf = [f16::ZERO; 64];
        for r in 0..4 {
            buf[r * stride..r * stride + 4].copy_from_slice(&data[r * 4..(r + 1) * 4]);
        }
        {
            let buf = D::F16Vec::make_array_slice_mut(&mut buf);
            do_reinterpreting_dct_4(d, buf, 1);
            D::F16Vec::transpose_square(d, buf, 1);
            do_reinterpreting_dct_4(d, buf, 1);
        }
        for y in 0..4 {
            output[y * 32..y * 32 + 4].copy_from_slice(&buf[y * stride..y * stride + 4]);
        }
        return;
    }
    assert!(4usize.is_multiple_of(D::F16Vec::LEN));
    assert!(4usize.is_multiple_of(D::F16Vec::LEN));
    {
        let data = D::F16Vec::make_array_slice_mut(data);
        let chunks = 4 / D::F16Vec::LEN;
        for i in 0..chunks {
            do_reinterpreting_dct_4(d, &mut data[i..], chunks);
        }
        for i in 0..chunks {
            D::F16Vec::transpose_square(d, &mut data[i * 4 + i..], chunks);
            for j in i + 1..chunks {
                D::F16Vec::transpose_square(d, &mut data[j * 4 + i..], chunks);
                D::F16Vec::transpose_square(d, &mut data[i * 4 + j..], chunks);
                for k in 0..D::F16Vec::LEN {
                    data.swap(i * 4 + j + k * chunks, j * 4 + i + k * chunks);
                }
            }
            do_reinterpreting_dct_4(d, &mut data[i..], chunks);
        }
    }
    for y in 0..4 {
        output[y * 32..y * 32 + 4].copy_from_slice(&data[y * 4..(y + 1) * 4]);
    }
}

#[inline(always)]
fn reinterpreting_dct2d_4_8_impl<D: SimdDescriptor>(d: D, data: &mut [f16], output: &mut [f16]) {
    assert_eq!(data.len(), 32, "Data length mismatch");
    assert!(output.len() > 199);
    if D::F16Vec::LEN > 4 {
        assert_eq!(D::F16Vec::LEN, 8);
        let stride = D::F16Vec::LEN;
        let mut buf = [f16::ZERO; 64];
        for r in 0..4 {
            buf[r * stride..r * stride + 8].copy_from_slice(&data[r * 8..(r + 1) * 8]);
        }
        {
            let buf = D::F16Vec::make_array_slice_mut(&mut buf);
            D::F16Vec::transpose_square(d, buf, 1);
            do_reinterpreting_dct_8(d, buf, 1);
            D::F16Vec::transpose_square(d, buf, 1);
            do_reinterpreting_dct_4(d, buf, 1);
        }
        for y in 0..4 {
            output[y * 64..y * 64 + 8].copy_from_slice(&buf[y * stride..y * stride + 8]);
        }
        return;
    }
    assert!(4usize.is_multiple_of(D::F16Vec::LEN));
    assert!(8usize.is_multiple_of(D::F16Vec::LEN));
    {
        let data = D::F16Vec::make_array_slice_mut(data);
        let column_chunks = 8 / D::F16Vec::LEN;
        let row_chunks = 4 / D::F16Vec::LEN;
        for i in 0..row_chunks {
            for j in 0..column_chunks {
                D::F16Vec::transpose_square(d, &mut data[i * 8 + j..], column_chunks);
            }
            do_reinterpreting_dct_8_rowblock(d, &mut data[i * 8..]);
        }
        for i in 0..column_chunks {
            for j in 0..row_chunks {
                D::F16Vec::transpose_square(d, &mut data[j * 8 + i..], column_chunks);
            }
            do_reinterpreting_dct_4(d, &mut data[i..], column_chunks)
        }
    }
    for y in 0..4 {
        output[y * 64..y * 64 + 8].copy_from_slice(&data[y * 8..(y + 1) * 8]);
    }
}

#[inline(always)]
fn reinterpreting_dct2d_8_4_impl<D: SimdDescriptor>(d: D, data: &mut [f16], output: &mut [f16]) {
    assert_eq!(data.len(), 32, "Data length mismatch");
    assert!(output.len() > 199);
    if D::F16Vec::LEN > 4 {
        assert_eq!(D::F16Vec::LEN, 8);
        let stride = D::F16Vec::LEN;
        let mut buf = [f16::ZERO; 64];
        for r in 0..8 {
            buf[r * stride..r * stride + 4].copy_from_slice(&data[r * 4..(r + 1) * 4]);
        }
        {
            let buf = D::F16Vec::make_array_slice_mut(&mut buf);
            do_reinterpreting_dct_8(d, buf, 1);
            D::F16Vec::transpose_square(d, buf, 1);
            do_reinterpreting_dct_4(d, buf, 1);
        }
        for y in 0..4 {
            output[y * 64..y * 64 + 8].copy_from_slice(&buf[y * stride..y * stride + 8]);
        }
        return;
    }
    assert!(8usize.is_multiple_of(D::F16Vec::LEN));
    assert!(4usize.is_multiple_of(D::F16Vec::LEN));
    {
        let data = D::F16Vec::make_array_slice_mut(data);
        let column_chunks = 4 / D::F16Vec::LEN;
        let row_chunks = 8 / D::F16Vec::LEN;
        for i in 0..column_chunks {
            do_reinterpreting_dct_8_trh(d, &mut data[i..]);
        }
        for l in 0..2 {
            for i in 0..column_chunks {
                let tr_block =
                    |data: &mut [<D::F16Vec as F16SimdVec>::UnderlyingArray], i, j, l| {
                        D::F16Vec::transpose_square(
                            d,
                            &mut data[i * 8 + j + l * column_chunks..],
                            row_chunks,
                        )
                    };
                tr_block(data, i, i, l);
                for j in i + 1..column_chunks {
                    tr_block(data, i, j, l);
                    tr_block(data, j, i, l);
                    for k in 0..D::F16Vec::LEN {
                        data.swap(
                            i * 8 + j + k * row_chunks + l * column_chunks,
                            j * 8 + i + k * row_chunks + l * column_chunks,
                        );
                    }
                }
                do_reinterpreting_dct_4(d, &mut data[i + l * column_chunks..], row_chunks);
            }
        }
    }
    for y in 0..4 {
        output[y * 64..y * 64 + 8].copy_from_slice(&data[y * 8..(y + 1) * 8]);
    }
}

#[inline(always)]
fn reinterpreting_dct2d_8_8_impl<D: SimdDescriptor>(d: D, data: &mut [f16], output: &mut [f16]) {
    assert_eq!(data.len(), 64, "Data length mismatch");
    assert!(output.len() > 455);
    assert!(8usize.is_multiple_of(D::F16Vec::LEN));
    assert!(8usize.is_multiple_of(D::F16Vec::LEN));
    {
        let data = D::F16Vec::make_array_slice_mut(data);
        let chunks = 8 / D::F16Vec::LEN;
        for i in 0..chunks {
            do_reinterpreting_dct_8(d, &mut data[i..], chunks);
        }
        for i in 0..chunks {
            D::F16Vec::transpose_square(d, &mut data[i * 8 + i..], chunks);
            for j in i + 1..chunks {
                D::F16Vec::transpose_square(d, &mut data[j * 8 + i..], chunks);
                D::F16Vec::transpose_square(d, &mut data[i * 8 + j..], chunks);
                for k in 0..D::F16Vec::LEN {
                    data.swap(i * 8 + j + k * chunks, j * 8 + i + k * chunks);
                }
            }
            do_reinterpreting_dct_8(d, &mut data[i..], chunks);
        }
    }
    for y in 0..8 {
        output[y * 64..y * 64 + 8].copy_from_slice(&data[y * 8..(y + 1) * 8]);
    }
}

#[inline(always)]
fn reinterpreting_dct2d_8_16_impl<D: SimdDescriptor>(d: D, data: &mut [f16], output: &mut [f16]) {
    assert_eq!(data.len(), 128, "Data length mismatch");
    assert!(output.len() > 911);
    assert!(8usize.is_multiple_of(D::F16Vec::LEN));
    assert!(16usize.is_multiple_of(D::F16Vec::LEN));
    {
        let data = D::F16Vec::make_array_slice_mut(data);
        let column_chunks = 16 / D::F16Vec::LEN;
        let row_chunks = 8 / D::F16Vec::LEN;
        for i in 0..row_chunks {
            for j in 0..column_chunks {
                D::F16Vec::transpose_square(d, &mut data[i * 16 + j..], column_chunks);
            }
            do_reinterpreting_dct_16_rowblock(d, &mut data[i * 16..]);
        }
        for i in 0..column_chunks {
            for j in 0..row_chunks {
                D::F16Vec::transpose_square(d, &mut data[j * 16 + i..], column_chunks);
            }
            do_reinterpreting_dct_8(d, &mut data[i..], column_chunks)
        }
    }
    for y in 0..8 {
        output[y * 128..y * 128 + 16].copy_from_slice(&data[y * 16..(y + 1) * 16]);
    }
}

#[inline(always)]
fn reinterpreting_dct2d_16_8_impl<D: SimdDescriptor>(d: D, data: &mut [f16], output: &mut [f16]) {
    assert_eq!(data.len(), 128, "Data length mismatch");
    assert!(output.len() > 911);
    assert!(16usize.is_multiple_of(D::F16Vec::LEN));
    assert!(8usize.is_multiple_of(D::F16Vec::LEN));
    {
        let data = D::F16Vec::make_array_slice_mut(data);
        let column_chunks = 8 / D::F16Vec::LEN;
        let row_chunks = 16 / D::F16Vec::LEN;
        for i in 0..column_chunks {
            do_reinterpreting_dct_16_trh(d, &mut data[i..]);
        }
        for l in 0..2 {
            for i in 0..column_chunks {
                let tr_block =
                    |data: &mut [<D::F16Vec as F16SimdVec>::UnderlyingArray], i, j, l| {
                        D::F16Vec::transpose_square(
                            d,
                            &mut data[i * 16 + j + l * column_chunks..],
                            row_chunks,
                        )
                    };
                tr_block(data, i, i, l);
                for j in i + 1..column_chunks {
                    tr_block(data, i, j, l);
                    tr_block(data, j, i, l);
                    for k in 0..D::F16Vec::LEN {
                        data.swap(
                            i * 16 + j + k * row_chunks + l * column_chunks,
                            j * 16 + i + k * row_chunks + l * column_chunks,
                        );
                    }
                }
                do_reinterpreting_dct_8(d, &mut data[i + l * column_chunks..], row_chunks);
            }
        }
    }
    for y in 0..8 {
        output[y * 128..y * 128 + 16].copy_from_slice(&data[y * 16..(y + 1) * 16]);
    }
}

#[inline(always)]
fn reinterpreting_dct2d_16_16_impl<D: SimdDescriptor>(d: D, data: &mut [f16], output: &mut [f16]) {
    assert_eq!(data.len(), 256, "Data length mismatch");
    assert!(output.len() > 1935);
    assert!(16usize.is_multiple_of(D::F16Vec::LEN));
    assert!(16usize.is_multiple_of(D::F16Vec::LEN));
    {
        let data = D::F16Vec::make_array_slice_mut(data);
        let chunks = 16 / D::F16Vec::LEN;
        for i in 0..chunks {
            do_reinterpreting_dct_16(d, &mut data[i..], chunks);
        }
        for i in 0..chunks {
            D::F16Vec::transpose_square(d, &mut data[i * 16 + i..], chunks);
            for j in i + 1..chunks {
                D::F16Vec::transpose_square(d, &mut data[j * 16 + i..], chunks);
                D::F16Vec::transpose_square(d, &mut data[i * 16 + j..], chunks);
                for k in 0..D::F16Vec::LEN {
                    data.swap(i * 16 + j + k * chunks, j * 16 + i + k * chunks);
                }
            }
            do_reinterpreting_dct_16(d, &mut data[i..], chunks);
        }
    }
    for y in 0..16 {
        output[y * 128..y * 128 + 16].copy_from_slice(&data[y * 16..(y + 1) * 16]);
    }
}

#[inline(always)]
fn reinterpreting_dct2d_16_32_impl<D: SimdDescriptor>(d: D, data: &mut [f16], output: &mut [f16]) {
    assert_eq!(data.len(), 512, "Data length mismatch");
    assert!(output.len() > 3871);
    assert!(16usize.is_multiple_of(D::F16Vec::LEN));
    assert!(32usize.is_multiple_of(D::F16Vec::LEN));
    {
        let data = D::F16Vec::make_array_slice_mut(data);
        let column_chunks = 32 / D::F16Vec::LEN;
        let row_chunks = 16 / D::F16Vec::LEN;
        for i in 0..row_chunks {
            for j in 0..column_chunks {
                D::F16Vec::transpose_square(d, &mut data[i * 32 + j..], column_chunks);
            }
            do_reinterpreting_dct_32_rowblock(d, &mut data[i * 32..]);
        }
        for i in 0..column_chunks {
            for j in 0..row_chunks {
                D::F16Vec::transpose_square(d, &mut data[j * 32 + i..], column_chunks);
            }
            do_reinterpreting_dct_16(d, &mut data[i..], column_chunks)
        }
    }
    for y in 0..16 {
        output[y * 256..y * 256 + 32].copy_from_slice(&data[y * 32..(y + 1) * 32]);
    }
}

#[inline(always)]
fn reinterpreting_dct2d_32_16_impl<D: SimdDescriptor>(d: D, data: &mut [f16], output: &mut [f16]) {
    assert_eq!(data.len(), 512, "Data length mismatch");
    assert!(output.len() > 3871);
    assert!(32usize.is_multiple_of(D::F16Vec::LEN));
    assert!(16usize.is_multiple_of(D::F16Vec::LEN));
    {
        let data = D::F16Vec::make_array_slice_mut(data);
        let column_chunks = 16 / D::F16Vec::LEN;
        let row_chunks = 32 / D::F16Vec::LEN;
        for i in 0..column_chunks {
            do_reinterpreting_dct_32_trh(d, &mut data[i..]);
        }
        for l in 0..2 {
            for i in 0..column_chunks {
                let tr_block =
                    |data: &mut [<D::F16Vec as F16SimdVec>::UnderlyingArray], i, j, l| {
                        D::F16Vec::transpose_square(
                            d,
                            &mut data[i * 32 + j + l * column_chunks..],
                            row_chunks,
                        )
                    };
                tr_block(data, i, i, l);
                for j in i + 1..column_chunks {
                    tr_block(data, i, j, l);
                    tr_block(data, j, i, l);
                    for k in 0..D::F16Vec::LEN {
                        data.swap(
                            i * 32 + j + k * row_chunks + l * column_chunks,
                            j * 32 + i + k * row_chunks + l * column_chunks,
                        );
                    }
                }
                do_reinterpreting_dct_16(d, &mut data[i + l * column_chunks..], row_chunks);
            }
        }
    }
    for y in 0..16 {
        output[y * 256..y * 256 + 32].copy_from_slice(&data[y * 32..(y + 1) * 32]);
    }
}

#[inline(always)]
fn reinterpreting_dct2d_32_32_impl<D: SimdDescriptor>(d: D, data: &mut [f16], output: &mut [f16]) {
    assert_eq!(data.len(), 1024, "Data length mismatch");
    assert!(output.len() > 7967);
    assert!(32usize.is_multiple_of(D::F16Vec::LEN));
    assert!(32usize.is_multiple_of(D::F16Vec::LEN));
    {
        let data = D::F16Vec::make_array_slice_mut(data);
        let chunks = 32 / D::F16Vec::LEN;
        for i in 0..chunks {
            do_reinterpreting_dct_32(d, &mut data[i..], chunks);
        }
        for i in 0..chunks {
            D::F16Vec::transpose_square(d, &mut data[i * 32 + i..], chunks);
            for j in i + 1..chunks {
                D::F16Vec::transpose_square(d, &mut data[j * 32 + i..], chunks);
                D::F16Vec::transpose_square(d, &mut data[i * 32 + j..], chunks);
                for k in 0..D::F16Vec::LEN {
                    data.swap(i * 32 + j + k * chunks, j * 32 + i + k * chunks);
                }
            }
            do_reinterpreting_dct_32(d, &mut data[i..], chunks);
        }
    }
    for y in 0..32 {
        output[y * 256..y * 256 + 32].copy_from_slice(&data[y * 32..(y + 1) * 32]);
    }
}

#[inline(always)]
#[allow(unused_variables)]
pub fn reinterpreting_dct2d_1_2<D: SimdDescriptor>(d: D, data: &mut [f16], output: &mut [f16]) {
    let d = d.maybe_downgrade_256bit();
    reinterpreting_dct2d_1_2_impl(d, data, output)
}

#[inline(always)]
#[allow(unused_variables)]
pub fn reinterpreting_dct2d_2_1<D: SimdDescriptor>(d: D, data: &mut [f16], output: &mut [f16]) {
    let d = d.maybe_downgrade_256bit();
    reinterpreting_dct2d_2_1_impl(d, data, output)
}

#[inline(always)]
#[allow(unused_variables)]
pub fn reinterpreting_dct2d_1_4<D: SimdDescriptor>(d: D, data: &mut [f16], output: &mut [f16]) {
    let d = d.maybe_downgrade_256bit();
    reinterpreting_dct2d_1_4_impl(d, data, output)
}

#[inline(always)]
#[allow(unused_variables)]
pub fn reinterpreting_dct2d_4_1<D: SimdDescriptor>(d: D, data: &mut [f16], output: &mut [f16]) {
    let d = d.maybe_downgrade_256bit();
    reinterpreting_dct2d_4_1_impl(d, data, output)
}

#[inline(always)]
#[allow(unused_variables)]
pub fn reinterpreting_dct2d_2_2<D: SimdDescriptor>(d: D, data: &mut [f16], output: &mut [f16]) {
    let d = d.maybe_downgrade_256bit();
    reinterpreting_dct2d_2_2_impl(d, data, output)
}

#[inline(always)]
#[allow(unused_variables)]
pub fn reinterpreting_dct2d_2_4<D: SimdDescriptor>(d: D, data: &mut [f16], output: &mut [f16]) {
    let d = d.maybe_downgrade_256bit();
    reinterpreting_dct2d_2_4_impl(d, data, output)
}

#[inline(always)]
#[allow(unused_variables)]
pub fn reinterpreting_dct2d_4_2<D: SimdDescriptor>(d: D, data: &mut [f16], output: &mut [f16]) {
    let d = d.maybe_downgrade_256bit();
    reinterpreting_dct2d_4_2_impl(d, data, output)
}

#[inline(always)]
#[allow(unused_variables)]
pub fn reinterpreting_dct2d_4_4<D: SimdDescriptor>(d: D, data: &mut [f16], output: &mut [f16]) {
    let d = d.maybe_downgrade_256bit();
    reinterpreting_dct2d_4_4_impl(d, data, output)
}

#[inline(always)]
#[allow(unused_variables)]
pub fn reinterpreting_dct2d_4_8<D: SimdDescriptor>(d: D, data: &mut [f16], output: &mut [f16]) {
    let d = d.maybe_downgrade_256bit();
    reinterpreting_dct2d_4_8_impl(d, data, output)
}

#[inline(always)]
#[allow(unused_variables)]
pub fn reinterpreting_dct2d_8_4<D: SimdDescriptor>(d: D, data: &mut [f16], output: &mut [f16]) {
    let d = d.maybe_downgrade_256bit();
    reinterpreting_dct2d_8_4_impl(d, data, output)
}

#[inline(always)]
#[allow(unused_variables)]
pub fn reinterpreting_dct2d_8_8<D: SimdDescriptor>(d: D, data: &mut [f16], output: &mut [f16]) {
    let d = d.maybe_downgrade_256bit();
    reinterpreting_dct2d_8_8_impl(d, data, output)
}

#[inline(always)]
#[allow(unused_variables)]
pub fn reinterpreting_dct2d_8_16<D: SimdDescriptor>(d: D, data: &mut [f16], output: &mut [f16]) {
    let d = d.maybe_downgrade_256bit();
    reinterpreting_dct2d_8_16_impl(d, data, output)
}

#[inline(always)]
#[allow(unused_variables)]
pub fn reinterpreting_dct2d_16_8<D: SimdDescriptor>(d: D, data: &mut [f16], output: &mut [f16]) {
    let d = d.maybe_downgrade_256bit();
    reinterpreting_dct2d_16_8_impl(d, data, output)
}

#[inline(always)]
#[allow(unused_variables)]
pub fn reinterpreting_dct2d_16_16<D: SimdDescriptor>(d: D, data: &mut [f16], output: &mut [f16]) {
    reinterpreting_dct2d_16_16_impl(d, data, output)
}

#[inline(always)]
#[allow(unused_variables)]
pub fn reinterpreting_dct2d_16_32<D: SimdDescriptor>(d: D, data: &mut [f16], output: &mut [f16]) {
    reinterpreting_dct2d_16_32_impl(d, data, output)
}

#[inline(always)]
#[allow(unused_variables)]
pub fn reinterpreting_dct2d_32_16<D: SimdDescriptor>(d: D, data: &mut [f16], output: &mut [f16]) {
    reinterpreting_dct2d_32_16_impl(d, data, output)
}

#[inline(always)]
#[allow(unused_variables)]
pub fn reinterpreting_dct2d_32_32<D: SimdDescriptor>(d: D, data: &mut [f16], output: &mut [f16]) {
    reinterpreting_dct2d_32_32_impl(d, data, output)
}
