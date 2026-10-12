#!/usr/bin/env python3
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

HEADER = """\
// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
"""

SIZES = [
    (1, 2),
    (2, 1),
    (1, 4),
    (4, 1),
    (2, 2),
    (2, 4),
    (4, 2),
    (4, 4),
    (4, 8),
    (8, 4),
    (8, 8),
    (8, 16),
    (16, 8),
    (16, 16),
    (16, 32),
    (32, 16),
    (32, 32),
]

print(HEADER)

print("use jxl_simd::{SimdDescriptor, F16SimdVec, f16};")
print("use crate::*;")


def impl_r_less_c(ROWS, COLS):
    print("let data = D::F16Vec::make_array_slice_mut(data);")
    print("let column_chunks = %d / D::F16Vec::LEN;" % COLS)
    print("let row_chunks = %d / D::F16Vec::LEN;" % ROWS)
    # Step 1: do rowblock-DCTs on the first K rows, transposing KxK blocks first.
    print("for i in 0..row_chunks {")
    print("for j in 0..column_chunks {")
    print(
        "D::F16Vec::transpose_square(d, &mut data[i * %d + j..], column_chunks);" % COLS
    )
    print("}")
    print("do_reinterpreting_dct_%d_rowblock(d, &mut data[i * %d..]);" % (COLS, COLS))
    print("}")
    # Step 2: do column-DCTs on groups of K columns, transposing KxK blocks back.
    print("for i in 0..column_chunks {")
    print("for j in 0..row_chunks {")
    print(
        "D::F16Vec::transpose_square(d, &mut data[j * %d + i..], column_chunks);" % COLS
    )
    print("}")
    print("do_reinterpreting_dct_%d(d, &mut data[i..], column_chunks)" % ROWS)
    print("}")


def impl_square(N):
    print("let data = D::F16Vec::make_array_slice_mut(data);")
    print("let chunks = %d / D::F16Vec::LEN;" % N)
    # Step 1: do column-DCTs on the first K columns.
    print("for i in 0..chunks {")
    print("do_reinterpreting_dct_%d(d, &mut data[i..], chunks);" % N)
    print("}")
    # Step 2: do column-DCTs on groups of K columns, transposing KxK blocks and
    # swapping them in their final place as we do so.
    print("for i in 0..chunks {")
    print("D::F16Vec::transpose_square(d, &mut data[i * %d + i..], chunks);" % N)
    print("for j in i+1..chunks {")
    print("D::F16Vec::transpose_square(d, &mut data[j * %d + i..], chunks);" % N)
    print("D::F16Vec::transpose_square(d, &mut data[i * %d + j..], chunks);" % N)
    print("for k in 0..D::F16Vec::LEN {")
    print("data.swap(i * %d + j + k * chunks, j * %d + i + k * chunks);" % (N, N))
    print("}")
    print("}")
    print("do_reinterpreting_dct_%d(d, &mut data[i..], chunks);" % N)
    print("}")


def impl_r_greater_c(ROWS, COLS):
    ratio = ROWS / COLS
    print("let data = D::F16Vec::make_array_slice_mut(data);")
    print("let column_chunks = %d / D::F16Vec::LEN;" % COLS)
    print("let row_chunks = %d / D::F16Vec::LEN;" % ROWS)
    # Step 1: do column-DCTs on columns, which does the first part of the full matrix transpose.
    print("for i in 0..column_chunks {")
    print("do_reinterpreting_dct_%d_trh(d, &mut data[i..]);" % ROWS)
    print("}")
    # Step 2: Incrementally transpose each square sub-block of the matrix, then do a column-IDCT.
    print("for l in 0..%d {" % ratio)
    print("for i in 0..column_chunks {")
    print(
        "let tr_block = |data: &mut [<D::F16Vec as F16SimdVec>::UnderlyingArray], i, j, l| {"
    )
    print(
        "D::F16Vec::transpose_square(d, &mut data[i * %d + j + l * column_chunks..], row_chunks)};"
        % ROWS
    )
    print("tr_block(data, i, i, l);")
    print("for j in i+1..column_chunks {")
    print("tr_block(data, i, j, l);")
    print("tr_block(data, j, i, l);")
    print("for k in 0..D::F16Vec::LEN {")
    print(
        "data.swap(i * %d + j + k * row_chunks + l * column_chunks, j * %d + i + k * row_chunks + l * column_chunks);"
        % (ROWS, ROWS)
    )
    print("}")
    print("}")
    print(
        "do_reinterpreting_dct_%d(d, &mut data[i + l * column_chunks..], row_chunks);"
        % COLS
    )
    print("}")
    print("}")


def impl_small(ROWS, COLS):
    L = min(ROWS, COLS)
    S = max(ROWS, COLS)
    if S == 8:
        print("assert_eq!(D::F16Vec::LEN, 8);")
    else:
        print("assert!(D::F16Vec::LEN >= %d && D::F16Vec::LEN <= 8);" % S)
    print("let stride = D::F16Vec::LEN;")
    print("let mut buf = [f16::ZERO; 64];")
    if ROWS == 1 or COLS == 1:
        print("for r in 0..%d {" % S)
        print("buf[r * stride] = data[r];")
        print("}")
        print("{")
        print("let buf = D::F16Vec::make_array_slice_mut(&mut buf);")
        print("do_reinterpreting_dct_%d(d, buf, 1);" % S)
        print("}")
        print("for x in 0..%d {" % S)
        print("output[x] = buf[x * stride];")
        print("}")
    elif ROWS < COLS:
        print("for r in 0..%d {" % ROWS)
        print("buf[r * stride..r * stride + %d].copy_from_slice(&data[r * %d..(r + 1) * %d]);" % (COLS, COLS, COLS))
        print("}")
        print("{")
        print("let buf = D::F16Vec::make_array_slice_mut(&mut buf);")
        print("D::F16Vec::transpose_square(d, buf, 1);")
        print("do_reinterpreting_dct_%d(d, buf, 1);" % COLS)
        print("D::F16Vec::transpose_square(d, buf, 1);")
        print("do_reinterpreting_dct_%d(d, buf, 1);" % ROWS)
        print("}")
        print("for y in 0..%d {" % L)
        print("output[y * %d..y * %d + %d].copy_from_slice(&buf[y * stride..y * stride + %d]);" % (S * 8, S * 8, S, S))
        print("}")
    else:
        print("for r in 0..%d {" % ROWS)
        print("buf[r * stride..r * stride + %d].copy_from_slice(&data[r * %d..(r + 1) * %d]);" % (COLS, COLS, COLS))
        print("}")
        print("{")
        print("let buf = D::F16Vec::make_array_slice_mut(&mut buf);")
        print("do_reinterpreting_dct_%d(d, buf, 1);" % ROWS)
        print("D::F16Vec::transpose_square(d, buf, 1);")
        print("do_reinterpreting_dct_%d(d, buf, 1);" % COLS)
        print("}")
        print("for y in 0..%d {" % L)
        print("output[y * %d..y * %d + %d].copy_from_slice(&buf[y * stride..y * stride + %d]);" % (S * 8, S * 8, S, S))
        print("}")


for ROWS, COLS in SIZES:
    print()
    SZ = ROWS * COLS
    L = min(ROWS, COLS)
    S = max(ROWS, COLS)
    print("#[inline(always)]")
    print(
        "fn reinterpreting_dct2d_%d_%d_impl<D: SimdDescriptor>(d: D, data: &mut[f16], output: &mut[f16]) {"
        % (ROWS, COLS)
    )
    print('assert_eq!(data.len(), %d, "Data length mismatch");' % SZ)
    print("assert!(output.len() > %d);" % ((L - 1) * S * 8 + S - 1))

    if ROWS < 8 or COLS < 8:
        print("if D::F16Vec::LEN > %d {" % L)
        impl_small(ROWS, COLS)
        print("return;")
        print("}")

    print("assert!(%dusize.is_multiple_of(D::F16Vec::LEN));" % ROWS)
    print("assert!(%dusize.is_multiple_of(D::F16Vec::LEN));" % COLS)
    print("{")
    if ROWS == 1 or COLS == 1:
        print("let data = D::F16Vec::make_array_slice_mut(data);")
        print("do_reinterpreting_dct_%d(d, data, 1);" % S)
    elif ROWS < COLS:
        impl_r_less_c(ROWS, COLS)
    elif ROWS == COLS:
        impl_square(ROWS)
    else:
        impl_r_greater_c(ROWS, COLS)
    print("}")

    print("for y in 0..%d {" % L)
    print(
        "output[y * %d..y * %d + %d].copy_from_slice(&data[y * %d..(y + 1) * %d]);"
        % (S * 8, S * 8, S, S, S)
    )
    print("}")
    print("}")

# Wrappers to reduce SIMD size.
for ROWS, COLS in SIZES:
    print()
    print("#[inline(always)]")
    print("#[allow(unused_variables)]")
    print(
        "pub fn reinterpreting_dct2d_%d_%d<D: SimdDescriptor>(d: D, data: &mut[f16], output: &mut[f16]) {"
        % (ROWS, COLS)
    )
    if ROWS < 16 or COLS < 16:
        print("let d = d.maybe_downgrade_256bit();")
        print("reinterpreting_dct2d_%d_%d_impl(d, data, output)" % (ROWS, COLS))
    else:
        print("reinterpreting_dct2d_%d_%d_impl(d, data, output)" % (ROWS, COLS))
    print("}")
