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
    (2, 2),
    (4, 4),
    (4, 8),
    (8, 4),
    (8, 8),
    (8, 16),
    (8, 32),
    (16, 8),
    (16, 16),
    (16, 32),
    (32, 8),
    (32, 16),
    (32, 32),
]

print(HEADER)

print("use jxl_simd::{SimdDescriptor, F16SimdVec, f16};")
print("use crate::*;")


def maybe_wrap(n, call):
    print("%s;" % call)


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
    maybe_wrap(COLS, "do_idct_%d_rowblock(d, &mut data[i * %d..])" % (COLS, COLS))
    print("}")
    # Step 2: do column-DCTs on groups of K columns, transposing KxK blocks back.
    print("for i in 0..column_chunks {")
    print("for j in 0..row_chunks {")
    print(
        "D::F16Vec::transpose_square(d, &mut data[j * %d + i..], column_chunks);" % COLS
    )
    print("}")
    maybe_wrap(ROWS, "do_idct_%d(d, &mut data[i..], column_chunks)" % ROWS)
    print("}")


def impl_square(N):
    print("let data = D::F16Vec::make_array_slice_mut(data);")
    print("let chunks = %d / D::F16Vec::LEN;" % N)
    # Step 1: do column-DCTs on the first K columns.
    print("for i in 0..chunks {")
    maybe_wrap(N, "do_idct_%d(d, &mut data[i..], chunks)" % N)
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
    maybe_wrap(N, "do_idct_%d(d, &mut data[i..], chunks)" % N)
    print("}")


def impl_r_greater_c(ROWS, COLS):
    ratio = ROWS / COLS
    print("let data = D::F16Vec::make_array_slice_mut(data);")
    print("let column_chunks = %d / D::F16Vec::LEN;" % COLS)
    print("let row_chunks = %d / D::F16Vec::LEN;" % ROWS)
    # Note: input is transposed, so in the beginning it has ROWS *columns* and COLS *rows*.
    # Step 1: do column-DCTs on columns.
    print("for i in 0..row_chunks {")
    maybe_wrap(COLS, "do_idct_%d(d, &mut data[i..], row_chunks)" % COLS)
    print("}")
    # Step 2: Incrementally transpose each square sub-block of the matrix, then do a column-IDCT which also completes the transpose.
    print("for i in 0..column_chunks {")
    print(
        "let tr_block = |data: &mut [<D::F16Vec as F16SimdVec>::UnderlyingArray], i, j, l| {"
    )
    print(
        "D::F16Vec::transpose_square(d, &mut data[i * %d + j + l * column_chunks..], row_chunks)};"
        % ROWS
    )
    print("(0..%d).for_each(|l| tr_block(data, i, i, l));" % ratio)
    print("for j in i+1..column_chunks {")
    print("(0..%d).for_each(|l| tr_block(data, i, j, l));" % ratio)
    print("(0..%d).for_each(|l| tr_block(data, j, i, l));" % ratio)
    print("for l in 0..%d {" % ratio)
    print("for k in 0..D::F16Vec::LEN {")
    print(
        "data.swap(i * %d + j + k * row_chunks + l * column_chunks, j * %d + i + k * row_chunks + l * column_chunks);"
        % (ROWS, ROWS)
    )
    print("}")
    print("}")
    print("}")
    if ratio == 2:
        maybe_wrap(ROWS, "do_idct_%d_trh(d, &mut data[i..])" % ROWS)
    else:
        maybe_wrap(ROWS, "do_idct_%d_trq(d, &mut data[i..])" % ROWS)
    print("}")


def impl_small(ROWS, COLS):
    S = max(ROWS, COLS)
    if S == 8:
        print("assert_eq!(D::F16Vec::LEN, 8);")
    else:
        print("assert!(D::F16Vec::LEN >= %d && D::F16Vec::LEN <= 8);" % S)
    print("let stride = D::F16Vec::LEN;")
    print("let mut buf = [f16::ZERO; 64];")
    if ROWS < COLS:
        print("for r in 0..%d {" % ROWS)
        print("buf[r * stride..r * stride + %d].copy_from_slice(&data[r * %d..(r + 1) * %d]);" % (COLS, COLS, COLS))
        print("}")
        print("{")
        print("let buf = D::F16Vec::make_array_slice_mut(&mut buf);")
        print("D::F16Vec::transpose_square(d, buf, 1);")
        maybe_wrap(COLS, "do_idct_%d(d, buf, 1)" % COLS)
        print("D::F16Vec::transpose_square(d, buf, 1);")
        maybe_wrap(ROWS, "do_idct_%d(d, buf, 1)" % ROWS)
        print("}")
    else:
        print("for r in 0..%d {" % COLS)
        print("buf[r * stride..r * stride + %d].copy_from_slice(&data[r * %d..(r + 1) * %d]);" % (ROWS, ROWS, ROWS))
        print("}")
        print("{")
        print("let buf = D::F16Vec::make_array_slice_mut(&mut buf);")
        maybe_wrap(COLS, "do_idct_%d(d, buf, 1)" % COLS)
        print("D::F16Vec::transpose_square(d, buf, 1);")
        maybe_wrap(ROWS, "do_idct_%d(d, buf, 1)" % ROWS)
        print("}")
    print("for r in 0..%d {" % ROWS)
    print("data[r * %d..(r + 1) * %d].copy_from_slice(&buf[r * stride..r * stride + %d]);" % (COLS, COLS, COLS))
    print("}")


for ROWS, COLS in SIZES:
    print()
    SZ = ROWS * COLS
    print("#[inline(always)]")
    print(
        "fn idct2d_%d_%d_impl<D: SimdDescriptor>(d: D, data: &mut[f16]) {"
        % (ROWS, COLS)
    )
    print('assert_eq!(data.len(), %d, "Data length mismatch");' % SZ)
    if ROWS < 8 or COLS < 8:
        print("if D::F16Vec::LEN > %d {" % min(ROWS, COLS))
        impl_small(ROWS, COLS)
        print("return;")
        print("}")
    print("assert!(%dusize.is_multiple_of(D::F16Vec::LEN));" % ROWS)
    print("assert!(%dusize.is_multiple_of(D::F16Vec::LEN));" % COLS)
    if ROWS < COLS:
        impl_r_less_c(ROWS, COLS)
    elif ROWS == COLS:
        impl_square(ROWS)
    else:
        impl_r_greater_c(ROWS, COLS)
    print("}")

# Wrappers to reduce SIMD size.
for ROWS, COLS in SIZES:
    print()
    print("#[inline(always)]")
    print("#[allow(unused_variables)]")
    print(
        "pub fn idct2d_%d_%d<D: SimdDescriptor>(d: D, data: &mut[f16]) {" % (ROWS, COLS)
    )
    if ROWS < 16 or COLS < 16:
        print("let d = d.maybe_downgrade_256bit();")
        print("idct2d_%d_%d_impl(d, data)" % (ROWS, COLS))
    else:
        print("idct2d_%d_%d_impl(d, data)" % (ROWS, COLS))
    print("}")
