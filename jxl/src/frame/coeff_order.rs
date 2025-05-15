// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    bit_reader::BitReader,
    error::Result,
    frame::transform_map::*,
    frame::Histograms,
    headers::permutation::Permutation,
    util::{tracing_wrappers::*, CeilLog2},
    BLOCK_DIM, BLOCK_SIZE,
};

use std::mem;

pub const NUM_ORDERS: usize = 13;

pub const TRANSFORM_TYPE_LUT: [HfTransformType; NUM_ORDERS] = [
    HfTransformType::DCT,
    HfTransformType::IDENTITY, // a.k.a. "Hornuss"
    HfTransformType::DCT16X16,
    HfTransformType::DCT32X32,
    HfTransformType::DCT8X16,
    HfTransformType::DCT8X32,
    HfTransformType::DCT16X32,
    HfTransformType::DCT64X64,
    HfTransformType::DCT32X64,
    HfTransformType::DCT128X128,
    HfTransformType::DCT64X128,
    HfTransformType::DCT256X256,
    HfTransformType::DCT128X256,
];

pub const ORDER_LUT: [usize; HfTransformType::VALUES.len()] = [
    /* DCT */ 0, /* IDENTITY */ 1, /* DCT2X2 */ 0, /* DCT4X4 */ 0,
    /* DCT16X16 */ 2, /* DCT32X32 */ 3, /* DCT16X8 */ 4, /* DCT8X16 */ 4,
    /* DCT32X8 */ 5, /* DCT8X32 */ 5, /* DCT32X16 */ 6, /* DCT16X32 */ 6,
    /* DCT4X8 */ 0, /* DCT8X4 */ 0, /* AFV0 */ 1, /* AFV1 */ 1,
    /* AFV2 */ 1, /* AFV3 */ 1, /* DCT64X64 */ 7, /* DCT64X32 */ 8,
    /* DCT32X64 */ 8, /* DCT128X128 */ 9, /* DCT128X64 */ 10,
    /* DCT64X128 */ 10, /* DCT256X256 */ 11, /* DCT256X128 */ 11,
    /* DCT128X256 */ 12,
];

pub const NUM_PERMUTATION_CONTEXTS: usize = 8;

pub fn natural_coeff_order(transform: HfTransformType) -> Vec<u32> {
    let cx = covered_blocks_x(transform) as usize;
    let cy = covered_blocks_y(transform) as usize;
    let xsize: usize = cx * BLOCK_DIM;
    assert!(cx >= cy);
    // We compute the zigzag order for a cx x cx block, then discard all the
    // lines that are not multiple of the ratio between cx and cy.
    let xs = cx / cy;
    let xsm = xs - 1;
    let xss = xs.ceil_log2();
    let mut out: Vec<u32> = vec![0; cx * cy * BLOCK_SIZE];
    // First half of the block
    let mut cur = cx * cy;
    for i in 0..xsize {
        for j in 0..(i + 1) {
            let mut x = j;
            let mut y = i - j;
            if i % 2 != 0 {
                mem::swap(&mut x, &mut y);
            }
            if (y & xsm) != 0 {
                continue;
            }
            y >>= xss;
            let val;
            if x < cx && y < cy {
                val = y * cx + x;
            } else {
                val = cur;
                cur += 1;
            }
            out[val] = (y * xsize + x) as u32;
        }
    }
    // Second half
    for ir in 1..xsize {
        let ip = xsize - ir;
        let i = ip - 1;
        for j in 0..(i + 1) {
            let mut x = xsize - 1 - (i - j);
            let mut y = xsize - 1 - j;
            if i % 2 != 0 {
                mem::swap(&mut x, &mut y);
            }
            if (y & xsm) != 0 {
                continue;
            }
            y >>= xss;
            let val = cur;
            cur += 1;
            out[val] = (y * xsize + x) as u32;
        }
    }
    out
}

pub fn decode_coeff_orders(used_orders: u32, br: &mut BitReader) -> Result<Vec<Permutation>> {
    // TODO(szabadka): Compute natural coefficient orders only for those transform that are used.
    let all_component_orders = 3 * NUM_ORDERS;
    let mut permutations: Vec<Permutation> = (0..all_component_orders)
        .map(|o| Permutation(natural_coeff_order(TRANSFORM_TYPE_LUT[o / 3])))
        .collect();
    if used_orders == 0 {
        return Ok(permutations);
    }
    let histograms = Histograms::decode(NUM_PERMUTATION_CONTEXTS, br, true)?;
    let mut reader = histograms.make_reader(br)?;
    for (ord, transform_type) in TRANSFORM_TYPE_LUT.iter().enumerate() {
        if used_orders & (1 << ord) == 0 {
            continue;
        }
        debug!(?transform_type);
        let num_blocks = covered_blocks_x(*transform_type) * covered_blocks_y(*transform_type);
        for c in 0..3 {
            let size = num_blocks * BLOCK_SIZE as u32;
            let permutation = Permutation::decode(size, num_blocks, br, &mut reader)?;
            let index = 3 * ord + c;
            permutations[index].compose(&permutation);
        }
    }
    reader.check_final_state()?;
    Ok(permutations)
}

#[test]
fn lut_consistency() {
    for (i, &transform_type) in TRANSFORM_TYPE_LUT.iter().enumerate() {
        assert_eq!(ORDER_LUT[transform_type as usize], i);
    }
}
