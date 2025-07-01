// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    BLOCK_DIM, BLOCK_SIZE,
    bit_reader::BitReader,
    entropy_coding::decode::SymbolReader,
    error::Result,
    frame::Histograms,
    frame::transform_map::*,
    headers::permutation::Permutation,
    util::{CeilLog2, tracing_wrappers::*},
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
    let mut reader = SymbolReader::new(&histograms, br, None)?;
    for (ord, transform_type) in TRANSFORM_TYPE_LUT.iter().enumerate() {
        if used_orders & (1 << ord) == 0 {
            continue;
        }
        debug!(?transform_type);
        let num_blocks = covered_blocks_x(*transform_type) * covered_blocks_y(*transform_type);
        for c in 0..3 {
            let size = num_blocks * BLOCK_SIZE as u32;
            let permutation = Permutation::decode(size, num_blocks, &histograms, br, &mut reader)?;
            let index = 3 * ord + c;
            permutations[index].compose(&permutation);
        }
    }
    reader.check_final_state(&histograms)?;
    Ok(permutations)
}
