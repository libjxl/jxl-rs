// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(dead_code)]

use crate::BLOCK_DIM;

pub const MAX_COEFF_BLOCKS: usize = 32;
pub const MAX_BLOCK_DIM: usize = BLOCK_DIM * MAX_COEFF_BLOCKS;
pub const MAX_COEFF_AREA: usize = MAX_BLOCK_DIM * MAX_BLOCK_DIM;

#[derive(Copy, Clone)]
pub enum HfStrategyType {
    Dct,
    Identity,
    Dct2x2,
    Dct4x4,
    Dct16x16,
    Dct32x32,
    Dct16x8,
    Dct8x16,
    Dct32x8,
    Dct8x32,
    Dct32x16,
    Dct16x32,
    Dct4x8,
    Dct8x4,
    Afv0,
    Afv1,
    Afv2,
    Afv3,
    Dct64x64,
    Dct64x32,
    Dct32x64,
    Dct128x128,
    Dct128x64,
    Dct64x128,
    Dct256x256,
    Dct256x128,
    Dct128x256,
}

impl HfStrategyType {
    pub const NUM_VALID_STRATEGIES: usize = Self::Dct128x256 as usize + 1;

    pub fn is_multiblock(&self) -> bool {
        matches!(
            self,
            HfStrategyType::Dct16x16
                | HfStrategyType::Dct32x32
                | HfStrategyType::Dct16x8
                | HfStrategyType::Dct8x16
                | HfStrategyType::Dct32x8
                | HfStrategyType::Dct8x32
                | HfStrategyType::Dct16x32
                | HfStrategyType::Dct32x16
                | HfStrategyType::Dct32x64
                | HfStrategyType::Dct64x32
                | HfStrategyType::Dct64x64
                | HfStrategyType::Dct64x128
                | HfStrategyType::Dct128x64
                | HfStrategyType::Dct128x128
                | HfStrategyType::Dct128x256
                | HfStrategyType::Dct256x128
                | HfStrategyType::Dct256x256
        )
    }

    pub fn covered_blocks_x(&self) -> usize {
        match self {
            HfStrategyType::Dct => 1,
            HfStrategyType::Identity => 1,
            HfStrategyType::Dct2x2 => 1,
            HfStrategyType::Dct4x4 => 1,
            HfStrategyType::Dct16x16 => 2,
            HfStrategyType::Dct32x32 => 4,
            HfStrategyType::Dct16x8 => 1,
            HfStrategyType::Dct8x16 => 2,
            HfStrategyType::Dct32x8 => 1,
            HfStrategyType::Dct8x32 => 4,
            HfStrategyType::Dct32x16 => 2,
            HfStrategyType::Dct16x32 => 4,
            HfStrategyType::Dct4x8 => 1,
            HfStrategyType::Dct8x4 => 1,
            HfStrategyType::Afv0 => 1,
            HfStrategyType::Afv1 => 1,
            HfStrategyType::Afv2 => 1,
            HfStrategyType::Afv3 => 1,
            HfStrategyType::Dct64x64 => 8,
            HfStrategyType::Dct64x32 => 4,
            HfStrategyType::Dct32x64 => 8,
            HfStrategyType::Dct128x128 => 16,
            HfStrategyType::Dct128x64 => 8,
            HfStrategyType::Dct64x128 => 16,
            HfStrategyType::Dct256x256 => 32,
            HfStrategyType::Dct256x128 => 16,
            HfStrategyType::Dct128x256 => 32,
        }
    }

    pub fn covered_blocks_y(&self) -> usize {
        match self {
            HfStrategyType::Dct => 1,
            HfStrategyType::Identity => 1,
            HfStrategyType::Dct2x2 => 1,
            HfStrategyType::Dct4x4 => 1,
            HfStrategyType::Dct16x16 => 2,
            HfStrategyType::Dct32x32 => 4,
            HfStrategyType::Dct16x8 => 2,
            HfStrategyType::Dct8x16 => 1,
            HfStrategyType::Dct32x8 => 4,
            HfStrategyType::Dct8x32 => 1,
            HfStrategyType::Dct32x16 => 4,
            HfStrategyType::Dct16x32 => 2,
            HfStrategyType::Dct4x8 => 1,
            HfStrategyType::Dct8x4 => 1,
            HfStrategyType::Afv0 => 1,
            HfStrategyType::Afv1 => 1,
            HfStrategyType::Afv2 => 1,
            HfStrategyType::Afv3 => 1,
            HfStrategyType::Dct64x64 => 8,
            HfStrategyType::Dct64x32 => 8,
            HfStrategyType::Dct32x64 => 4,
            HfStrategyType::Dct128x128 => 16,
            HfStrategyType::Dct128x64 => 16,
            HfStrategyType::Dct64x128 => 8,
            HfStrategyType::Dct256x256 => 32,
            HfStrategyType::Dct256x128 => 32,
            HfStrategyType::Dct128x256 => 16,
        }
    }

    pub fn log2_covered_blocks(&self) -> usize {
        match self {
            HfStrategyType::Dct => 0,
            HfStrategyType::Identity => 0,
            HfStrategyType::Dct2x2 => 0,
            HfStrategyType::Dct4x4 => 0,
            HfStrategyType::Dct16x16 => 2,
            HfStrategyType::Dct32x32 => 4,
            HfStrategyType::Dct16x8 => 1,
            HfStrategyType::Dct8x16 => 1,
            HfStrategyType::Dct32x8 => 2,
            HfStrategyType::Dct8x32 => 2,
            HfStrategyType::Dct32x16 => 3,
            HfStrategyType::Dct16x32 => 3,
            HfStrategyType::Dct4x8 => 0,
            HfStrategyType::Dct8x4 => 0,
            HfStrategyType::Afv0 => 0,
            HfStrategyType::Afv1 => 0,
            HfStrategyType::Afv2 => 0,
            HfStrategyType::Afv3 => 0,
            HfStrategyType::Dct64x64 => 6,
            HfStrategyType::Dct64x32 => 5,
            HfStrategyType::Dct32x64 => 5,
            HfStrategyType::Dct128x128 => 8,
            HfStrategyType::Dct128x64 => 7,
            HfStrategyType::Dct64x128 => 7,
            HfStrategyType::Dct256x256 => 10,
            HfStrategyType::Dct256x128 => 9,
            HfStrategyType::Dct128x256 => 9,
        }
    }
}
