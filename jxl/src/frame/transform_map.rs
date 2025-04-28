// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::error::{Error, Result};
use enum_iterator::{cardinality, Sequence};

#[allow(clippy::upper_case_acronyms)]
#[derive(Copy, Clone, Debug, PartialEq, Sequence)]
pub enum HfTransformType {
    // Regular block size DCT
    DCT = 0,
    // Encode pixels without transforming
    IDENTITY = 1,
    // Use 2-by-2 DCT
    DCT2X2 = 2,
    // Use 4-by-4 DCT
    DCT4X4 = 3,
    // Use 16-by-16 DCT
    DCT16X16 = 4,
    // Use 32-by-32 DCT
    DCT32X32 = 5,
    // Use 16-by-8 DCT
    DCT16X8 = 6,
    // Use 8-by-16 DCT
    DCT8X16 = 7,
    // Use 32-by-8 DCT
    DCT32X8 = 8,
    // Use 8-by-32 DCT
    DCT8X32 = 9,
    // Use 32-by-16 DCT
    DCT32X16 = 10,
    // Use 16-by-32 DCT
    DCT16X32 = 11,
    // 4x8 and 8x4 DCT
    DCT4X8 = 12,
    DCT8X4 = 13,
    // Corner-DCT.
    AFV0 = 14,
    AFV1 = 15,
    AFV2 = 16,
    AFV3 = 17,
    // Larger DCTs
    DCT64X64 = 18,
    DCT64X32 = 19,
    DCT32X64 = 20,
    // No transforms smaller than 64x64 are allowed below.
    DCT128X128 = 21,
    DCT128X64 = 22,
    DCT64X128 = 23,
    DCT256X256 = 24,
    DCT256X128 = 25,
    DCT128X256 = 26,
}

pub const INVALID_TRANSFORM: u8 = cardinality::<HfTransformType>() as u8;

pub fn get_transform_type(raw_type: i32) -> Result<HfTransformType, Error> {
    let lut: [HfTransformType; cardinality::<HfTransformType>()] = [
        HfTransformType::DCT,
        HfTransformType::IDENTITY,
        HfTransformType::DCT2X2,
        HfTransformType::DCT4X4,
        HfTransformType::DCT16X16,
        HfTransformType::DCT32X32,
        HfTransformType::DCT16X8,
        HfTransformType::DCT8X16,
        HfTransformType::DCT32X8,
        HfTransformType::DCT8X32,
        HfTransformType::DCT32X16,
        HfTransformType::DCT16X32,
        HfTransformType::DCT4X8,
        HfTransformType::DCT8X4,
        HfTransformType::AFV0,
        HfTransformType::AFV1,
        HfTransformType::AFV2,
        HfTransformType::AFV3,
        HfTransformType::DCT64X64,
        HfTransformType::DCT64X32,
        HfTransformType::DCT32X64,
        HfTransformType::DCT128X128,
        HfTransformType::DCT128X64,
        HfTransformType::DCT64X128,
        HfTransformType::DCT256X256,
        HfTransformType::DCT256X128,
        HfTransformType::DCT128X256,
    ];
    if raw_type < 0 || raw_type >= INVALID_TRANSFORM.into() {
        Err(Error::InvalidVarDCTTransform(raw_type))
    } else {
        Ok(lut[raw_type as usize])
    }
}

pub fn covered_blocks_x(transform: HfTransformType) -> u32 {
    let lut: [u32; cardinality::<HfTransformType>()] = [
        1, 1, 1, 1, 2, 4, 1, 2, 1, 4, 2, 4, 1, 1, 1, 1, 1, 1, 8, 4, 8, 16, 8, 16, 32, 16, 32,
    ];
    lut[transform as usize]
}

pub fn covered_blocks_y(transform: HfTransformType) -> u32 {
    let lut: [u32; cardinality::<HfTransformType>()] = [
        1, 1, 1, 1, 2, 4, 2, 1, 4, 1, 4, 2, 1, 1, 1, 1, 1, 1, 8, 8, 4, 16, 16, 8, 32, 32, 16,
    ];
    lut[transform as usize]
}
