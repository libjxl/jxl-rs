// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use thiserror::Error;

use crate::entropy_coding::huffman::HUFFMAN_MAX_BITS;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Read out of bounds")]
    OutOfBounds,
    #[error("Non-zero padding bits")]
    NonZeroPadding,
    #[error("Invalid signature {0:02x}{1:02x}, expected ff0a")]
    InvalidSignature(u8, u8),
    #[error("Invalid exponent_bits_per_sample: {0}")]
    InvalidExponent(u32),
    #[error("Invalid mantissa_bits: {0}")]
    InvalidMantissa(i32),
    #[error("Invalid bits_per_sample: {0}")]
    InvalidBitsPerSample(u32),
    #[error("Invalid enum value {0} for {1}")]
    InvalidEnum(u32, String),
    #[error("Value of dim_shift {0} is too large")]
    DimShiftTooLarge(u32),
    #[error("Float is NaN or Inf")]
    FloatNaNOrInf,
    #[error("Invalid gamma value: {0}")]
    InvalidGamma(f32),
    #[error("Invalid color encoding: no ICC and unknown TF / ColorSpace")]
    InvalidColorEncoding,
    #[error("Invalid intensity_target: {0}")]
    InvalidIntensityTarget(f32),
    #[error("Invalid min_nits: {0}")]
    InvalidMinNits(f32),
    #[error("Invalid linear_below {1}, relative_to_max_display is {0}")]
    InvalidLinearBelow(bool, f32),
    #[error("Overflow when computing a bitstream size")]
    SizeOverflow,
    #[error("File truncated")]
    FileTruncated,
    #[error("Invalid ISOBMMF container")]
    InvalidBox,
    #[error("ICC is too large")]
    ICCTooLarge,
    #[error("Invalid HybridUintConfig: {0} {1} {2:?}")]
    InvalidUintConfig(u32, u32, Option<u32>),
    #[error("LZ77 enabled when explicitly disallowed")]
    LZ77Disallowed,
    #[error("Huffman alphabet too large: {0}, max is {}", 1 << HUFFMAN_MAX_BITS)]
    AlphabetTooLargeHuff(usize),
    #[error("Invalid Huffman code")]
    InvalidHuffman,
    #[error("Integer too large: nbits {0} > 29")]
    IntegerTooLarge(u32),
    #[error("Invalid context map: context id {0} > 255")]
    InvalidContextMap(u32),
    #[error("Invalid context map: number of histogram {0}, number of distinct histograms {1}")]
    InvalidContextMapHole(u32, u32),
}
