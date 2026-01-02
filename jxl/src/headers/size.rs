// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{bit_reader::BitReader, error::Error, headers::encodings::*};
use jxl_macros::UnconditionalCoder;
use num_derive::FromPrimitive;

#[derive(UnconditionalCoder, Copy, Clone, PartialEq, Debug, FromPrimitive, Default)]
enum AspectRatio {
    #[default]
    Unknown = 0,
    Ratio1Over1 = 1,
    Ratio12Over10 = 2,
    Ratio4Over3 = 3,
    Ratio3Over2 = 4,
    Ratio16Over9 = 5,
    Ratio5Over4 = 6,
    Ratio2Over1 = 7,
}

#[derive(UnconditionalCoder, Debug, Clone, Default)]
pub struct Size {
    small: bool,
    #[condition(small)]
    #[coder(Bits(5) + 1)]
    ysize_div8: Option<u32>,
    #[condition(!small)]
    #[coder(1 + u2S(Bits(9), Bits(13), Bits(18), Bits(30)))]
    ysize: Option<u32>,
    #[coder(Bits(3))]
    ratio: AspectRatio,
    #[condition(small && ratio == AspectRatio::Unknown)]
    #[coder(Bits(5) + 1)]
    xsize_div8: Option<u32>,
    #[condition(!small && ratio == AspectRatio::Unknown)]
    #[coder(1 + u2S(Bits(9), Bits(13), Bits(18), Bits(30)))]
    xsize: Option<u32>,
}

#[derive(UnconditionalCoder, Debug, Clone)]
pub struct Preview {
    div8: bool,
    #[condition(div8)]
    #[coder(u2S(16, 32, Bits(5) + 1, Bits(9) + 33))]
    ysize_div8: Option<u32>,
    #[condition(!div8)]
    #[coder(1 + u2S(Bits(6), Bits(8) + 64, Bits(10) + 320, Bits(12) + 1344))]
    ysize: Option<u32>,
    #[coder(Bits(3))]
    ratio: AspectRatio,
    #[condition(div8 && ratio == AspectRatio::Unknown)]
    #[coder(u2S(16, 32, Bits(5) + 1, Bits(9) + 33))]
    xsize_div8: Option<u32>,
    #[condition(!div8 && ratio == AspectRatio::Unknown)]
    #[coder(1 + u2S(Bits(6), Bits(8) + 64, Bits(10) + 320, Bits(12) + 1344))]
    xsize: Option<u32>,
}

fn map_aspect_ratio<T: Fn() -> u32>(ysize: u32, ratio: AspectRatio, fallback: T) -> u32 {
    match ratio {
        AspectRatio::Unknown => fallback(),
        AspectRatio::Ratio1Over1 => ysize,
        AspectRatio::Ratio12Over10 => (ysize as u64 * 12 / 10) as u32,
        AspectRatio::Ratio4Over3 => (ysize as u64 * 4 / 3) as u32,
        AspectRatio::Ratio3Over2 => (ysize as u64 * 3 / 2) as u32,
        AspectRatio::Ratio16Over9 => (ysize as u64 * 16 / 9) as u32,
        AspectRatio::Ratio5Over4 => (ysize as u64 * 5 / 4) as u32,
        AspectRatio::Ratio2Over1 => ysize * 2,
    }
}

impl Size {
    pub fn ysize(&self) -> u32 {
        if self.small {
            self.ysize_div8.unwrap() * 8
        } else {
            self.ysize.unwrap()
        }
    }

    pub fn xsize(&self) -> u32 {
        map_aspect_ratio(self.ysize(), self.ratio, /* fallback */ || {
            if self.small {
                self.xsize_div8.unwrap() * 8
            } else {
                self.xsize.unwrap()
            }
        })
    }
}

impl Preview {
    pub fn ysize(&self) -> u32 {
        if self.div8 {
            self.ysize_div8.unwrap() * 8
        } else {
            self.ysize.unwrap()
        }
    }

    pub fn xsize(&self) -> u32 {
        map_aspect_ratio(self.ysize(), self.ratio, /* fallback */ || {
            if self.div8 {
                self.xsize_div8.unwrap() * 8
            } else {
                self.xsize.unwrap()
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test that aspect ratio calculations handle large ysize values correctly.
    ///
    /// This test demonstrates integer overflow vulnerabilities in the current
    /// implementation. With large ysize values, the aspect ratio calculations
    /// either overflow (Ratio2Over1) or truncate (other ratios that calculate
    /// in u64 then cast to u32).
    ///
    /// Expected behavior: calculations should not wrap/truncate, and should
    /// either return correct u64 values or reject invalid dimensions early.
    #[test]
    fn test_aspect_ratio_large_values_no_overflow() {
        // Test Ratio2Over1 with value that would overflow u32
        // ysize * 2 where ysize > 2^31 will wrap in release, panic in debug
        let large_ysize: u32 = 0x8000_0000; // 2^31
        let size = Size {
            small: false,
            ysize_div8: None,
            ysize: Some(large_ysize),
            ratio: AspectRatio::Ratio2Over1,
            xsize_div8: None,
            xsize: None,
        };

        // The mathematically correct result is 2^32 = 4294967296
        // But the current implementation does ysize * 2 which wraps to 0
        let xsize = size.xsize();

        // This assertion will FAIL on current code (xsize wraps to 0)
        // After fix, xsize() should return u64 and this test needs updating
        assert!(
            xsize >= large_ysize,
            "xsize ({}) should be >= ysize ({}) for Ratio2Over1, but wrapped due to overflow",
            xsize,
            large_ysize
        );
    }

    /// Test that Ratio16Over9 doesn't truncate for large values.
    #[test]
    fn test_aspect_ratio_16_9_no_truncation() {
        // For ysize near u32::MAX, the result of ysize * 16 / 9 exceeds u32::MAX
        let large_ysize: u32 = 3_000_000_000;
        let size = Size {
            small: false,
            ysize_div8: None,
            ysize: Some(large_ysize),
            ratio: AspectRatio::Ratio16Over9,
            xsize_div8: None,
            xsize: None,
        };

        // Correct result: 3_000_000_000 * 16 / 9 = 5_333_333_333 (exceeds u32::MAX)
        // Current code: (3_000_000_000u64 * 16 / 9) as u32 = truncated value
        let xsize = size.xsize();
        let expected = (large_ysize as u64) * 16 / 9;

        // This will FAIL because xsize is truncated to u32
        assert_eq!(
            xsize as u64, expected,
            "xsize should be {} but was {} (truncated from u64 to u32)",
            expected, xsize
        );
    }
}
