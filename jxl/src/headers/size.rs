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

/// Maps ysize to xsize based on aspect ratio.
/// Returns None if the calculation would overflow u32.
fn map_aspect_ratio_checked(ysize: u32, ratio: AspectRatio) -> Option<u32> {
    let result = match ratio {
        AspectRatio::Unknown => return None, // Caller must use fallback
        AspectRatio::Ratio1Over1 => ysize as u64,
        AspectRatio::Ratio12Over10 => ysize as u64 * 12 / 10,
        AspectRatio::Ratio4Over3 => ysize as u64 * 4 / 3,
        AspectRatio::Ratio3Over2 => ysize as u64 * 3 / 2,
        AspectRatio::Ratio16Over9 => ysize as u64 * 16 / 9,
        AspectRatio::Ratio5Over4 => ysize as u64 * 5 / 4,
        AspectRatio::Ratio2Over1 => ysize as u64 * 2,
    };
    u32::try_from(result).ok()
}

fn map_aspect_ratio<T: Fn() -> u32>(ysize: u32, ratio: AspectRatio, fallback: T) -> u32 {
    match ratio {
        AspectRatio::Unknown => fallback(),
        _ => map_aspect_ratio_checked(ysize, ratio).unwrap_or({
            // This can only happen with ysize > 2^31 and a multiplying ratio.
            // In practice, such sizes would fail allocation anyway.
            // Return u32::MAX to trigger downstream size validation errors.
            u32::MAX
        }),
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

    /// Test that aspect ratio calculations don't overflow for large ysize values.
    ///
    /// Previously, Ratio2Over1 used `ysize * 2` which would overflow for values > 2^31.
    /// The other ratios used u64 intermediate but could still cause issues when
    /// cast back to u32. Now all ratios saturate to u32::MAX on overflow.
    #[test]
    fn test_aspect_ratio_no_overflow() {
        // Test with a value that would overflow if multiplied by 2
        let large_ysize = u32::MAX;

        // All these should return u32::MAX (saturated) instead of wrapping/overflowing
        assert_eq!(
            map_aspect_ratio(large_ysize, AspectRatio::Ratio2Over1, || 0),
            u32::MAX
        );
        assert_eq!(
            map_aspect_ratio(large_ysize, AspectRatio::Ratio16Over9, || 0),
            u32::MAX
        );

        // Ratio12Over10 with large value
        let result = map_aspect_ratio(large_ysize, AspectRatio::Ratio12Over10, || 0);
        assert_eq!(result, u32::MAX); // 4294967295 * 12 / 10 overflows, saturates

        // Test values that don't overflow
        let small_ysize = 1000u32;
        assert_eq!(
            map_aspect_ratio(small_ysize, AspectRatio::Ratio2Over1, || 0),
            2000
        );
        assert_eq!(
            map_aspect_ratio(small_ysize, AspectRatio::Ratio4Over3, || 0),
            1333
        );

        // Test Ratio1Over1
        assert_eq!(
            map_aspect_ratio(small_ysize, AspectRatio::Ratio1Over1, || 0),
            1000
        );
    }

    /// Test Unknown aspect ratio uses fallback correctly.
    #[test]
    fn test_aspect_ratio_fallback() {
        assert_eq!(map_aspect_ratio(1000, AspectRatio::Unknown, || 500), 500);
    }
}
