// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{bit_reader::BitReader, error::Error, headers::encodings::*};
use jxl_macros::UnconditionalCoder;

#[derive(UnconditionalCoder, Debug, Clone, Copy, PartialEq, Eq)]
#[validate]
pub struct BitDepth {
    #[default(false)]
    floating_point_sample: bool,
    #[select_coder(floating_point_sample)]
    #[coder_true(u2S(32, 16, 24, Bits(6)+1))]
    #[coder_false(u2S(8, 10, 12, Bits(6)+1))]
    #[default(8)]
    bits_per_sample: u32,
    #[condition(floating_point_sample)]
    #[default(0)]
    #[coder(Bits(4)+1)]
    exponent_bits_per_sample: u32,
}

impl BitDepth {
    pub fn integer_samples(bits_per_sample: u32) -> BitDepth {
        BitDepth {
            floating_point_sample: false,
            bits_per_sample,
            exponent_bits_per_sample: 0,
        }
    }
    #[cfg(test)]
    pub fn f32() -> BitDepth {
        BitDepth {
            floating_point_sample: true,
            bits_per_sample: 32,
            exponent_bits_per_sample: 8,
        }
    }
    pub fn bits_per_sample(&self) -> u32 {
        self.bits_per_sample
    }
    pub fn exponent_bits_per_sample(&self) -> u32 {
        self.exponent_bits_per_sample
    }
    fn check(&self, _: &Empty) -> Result<(), Error> {
        if self.floating_point_sample {
            if self.exponent_bits_per_sample < 2 || self.exponent_bits_per_sample > 8 {
                Err(Error::InvalidExponent(self.exponent_bits_per_sample))
            } else {
                let mantissa_bits =
                    self.bits_per_sample as i32 - self.exponent_bits_per_sample as i32 - 1;
                if !(2..=23).contains(&mantissa_bits) {
                    Err(Error::InvalidMantissa(mantissa_bits))
                } else {
                    Ok(())
                }
            }
        } else if self.bits_per_sample > 31 {
            Err(Error::InvalidBitsPerSample(self.bits_per_sample))
        } else {
            Ok(())
        }
    }
}
