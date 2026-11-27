// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::bit_reader::BitReader;
use crate::error::Error;
use crate::util::CeilLog2;

#[derive(Debug, Clone, Copy)]
pub struct HybridUint {
    split_token: u32,
    split_exponent: u32,
    msb_in_token: u32,
    lsb_in_token: u32,
}

impl HybridUint {
    pub(super) fn is_split_exponent_zero(&self) -> bool {
        self.split_exponent == 0
    }

    pub fn split_token(&self) -> u32 {
        self.split_token
    }

    pub fn decode(log_alpha_size: usize, br: &mut BitReader) -> Result<HybridUint, Error> {
        let split_exponent = br.read((log_alpha_size + 1).ceil_log2())? as u32;
        let split_token = 1u32 << split_exponent;
        let msb_in_token;
        let lsb_in_token;
        if split_exponent != log_alpha_size as u32 {
            let nbits = (split_exponent + 1).ceil_log2() as usize;
            msb_in_token = br.read(nbits)? as u32;
            if msb_in_token > split_exponent {
                return Err(Error::InvalidUintConfig(split_exponent, msb_in_token, None));
            }
            let nbits = (split_exponent - msb_in_token + 1).ceil_log2() as usize;
            lsb_in_token = br.read(nbits)? as u32;
        } else {
            msb_in_token = 0;
            lsb_in_token = 0;
        }
        if lsb_in_token + msb_in_token > split_exponent {
            return Err(Error::InvalidUintConfig(
                split_exponent,
                msb_in_token,
                Some(lsb_in_token),
            ));
        }
        Ok(HybridUint {
            split_token,
            split_exponent,
            msb_in_token,
            lsb_in_token,
        })
    }

    /// Returns true if this config matches the 420 pattern (common in e3 images):
    /// split_exponent=4, msb_in_token=2, lsb_in_token=0
    #[inline(always)]
    pub fn is_config_420(&self) -> bool {
        self.split_exponent == 4
            && self.split_token == 16
            && self.msb_in_token == 2
            && self.lsb_in_token == 0
    }

    /// Specialized fast path for 420 config:
    /// split_exponent=4, msb_in_token=2, lsb_in_token=0
    #[inline(always)]
    pub fn read_config_420(symbol: u32, br: &mut BitReader) -> u32 {
        if symbol < 16 {
            return symbol;
        }

        // Equivalent to: 2 + ((symbol - 16) >> 2)
        let nbits = (symbol >> 2) - 2;
        let nbits = nbits & 31;
        let bits = br.read_optimistic(nbits as usize) as u32;
        let hi = (symbol & 3) | 4;

        (hi << nbits) | bits
    }

    /// Returns true if this config matches `(split_exponent, 0, 0)`:
    /// `msb_in_token == 0` and `lsb_in_token == 0`.
    #[inline(always)]
    pub fn is_config_n00(&self, split_exponent: u32) -> bool {
        self.split_exponent == split_exponent
            && self.split_token == (1u32 << split_exponent)
            && self.msb_in_token == 0
            && self.lsb_in_token == 0
    }

    /// Specialized fast path for `(SPLIT_EXPONENT, 0, 0)` config:
    /// `msb_in_token == 0`, `lsb_in_token == 0`.
    #[inline(always)]
    pub fn read_config_n00<const SPLIT_EXPONENT: u32>(symbol: u32, br: &mut BitReader) -> u32 {
        let split_token = 1u32 << SPLIT_EXPONENT;
        if symbol < split_token {
            return symbol;
        }
        let nbits = (symbol - split_token).wrapping_add(SPLIT_EXPONENT) & 31;
        let bits = br.read_optimistic(nbits as usize) as u32;
        (1u32 << nbits) | bits
    }

    #[inline(always)]
    pub fn read(&self, symbol: u32, br: &mut BitReader) -> u32 {
        if symbol < self.split_token {
            return symbol;
        }
        let bits_in_token = self.lsb_in_token + self.msb_in_token;
        let nbits =
            self.split_exponent - bits_in_token + ((symbol - self.split_token) >> bits_in_token);
        // The bitstream is invalid if nbits >= 32. We do not report errors, and just pretend we
        // decoded a number <32.
        let nbits = nbits & 31;
        let low = symbol & ((1 << self.lsb_in_token) - 1);
        let symbol_nolow = symbol >> self.lsb_in_token;
        let bits = br.read_optimistic(nbits as usize) as u32;
        let hi = (symbol_nolow & ((1 << self.msb_in_token) - 1)) | (1 << self.msb_in_token);
        (((hi << nbits) | bits) << self.lsb_in_token) | low
    }

    pub fn max_bits_for_symbol(&self, symbol: u32) -> usize {
        if symbol < self.split_token {
            if symbol == 0 {
                return 0;
            }
            return (symbol.ilog2() + 1) as usize;
        }
        let bits_in_token = self.lsb_in_token + self.msb_in_token;
        let token_offset = (symbol - self.split_token) >> bits_in_token;
        let nbits = self.split_exponent - bits_in_token + token_offset;
        if nbits >= 32 {
            return 32;
        }
        (self.split_exponent + token_offset + 1) as usize
    }
}

#[cfg(test)]
impl HybridUint {
    pub fn new(split_exponent: u32, msb_in_token: u32, lsb_in_token: u32) -> Self {
        Self {
            split_token: 1 << split_exponent,
            split_exponent,
            msb_in_token,
            lsb_in_token,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_hybrid_uint_decode_invalid() {
        let mut br = BitReader::new(&[10, 75, 10, 75, 168, 139, 132, 255, 244]);
        br.skip_bits(1).unwrap();
        if let Ok(uint) = HybridUint::decode(15, &mut br) {
            uint.read(1022, &mut br);
        }
    }

    #[test]
    fn test_max_bits_for_symbol() {
        let config420 = HybridUint::new(4, 2, 0);
        assert_eq!(config420.max_bits_for_symbol(0), 0);
        assert_eq!(config420.max_bits_for_symbol(1), 1);
        assert_eq!(config420.max_bits_for_symbol(15), 4);
        assert_eq!(config420.max_bits_for_symbol(16), 5);
        assert_eq!(config420.max_bits_for_symbol(19), 5);
        assert_eq!(config420.max_bits_for_symbol(20), 6);
    }

    #[test]
    fn test_read_config_n00_matches_generic() {
        let data = [
            0x5a, 0xa5, 0x3c, 0xc3, 0xf0, 0x0f, 0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf1,
            0x23, 0x45,
        ];
        for (exp, read_fast) in [
            (
                0u32,
                HybridUint::read_config_n00::<0> as fn(u32, &mut BitReader) -> u32,
            ),
            (4u32, HybridUint::read_config_n00::<4>),
        ] {
            let cfg = HybridUint::new(exp, 0, 0);
            assert!(cfg.is_config_n00(exp) && !cfg.is_config_n00(exp ^ 4));
            for sym in 0..64u32 {
                let (mut br_ref, mut br_fast) = (BitReader::new(&data), BitReader::new(&data));
                assert_eq!(read_fast(sym, &mut br_fast), cfg.read(sym, &mut br_ref));
                assert_eq!(br_fast.total_bits_read(), br_ref.total_bits_read());
            }
        }
    }
}
