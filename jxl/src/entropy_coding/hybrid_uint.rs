// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::bit_reader::BitReader;
use crate::error::Error;

use crate::util::*;

#[derive(Debug)]
pub struct HybridUint {
    split_token: u32,
    split_exponent: u32,
    msb_in_token: u32,
    lsb_in_token: u32,
}

impl HybridUint {
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
    pub fn read(&self, symbol: u32, br: &mut BitReader) -> Result<u32, Error> {
        if symbol < self.split_token {
            return Ok(symbol);
        }
        let bits_in_token = self.lsb_in_token + self.msb_in_token;
        let nbits =
            self.split_exponent - bits_in_token + ((symbol - self.split_token) >> bits_in_token);
        if nbits > 29 {
            return Err(Error::IntegerTooLarge(nbits));
        }
        let low = symbol & ((1 << self.lsb_in_token) - 1);
        let symbol_nolow = symbol >> self.lsb_in_token;
        let bits = br.read(nbits as usize)? as u32;
        let hi = (symbol_nolow & ((1 << self.msb_in_token) - 1)) | (1 << self.msb_in_token);
        Ok((((hi << nbits) | bits) << self.lsb_in_token) | low)
    }
}
