// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::bit_reader::BitReader;
use crate::error::Error;

pub struct Bool;

impl Bool {
    pub fn new() -> Bool {
        Bool {}
    }
    pub fn read(&self, br: &mut BitReader) -> Result<bool, Error> {
        Ok(br.read(1)? != 0)
    }
}

pub enum U32 {
    Bits(usize),
    BitsOffset { n: usize, off: u32 },
    Val(u32),
}

impl U32 {
    pub fn read(&self, br: &mut BitReader) -> Result<u32, Error> {
        match self {
            &U32::Bits(n) => Ok(br.read(n)? as u32),
            &U32::BitsOffset { n, off } => Ok(br.read(n)? as u32 + off),
            &U32::Val(val) => Ok(val),
        }
    }
}

pub struct U32Coder(pub U32, pub U32, pub U32, pub U32);

impl U32Coder {
    pub fn read(&self, br: &mut BitReader) -> Result<u32, Error> {
        let selector = br.read(2)?;
        match selector {
            0 => self.0.read(br),
            1 => self.1.read(br),
            2 => self.2.read(br),
            3 => self.3.read(br),
            _ => panic!("Read two bits and got {}", selector),
        }
    }
}
