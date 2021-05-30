// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::bit_reader::BitReader;
use crate::error::Error;

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

pub enum U32Coder {
    Direct(U32),
    Select(U32, U32, U32, U32),
}

trait CoderConfig
where
    Self: Sized,
{
}

pub trait UnconditionalCoder<Config>
where
    Self: Sized,
{
    fn read_unconditional(config: Config, br: &mut BitReader) -> Result<Self, Error>;
}

impl UnconditionalCoder<()> for bool {
    fn read_unconditional(_: (), br: &mut BitReader) -> Result<bool, Error> {
        Ok(br.read(1)? != 0)
    }
}

impl UnconditionalCoder<U32Coder> for u32 {
    fn read_unconditional(config: U32Coder, br: &mut BitReader) -> Result<u32, Error> {
        match config {
            U32Coder::Direct(u) => u.read(br),
            U32Coder::Select(u0, u1, u2, u3) => {
                let selector = br.read(2)?;
                match selector {
                    0 => u0.read(br),
                    1 => u1.read(br),
                    2 => u2.read(br),
                    3 => u3.read(br),
                    _ => panic!("Read two bits and got {}", selector),
                }
            }
        }
    }
}

pub struct SelectCoder<T: Sized> {
    pub use_true: bool,
    pub coder_true: T,
    pub coder_false: T,
}

impl<T, U: UnconditionalCoder<T>> UnconditionalCoder<SelectCoder<T>> for U {
    fn read_unconditional(config: SelectCoder<T>, br: &mut BitReader) -> Result<U, Error> {
        if config.use_true {
            U::read_unconditional(config.coder_true, br)
        } else {
            U::read_unconditional(config.coder_false, br)
        }
    }
}

pub trait ConditionalCoder<Config>
where
    Self: Sized,
{
    fn read_conditional(config: Config, condition: bool, br: &mut BitReader)
        -> Result<Self, Error>;
}

impl<Config, T: UnconditionalCoder<Config>> ConditionalCoder<Config> for Option<T> {
    fn read_conditional(
        config: Config,
        condition: bool,
        br: &mut BitReader,
    ) -> Result<Option<T>, Error> {
        if condition {
            Ok(Some(T::read_unconditional(config, br)?))
        } else {
            Ok(None)
        }
    }
}

pub trait DefaultedCoder<Config>
where
    Self: Sized,
{
    fn read_defaulted(
        config: Config,
        condition: bool,
        default: Self,
        br: &mut BitReader,
    ) -> Result<Self, Error>;
}

impl<Config, T: UnconditionalCoder<Config>> DefaultedCoder<Config> for T {
    fn read_defaulted(
        config: Config,
        condition: bool,
        default: Self,
        br: &mut BitReader,
    ) -> Result<T, Error> {
        if condition {
            Ok(T::read_unconditional(config, br)?)
        } else {
            Ok(default)
        }
    }
}
