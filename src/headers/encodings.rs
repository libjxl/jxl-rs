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

pub enum U32Coder {
    Direct(U32),
    Select(U32, U32, U32, U32),
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

pub trait JxlHeader
where
    Self: Sized,
{
    fn read(br: &mut BitReader) -> Result<Self, Error>;
}

pub trait UnconditionalCoder
where
    Self: Sized,
{
    type Config;

    fn read_unconditional(config: Self::Config, br: &mut BitReader) -> Result<Self, Error>;
}

impl<T: JxlHeader> UnconditionalCoder for T {
    type Config = ();

    fn read_unconditional(_: Self::Config, br: &mut BitReader) -> Result<Self, Error> {
        T::read(br)
    }
}

impl UnconditionalCoder for bool {
    type Config = ();

    fn read_unconditional(_: Self::Config, br: &mut BitReader) -> Result<bool, Error> {
        Ok(br.read(1)? != 0)
    }
}

impl UnconditionalCoder for u32 {
    type Config = U32Coder;

    fn read_unconditional(config: Self::Config, br: &mut BitReader) -> Result<u32, Error> {
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

pub trait ConditionalCoder
where
    Self: Sized,
{
    type Config;

    fn read_conditional(
        config: Self::Config,
        condition: bool,
        br: &mut BitReader,
    ) -> Result<Self, Error>;
}

impl<T: UnconditionalCoder> ConditionalCoder for Option<T> {
    type Config = T::Config;

    fn read_conditional(
        config: Self::Config,
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

pub trait DefaultedCoder
where
    Self: Sized,
{
    type Config;

    fn read_defaulted(
        config: Self::Config,
        condition: bool,
        default: Self,
        br: &mut BitReader,
    ) -> Result<Self, Error>;
}

impl<T: UnconditionalCoder> DefaultedCoder for T {
    type Config = T::Config;

    fn read_defaulted(
        config: Self::Config,
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
