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
        match *self {
            U32::Bits(n) => Ok(br.read(n)? as u32),
            U32::BitsOffset { n, off } => Ok(br.read(n)? as u32 + off),
            U32::Val(val) => Ok(val),
        }
    }
}

pub enum U32Coder {
    Direct(U32),
    Select(U32, U32, U32, U32),
}

#[derive(Default)]
pub struct Empty {}

pub trait UnconditionalCoder<Config>
where
    Self: Sized,
{
    type Nonserialized;
    fn read_unconditional(
        config: &Config,
        br: &mut BitReader,
        nonserialized: &Self::Nonserialized,
    ) -> Result<Self, Error>;
}

impl UnconditionalCoder<()> for bool {
    type Nonserialized = Empty;
    fn read_unconditional(
        _: &(),
        br: &mut BitReader,
        _: &Self::Nonserialized,
    ) -> Result<bool, Error> {
        Ok(br.read(1)? != 0)
    }
}

impl UnconditionalCoder<()> for f32 {
    type Nonserialized = Empty;
    fn read_unconditional(
        _: &(),
        br: &mut BitReader,
        _: &Self::Nonserialized,
    ) -> Result<f32, Error> {
        use half::f16;
        let ret = f16::from_bits(br.read(16)? as u16);
        if !ret.is_finite() {
            Err(Error::FloatNaNOrInf)
        } else {
            Ok(ret.to_f32())
        }
    }
}

impl UnconditionalCoder<U32Coder> for u32 {
    type Nonserialized = Empty;
    fn read_unconditional(
        config: &U32Coder,
        br: &mut BitReader,
        _: &Self::Nonserialized,
    ) -> Result<u32, Error> {
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

impl UnconditionalCoder<U32Coder> for i32 {
    type Nonserialized = Empty;
    fn read_unconditional(
        config: &U32Coder,
        br: &mut BitReader,
        nonserialized: &Self::Nonserialized,
    ) -> Result<i32, Error> {
        let u = u32::read_unconditional(config, br, nonserialized)?;
        Ok(((u >> 1) ^ (((!u) & 1).wrapping_rem(1))) as i32)
    }
}

impl UnconditionalCoder<()> for u64 {
    type Nonserialized = Empty;
    fn read_unconditional(
        _: &(),
        br: &mut BitReader,
        _: &Self::Nonserialized,
    ) -> Result<u64, Error> {
        match br.read(2)? {
            0 => Ok(0),
            1 => Ok(1 + br.read(4)?),
            2 => Ok(17 + br.read(8)?),
            _ => {
                let mut result: u64 = br.read(12)? as u64;
                let mut shift = 12;
                while br.read(1)? == 1 {
                    if shift >= 60 {
                        assert_eq!(shift, 60);
                        return Ok(result | ((br.read(4)? as u64) << shift));
                    }
                    result |= (br.read(8)? as u64) << shift;
                    shift += 8;
                }
                Ok(result)
            }
        }
    }
}

impl UnconditionalCoder<()> for String {
    type Nonserialized = Empty;
    fn read_unconditional(
        _: &(),
        br: &mut BitReader,
        nonserialized: &Self::Nonserialized,
    ) -> Result<String, Error> {
        let len = u32::read_unconditional(
            &U32Coder::Select(
                U32::Val(0),
                U32::Bits(4),
                U32::BitsOffset { n: 5, off: 16 },
                U32::BitsOffset { n: 10, off: 48 },
            ),
            br,
            nonserialized,
        )?;
        let mut ret = String::new();
        ret.reserve(len as usize);
        for _ in 0..len {
            ret.push(br.read(8)? as u8 as char);
        }
        Ok(ret)
    }
}

impl<T: UnconditionalCoder<Config>, Config, const N: usize> UnconditionalCoder<Config> for [T; N] {
    type Nonserialized = T::Nonserialized;
    fn read_unconditional(
        config: &Config,
        br: &mut BitReader,
        nonserialized: &Self::Nonserialized,
    ) -> Result<[T; N], Error> {
        use array_init::try_array_init;
        try_array_init(|_| T::read_unconditional(config, br, nonserialized))
    }
}

pub struct VectorCoder<T: Sized> {
    pub size_coder: U32Coder,
    pub value_coder: T,
}

impl<Config, T: UnconditionalCoder<Config>> UnconditionalCoder<VectorCoder<Config>> for Vec<T> {
    type Nonserialized = T::Nonserialized;
    fn read_unconditional(
        config: &VectorCoder<Config>,
        br: &mut BitReader,
        nonserialized: &Self::Nonserialized,
    ) -> Result<Vec<T>, Error> {
        let len = u32::read_unconditional(&config.size_coder, br, &Empty {})?;
        let mut ret: Vec<T> = Vec::new();
        ret.reserve_exact(len as usize);
        for _ in 0..len {
            ret.push(T::read_unconditional(
                &config.value_coder,
                br,
                nonserialized,
            )?);
        }
        Ok(ret)
    }
}

pub struct SelectCoder<T: Sized> {
    pub use_true: bool,
    pub coder_true: T,
    pub coder_false: T,
}

// Marker trait to avoid conflicting declarations for [T; N].
pub trait Selectable {}
impl Selectable for u32 {}

impl<Config, T: UnconditionalCoder<Config> + Selectable> UnconditionalCoder<SelectCoder<Config>>
    for T
{
    type Nonserialized = <T as UnconditionalCoder<Config>>::Nonserialized;
    fn read_unconditional(
        config: &SelectCoder<Config>,
        br: &mut BitReader,
        nonserialized: &Self::Nonserialized,
    ) -> Result<T, Error> {
        if config.use_true {
            T::read_unconditional(&config.coder_true, br, nonserialized)
        } else {
            T::read_unconditional(&config.coder_false, br, nonserialized)
        }
    }
}

pub trait ConditionalCoder<Config>
where
    Self: Sized,
{
    type Nonserialized;
    fn read_conditional(
        config: &Config,
        condition: bool,
        br: &mut BitReader,
        nonserialized: &Self::Nonserialized,
    ) -> Result<Self, Error>;
}

impl<Config, T: UnconditionalCoder<Config>> ConditionalCoder<Config> for Option<T> {
    type Nonserialized = T::Nonserialized;
    fn read_conditional(
        config: &Config,
        condition: bool,
        br: &mut BitReader,
        nonserialized: &Self::Nonserialized,
    ) -> Result<Option<T>, Error> {
        if condition {
            Ok(Some(T::read_unconditional(config, br, nonserialized)?))
        } else {
            Ok(None)
        }
    }
}

impl ConditionalCoder<()> for String {
    type Nonserialized = Empty;
    fn read_conditional(
        _: &(),
        condition: bool,
        br: &mut BitReader,
        nonserialized: &Empty,
    ) -> Result<String, Error> {
        if condition {
            String::read_unconditional(&(), br, nonserialized)
        } else {
            Ok(String::new())
        }
    }
}

impl<Config, T: UnconditionalCoder<Config>> ConditionalCoder<VectorCoder<Config>> for Vec<T> {
    type Nonserialized = T::Nonserialized;
    fn read_conditional(
        config: &VectorCoder<Config>,
        condition: bool,
        br: &mut BitReader,
        nonserialized: &Self::Nonserialized,
    ) -> Result<Vec<T>, Error> {
        if condition {
            Vec::read_unconditional(config, br, nonserialized)
        } else {
            Ok(Vec::new())
        }
    }
}

pub trait DefaultedElementCoder<Config, T>
where
    Self: Sized,
{
    type Nonserialized;
    fn read_defaulted_element(
        config: &Config,
        condition: bool,
        default: T,
        br: &mut BitReader,
        nonserialized: &Self::Nonserialized,
    ) -> Result<Self, Error>;
}

impl<Config, T> DefaultedElementCoder<VectorCoder<Config>, T> for Vec<T>
where
    T: UnconditionalCoder<Config> + Clone,
{
    type Nonserialized = T::Nonserialized;

    fn read_defaulted_element(
        config: &VectorCoder<Config>,
        condition: bool,
        default: T,
        br: &mut BitReader,
        nonserialized: &Self::Nonserialized,
    ) -> Result<Self, Error> {
        let len = u32::read_unconditional(&config.size_coder, br, &Empty {})?;
        if condition {
            let mut ret: Vec<T> = Vec::new();
            ret.reserve_exact(len as usize);
            for _ in 0..len {
                ret.push(T::read_unconditional(
                    &config.value_coder,
                    br,
                    nonserialized,
                )?);
            }
            Ok(ret)
        } else {
            Ok(vec![default; len as usize])
        }
    }
}

pub trait DefaultedCoder<Config>
where
    Self: Sized,
{
    type Nonserialized;
    fn read_defaulted(
        config: &Config,
        condition: bool,
        default: Self,
        br: &mut BitReader,
        nonserialized: &Self::Nonserialized,
    ) -> Result<Self, Error>;
}

impl<Config, T: UnconditionalCoder<Config>> DefaultedCoder<Config> for T {
    type Nonserialized = T::Nonserialized;
    fn read_defaulted(
        config: &Config,
        condition: bool,
        default: Self,
        br: &mut BitReader,
        nonserialized: &Self::Nonserialized,
    ) -> Result<T, Error> {
        if condition {
            Ok(T::read_unconditional(config, br, nonserialized)?)
        } else {
            Ok(default)
        }
    }
}

// TODO(veluca93): this will likely need to be implemented differently if
// there are extensions.
#[derive(Debug, PartialEq, Default)]
pub struct Extensions {}

impl UnconditionalCoder<()> for Extensions {
    type Nonserialized = Empty;
    fn read_unconditional(
        _: &(),
        br: &mut BitReader,
        _: &Self::Nonserialized,
    ) -> Result<Extensions, Error> {
        use std::convert::TryFrom;
        let selector = u64::read_unconditional(&(), br, &Empty {})?;
        let mut total_size: u64 = 0;
        for i in 0..64 {
            if (selector & (1u64 << i)) != 0 {
                let size = u64::read_unconditional(&(), br, &Empty {})?;
                let sum = total_size.checked_add(size);
                if let Some(s) = sum {
                    total_size = s;
                } else {
                    return Err(Error::SizeOverflow);
                }
            }
        }
        let total_size = usize::try_from(total_size);
        if let Ok(ts) = total_size {
            br.skip_bits(ts)?;
        } else {
            return Err(Error::SizeOverflow);
        }
        Ok(Extensions {})
    }
}
