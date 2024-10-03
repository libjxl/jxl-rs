// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::fmt::Debug;

use crate::{
    error::{Error, Result},
    util::{tracing::*, ShiftRightCeil},
};

mod private {
    pub trait Sealed {}
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum DataTypeTag {
    U8,
    U16,
    U32,
    F32,
    I8,
    I16,
    I32,
    F16,
    F64,
}

pub trait ImageDataType: private::Sealed + Copy + Default + 'static + Debug + PartialEq {
    /// ID of this data type. Different types *must* have different values.
    const DATA_TYPE_ID: DataTypeTag;

    fn from_f64(f: f64) -> Self;
    fn to_f64(self) -> f64;
    #[cfg(test)]
    fn random<R: rand::Rng>(rng: &mut R) -> Self;
}

macro_rules! impl_image_data_type {
    ($ty: ty, $id: ident) => {
        impl private::Sealed for $ty {}
        impl ImageDataType for $ty {
            const DATA_TYPE_ID: DataTypeTag = DataTypeTag::$id;
            fn from_f64(f: f64) -> $ty {
                f as $ty
            }
            fn to_f64(self) -> f64 {
                self as f64
            }
            #[cfg(test)]
            fn random<R: rand::Rng>(rng: &mut R) -> Self {
                use rand::distributions::{Distribution, Uniform};
                Uniform::new(<$ty>::MIN, <$ty>::MAX).sample(rng)
            }
        }
    };
}

impl_image_data_type!(u8, U8);
impl_image_data_type!(u16, U16);
impl_image_data_type!(u32, U32);
impl_image_data_type!(f32, F32);
impl_image_data_type!(i8, I8);
impl_image_data_type!(i16, I16);
impl_image_data_type!(i32, I32);

impl private::Sealed for half::f16 {}
impl ImageDataType for half::f16 {
    const DATA_TYPE_ID: DataTypeTag = DataTypeTag::F16;
    fn from_f64(f: f64) -> half::f16 {
        half::f16::from_f64(f)
    }
    fn to_f64(self) -> f64 {
        <half::f16>::to_f64(self)
    }
    #[cfg(test)]
    fn random<R: rand::Rng>(rng: &mut R) -> Self {
        use rand::distributions::{Distribution, Uniform};
        Self::from_f64(Uniform::new(Self::MIN.to_f64(), Self::MAX.to_f64()).sample(rng))
    }
}

// Meant to be used by the simple render pipeline and in general for
// testing purposes.
impl_image_data_type!(f64, F64);

pub struct Image<T: ImageDataType> {
    size: (usize, usize),
    data: Vec<T>,
}

#[derive(Clone, Copy)]
pub struct ImageRect<'a, T: ImageDataType> {
    origin: (usize, usize),
    size: (usize, usize),
    image: &'a Image<T>,
}

pub struct ImageRectMut<'a, T: ImageDataType> {
    origin: (usize, usize),
    size: (usize, usize),
    image: &'a mut Image<T>,
}

impl<'a, T: ImageDataType> Debug for ImageRect<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:?} {}x{}+{}+{}",
            T::DATA_TYPE_ID,
            self.size.0,
            self.size.1,
            self.origin.0,
            self.origin.1
        )
    }
}

impl<'a, T: ImageDataType> Debug for ImageRectMut<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "mut {:?} {}x{}+{}+{}",
            T::DATA_TYPE_ID,
            self.size.0,
            self.size.1,
            self.origin.0,
            self.origin.1
        )
    }
}

impl<T: ImageDataType> Image<T> {
    #[instrument(err)]
    pub fn new(size: (usize, usize)) -> Result<Image<T>> {
        let (xsize, ysize) = size;
        // These limits let us not worry about overflows.
        if xsize as u64 >= i64::MAX as u64 / 4 || ysize as u64 >= i64::MAX as u64 / 4 {
            return Err(Error::ImageSizeTooLarge(xsize, ysize));
        }
        let total_size = xsize
            .checked_mul(ysize)
            .ok_or(Error::ImageSizeTooLarge(xsize, ysize))?;
        if xsize == 0 || ysize == 0 {
            return Err(Error::InvalidImageSize(xsize, ysize));
        }
        debug!("trying to allocate image");
        let mut data = vec![];
        data.try_reserve_exact(total_size)?;
        data.resize(total_size, T::default());
        Ok(Image {
            size: (xsize, ysize),
            data,
        })
    }

    #[cfg(test)]
    pub fn new_random<R: rand::Rng>(size: (usize, usize), rng: &mut R) -> Result<Image<T>> {
        let mut img = Self::new(size)?;
        img.data.iter_mut().for_each(|x| *x = T::random(rng));
        Ok(img)
    }

    pub fn size(&self) -> (usize, usize) {
        self.size
    }

    pub fn group_rect(&self, group_id: usize, log_group_size: (usize, usize)) -> ImageRect<'_, T> {
        let xgroups = self.size.0.shrc(log_group_size.0);
        let group = (group_id % xgroups, group_id / xgroups);
        let origin = (group.0 << log_group_size.0, group.1 << log_group_size.1);
        let size = (
            (self.size.0 - origin.0).min(1 << log_group_size.0),
            (self.size.1 - origin.1).min(1 << log_group_size.1),
        );
        trace!(
            "making rect {}x{}+{}+{} for group {group_id} in image of size {:?}, log group sizes {:?}",
            size.0, size.1, origin.0, origin.1, self.size, log_group_size
        );
        self.as_rect().rect(origin, size).unwrap()
    }

    pub fn group_rect_mut(
        &mut self,
        group_id: usize,
        log_group_size: usize,
    ) -> ImageRectMut<'_, T> {
        let xgroups = self.size.0.shrc(log_group_size);
        let group = (group_id % xgroups, group_id / xgroups);
        let origin = (group.0 << log_group_size, group.1 << log_group_size);
        let size = (
            (self.size.0 - origin.0).min(1 << log_group_size),
            (self.size.1 - origin.1).min(1 << log_group_size),
        );
        self.as_rect_mut().into_rect(origin, size).unwrap()
    }

    pub fn as_rect(&self) -> ImageRect<'_, T> {
        ImageRect {
            origin: (0, 0),
            size: self.size,
            image: self,
        }
    }

    pub fn as_rect_mut(&mut self) -> ImageRectMut<'_, T> {
        ImageRectMut {
            origin: (0, 0),
            size: self.size,
            image: self,
        }
    }
}

fn rect_size_check(
    origin: (usize, usize),
    size: (usize, usize),
    ssize: (usize, usize),
) -> Result<()> {
    if origin
        .0
        .checked_add(size.0)
        .ok_or(Error::ArithmeticOverflow)?
        > ssize.0
        || origin
            .1
            .checked_add(size.1)
            .ok_or(Error::ArithmeticOverflow)?
            > ssize.1
    {
        Err(Error::RectOutOfBounds(
            size.0, size.1, origin.0, origin.1, ssize.0, ssize.1,
        ))
    } else {
        Ok(())
    }
}

impl<'a, T: ImageDataType> ImageRect<'a, T> {
    pub fn rect(self, origin: (usize, usize), size: (usize, usize)) -> Result<ImageRect<'a, T>> {
        rect_size_check(origin, size, self.size)?;
        Ok(ImageRect {
            origin: (origin.0 + self.origin.0, origin.1 + self.origin.1),
            size,
            image: self.image,
        })
    }

    pub fn size(&self) -> (usize, usize) {
        self.size
    }

    pub fn row(&self, row: usize) -> &'a [T] {
        debug_assert!(row < self.size.1);
        let start = (row + self.origin.1) * self.image.size.0 + self.origin.0;
        trace!(
            "{self:?} img size {:?} row {row} start {}",
            self.image.size,
            start
        );
        &self.image.data[start..start + self.size.0]
    }

    pub fn to_image(&self) -> Result<Image<T>> {
        let total_size = self.size.0 * self.size.1;
        let mut data = vec![];
        data.try_reserve_exact(total_size)?;
        data.extend((0..self.size.1).flat_map(|x| self.row(x).iter()));
        Ok(Image {
            size: self.size,
            data,
        })
    }

    pub fn iter(&self) -> impl Iterator<Item = T> + '_ {
        (0..self.size.1).flat_map(|x| self.row(x).iter().cloned())
    }

    #[cfg(test)]
    pub fn check_equal(&self, other: ImageRect<T>) {
        assert_eq!(self.size, other.size);
        for y in 0..self.size.1 {
            for x in 0..self.size.0 {
                if self.row(y)[x] != other.row(y)[x] {
                    let mut msg = format!(
                        "mismatch at position {x}x{y}, values {:?} and {:?}",
                        self.row(y)[x],
                        other.row(y)[x]
                    );
                    if self.origin != (0, 0) {
                        msg = format!(
                            "; position in ground truth {}x{}",
                            x + self.origin.0,
                            y + self.origin.1
                        );
                    }
                    if other.origin != (0, 0) {
                        msg = format!(
                            "; position in checked img {}x{}",
                            x + other.origin.0,
                            y + other.origin.1
                        );
                    }
                    panic!("{}", msg);
                }
            }
        }
    }
}

impl<'a, T: ImageDataType> PartialEq<ImageRect<'a, T>> for ImageRect<'a, T> {
    fn eq(&self, other: &ImageRect<'a, T>) -> bool {
        self.iter().zip(other.iter()).all(|(x, y)| x == y)
    }
}

impl<'a, T: ImageDataType + Eq> Eq for ImageRect<'a, T> {}

impl<'a, T: ImageDataType> ImageRectMut<'a, T> {
    pub fn rect(
        &'a mut self,
        origin: (usize, usize),
        size: (usize, usize),
    ) -> Result<ImageRectMut<'a, T>> {
        rect_size_check(origin, size, self.size)?;
        Ok(ImageRectMut {
            origin: (origin.0 + self.origin.0, origin.1 + self.origin.1),
            size,
            image: self.image,
        })
    }

    pub fn into_rect(
        self,
        origin: (usize, usize),
        size: (usize, usize),
    ) -> Result<ImageRectMut<'a, T>> {
        rect_size_check(origin, size, self.size)?;
        Ok(ImageRectMut {
            origin: (origin.0 + self.origin.0, origin.1 + self.origin.1),
            size,
            image: self.image,
        })
    }

    pub fn size(&self) -> (usize, usize) {
        self.size
    }

    #[instrument(skip_all)]
    pub fn copy_from(&mut self, other: ImageRect<'_, T>) -> Result<()> {
        if other.size != self.size {
            return Err(Error::CopyOfDifferentSize(
                other.size.0,
                other.size.1,
                self.size.0,
                self.size.1,
            ));
        }

        for i in 0..self.size.1 {
            trace!("copying row {i} of {}", self.size.1);
            self.row(i).copy_from_slice(other.row(i));
        }

        Ok(())
    }

    fn row_offset(&self, row: usize) -> usize {
        debug_assert!(row < self.size.1);
        (row + self.origin.1) * self.image.size.0 + self.origin.0
    }

    pub fn row(&mut self, row: usize) -> &mut [T] {
        debug_assert!(row < self.size.1);
        let start = self.row_offset(row);
        trace!(
            "{self:?} img size {:?} row {row} start {}",
            self.image.size,
            start
        );
        &mut self.image.data[start..start + self.size.0]
    }

    pub fn as_rect(&'a self) -> ImageRect<'a, T> {
        ImageRect {
            origin: self.origin,
            size: self.size,
            image: self.image,
        }
    }

    /// Applies `f` to all the pixels in this rect. As side information, `f` is passed the
    /// coordinates of the pixel in the full image.
    pub fn apply<F>(&mut self, mut f: F)
    where
        F: for<'b> FnMut((usize, usize), &'b mut T),
    {
        let origin = self.origin;
        (0..self.size.1).for_each(|x| {
            self.row(x)
                .iter_mut()
                .enumerate()
                .for_each(|(y, v)| f((origin.0 + x, origin.1 + y), v))
        });
    }
}

#[cfg(feature = "debug_tools")]
pub mod debug_tools {
    use super::{ImageDataType, ImageRect};
    pub trait ToU8ForWriting {
        fn to_u8_for_writing(self) -> u8;
    }

    impl ToU8ForWriting for u8 {
        fn to_u8_for_writing(self) -> u8 {
            self
        }
    }

    impl ToU8ForWriting for u16 {
        fn to_u8_for_writing(self) -> u8 {
            ((self as u32 * 0xff + 0x8000) / 0xffff) as u8
        }
    }

    impl ToU8ForWriting for f32 {
        fn to_u8_for_writing(self) -> u8 {
            (self * 255.0).clamp(0.0, 255.0).round() as u8
        }
    }

    impl ToU8ForWriting for u32 {
        fn to_u8_for_writing(self) -> u8 {
            ((self as u64 * 0xff + 0x80000000) / 0xffffffff) as u8
        }
    }

    impl ToU8ForWriting for half::f16 {
        fn to_u8_for_writing(self) -> u8 {
            self.to_f32().to_u8_for_writing()
        }
    }

    impl<'a, T: ImageDataType + ToU8ForWriting> ImageRect<'a, T> {
        pub fn to_pgm(&self) -> Vec<u8> {
            use std::io::Write;
            let mut ret = vec![];
            write!(&mut ret, "P5\n{} {}\n255\n", self.size.0, self.size.1).unwrap();
            ret.extend(
                (0..self.size.1)
                    .flat_map(|x| self.row(x).iter())
                    .map(|x| x.to_u8_for_writing()),
            );
            ret
        }
    }

    #[cfg(test)]
    mod test {
        use super::super::Image;
        use super::ToU8ForWriting;
        use crate::error::Result;

        #[test]
        fn to_pgm() -> Result<()> {
            let image = Image::<u8>::new((32, 32))?;
            assert!(image.as_rect().to_pgm().starts_with(b"P5\n32 32\n255\n"));
            Ok(())
        }

        #[test]
        fn u16_to_u8() {
            let mut left_source_u16 = 0xffffu16 / 510;
            for want_u8 in 0x00u8..0xffu8 {
                assert!(left_source_u16.to_u8_for_writing() == want_u8);
                assert!((left_source_u16 + 1).to_u8_for_writing() == want_u8 + 1);
                // Since we have 256 u8 values, but 0x00 and 0xff only have half the
                // range, we actually get whole ranges of size 0xffff / 255.
                left_source_u16 = left_source_u16.wrapping_add(0xffff / 255);
            }
        }

        #[test]
        fn f32_to_u8() {
            let epsilon = 1e-4f32;
            for want_u8 in 0x00u8..0xffu8 {
                let threshold = 1f32 / 510f32 + (1f32 / 255f32) * (want_u8 as f32);
                assert!((threshold - epsilon).to_u8_for_writing() == want_u8);
                assert!((threshold + epsilon).to_u8_for_writing() == want_u8 + 1);
            }
        }

        #[test]
        fn u32_to_u8() {
            let mut left_source_u32 = 0xffffffffu32 / 510;
            for want_u8 in 0x00u8..0xffu8 {
                assert!(left_source_u32.to_u8_for_writing() == want_u8);
                assert!((left_source_u32 + 1).to_u8_for_writing() == want_u8 + 1);
                // Since we have 256 u8 values, but 0x00 and 0xff only have half the
                // range, we actually get whole ranges of size 0xffffffff / 255.
                left_source_u32 = left_source_u32.wrapping_add(0xffffffffu32 / 255);
            }
        }

        #[test]
        fn f16_to_u8() {
            let epsilon = half::f16::from_f32(1e-3f32);
            for want_u8 in 0x00u8..0xffu8 {
                let threshold =
                    half::f16::from_f32(1f32 / 510f32 + (1f32 / 255f32) * (want_u8 as f32));
                assert!((threshold - epsilon).to_u8_for_writing() == want_u8);
                assert!((threshold + epsilon).to_u8_for_writing() == want_u8 + 1);
            }
        }
    }
}

#[cfg(test)]
mod test {
    use arbtest::arbitrary::Arbitrary;

    use crate::error::Result;

    use super::{Image, ImageDataType};

    #[test]
    fn huge_image() {
        assert!(Image::<u8>::new((1 << 28, 1 << 28)).is_err());
    }

    #[test]
    fn rect_basic() -> Result<()> {
        let mut image = Image::<u8>::new((32, 42))?;
        assert_eq!(image.as_rect_mut().rect((31, 40), (1, 1))?.size(), (1, 1));
        assert_eq!(image.as_rect_mut().rect((0, 0), (1, 1))?.size(), (1, 1));
        assert!(image.as_rect_mut().rect((30, 30), (3, 3)).is_err());
        image.as_rect_mut().rect((30, 30), (1, 1))?.row(0)[0] = 1;
        assert_eq!(image.as_rect_mut().row(30)[30], 1);
        Ok(())
    }

    fn f64_conversions<T: ImageDataType + Eq + for<'a> Arbitrary<'a>>() {
        arbtest::arbtest(|u| {
            let t = T::arbitrary(u)?;
            assert_eq!(t, T::from_f64(t.to_f64()));
            Ok(())
        });
    }

    #[test]
    fn u8_f64_conv() {
        f64_conversions::<u8>();
    }

    #[test]
    fn u16_f64_conv() {
        f64_conversions::<u16>();
    }

    #[test]
    fn u32_f64_conv() {
        f64_conversions::<u32>();
    }

    #[test]
    fn i8_f64_conv() {
        f64_conversions::<i8>();
    }

    #[test]
    fn i16_f64_conv() {
        f64_conversions::<i16>();
    }

    #[test]
    fn i32_f64_conv() {
        f64_conversions::<i32>();
    }

    #[test]
    fn f32_f64_conv() {
        arbtest::arbtest(|u| {
            let t = f32::arbitrary(u)?;
            if !t.is_nan() {
                assert_eq!(t, f32::from_f64(t.to_f64()));
            }
            Ok(())
        });
    }
}
