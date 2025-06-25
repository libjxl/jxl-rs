// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::fmt::Debug;

use crate::{
    error::{Error, Result},
    util::{ShiftRightCeil, tracing_wrappers::*},
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

#[cfg(test)]
macro_rules! type_min {
    (f32) => {
        0.0f32
    };
    ($ty: ty) => {
        <$ty>::MIN
    };
}

#[cfg(test)]
macro_rules! type_max {
    (f32) => {
        1.0f32
    };
    ($ty: ty) => {
        <$ty>::MAX
    };
}

macro_rules! impl_image_data_type {
    ($ty: ident, $id: ident) => {
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
                let min = type_min!($ty);
                let max = type_max!($ty);
                Uniform::new_inclusive(min, max).sample(rng)
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
        Self::from_f64(Uniform::new(0.0f32, 1.0f32).sample(rng) as f64)
    }
}

// Meant to be used by the simple render pipeline and in general for
// testing purposes.
impl_image_data_type!(f64, F64);

#[derive(Clone)]
pub struct Image<T: ImageDataType> {
    // width, height
    size: (usize, usize),
    data: Vec<T>,
}

impl<T: ImageDataType> Debug for Image<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?} {}x{}", T::DATA_TYPE_ID, self.size.0, self.size.1,)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Rect {
    pub origin: (usize, usize),
    // width, height
    pub size: (usize, usize),
}

impl Rect {
    pub fn is_within(&self, size: (usize, usize)) -> Result<()> {
        if self
            .origin
            .0
            .checked_add(self.size.0)
            .ok_or(Error::ArithmeticOverflow)?
            > size.0
            || self
                .origin
                .1
                .checked_add(self.size.1)
                .ok_or(Error::ArithmeticOverflow)?
                > size.1
        {
            Err(Error::RectOutOfBounds(
                self.size.0,
                self.size.1,
                self.origin.0,
                self.origin.1,
                size.0,
                size.1,
            ))
        } else {
            Ok(())
        }
    }
}

#[derive(Clone, Copy)]
pub struct ImageRect<'a, T: ImageDataType> {
    pub rect: Rect,
    image: &'a Image<T>,
}

pub struct ImageRectMut<'a, T: ImageDataType> {
    rect: Rect,
    image: &'a mut Image<T>,
}

impl<T: ImageDataType> Debug for ImageRect<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:?} {}x{}+{}+{}",
            T::DATA_TYPE_ID,
            self.rect.size.0,
            self.rect.size.1,
            self.rect.origin.0,
            self.rect.origin.1
        )
    }
}

impl<T: ImageDataType> Debug for ImageRectMut<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "mut {:?} {}x{}+{}+{}",
            T::DATA_TYPE_ID,
            self.rect.size.0,
            self.rect.size.1,
            self.rect.origin.0,
            self.rect.origin.1
        )
    }
}

impl<T: ImageDataType> Image<T> {
    #[instrument(err)]
    pub fn new_with_default(size: (usize, usize), default: T) -> Result<Image<T>> {
        let (xsize, ysize) = size;
        // These limits let us not worry about overflows.
        if xsize as u64 >= i64::MAX as u64 / 4 || ysize as u64 >= i64::MAX as u64 / 4 {
            return Err(Error::ImageSizeTooLarge(xsize, ysize));
        }
        let total_size = xsize
            .checked_mul(ysize)
            .ok_or(Error::ImageSizeTooLarge(xsize, ysize))?;
        // To simplify modular transform logic, we allow empty images, because some modular
        // meta-images can have 0 xsize or ysize (e.g. delta-palette, reference property image).
        debug!("trying to allocate image");
        let mut data = vec![];
        data.try_reserve_exact(total_size)?;
        data.resize(total_size, default);
        Ok(Image {
            size: (xsize, ysize),
            data,
        })
    }

    pub fn new(size: (usize, usize)) -> Result<Image<T>> {
        Self::new_with_default(size, T::default())
    }

    #[cfg(test)]
    pub fn new_with_data(size: (usize, usize), data: Vec<T>) -> Image<T> {
        Image { size, data }
    }

    #[cfg(test)]
    pub fn new_random<R: rand::Rng>(size: (usize, usize), rng: &mut R) -> Result<Image<T>> {
        let mut img = Self::new(size)?;
        img.data.iter_mut().for_each(|x| *x = T::random(rng));
        Ok(img)
    }

    #[cfg(test)]
    pub fn new_range(size: (usize, usize), start: f32, step: f32) -> Result<Image<T>> {
        let mut img = Self::new(size)?;
        img.data
            .iter_mut()
            .enumerate()
            .for_each(|(index, x)| *x = T::from_f64((start + step * index as f32) as f64));
        Ok(img)
    }

    #[cfg(test)]
    pub fn new_constant(size: (usize, usize), val: T) -> Result<Image<T>> {
        let mut img = Self::new(size)?;
        img.data.iter_mut().for_each(|x| *x = val);
        Ok(img)
    }

    pub fn size(&self) -> (usize, usize) {
        self.size
    }

    pub fn fill(&mut self, v: T) {
        let size = self.size;
        let mut rect = self.as_rect_mut();
        for y in 0..size.1 {
            for x in 0..size.0 {
                rect.row(y)[x] = v;
            }
        }
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
        self.as_rect().rect(Rect { origin, size }).unwrap()
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
        self.as_rect_mut().into_rect(Rect { origin, size }).unwrap()
    }

    pub fn as_rect(&self) -> ImageRect<'_, T> {
        ImageRect {
            rect: Rect {
                origin: (0, 0),
                size: self.size,
            },
            image: self,
        }
    }

    pub fn as_rect_mut(&mut self) -> ImageRectMut<'_, T> {
        ImageRectMut {
            rect: Rect {
                origin: (0, 0),
                size: self.size,
            },
            image: self,
        }
    }

    pub fn try_clone(&self) -> Result<Self> {
        self.as_rect().to_image()
    }
}

impl<'a, T: ImageDataType> ImageRect<'a, T> {
    pub fn rect(self, rect: Rect) -> Result<ImageRect<'a, T>> {
        rect.is_within(self.rect.size)?;
        Ok(ImageRect {
            rect: Rect {
                origin: (
                    rect.origin.0 + self.rect.origin.0,
                    rect.origin.1 + self.rect.origin.1,
                ),
                size: rect.size,
            },
            image: self.image,
        })
    }

    pub fn size(&self) -> (usize, usize) {
        self.rect.size
    }

    pub fn row(&self, row: usize) -> &'a [T] {
        debug_assert!(row < self.rect.size.1);
        let start = (row + self.rect.origin.1) * self.image.size.0 + self.rect.origin.0;
        trace!(
            "{self:?} img size {:?} rect size {:?} row {row} start {}",
            self.image.size, self.rect.size, start
        );
        &self.image.data[start..start + self.rect.size.0]
    }

    pub fn to_image(&self) -> Result<Image<T>> {
        let total_size = self.rect.size.0 * self.rect.size.1;
        let mut data = vec![];
        data.try_reserve_exact(total_size)?;
        data.extend((0..self.rect.size.1).flat_map(|x| self.row(x).iter()));
        Ok(Image {
            size: self.rect.size,
            data,
        })
    }

    pub fn iter(&self) -> impl Iterator<Item = T> + '_ {
        (0..self.rect.size.1).flat_map(|x| self.row(x).iter().cloned())
    }

    #[cfg(test)]
    pub fn check_equal(&self, other: ImageRect<T>) {
        assert_eq!(self.rect.size, other.rect.size);
        let mismatch_info = |x: usize, y: usize| -> String {
            let mut msg = format!(
                "mismatch at position {x}x{y}, values {:?} and {:?}",
                self.row(y)[x],
                other.row(y)[x]
            );
            if self.rect.origin != (0, 0) {
                msg = format!(
                    "{}; position in ground truth {}x{}",
                    msg,
                    x + self.rect.origin.0,
                    y + self.rect.origin.1
                );
            }
            if other.rect.origin != (0, 0) {
                msg = format!(
                    "{}; position in checked img {}x{}",
                    msg,
                    x + other.rect.origin.0,
                    y + other.rect.origin.1
                );
            }
            msg
        };
        for y in 0..self.rect.size.1 {
            for x in 0..self.rect.size.0 {
                assert_eq!(self.row(y)[x], other.row(y)[x], "{}", mismatch_info(x, y));
            }
        }
    }
}

impl<'a, T: ImageDataType> PartialEq<ImageRect<'a, T>> for ImageRect<'a, T> {
    fn eq(&self, other: &ImageRect<'a, T>) -> bool {
        self.iter().zip(other.iter()).all(|(x, y)| x == y)
    }
}

impl<T: ImageDataType + Eq> Eq for ImageRect<'_, T> {}

impl<'a, T: ImageDataType> ImageRectMut<'a, T> {
    pub fn rect(&'a mut self, rect: Rect) -> Result<ImageRectMut<'a, T>> {
        rect.is_within(self.rect.size)?;
        Ok(ImageRectMut {
            rect: Rect {
                origin: (
                    rect.origin.0 + self.rect.origin.0,
                    rect.origin.1 + self.rect.origin.1,
                ),
                size: rect.size,
            },
            image: self.image,
        })
    }

    pub fn into_rect(self, rect: Rect) -> Result<ImageRectMut<'a, T>> {
        rect.is_within(self.rect.size)?;
        Ok(ImageRectMut {
            rect: Rect {
                origin: (
                    rect.origin.0 + self.rect.origin.0,
                    rect.origin.1 + self.rect.origin.1,
                ),
                size: rect.size,
            },
            image: self.image,
        })
    }

    pub fn size(&self) -> (usize, usize) {
        self.rect.size
    }

    #[instrument(skip_all)]
    pub fn copy_from(&mut self, other: ImageRect<'_, T>) -> Result<()> {
        if other.rect.size != self.rect.size {
            return Err(Error::CopyOfDifferentSize(
                other.rect.size.0,
                other.rect.size.1,
                self.rect.size.0,
                self.rect.size.1,
            ));
        }

        for i in 0..self.rect.size.1 {
            trace!("copying row {i} of {}", self.rect.size.1);
            self.row(i).copy_from_slice(other.row(i));
        }

        Ok(())
    }

    fn row_offset(&self, row: usize) -> usize {
        debug_assert!(row < self.rect.size.1);
        (row + self.rect.origin.1) * self.image.size.0 + self.rect.origin.0
    }

    pub fn row(&mut self, row: usize) -> &mut [T] {
        debug_assert!(row < self.rect.size.1);
        let start = self.row_offset(row);
        trace!(
            "{self:?} img size {:?} row {row} start {}",
            self.image.size, start
        );
        &mut self.image.data[start..start + self.rect.size.0]
    }

    pub fn as_rect(&'a self) -> ImageRect<'a, T> {
        ImageRect {
            rect: self.rect,
            image: self.image,
        }
    }

    /// Applies `f` to all the pixels in this rect. As side information, `f` is passed the
    /// coordinates of the pixel in the full image.
    pub fn apply<F>(&mut self, mut f: F)
    where
        F: for<'b> FnMut((usize, usize), &'b mut T),
    {
        let origin = self.rect.origin;
        (0..self.rect.size.1).for_each(|x| {
            self.row(x)
                .iter_mut()
                .enumerate()
                .for_each(|(y, v)| f((origin.0 + x, origin.1 + y), v))
        });
    }
}

#[cfg(test)]
mod test {
    use arbtest::arbitrary::Arbitrary;

    use crate::error::Result;

    use super::{Image, ImageDataType, Rect};

    #[test]
    fn huge_image() {
        assert!(Image::<u8>::new((1 << 28, 1 << 28)).is_err());
    }

    #[test]
    fn rect_basic() -> Result<()> {
        let mut image = Image::<u8>::new((32, 42))?;
        assert_eq!(
            image
                .as_rect_mut()
                .rect(Rect {
                    origin: (31, 40),
                    size: (1, 1)
                })?
                .size(),
            (1, 1)
        );
        assert_eq!(
            image
                .as_rect_mut()
                .rect(Rect {
                    origin: (0, 0),
                    size: (1, 1)
                })?
                .size(),
            (1, 1)
        );
        assert!(
            image
                .as_rect_mut()
                .rect(Rect {
                    origin: (30, 30),
                    size: (3, 3)
                })
                .is_err()
        );
        image
            .as_rect_mut()
            .rect(Rect {
                origin: (30, 30),
                size: (1, 1),
            })?
            .row(0)[0] = 1;
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
