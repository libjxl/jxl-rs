// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::{fmt::Debug, marker::PhantomData};

use crate::{error::Result, simd::CACHE_LINE_BYTE_SIZE, util::tracing_wrappers::*};

use super::{ImageDataType, OwnedRawImage, RawImageRect, RawImageRectMut, Rect};

#[repr(transparent)]
pub struct Image<T: ImageDataType> {
    // Safety invariant: self.raw.data.is_aligned(T::DATA_TYPE_ID.size()) is true.
    raw: OwnedRawImage,
    _ph: PhantomData<T>,
}

impl<T: ImageDataType> Image<T> {
    #[instrument(ret, err)]
    pub fn new(size: (usize, usize)) -> Result<Image<T>> {
        Ok(Self::from_raw(OwnedRawImage::new_zeroed((
            size.0 * T::DATA_TYPE_ID.size(),
            size.1,
        ))?))
    }

    pub fn new_with_value(size: (usize, usize), value: T) -> Result<Image<T>> {
        // TODO(veluca): skip zero-initializing the allocation if this becomes
        // performance-sensitive.
        let mut ret = Self::new(size)?;
        ret.fill(value);
        Ok(ret)
    }

    pub fn size(&self) -> (usize, usize) {
        (
            self.raw.byte_size().0 / T::DATA_TYPE_ID.size(),
            self.raw.byte_size().1,
        )
    }

    pub fn fill(&mut self, v: T) {
        for y in 0..self.size().1 {
            self.as_rect_mut().row(y).fill(v);
        }
    }

    pub fn as_rect(&self) -> ImageRect<'_, T> {
        ImageRect::from_raw(self.raw.as_rect())
    }

    pub fn as_rect_mut(&mut self) -> ImageRectMut<'_, T> {
        ImageRectMut::from_raw(self.raw.as_rect_mut())
    }

    pub fn try_clone(&self) -> Result<Self> {
        Ok(Self::from_raw(self.raw.try_clone()?))
    }

    pub fn into_raw(self) -> OwnedRawImage {
        self.raw
    }

    pub fn from_raw(raw: OwnedRawImage) -> Self {
        const { assert!(CACHE_LINE_BYTE_SIZE.is_multiple_of(T::DATA_TYPE_ID.size())) };
        assert!(raw.data.is_aligned(T::DATA_TYPE_ID.size()));
        Image {
            // Safety note: we just checked alignment.
            raw,
            _ph: PhantomData,
        }
    }
}

#[derive(Clone, Copy)]
pub struct ImageRect<'a, T: ImageDataType> {
    // Safety invariant: self.raw.data.is_aligned(T::DATA_TYPE_ID.size()) is true.
    raw: RawImageRect<'a>,
    _ph: PhantomData<T>,
}

impl<'a, T: ImageDataType> ImageRect<'a, T> {
    pub fn rect(&self, rect: Rect) -> Result<ImageRect<'a, T>> {
        rect.is_within(self.size())?;
        Ok(Self::from_raw(
            self.raw.rect(rect.to_byte_rect(T::DATA_TYPE_ID)),
        ))
    }

    pub fn size(&self) -> (usize, usize) {
        (
            self.raw.byte_size().0 / T::DATA_TYPE_ID.size(),
            self.raw.byte_size().1,
        )
    }

    pub fn row(&self, row: usize) -> &'a [T] {
        let row = self.raw.row(row);
        // SAFETY: Since self.raw.data.is_aligned(T::DATA_TYPE_ID.size()), the returned slice is
        // aligned to T::DATA_TYPE_ID.size(), and sizeof(T) == T::DATA_TYPE_ID.size()
        // by the requirements of ImageDataType; moreover ImageDataType requires T to be a
        // bag-of-bits type with no padding, so the implicit transmute is not an issue.
        unsafe {
            std::slice::from_raw_parts(row.as_ptr() as *const T, row.len() / T::DATA_TYPE_ID.size())
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = T> + '_ {
        (0..self.size().1).flat_map(|x| self.row(x).iter().cloned())
    }

    pub fn into_raw(self) -> RawImageRect<'a> {
        self.raw
    }

    pub fn from_raw(raw: RawImageRect<'a>) -> Self {
        const { assert!(CACHE_LINE_BYTE_SIZE.is_multiple_of(T::DATA_TYPE_ID.size())) };
        assert!(raw.data.is_aligned(T::DATA_TYPE_ID.size()));
        ImageRect {
            // Safety note: we just checked alignment.
            raw,
            _ph: PhantomData,
        }
    }
}

pub struct ImageRectMut<'a, T: ImageDataType> {
    // Safety invariant: self.raw.data.is_aligned(T::DATA_TYPE_ID.size()) is true.
    raw: RawImageRectMut<'a>,
    _ph: PhantomData<T>,
}

impl<'a, T: ImageDataType> ImageRectMut<'a, T> {
    pub fn rect(&'a mut self, rect: Rect) -> Result<ImageRectMut<'a, T>> {
        rect.is_within(self.size())?;
        Ok(Self::from_raw(
            self.raw.rect_mut(rect.to_byte_rect(T::DATA_TYPE_ID)),
        ))
    }

    pub fn size(&self) -> (usize, usize) {
        (
            self.raw.byte_size().0 / T::DATA_TYPE_ID.size(),
            self.raw.byte_size().1,
        )
    }

    pub fn row(&mut self, row: usize) -> &mut [T] {
        let row = self.raw.row(row);
        // SAFETY: Since self.raw.data.is_aligned(T::DATA_TYPE_ID.size()), the returned slice is
        // aligned to T::DATA_TYPE_ID.size(), and sizeof(T) == T::DATA_TYPE_ID.size()
        // by the requirements of ImageDataType; moreover ImageDataType requires T to be a
        // bag-of-bits type with no padding, so the implicit transmute is not an issue.
        unsafe {
            std::slice::from_raw_parts_mut(
                row.as_mut_ptr() as *mut T,
                row.len() / T::DATA_TYPE_ID.size(),
            )
        }
    }

    pub fn as_rect(&'a self) -> ImageRect<'a, T> {
        ImageRect::from_raw(self.raw.as_rect())
    }

    pub fn into_raw(self) -> RawImageRectMut<'a> {
        self.raw
    }

    pub fn from_raw(raw: RawImageRectMut<'a>) -> Self {
        const { assert!(CACHE_LINE_BYTE_SIZE.is_multiple_of(T::DATA_TYPE_ID.size())) };
        assert!(raw.data.is_aligned(T::DATA_TYPE_ID.size()));
        ImageRectMut {
            // Safety note: we just checked alignment.
            raw,
            _ph: PhantomData,
        }
    }
}

impl<T: ImageDataType> Debug for Image<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:?} {}x{}",
            T::DATA_TYPE_ID,
            self.size().0,
            self.size().1
        )
    }
}

impl<T: ImageDataType> Debug for ImageRect<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:?} rect {}x{}",
            T::DATA_TYPE_ID,
            self.size().0,
            self.size().1
        )
    }
}

impl<T: ImageDataType> Debug for ImageRectMut<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:?} mutrect {}x{}",
            T::DATA_TYPE_ID,
            self.size().0,
            self.size().1
        )
    }
}
