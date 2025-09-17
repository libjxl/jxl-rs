// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::{any::Any, ops::Range};

use half::f16;

use crate::{
    error::Result,
    image::{DataTypeTag, ImageDataType},
    simd::CACHE_LINE_BYTE_SIZE,
    util::ShiftRightCeil,
};

pub struct RowBuffer {
    // TODO(veluca): consider making this into a Vec<u8> and using casts instead, if we want to get
    // rid of the double allocation & the typeid checking on access.
    buffer: Box<dyn Any>,
    pub(super) row_len: usize,
    pub(super) row_stride: usize,
    pub(super) next_row: usize,
    pub(super) row_range: Range<usize>,
}

fn make_buffer<T: Default + Clone + 'static>(len: usize) -> Result<Box<dyn Any>> {
    let mut vec = Vec::<T>::new();
    vec.try_reserve(len)?;
    vec.resize(len, Default::default());
    Ok(Box::new(vec))
}

impl RowBuffer {
    pub fn new(
        data_type: DataTypeTag,
        next_y_border: usize,
        y_shift: usize,
        row_len: usize,
    ) -> Result<Self> {
        // This is slightly wasteful (i.e. if y_shift = 2 and next_y_border = 1, it uses 4 more
        // rows than would be necessary), but certainly sufficient.
        let num_rows = (1 << y_shift) + 2 * (next_y_border.shrc(y_shift) << y_shift);
        let row_stride = row_len + 4 * (CACHE_LINE_BYTE_SIZE / data_type.size());
        let buffer: Box<dyn Any> = match data_type {
            DataTypeTag::U8 => make_buffer::<u8>(row_stride * num_rows)?,
            DataTypeTag::I8 => make_buffer::<i8>(row_stride * num_rows)?,
            DataTypeTag::U16 => make_buffer::<u16>(row_stride * num_rows)?,
            DataTypeTag::F16 => make_buffer::<f16>(row_stride * num_rows)?,
            DataTypeTag::I16 => make_buffer::<i16>(row_stride * num_rows)?,
            DataTypeTag::U32 => make_buffer::<u32>(row_stride * num_rows)?,
            DataTypeTag::F32 => make_buffer::<f32>(row_stride * num_rows)?,
            DataTypeTag::I32 => make_buffer::<i32>(row_stride * num_rows)?,
            DataTypeTag::F64 => make_buffer::<f64>(row_stride * num_rows)?,
        };

        Ok(Self {
            buffer,
            row_stride,
            row_len,
            next_row: 0,
            row_range: 0..0,
        })
    }

    pub fn get_buf<T: ImageDataType>(&mut self) -> &mut [T] {
        &mut *self
            .buffer
            .downcast_mut::<Vec<T>>()
            .expect("called get_buf with the wrong buffer type")
    }
}
