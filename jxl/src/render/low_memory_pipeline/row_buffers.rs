// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::any::Any;

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
    pub(super) next_row: usize,
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
        y_border: usize,
        input_y_shift: usize,
        chunk_size: usize,
    ) -> Result<Self> {
        let num_rows = ((y_border + 1).max(1 << input_y_shift) + y_border).shrc(input_y_shift)
            << input_y_shift;
        let row_len = chunk_size + 4 * (CACHE_LINE_BYTE_SIZE / data_type.size());
        let buffer: Box<dyn Any> = match data_type {
            DataTypeTag::U8 => make_buffer::<u8>(row_len * num_rows)?,
            DataTypeTag::I8 => make_buffer::<i8>(row_len * num_rows)?,
            DataTypeTag::U16 => make_buffer::<u16>(row_len * num_rows)?,
            DataTypeTag::F16 => make_buffer::<f16>(row_len * num_rows)?,
            DataTypeTag::I16 => make_buffer::<i16>(row_len * num_rows)?,
            DataTypeTag::U32 => make_buffer::<u32>(row_len * num_rows)?,
            DataTypeTag::F32 => make_buffer::<f32>(row_len * num_rows)?,
            DataTypeTag::I32 => make_buffer::<i32>(row_len * num_rows)?,
            DataTypeTag::F64 => make_buffer::<f64>(row_len * num_rows)?,
        };

        Ok(Self {
            buffer,
            row_len,
            next_row: 0,
        })
    }

    pub fn get_buf<T: ImageDataType>(&mut self) -> &mut [T] {
        &mut *self
            .buffer
            .downcast_mut::<Vec<T>>()
            .expect("called get_buf with the wrong buffer type")
    }
}
