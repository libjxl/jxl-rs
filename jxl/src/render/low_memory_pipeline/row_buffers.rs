// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::{any::Any, ops::Range};

use half::f16;

use crate::{
    error::Result,
    image::{DataTypeTag, ImageDataType},
    simd::{CACHE_LINE_BYTE_SIZE, num_per_cache_line},
    util::ShiftRightCeil,
};

/// Temporary storage for data rows. Note that the first pixel of the group is expected to be
/// located *two cachelines worth of data* inside the row.
pub struct RowBuffer {
    // TODO(veluca): consider making this into a Vec<u8> and using casts instead, if we want to get
    // rid of the double allocation & the typeid checking on access.
    buffer: Box<dyn Any>,
    // Distance (in number of elements) between the start of two rows.
    row_stride: usize,
    // Number of rows that are actually stored.
    // TODO(veluca): consider padding this to a power of 2 and using & here. In *most* cases,
    // that's not a huge loss in memory usage (for most images, num_rows is 1/3/5/7, which would
    // become 1/4/8/8).
    num_rows: usize,
}

fn make_buffer<T: Default + Clone + 'static>(len: usize) -> Result<Box<dyn Any>> {
    // TODO(veluca): allocate this aligned to a cache line.
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
        // Input offset is at *two* cachelines, and we need up to *three* cachelines on the other
        // side as the data might exceed xsize slightly.
        let row_stride = row_len + 5 * (CACHE_LINE_BYTE_SIZE / data_type.size());
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
            num_rows,
        })
    }

    pub fn get_buf_mut<T: ImageDataType>(&mut self) -> &mut [T] {
        &mut *self
            .buffer
            .downcast_mut::<Vec<T>>()
            .expect("called get_buf with the wrong buffer type")
    }

    pub fn get_buf<T: ImageDataType>(&self) -> &[T] {
        self.buffer
            .downcast_ref::<Vec<T>>()
            .expect("called get_buf with the wrong buffer type")
    }

    pub fn get_row<T: ImageDataType>(&self, row: usize) -> &[T] {
        let row_idx = row % self.num_rows;
        let start = row_idx * self.row_stride;
        &self.get_buf()[start..start + self.row_stride]
    }

    pub fn get_row_mut<T: ImageDataType>(&mut self, row: usize) -> &mut [T] {
        let row_idx = row % self.num_rows;
        let stride = self.row_stride;
        let start = row_idx * stride;
        &mut self.get_buf_mut()[start..start + stride]
    }

    // TODO(veluca): use some kind of smallvec.
    pub fn get_rows_mut<T: ImageDataType>(
        &mut self,
        y: Range<usize>,
        xoffset: usize,
    ) -> Vec<&mut [T]> {
        assert!(y.clone().count() <= self.num_rows);
        let first_row_idx = y.start % self.num_rows;
        let stride = self.row_stride;
        let start = first_row_idx * stride;
        let num_pre = (y.clone().count() + first_row_idx).saturating_sub(self.num_rows);
        let num_post = y.clone().count() - num_pre;
        let buf = self.get_buf_mut::<T>();
        let (pre, post) = buf.split_at_mut(start);
        let pre_rows = pre.chunks_exact_mut(stride).take(num_pre);
        let post_rows = post.chunks_exact_mut(stride).take(num_post);
        post_rows
            .chain(pre_rows)
            .map(|x| &mut x[xoffset..])
            .collect()
    }

    pub const fn x0_offset<T: ImageDataType>() -> usize {
        2 * num_per_cache_line::<T>()
    }
}
