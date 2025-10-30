// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::ops::Range;

use crate::{
    error::Result,
    image::{DataTypeTag, ImageDataType},
    util::{
        CACHE_LINE_BYTE_SIZE, CacheLine, num_per_cache_line, slice_from_cachelines,
        slice_from_cachelines_mut,
    },
};

/// Temporary storage for data rows. Note that the first pixel of the group is expected to be
/// located *two cachelines worth of data* inside the row.
pub struct RowBuffer {
    buffer: Box<[CacheLine]>,
    // Distance (in number of *cache lines*) between the start of two rows.
    row_stride: usize,
    // Number of rows that are actually stored.
    // TODO(veluca): consider padding this to a power of 2 and using & here. In *most* cases,
    // that's not a huge loss in memory usage (for most images, num_rows is 1/3/5/7, which would
    // become 1/4/8/8).
    num_rows: usize,
}

impl RowBuffer {
    pub fn new(
        data_type: DataTypeTag,
        next_y_border: usize,
        y_shift: usize,
        row_len: usize,
    ) -> Result<Self> {
        let num_rows = (1 << y_shift) + 2 * next_y_border;
        // Input offset is at *two* cachelines, and we need up to *three* cachelines on the other
        // side as the data might exceed xsize slightly.
        let row_stride = (row_len * data_type.size()).div_ceil(CACHE_LINE_BYTE_SIZE) + 5;
        let mut buffer = Vec::<CacheLine>::new();
        buffer.try_reserve_exact(row_stride * num_rows)?;
        buffer.resize(row_stride * num_rows, CacheLine::default());
        let buffer = buffer.into_boxed_slice();
        Ok(Self {
            buffer,
            row_stride,
            num_rows,
        })
    }

    pub fn get_row<T: ImageDataType>(&self, row: usize) -> &[T] {
        let row_idx = row % self.num_rows;
        let start = row_idx * self.row_stride;
        slice_from_cachelines(&self.buffer[start..start + self.row_stride])
    }

    pub fn get_row_mut<T: ImageDataType>(&mut self, row: usize) -> &mut [T] {
        let row_idx = row % self.num_rows;
        let stride = self.row_stride;
        let start = row_idx * stride;
        slice_from_cachelines_mut(&mut self.buffer[start..start + stride])
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
        let buf = &mut self.buffer[..];
        let (pre, post) = buf.split_at_mut(start);
        let pre_rows = pre.chunks_exact_mut(stride).take(num_pre);
        let post_rows = post.chunks_exact_mut(stride).take(num_post);
        post_rows
            .chain(pre_rows)
            .map(|x| &mut slice_from_cachelines_mut(x)[xoffset..])
            .collect()
    }

    pub const fn x0_offset<T: ImageDataType>() -> usize {
        2 * num_per_cache_line::<T>()
    }

    pub const fn x0_byte_offset() -> usize {
        2 * CACHE_LINE_BYTE_SIZE
    }
}
