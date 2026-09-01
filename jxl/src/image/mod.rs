// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(unsafe_code)]

mod data_type;
mod internal;
mod output_buffer;
mod raw;
mod rect;
mod recycler;
#[cfg(test)]
mod test;
mod typed;

pub use data_type::{DataTypeTag, ImageDataType};
pub use output_buffer::JxlOutputBuffer;
pub use raw::{OwnedRawImage, RawImageRect, RawImageRectMut};
pub use recycler::{
    BUFFER_KIND_GROUP, BUFFER_KIND_LEFTRIGHT, BUFFER_KIND_TOPBOTTOM, BufferRecycler, KIND_GROUP,
    KIND_LEFTRIGHT, KIND_TOPBOTTOM, NUM_BUCKETS, NUM_GROUP_SHIFTS, NUM_KINDS, NUM_LEFTRIGHT_SHIFTS,
    NUM_TOPBOTTOM_SHIFTS, bucket_id, data_type_size_to_bucket, group_shift_to_bucket,
    leftright_shift_to_bucket, topbottom_shift_to_bucket,
};
pub use rect::Rect;
pub use typed::{Image, ImageRect, ImageRectMut};
