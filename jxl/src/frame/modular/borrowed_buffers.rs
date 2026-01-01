// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::ops::DerefMut;

use crate::{
    error::Result,
    frame::modular::{IMAGE_OFFSET, IMAGE_PADDING},
    image::Image,
    util::AtomicRefMut,
};

use super::{ModularBufferInfo, ModularChannel, ModularChannelI16};

pub fn with_buffers<T>(
    buffers: &[ModularBufferInfo],
    indices: &[usize],
    grid: usize,
    skip_empty: bool,
    f: impl FnOnce(Vec<&mut ModularChannel>) -> Result<T>,
) -> Result<T> {
    let mut bufs = vec![];
    for i in indices {
        // Allocate buffers if they are not present.
        let buf = &buffers[*i];
        let b = &buf.buffer_grid[grid];

        // Ensure data is in i32 format (may have been decoded as i16)
        b.ensure_i32()?;

        let mut data = b.data.borrow_mut();
        if data.is_none() {
            *data = Some(ModularChannel {
                data: Image::new_with_padding(b.size, IMAGE_OFFSET, IMAGE_PADDING)?,
                auxiliary_data: None,
                shift: buf.info.shift,
                bit_depth: buf.info.bit_depth,
            });
        }

        // Skip zero-sized buffers when decoding - they don't contribute to the bitstream.
        // This matches libjxl's behavior in DecodeGroup where zero-sized rects are skipped.
        // The buffer is still allocated above so transforms can access it.
        if skip_empty && (b.size.0 == 0 || b.size.1 == 0) {
            continue;
        }

        bufs.push(AtomicRefMut::map(data, |x| x.as_mut().unwrap()));
    }
    f(bufs.iter_mut().map(|x| x.deref_mut()).collect())
}

/// Check if i16 decoding can be used for the given buffers.
/// i16 is appropriate when all channels have bit_depth <= 16.
pub fn can_use_i16(buffers: &[ModularBufferInfo], indices: &[usize], grid: usize) -> bool {
    for i in indices {
        let buf = &buffers[*i];
        let b = &buf.buffer_grid[grid];
        // Skip empty buffers
        if b.size.0 == 0 || b.size.1 == 0 {
            continue;
        }
        // Check bit depth - must fit in i16 (we use signed, so max 15 bits for unsigned values)
        // For signed values, 16 bits is fine. We're conservative and allow up to 16 bits.
        if buf.info.bit_depth.bits_per_sample() > 16 {
            return false;
        }
    }
    true
}

/// Like with_buffers, but allocates i16 buffers for decoding.
/// Should only be called when can_use_i16 returns true.
pub fn with_buffers_i16<T>(
    buffers: &[ModularBufferInfo],
    indices: &[usize],
    grid: usize,
    skip_empty: bool,
    f: impl FnOnce(Vec<&mut ModularChannelI16>) -> Result<T>,
) -> Result<T> {
    let mut bufs = vec![];
    for i in indices {
        let buf = &buffers[*i];
        let b = &buf.buffer_grid[grid];

        let mut data = b.data_i16.borrow_mut();
        if data.is_none() {
            *data = Some(ModularChannelI16 {
                data: Image::new_with_padding(b.size, IMAGE_OFFSET, IMAGE_PADDING)?,
                auxiliary_data: None,
                shift: buf.info.shift,
                bit_depth: buf.info.bit_depth,
            });
        }

        // Skip zero-sized buffers when decoding
        if skip_empty && (b.size.0 == 0 || b.size.1 == 0) {
            continue;
        }

        bufs.push(AtomicRefMut::map(data, |x| x.as_mut().unwrap()));
    }
    f(bufs.iter_mut().map(|x| x.deref_mut()).collect())
}
