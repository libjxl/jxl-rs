// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::cell::RefMut;

use crate::{error::Result, image::Image};

use super::ModularBufferInfo;

// Auxiliary struct to aid extracting mutable references to specific buffers.
#[derive(Debug)]
pub struct MutablyBorrowedModularBuffers<'a> {
    pub bufs: Vec<RefMut<'a, Image<i32>>>,
    pub channel_ids: Vec<usize>,
}

impl<'a> MutablyBorrowedModularBuffers<'a> {
    pub fn new(buffers: &'a [ModularBufferInfo], indices: &[usize], grid: usize) -> Result<Self> {
        let mut bufs = vec![];
        let mut channel_ids = vec![];
        for i in indices {
            // Allocate buffers if they are not present.
            let buf = &buffers[*i];
            channel_ids.push(buf.channel_id);
            let b = &buf.buffer_grid[grid];
            let mut data = b.data.borrow_mut();
            if data.is_none() {
                *data = Some(Image::new(b.size)?)
            }
            bufs.push(RefMut::map(data, |x| x.as_mut().unwrap()));
        }
        Ok(MutablyBorrowedModularBuffers { bufs, channel_ids })
    }
}
