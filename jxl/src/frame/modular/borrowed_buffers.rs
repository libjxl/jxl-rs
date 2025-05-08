// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::{cell::RefMut, ops::DerefMut};

use crate::{error::Result, image::Image};

use super::{ModularBufferInfo, ModularChannel};

pub fn with_buffers<T>(
    buffers: &[ModularBufferInfo],
    indices: &[usize],
    grid: usize,
    f: impl FnOnce(Vec<&mut ModularChannel>) -> Result<T>,
) -> Result<T> {
    let mut bufs = vec![];
    for i in indices {
        // Allocate buffers if they are not present.
        let buf = &buffers[*i];
        let b = &buf.buffer_grid[grid];
        let mut data = b.data.borrow_mut();
        if data.is_none() {
            *data = Some(ModularChannel {
                data: Image::new(b.size)?,
                auxiliary_data: None,
                shift: buf.info.shift,
                bit_depth: buf.info.bit_depth,
            });
        }
        bufs.push(RefMut::map(data, |x| x.as_mut().unwrap()));
    }
    f(bufs.iter_mut().map(|x| x.deref_mut()).collect())
}
