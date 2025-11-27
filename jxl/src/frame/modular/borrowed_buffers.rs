// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::{cell::RefMut, collections::HashSet, ops::DerefMut};

use crate::{
    error::{Error, Result},
    frame::modular::{IMAGE_OFFSET, IMAGE_PADDING},
    image::Image,
};

use super::{ModularBufferInfo, ModularChannel};

pub fn with_buffers<T>(
    buffers: &[ModularBufferInfo],
    indices: &[usize],
    grid: usize,
    f: impl FnOnce(Vec<&mut ModularChannel>) -> Result<T>,
) -> Result<T> {
    let mut bufs = vec![];
    let mut check_dups = HashSet::new();
    for i in indices {
        if !check_dups.insert(*i) {
            return Err(Error::DuplicateChannelIndex(*i));
        }
        // Allocate buffers if they are not present.
        let buf = &buffers[*i];
        let b = &buf.buffer_grid[grid];
        let mut data = b.data.borrow_mut();
        if data.is_none() {
            *data = Some(ModularChannel {
                data: Image::new_with_padding(b.size, IMAGE_OFFSET, IMAGE_PADDING)?,
                auxiliary_data: None,
                shift: buf.info.shift,
                bit_depth: buf.info.bit_depth,
            });
        }

        bufs.push(RefMut::map(data, |x| x.as_mut().unwrap()));
    }
    f(bufs.iter_mut().map(|x| x.deref_mut()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;
    use crate::frame::modular::{ChannelInfo, ModularBuffer, ModularGridKind};
    use crate::headers::bit_depth::BitDepth;
    use std::cell::RefCell;

    #[test]
    fn test_with_buffers_duplicate_indices_errors() {
        let mut buffers = vec![];
        let info = ChannelInfo {
            output_channel_idx: -1,
            size: (1, 1),
            shift: Some((0, 0)),
            bit_depth: BitDepth::integer_samples(8),
        };
        let buffer_info = ModularBufferInfo {
            info,
            coded_channel_id: 0,
            description: "".to_string(),
            grid_kind: ModularGridKind::None,
            grid_shape: (1, 1),
            buffer_grid: vec![ModularBuffer {
                size: (1, 1),
                data: RefCell::new(None),
                remaining_uses: 2,
                used_by_transforms: vec![],
            }],
        };
        buffers.push(buffer_info);

        let indices = vec![0, 0];
        let result = with_buffers(&buffers, &indices, 0, |_| Ok(()));
        assert!(matches!(result, Err(Error::DuplicateChannelIndex(0))));
    }
}
