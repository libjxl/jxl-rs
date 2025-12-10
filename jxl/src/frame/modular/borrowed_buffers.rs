// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    error::Result,
    frame::modular::{IMAGE_OFFSET, IMAGE_PADDING},
    image::Image,
};

use super::{ModularBufferInfo, ModularChannel};

pub struct CollectResult<'buf> {
    pub immutable_grids: Vec<Vec<&'buf ModularChannel>>,
    pub mutable_grids: Vec<Vec<&'buf mut ModularChannel>>,
}

pub fn collect_buffers<'buf>(
    buffers: &'buf mut [ModularBufferInfo],
    indices_and_grids: &[(usize, &[usize])],
    indices_and_grids_mut: &[(usize, &[usize])],
    skip_empty: bool,
) -> Result<CollectResult<'buf>> {
    let mut buffer_refs: Vec<_> = buffers.iter_mut().map(Some).collect();

    let bufs = indices_and_grids
        .iter()
        .map(|&(idx, grids)| -> Result<_> {
            let buf = buffer_refs[idx].take().unwrap();
            let info = &buf.info;

            let mut collected = Vec::new();
            for &grid in grids {
                let b = &mut buf.buffer_grid[grid];
                if b.data.is_none() {
                    b.data = Some(ModularChannel {
                        data: Image::new_with_padding(b.size, IMAGE_OFFSET, IMAGE_PADDING)?,
                        auxiliary_data: None,
                        shift: info.shift,
                        bit_depth: info.bit_depth,
                    });
                }
            }

            for &grid in grids {
                let b = &buf.buffer_grid[grid];

                // Skip zero-sized buffers when decoding - they don't contribute to the bitstream.
                // This matches libjxl's behavior in DecodeGroup where zero-sized rects are skipped.
                // The buffer is still allocated above so transforms can access it.
                if skip_empty && (b.size.0 == 0 || b.size.1 == 0) {
                    continue;
                }

                collected.push(b.data.as_ref().unwrap());
            }

            Ok(collected)
        })
        .collect::<Result<Vec<_>>>()?;

    let bufs_mut = indices_and_grids_mut
        .iter()
        .map(|&(idx, grids)| -> Result<_> {
            let buf = buffer_refs[idx].take().unwrap();
            let info = &buf.info;
            let mut grid_refs: Vec<_> = buf.buffer_grid.iter_mut().map(Some).collect();

            let mut collected = Vec::new();
            for &grid in grids {
                let b = grid_refs[grid].take().unwrap();
                let data = &mut b.data;

                if data.is_none() {
                    *data = Some(ModularChannel {
                        data: Image::new_with_padding(b.size, IMAGE_OFFSET, IMAGE_PADDING)?,
                        auxiliary_data: None,
                        shift: info.shift,
                        bit_depth: info.bit_depth,
                    });
                }

                // Skip zero-sized buffers when decoding - they don't contribute to the bitstream.
                // This matches libjxl's behavior in DecodeGroup where zero-sized rects are skipped.
                // The buffer is still allocated above so transforms can access it.
                if skip_empty && (b.size.0 == 0 || b.size.1 == 0) {
                    continue;
                }

                collected.push(data.as_mut().unwrap());
            }

            Ok(collected)
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(CollectResult {
        immutable_grids: bufs,
        mutable_grids: bufs_mut,
    })
}

pub fn with_buffers<T>(
    buffers: &mut [ModularBufferInfo],
    indices: &[usize],
    grid: usize,
    skip_empty: bool,
    f: impl FnOnce(Vec<&mut ModularChannel>) -> Result<T>,
) -> Result<T> {
    let grid = [grid];
    let grid = grid.as_ref();
    let indices_and_grids: Vec<_> = indices.iter().map(|&idx| (idx, grid)).collect();
    let b = collect_buffers(buffers, &[], &indices_and_grids, skip_empty)?
        .mutable_grids
        .into_iter()
        .flatten()
        .collect();
    f(b)
}
