// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
use crate::{
    error::Result,
    frame::modular::{
        transforms::{RctOp, RctPermutation},
        ModularBufferInfo,
    },
    image::Image,
    util::tracing_wrappers::*,
};

use super::TransformStep;

#[instrument(skip(buffers), ret)]
pub fn do_rct_step(
    step: TransformStep,
    buffers: &mut [ModularBufferInfo],
    (gx, gy): (usize, usize),
) -> Result<Vec<(usize, usize)>> {
    let TransformStep::Rct {
        buf_in,
        buf_out,
        op,
        perm,
    } = step
    else {
        unreachable!("asking to RCT a non-RCT step");
    };

    let grid = buffers[buf_in[0]].grid_shape.0 * gy + gx;
    for i in 0..3 {
        assert_eq!(buffers[buf_in[0]].grid_kind, buffers[buf_in[i]].grid_kind);
        assert_eq!(buffers[buf_in[0]].grid_kind, buffers[buf_out[i]].grid_kind);
        assert_eq!(buffers[buf_in[0]].info.size, buffers[buf_in[i]].info.size);
        assert_eq!(buffers[buf_in[0]].info.size, buffers[buf_out[i]].info.size);
        assert!(buffers[buf_in[i]].buffer_grid[grid].data.is_some());
        assert!(buffers[buf_out[i]].buffer_grid[grid].data.is_none());
    }
    if op != RctOp::YCoCg && perm != RctPermutation::Rgb {
        unimplemented!("non-YCoCg not implemented");
    }

    // TODO(veluca): figure out a good way to re-use the buffers in place if possible.
    let buf_size = buffers[buf_in[0]].buffer_grid[grid]
        .data
        .as_ref()
        .map(|x| x.size())
        .unwrap();

    for i in 0..3 {
        buffers[buf_out[i]].buffer_grid[grid].data = Some(Image::new(buf_size)?);
    }

    for pos_y in 0..buf_size.1 {
        for pos_x in 0..buf_size.0 {
            let chan_indices = [0, 1, 2];
            let [y, co, cg] = chan_indices.map(|x| {
                buffers[buf_in[x]].buffer_grid[grid]
                    .data
                    .as_ref()
                    .unwrap()
                    .as_rect()
                    .row(pos_y)[pos_x]
            });
            let y = y - (cg >> 1);
            let g = cg + y;
            let b = y - (co >> 1);
            let r = y + co;
            for (i, p) in [r, g, b].iter().enumerate() {
                buffers[buf_out[i]].buffer_grid[grid]
                    .data
                    .as_mut()
                    .unwrap()
                    .as_rect_mut()
                    .row(pos_y)[pos_x] = *p;
            }
        }
    }

    for i in 0..3 {
        buffers[buf_in[i]].buffer_grid[grid].mark_used();
    }

    Ok(buf_out.iter().map(|x| (*x, grid)).collect())
}
