// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    frame::modular::transforms::{RctOp, RctPermutation},
    image::Image,
    util::tracing_wrappers::*,
};
use std::ops::DerefMut;

// Applies a RCT in-place to the given buffers.
#[instrument(skip(buffers), ret)]
pub fn do_rct_step(
    buffers: &mut [impl DerefMut<Target = Image<i32>>],
    op: RctOp,
    perm: RctPermutation,
) {
    if op != RctOp::YCoCg && perm != RctPermutation::Rgb {
        unimplemented!("non-YCoCg not implemented");
    }

    let size = buffers[0].size();

    let [b0, b1, b2] = buffers else {
        unreachable!("incorrect buffer count for RCT");
    };

    let buffers = [b0.deref_mut(), b1.deref_mut(), b2.deref_mut()];

    for pos_y in 0..size.1 {
        for pos_x in 0..size.0 {
            let chan_indices = [0, 1, 2];
            let [y, co, cg] = chan_indices.map(|x| buffers[x].as_rect().row(pos_y)[pos_x]);
            let y = y - (cg >> 1);
            let g = cg + y;
            let b = y - (co >> 1);
            let r = y + co;
            for (i, p) in [r, g, b].iter().enumerate() {
                buffers[i].as_rect_mut().row(pos_y)[pos_x] = *p;
            }
        }
    }
}
