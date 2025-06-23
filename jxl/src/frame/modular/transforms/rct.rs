// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    frame::modular::{
        ModularChannel,
        transforms::{RctOp, RctPermutation},
    },
    util::tracing_wrappers::*,
};

// Applies a RCT in-place to the given buffers.
#[instrument(level = "debug", skip(buffers), ret)]
pub fn do_rct_step(buffers: &mut [&mut ModularChannel], op: RctOp, perm: RctPermutation) {
    let size = buffers[0].data.size();

    let [r, g, b] = buffers else {
        unreachable!("incorrect buffer count for RCT");
    };

    let buffers = [r, g, b];

    'rct: {
        let apply_rct: fn(i32, i32, i32) -> (i32, i32, i32) = match op {
            RctOp::Noop => break 'rct,
            RctOp::YCoCg => |y, co, cg| {
                let y = y.wrapping_sub(cg >> 1);
                let g = cg.wrapping_add(y);
                let y = y.wrapping_sub(co >> 1);
                let r = y.wrapping_add(co);
                (r, g, y)
            },
            RctOp::AddFirstToThird => |v0, v1, v2| (v0, v1, v2.wrapping_add(v0)),
            RctOp::AddFirstToSecond => |v0, v1, v2| (v0, v1.wrapping_add(v0), v2),
            RctOp::AddFirstToSecondAndThird => {
                |v0, v1, v2| (v0, v1.wrapping_add(v0), v2.wrapping_add(v0))
            }
            RctOp::AddAvgToSecond => {
                |v0, v1, v2| (v0, v1.wrapping_add((v0.wrapping_add(v2)) >> 1), v2)
            }
            RctOp::AddFirstToThirdAndAvgToSecond => |v0, v1, v2| {
                let v2 = v0.wrapping_add(v2);
                (v0, v1.wrapping_add((v0.wrapping_add(v2)) >> 1), v2)
            },
        };

        for pos_y in 0..size.1 {
            for pos_x in 0..size.0 {
                let [v0, v1, v2] = [0, 1, 2].map(|x| buffers[x].data.as_rect().row(pos_y)[pos_x]);
                let (w0, w1, w2) = apply_rct(v0, v1, v2);
                for (i, p) in [w0, w1, w2].iter().enumerate() {
                    buffers[i].data.as_rect_mut().row(pos_y)[pos_x] = *p;
                }
            }
        }
    }

    let [r, g, b] = buffers;

    // Note: Gbr and Brg use the *inverse* permutation compared to libjxl, because we *first* write
    // to the buffers and then permute them, while in libjxl the buffers to be written to are
    // permuted first.
    // The same is true for Rbg/Grb/Bgr, but since those are involutions it doesn't change
    // anything.
    match perm {
        RctPermutation::Rgb => {}
        RctPermutation::Gbr => {
            // out[1, 2, 0] = in[0, 1, 2]
            std::mem::swap(&mut g.data, &mut b.data); // [1, 0, 2]
            std::mem::swap(&mut r.data, &mut g.data);
        }
        RctPermutation::Brg => {
            // out[2, 0, 1] = in[0, 1, 2]
            std::mem::swap(&mut r.data, &mut b.data); // [1, 0, 2]
            std::mem::swap(&mut r.data, &mut g.data);
        }
        RctPermutation::Rbg => {
            // out[0, 2, 1] = in[0, 1, 2]
            std::mem::swap(&mut b.data, &mut g.data);
        }
        RctPermutation::Grb => {
            // out[1, 0, 2] = in[0, 1, 2]
            std::mem::swap(&mut r.data, &mut g.data);
        }
        RctPermutation::Bgr => {
            // out[2, 1, 0] = in[0, 1, 2]
            std::mem::swap(&mut r.data, &mut b.data);
        }
    }
}
