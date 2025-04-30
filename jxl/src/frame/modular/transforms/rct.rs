// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    frame::modular::{
        transforms::{RctOp, RctPermutation},
        ModularChannel,
    },
    util::tracing_wrappers::*,
};

// Applies a RCT in-place to the given buffers.
#[instrument(skip(buffers), ret)]
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
                let y = y - (cg >> 1);
                let g = cg + y;
                let b = y - (co >> 1);
                let r = y + co;
                (r, g, b)
            },
            RctOp::AddFirstToThird => |v0, v1, v2| (v0, v1, v2 + v0),
            RctOp::AddFirstToSecond => |v0, v1, v2| (v0, v1 + v0, v2),
            RctOp::AddFirstToSecondAndThird => |v0, v1, v2| (v0, v1 + v0, v2 + v0),
            RctOp::AddAvgToSecond => |v0, v1, v2| (v0, v1 + ((v0 + v2) >> 1), v2),
            RctOp::AddFirstToThirdAndAvgToSecond => |v0, v1, v2| {
                let v2 = v0 + v2;
                (v0, v1 + ((v0 + v2) >> 1), v2)
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

    match perm {
        RctPermutation::Rgb => {}
        RctPermutation::Gbr => {
            std::mem::swap(&mut r.data, &mut g.data); // Grb
            std::mem::swap(&mut g.data, &mut b.data);
        }
        RctPermutation::Brg => {
            std::mem::swap(&mut r.data, &mut g.data); // Grb
            std::mem::swap(&mut r.data, &mut b.data);
        }
        RctPermutation::Rbg => {
            std::mem::swap(&mut b.data, &mut g.data);
        }
        RctPermutation::Grb => {
            std::mem::swap(&mut r.data, &mut g.data);
        }
        RctPermutation::Bgr => {
            std::mem::swap(&mut r.data, &mut b.data);
        }
    }
}
