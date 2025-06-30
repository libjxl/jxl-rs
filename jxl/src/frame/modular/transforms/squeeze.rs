// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::cell::Ref;

use crate::{
    error::{Error, Result},
    frame::modular::{ChannelInfo, ModularChannel},
    headers::modular::SqueezeParams,
    image::ImageRect,
};

use crate::util::tracing_wrappers::*;

#[instrument(level = "trace", err)]
pub fn check_squeeze_params(
    channels: &[(usize, ChannelInfo)],
    params: &SqueezeParams,
) -> Result<()> {
    let end_channel = (params.begin_channel + params.num_channels) as usize;
    if end_channel > channels.len() {
        return Err(Error::InvalidChannelRange(
            params.begin_channel as usize,
            params.num_channels as usize,
            channels.len(),
        ));
    }
    if channels[params.begin_channel as usize].1.is_meta() != channels[end_channel - 1].1.is_meta()
    {
        return Err(Error::MixingDifferentChannels);
    }
    if channels[params.begin_channel as usize].1.is_meta() && !params.in_place {
        return Err(Error::MetaSqueezeRequiresInPlace);
    }
    Ok(())
}

pub fn default_squeeze(data_channel_info: &[(usize, ChannelInfo)]) -> Vec<SqueezeParams> {
    let mut w = data_channel_info[0].1.size.0;
    let mut h = data_channel_info[0].1.size.1;
    let nc = data_channel_info.len();

    let mut params = vec![];

    let num_meta_channels = data_channel_info
        .iter()
        .take_while(|x| x.1.is_meta())
        .count();

    if nc > 2 && data_channel_info[1].1.size == (w, h) {
        // 420 previews
        let sp = SqueezeParams {
            horizontal: true,
            in_place: false,
            begin_channel: num_meta_channels as u32 + 1,
            num_channels: 2,
        };
        params.push(sp);
        params.push(SqueezeParams {
            horizontal: false,
            ..sp
        });
    }

    const MAX_FIRST_PREVIEW_SIZE: usize = 8;

    let sp = SqueezeParams {
        begin_channel: num_meta_channels as u32,
        num_channels: nc as u32,
        in_place: true,
        horizontal: false,
    };

    // vertical first on tall images
    if w <= h && h > MAX_FIRST_PREVIEW_SIZE {
        params.push(SqueezeParams {
            horizontal: false,
            ..sp
        });
        h = h.div_ceil(2);
    }
    while w > MAX_FIRST_PREVIEW_SIZE || h > MAX_FIRST_PREVIEW_SIZE {
        if w > MAX_FIRST_PREVIEW_SIZE {
            params.push(SqueezeParams {
                horizontal: true,
                ..sp
            });
            w = w.div_ceil(2);
        }
        if h > MAX_FIRST_PREVIEW_SIZE {
            params.push(SqueezeParams {
                horizontal: false,
                ..sp
            });
            h = h.div_ceil(2);
        }
    }

    params
}

#[inline(always)]
fn smooth_tendency(b: i64, a: i64, n: i64) -> i64 {
    let mut diff = 0;
    if b >= a && a >= n {
        diff = (4 * b - 3 * n - a + 6) / 12;
        //      2c = a<<1 + diff - diff&1 <= 2b  so diff - diff&1 <= 2b - 2a
        //      2d = a<<1 - diff - diff&1 >= 2n  so diff + diff&1 <= 2a - 2n
        if diff - (diff & 1) > 2 * (b - a) {
            diff = 2 * (b - a) + 1;
        }
        if diff + (diff & 1) > 2 * (a - n) {
            diff = 2 * (a - n);
        }
    } else if b <= a && a <= n {
        diff = (4 * b - 3 * n - a - 6) / 12;
        //      2c = a<<1 + diff + diff&1 >= 2b  so diff + diff&1 >= 2b - 2a
        //      2d = a<<1 - diff + diff&1 <= 2n  so diff - diff&1 >= 2a - 2n
        if diff + (diff & 1) < 2 * (b - a) {
            diff = 2 * (b - a) - 1;
        }
        if diff - (diff & 1) < 2 * (a - n) {
            diff = 2 * (a - n);
        }
    }
    diff
}

#[inline(always)]
fn unsqueeze(avg: i32, res: i32, next_avg: i32, prev: i32) -> (i32, i32) {
    let tendency = smooth_tendency(prev as i64, avg as i64, next_avg as i64);
    let diff = (res as i64) + tendency;
    let a = (avg as i64) + (diff / 2);
    let b = a - diff;
    (a as i32, b as i32)
}

pub fn do_hsqueeze_step(
    in_avg: &ImageRect<'_, i32>,
    in_res: &ImageRect<'_, i32>,
    in_next_avg: &Option<ImageRect<'_, i32>>,
    out_prev: &Option<Ref<'_, ModularChannel>>,
    buffers: &mut [&mut ModularChannel],
) {
    trace!("hsqueeze step in_avg: {in_avg:?} in_res: {in_res:?} in_next_avg: {in_next_avg:?}");
    let out = buffers.first_mut().unwrap();
    // Shortcut: guarantees that row is at least 1px in the main loop
    if out.data.size().0 == 0 {
        return;
    }
    let (w, h) = in_res.size();
    // Another shortcut: when output row has just 1px
    if w == 0 {
        for y in 0..h {
            out.data.as_rect_mut().row(y)[0] = in_avg.row(y)[0];
        }
        return;
    }
    // Otherwise: 2 or more in in row

    debug_assert!(w >= 1);
    let has_tail = out.data.size().0 & 1 == 1;
    if has_tail {
        debug_assert!(in_avg.size().0 == w + 1);
        debug_assert!(out.data.size().0 == 2 * w + 1);
    }

    for y in 0..h {
        let avg_row = in_avg.row(y);
        let res_row = in_res.row(y);
        let mut prev_b = match out_prev {
            None => avg_row[0],
            Some(mc) => mc.data.as_rect().row(y)[mc.data.size().0 - 1],
        };
        // Guarantee that `avg_row[x + 1]` is available.
        let x_end = if has_tail { w } else { w - 1 };
        for x in 0..x_end {
            let (a, b) = unsqueeze(avg_row[x], res_row[x], avg_row[x + 1], prev_b);
            out.data.as_rect_mut().row(y)[2 * x] = a;
            out.data.as_rect_mut().row(y)[2 * x + 1] = b;
            prev_b = b;
        }
        if !has_tail {
            let last_avg = match in_next_avg {
                None => avg_row[w - 1],
                Some(mc) => mc.row(y)[0],
            };
            let (a, b) = unsqueeze(avg_row[w - 1], res_row[w - 1], last_avg, prev_b);
            out.data.as_rect_mut().row(y)[2 * w - 2] = a;
            out.data.as_rect_mut().row(y)[2 * w - 1] = b;
        } else {
            // 1 last pixel
            out.data.as_rect_mut().row(y)[2 * w] = in_avg.row(y)[w];
        }
    }
}

pub fn do_vsqueeze_step(
    in_avg: &ImageRect<'_, i32>,
    in_res: &ImageRect<'_, i32>,
    in_next_avg: &Option<ImageRect<'_, i32>>,
    out_prev: &Option<Ref<'_, ModularChannel>>,
    buffers: &mut [&mut ModularChannel],
) {
    trace!("vsqueeze step in_avg: {in_avg:?} in_res: {in_res:?} in_next_avg: {in_next_avg:?}");
    let mut out = buffers.first_mut().unwrap().data.as_rect_mut();
    // Shortcut: guarantees that there at least 1 output row
    if out.size().1 == 0 {
        return;
    }
    let (w, h) = in_res.size();
    // Another shortcut: when there is one output row
    if h == 0 {
        out.row(0).copy_from_slice(in_avg.row(0));
        return;
    }
    // Otherwise: 2 or more rows

    debug_assert!(h > 0); // i.e. h - 1 >= 0
    let has_tail = out.size().1 & 1 == 1;
    if has_tail {
        debug_assert!(in_avg.size().1 == h + 1);
        debug_assert!(out.size().1 == 2 * h + 1);
    }

    {
        let prev_b_row = match out_prev {
            None => in_avg.row(0),
            Some(mc) => mc.data.as_rect().row(mc.data.size().1 - 1),
        };
        let avg_row = in_avg.row(0);
        let res_row = in_res.row(0);
        let avg_row_next = if !has_tail && (h == 1) {
            debug_assert!(in_next_avg.is_none());
            in_avg.row(0)
        } else {
            in_avg.row(1)
        };
        for x in 0..w {
            let (a, b) = unsqueeze(avg_row[x], res_row[x], avg_row_next[x], prev_b_row[x]);
            out.row(0)[x] = a;
            out.row(1)[x] = b;
        }
    }
    for y in 1..h {
        let avg_row = in_avg.row(y);
        let res_row = in_res.row(y);
        let avg_row_next = if has_tail || y < h - 1 {
            in_avg.row(y + 1)
        } else {
            match in_next_avg {
                None => avg_row,
                Some(mc) => mc.row(0),
            }
        };
        for x in 0..w {
            let (a, b) = unsqueeze(
                avg_row[x],
                res_row[x],
                avg_row_next[x],
                out.row(2 * y - 1)[x],
            );
            out.row(2 * y)[x] = a;
            out.row(2 * y + 1)[x] = b;
        }
    }
    if has_tail {
        out.row(2 * h).copy_from_slice(in_avg.row(h));
    }
}
