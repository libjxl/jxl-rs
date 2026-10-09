// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::frame::modular::predict::clamped_gradient;
use crate::frame::modular::{ModularChannel, ModularStorage};
use crate::frame::quantizer::NUM_QUANT_TABLES;
use crate::headers::frame_header::FrameHeader;
use crate::image::ImageRect;

#[derive(Debug)]
pub(in crate::frame::modular) struct References<'a> {
    pub(in crate::frame::modular) data: &'a mut [i32],
    pub(in crate::frame::modular) num_ref_props: usize,
}

impl<'a> References<'a> {
    pub(in crate::frame::modular) fn new(
        scratch: &'a mut Vec<i32>,
        num_ref_props: usize,
        xsize: usize,
    ) -> Self {
        let len = num_ref_props * xsize;
        scratch.clear();
        scratch.resize(len, 0);
        Self {
            data: &mut scratch[..len],
            num_ref_props,
        }
    }
}

#[derive(Debug)]
pub enum ModularStreamId {
    GlobalData,
    VarDCTLF(usize),
    ModularLF(usize),
    LFMeta(usize),
    QuantTable(usize),
    ModularHF { pass: usize, group: usize },
}

impl ModularStreamId {
    pub fn get_id(&self, frame_header: &FrameHeader) -> usize {
        match self {
            Self::GlobalData => 0,
            Self::VarDCTLF(g) => 1 + g,
            Self::ModularLF(g) => 1 + frame_header.num_lf_groups() + g,
            Self::LFMeta(g) => 1 + frame_header.num_lf_groups() * 2 + g,
            Self::QuantTable(q) => 1 + frame_header.num_lf_groups() * 3 + q,
            Self::ModularHF { pass, group } => {
                1 + frame_header.num_lf_groups() * 3
                    + NUM_QUANT_TABLES
                    + frame_header.num_groups() * *pass
                    + *group
            }
        }
    }
}

pub(super) fn precompute_references(
    buffers: &mut [&mut ModularChannel],
    chan: usize,
    y: usize,
    references: &mut References<'_>,
    storage: ModularStorage,
) {
    let num_extra_props = references.num_ref_props;
    if num_extra_props == 0 {
        return;
    }
    let xsize = buffers[chan].size(storage).0;
    let ref_data = &mut references.data[..num_extra_props * xsize];
    ref_data.fill(0);
    let mut offset = 0;
    for i in 0..chan {
        if offset >= num_extra_props {
            break;
        }
        let j = chan - i - 1;
        if buffers[j].size(storage) != buffers[chan].size(storage)
            || buffers[j].shift != buffers[chan].shift
        {
            continue;
        }
        if storage == ModularStorage::I16 {
            let ref_rect = ImageRect::<i16>::from_raw(buffers[j].data.as_rect());
            let ref_chan_row = &ref_rect.row(y)[..xsize];
            let ref_chan_prev = &ref_rect.row(y.saturating_sub(1))[..xsize];
            for (x, ref_pixel) in ref_data.chunks_exact_mut(num_extra_props).enumerate() {
                let ref_row = &mut ref_pixel[offset..offset + 4];
                let v = ref_chan_row[x] as i32;
                ref_row[0] = v.wrapping_abs();
                ref_row[1] = v;
                let vleft = if x > 0 { ref_chan_row[x - 1] as i32 } else { 0 };
                let vtop = if y > 0 {
                    ref_chan_prev[x] as i32
                } else {
                    vleft
                };
                let vtopleft = if x > 0 && y > 0 {
                    ref_chan_prev[x - 1] as i32
                } else {
                    vleft
                };
                let vpredicted = clamped_gradient(vleft as i64, vtop as i64, vtopleft as i64);
                ref_row[2] = (v as i64 - vpredicted).wrapping_abs() as i32;
                ref_row[3] = (v as i64 - vpredicted) as i32;
            }
        } else {
            let ref_rect = ImageRect::<i32>::from_raw(buffers[j].data.as_rect());
            let ref_chan_row = &ref_rect.row(y)[..xsize];
            let ref_chan_prev = &ref_rect.row(y.saturating_sub(1))[..xsize];
            for (x, ref_pixel) in ref_data.chunks_exact_mut(num_extra_props).enumerate() {
                let ref_row = &mut ref_pixel[offset..offset + 4];
                let v = ref_chan_row[x];
                ref_row[0] = v.wrapping_abs();
                ref_row[1] = v;
                let vleft = if x > 0 { ref_chan_row[x - 1] } else { 0 };
                let vtop = if y > 0 { ref_chan_prev[x] } else { vleft };
                let vtopleft = if x > 0 && y > 0 {
                    ref_chan_prev[x - 1]
                } else {
                    vleft
                };
                let vpredicted = clamped_gradient(vleft as i64, vtop as i64, vtopleft as i64);
                ref_row[2] = (v as i64 - vpredicted).wrapping_abs() as i32;
                ref_row[3] = (v as i64 - vpredicted) as i32;
            }
        }
        offset += 4;
    }
}

#[inline(always)]
pub(super) fn make_pixel(dec: i32, mul: u32, guess: i64) -> i32 {
    (guess + (mul as i64) * (dec as i64)) as i32
}
