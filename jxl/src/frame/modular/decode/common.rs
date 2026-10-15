// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::frame::modular::predict::clamped_gradient;
use crate::frame::modular::tree::{NUM_NONREF_PROPERTIES, PROPERTIES_PER_PREVCHAN};
use crate::frame::modular::{ModularChannel, ModularStorage};
use crate::frame::quantizer::NUM_QUANT_TABLES;
use crate::headers::frame_header::FrameHeader;
use crate::image::ImageRect;

#[derive(Debug)]
pub(crate) struct References<'a> {
    pub(crate) data: &'a mut [i32],
    pub(crate) num_ref_props: usize,
}

impl<'a> References<'a> {
    pub(crate) fn new(scratch: &'a mut Vec<i32>, num_ref_props: usize, xsize: usize) -> Self {
        let len = num_ref_props * xsize;
        scratch.clear();
        scratch.resize(len, 0);
        Self {
            data: &mut scratch[..len],
            num_ref_props,
        }
    }
}

pub(crate) fn init_properties_and_references<'a>(
    references: &'a mut Vec<i32>,
    property_buffer: &'a mut Vec<i32>,
    channel: usize,
    stream: usize,
    xsize: usize,
    max_property_count: usize,
) -> (References<'a>, &'a mut [i32; 256]) {
    let num_ref_props = max_property_count
        .saturating_sub(NUM_NONREF_PROPERTIES)
        .next_multiple_of(PROPERTIES_PER_PREVCHAN)
        .min(channel * PROPERTIES_PER_PREVCHAN);
    let references = References::new(references, num_ref_props, xsize);
    property_buffer.clear();
    property_buffer.resize(256, 0);
    let property_buffer: &mut [i32; 256] = (&mut property_buffer[..256]).try_into().unwrap();
    property_buffer[0] = channel as i32;
    property_buffer[1] = stream as i32;
    (references, property_buffer)
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
    references: &mut References,
    storage: ModularStorage,
) {
    let num_extra_props = references.num_ref_props;
    if num_extra_props == 0 {
        return;
    }
    references.data.fill(0);
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
            let ref_chan_row = ref_rect.row(y);
            let ref_chan_prev = ref_rect.row(y.saturating_sub(1));
            for x in 0..buffers[chan].size(storage).0 {
                let ref_row = &mut references.data[x * num_extra_props..(x + 1) * num_extra_props];
                let v = ref_chan_row[x] as i32;
                ref_row[offset] = v.wrapping_abs();
                ref_row[offset + 1] = v;
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
                ref_row[offset + 2] = (v as i64 - vpredicted).wrapping_abs() as i32;
                ref_row[offset + 3] = (v as i64 - vpredicted) as i32;
            }
        } else {
            let ref_rect = ImageRect::<i32>::from_raw(buffers[j].data.as_rect());
            let ref_chan_row = ref_rect.row(y);
            let ref_chan_prev = ref_rect.row(y.saturating_sub(1));
            for x in 0..buffers[chan].size(storage).0 {
                let ref_row = &mut references.data[x * num_extra_props..(x + 1) * num_extra_props];
                let v = ref_chan_row[x];
                ref_row[offset] = v.wrapping_abs();
                ref_row[offset + 1] = v;
                let vleft = if x > 0 { ref_chan_row[x - 1] } else { 0 };
                let vtop = if y > 0 { ref_chan_prev[x] } else { vleft };
                let vtopleft = if x > 0 && y > 0 {
                    ref_chan_prev[x - 1]
                } else {
                    vleft
                };
                let vpredicted = clamped_gradient(vleft as i64, vtop as i64, vtopleft as i64);
                ref_row[offset + 2] = (v as i64 - vpredicted).wrapping_abs() as i32;
                ref_row[offset + 3] = (v as i64 - vpredicted) as i32;
            }
        }
        offset += 4;
    }
}

#[inline(always)]
pub(super) fn make_pixel(dec: i32, mul: u32, guess: i64) -> i32 {
    (guess + (mul as i64) * (dec as i64)) as i32
}
