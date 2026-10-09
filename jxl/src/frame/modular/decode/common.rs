// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::error::Result;
use crate::frame::modular::predict::clamped_gradient;
use crate::frame::modular::{ModularChannel, ModularStorage};
use crate::frame::quantizer::NUM_QUANT_TABLES;
use crate::headers::frame_header::FrameHeader;
use crate::image::{ImageDataType, ImageRect};

#[derive(Debug)]
pub(crate) struct References<'a> {
    pub data: &'a mut [i32],
    pub num_ref_props: usize,
    pub xsize: usize,
}

impl<'a> References<'a> {
    pub fn new(storage: &'a mut Vec<i32>, num_ref_props: usize, xsize: usize) -> Result<Self> {
        let len = num_ref_props.checked_mul(xsize).unwrap();
        storage.clear();
        storage.try_reserve(len)?;
        storage.resize(len, 0);
        Ok(Self {
            data: &mut storage[..len],
            num_ref_props,
            xsize,
        })
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

#[inline(always)]
fn precompute_references_row<T: ImageDataType + Into<i32> + Copy>(
    ref_buf: &ModularChannel,
    y: usize,
    xsize: usize,
    num_extra_props: usize,
    offset: usize,
    ref_slice: &mut [i32],
) {
    let ref_rect = ImageRect::<T>::from_raw(ref_buf.data.as_rect());
    let ref_chan_row = &ref_rect.row(y)[..xsize];
    let ref_chan_prev = &ref_rect.row(y.saturating_sub(1))[..xsize];
    let mut prev_v = 0i32;
    let mut prev_top = 0i32;
    for (x, (&v_raw, ref_row)) in ref_chan_row
        .iter()
        .zip(ref_slice.chunks_exact_mut(num_extra_props))
        .enumerate()
    {
        let ref_row = &mut ref_row[offset..][..4];
        let v: i32 = v_raw.into();
        let vtop = if y > 0 {
            ref_chan_prev[x].into()
        } else {
            prev_v
        };
        let vtopleft = if x > 0 && y > 0 { prev_top } else { prev_v };
        let vpredicted = clamped_gradient(prev_v as i64, vtop as i64, vtopleft as i64);
        ref_row[0] = v.wrapping_abs();
        ref_row[1] = v;
        ref_row[2] = (v as i64 - vpredicted).wrapping_abs() as i32;
        ref_row[3] = (v as i64 - vpredicted) as i32;
        prev_v = v;
        prev_top = vtop;
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
    if num_extra_props == 0 || references.xsize == 0 {
        return;
    }
    let mut offset = 0;
    let xsize = buffers[chan].size(storage).0;
    let ref_slice = &mut references.data[..xsize * num_extra_props];
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
            precompute_references_row::<i16>(
                buffers[j],
                y,
                xsize,
                num_extra_props,
                offset,
                ref_slice,
            );
        } else {
            precompute_references_row::<i32>(
                buffers[j],
                y,
                xsize,
                num_extra_props,
                offset,
                ref_slice,
            );
        }
        offset += 4;
    }
}

#[inline(always)]
pub(super) fn make_pixel(dec: i32, mul: u32, guess: i64) -> i32 {
    (guess + (mul as i64) * (dec as i64)) as i32
}
