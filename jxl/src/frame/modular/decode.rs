// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    bit_reader::BitReader,
    entropy_coding::decode::SymbolReader,
    error::{Error, Result},
    frame::quantizer::NUM_QUANT_TABLES,
    headers::{JxlHeader, frame_header::FrameHeader, modular::GroupHeader},
    image::Image,
    util::tracing_wrappers::*,
};

use super::{
    ModularChannel, Tree,
    predict::{WeightedPredictorState, clamped_gradient},
    transforms::apply::meta_apply_local_transforms,
    tree::NUM_NONREF_PROPERTIES,
};

use num_traits::abs;
use std::cmp::max;

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

fn precompute_references(
    buffers: &mut [&mut ModularChannel],
    chan: usize,
    y: usize,
    references: &mut Image<i32>,
) {
    references.as_rect_mut().apply(|_, v: &mut i32| *v = 0);
    let mut offset = 0;
    let num_extra_props = references.size().0;
    for i in 0..chan {
        if offset >= num_extra_props {
            break;
        }
        let j = chan - i - 1;
        if buffers[j].data.size() != buffers[chan].data.size()
            || buffers[j].shift != buffers[chan].shift
        {
            continue;
        }
        let mut refs = references.as_rect_mut();
        let ref_chan = buffers[j].data.as_rect();
        for x in 0..buffers[chan].data.size().0 {
            let v = ref_chan.row(y)[x];
            refs.row(x)[offset] = abs(v);
            refs.row(x)[offset + 1] = v;
            let vleft = if x > 0 { ref_chan.row(y)[x - 1] } else { 0 };
            let vtop = if y > 0 { ref_chan.row(y - 1)[x] } else { vleft };
            let vtopleft = if x > 0 && y > 0 {
                ref_chan.row(y - 1)[x - 1]
            } else {
                vleft
            };
            let vpredicted = clamped_gradient(vleft as i64, vtop as i64, vtopleft as i64);
            refs.row(x)[offset + 2] = abs(v as i64 - vpredicted) as i32;
            refs.row(x)[offset + 3] = (v as i64 - vpredicted) as i32;
        }
        offset += 4;
    }
}

#[allow(clippy::too_many_arguments)]
#[instrument(level = "debug", skip(buffers, reader, tree, br))]
fn decode_modular_channel(
    buffers: &mut [&mut ModularChannel],
    chan: usize,
    stream_id: usize,
    header: &GroupHeader,
    tree: &Tree,
    reader: &mut SymbolReader,
    br: &mut BitReader,
) -> Result<()> {
    debug!("reading channel");
    let size = buffers[chan].data.size();
    let mut wp_state = WeightedPredictorState::new(&header.wp_header, size.0);
    let mut num_ref_props =
        max(0, tree.max_property() as i32 - NUM_NONREF_PROPERTIES as i32) as usize;
    num_ref_props = num_ref_props.div_ceil(4) * 4;
    let mut references = Image::<i32>::new((num_ref_props, size.0))?;
    let num_properties = NUM_NONREF_PROPERTIES + num_ref_props;
    let make_pixel =
        |dec: i32, mul: u32, guess: i64| -> i32 { (guess + (mul as i64) * (dec as i64)) as i32 };
    for y in 0..size.1 {
        precompute_references(buffers, chan, y, &mut references);
        let mut property_buffer: Vec<i32> = vec![0; num_properties];
        property_buffer[0] = chan as i32;
        property_buffer[1] = stream_id as i32;
        for x in 0..size.0 {
            let prediction_result = tree.predict(
                buffers,
                chan,
                &mut wp_state,
                x,
                y,
                &references,
                &mut property_buffer,
            );
            let dec =
                reader.read_signed(&tree.histograms, br, prediction_result.context as usize)?;
            let val = make_pixel(dec, prediction_result.multiplier, prediction_result.guess);
            buffers[chan].data.as_rect_mut().row(y)[x] = val;
            trace!(y, x, val, dec, ?property_buffer, ?prediction_result);
            wp_state.update_errors(val, (x, y), size.0);
        }
    }

    Ok(())
}

// This function will decode a header and apply local transforms if a header is not given.
// The intended use of passing a header is for the DcGlobal section.
pub fn decode_modular_subbitstream(
    buffers: Vec<&mut ModularChannel>,
    stream_id: usize,
    header: Option<GroupHeader>,
    global_tree: &Option<Tree>,
    br: &mut BitReader,
) -> Result<()> {
    if buffers.is_empty() {
        return Ok(());
    }
    let mut transform_steps = vec![];
    let mut buffer_storage = vec![];

    let buffers = buffers.into_iter().collect::<Vec<_>>();
    let (header, mut buffers) = match header {
        Some(h) => (h, buffers),
        None => {
            let h = GroupHeader::read(br)?;
            if !h.transforms.is_empty() {
                // Note: reassigning to `buffers` here convinces the borrow checker that the borrow of
                // `buffer_storage` ought to outlive `buffers[..]`'s lifetime, which obviously breaks
                // applying transforms later.
                let new_bufs;
                (new_bufs, transform_steps) =
                    meta_apply_local_transforms(buffers, &mut buffer_storage, &h)?;
                (h, new_bufs)
            } else {
                (h, buffers)
            }
        }
    };

    if header.use_global_tree && global_tree.is_none() {
        return Err(Error::NoGlobalTree);
    }
    let local_tree = if !header.use_global_tree {
        Some(Tree::read(br, 1024)?)
    } else {
        None
    };
    let tree = if header.use_global_tree {
        global_tree.as_ref().unwrap()
    } else {
        local_tree.as_ref().unwrap()
    };

    let image_width = buffers
        .iter()
        .map(|info| info.channel_info().size.0)
        .max()
        .unwrap_or(0);
    let mut reader = SymbolReader::new(&tree.histograms, br, Some(image_width))?;

    for i in 0..buffers.len() {
        decode_modular_channel(&mut buffers, i, stream_id, &header, tree, &mut reader, br)?;
    }

    reader.check_final_state(&tree.histograms)?;

    drop(buffers);

    for step in transform_steps.iter().rev() {
        step.local_apply(&mut buffer_storage)?;
    }

    Ok(())
}
