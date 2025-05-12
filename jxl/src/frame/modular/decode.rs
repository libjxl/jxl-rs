// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    bit_reader::BitReader,
    entropy_coding::decode::Reader,
    error::{Error, Result},
    frame::quantizer::NUM_QUANT_TABLES,
    headers::{frame_header::FrameHeader, modular::GroupHeader, JxlHeader},
    util::tracing_wrappers::*,
};

use super::{
    predict::WeightedPredictorState, transforms::apply::meta_apply_local_transforms,
    tree::NUM_NONREF_PROPERTIES, ModularChannel, Tree,
};

#[allow(unused)]
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

#[allow(clippy::too_many_arguments)]
#[instrument(level = "debug", skip(buffers, reader, br))]
fn decode_modular_channel(
    buffers: &mut [&mut ModularChannel],
    chan: usize,
    stream_id: usize,
    header: &GroupHeader,
    tree: &Tree,
    reader: &mut Reader,
    br: &mut BitReader,
) -> Result<()> {
    debug!("reading channel");
    let size = buffers[chan].data.size();
    let mut wp_state = WeightedPredictorState::new(header, size.0);
    for y in 0..size.1 {
        let mut property_buffer = [0; 256];
        property_buffer[0] = chan as i32;
        property_buffer[1] = stream_id as i32;
        for x in 0..size.0 {
            let prediction_result =
                tree.predict(buffers, chan, &mut wp_state, x, y, &mut property_buffer);
            let dec = reader.read_signed(br, prediction_result.context as usize)?;
            let val =
                prediction_result.guess + (prediction_result.multiplier as i64) * (dec as i64);
            buffers[chan].data.as_rect_mut().row(y)[x] = val as i32;
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
                    meta_apply_local_transforms(buffers, &mut buffer_storage, &h.transforms)?;
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

    if tree.max_property() >= NUM_NONREF_PROPERTIES {
        todo!(
            "reference properties are not implemented yet, max property: {}",
            tree.max_property()
        );
    }
    let image_width = buffers
        .iter()
        .map(|info| info.channel_info().size.0)
        .max()
        .unwrap_or(0);
    let mut reader = tree.histograms.make_reader_with_width(br, image_width)?;

    for i in 0..buffers.len() {
        decode_modular_channel(&mut buffers, i, stream_id, &header, tree, &mut reader, br)?;
    }

    reader.check_final_state()?;

    drop(buffers);

    for step in transform_steps.iter().rev() {
        step.local_apply(&mut buffer_storage)?;
    }

    Ok(())
}
