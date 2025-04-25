// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::ops::DerefMut;

use crate::{
    bit_reader::BitReader,
    entropy_coding::decode::Reader,
    error::{Error, Result},
    frame::quantizer::NUM_QUANT_TABLES,
    headers::{frame_header::FrameHeader, modular::GroupHeader, JxlHeader},
    image::Image,
    util::tracing_wrappers::*,
};

use super::{predict::WeightedPredictorState, tree::NUM_NONREF_PROPERTIES, Tree};

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
    buffers: &mut [impl DerefMut<Target = Image<i32>>],
    channel_indices: &[usize],
    index: usize,
    stream_id: usize,
    header: &GroupHeader,
    tree: &Tree,
    reader: &mut Reader,
    br: &mut BitReader,
) -> Result<()> {
    debug!("reading channel");
    let chan = channel_indices[index];
    let size = buffers[index].size();
    let mut wp_state = WeightedPredictorState::new(header);
    for y in 0..size.1 {
        let mut property_buffer = [0; 256];
        property_buffer[0] = chan as i32;
        property_buffer[1] = stream_id as i32;
        for x in 0..size.0 {
            let prediction_result =
                tree.predict(buffers, index, &mut wp_state, x, y, &mut property_buffer);
            let dec = reader.read_signed(br, prediction_result.context as usize)?;
            let val =
                prediction_result.guess + (prediction_result.multiplier as i64) * (dec as i64);
            buffers[index].as_rect_mut().row(y)[x] = val as i32;
            trace!(y, x, val, dec, ?property_buffer, ?prediction_result);
            // TODO(veluca): update WP errors.
        }
    }

    Ok(())
}

// This function will decode a header and apply local transforms if a header is not given.
// The intended use of passing a header is for the DcGlobal section.
pub fn decode_modular_subbitstream(
    buffers: &mut [impl DerefMut<Target = Image<i32>>],
    channel_indices: &[usize],
    stream_id: usize,
    header: Option<GroupHeader>,
    global_tree: &Option<Tree>,
    br: &mut BitReader,
) -> Result<()> {
    let (header, apply_transforms) = if let Some(header) = header {
        (header, false)
    } else {
        let header = GroupHeader::read(br)?;
        if !header.transforms.is_empty() {
            // TODO(veluca): meta-apply local transforms.
            todo!("Local transforms are not implemented yet");
        }
        (header, true)
    };
    if buffers.is_empty() {
        return Ok(());
    }
    assert_eq!(buffers.len(), channel_indices.len());
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

    if tree.max_property() >= NUM_NONREF_PROPERTIES - 2 {
        todo!(
            "WP and reference properties are not implemented yet, max property: {}",
            tree.max_property()
        );
    }

    let mut reader = tree.histograms.make_reader(br)?;

    for i in 0..buffers.len() {
        decode_modular_channel(
            buffers,
            channel_indices,
            i,
            stream_id,
            &header,
            tree,
            &mut reader,
            br,
        )?;
    }

    reader.check_final_state()?;

    if apply_transforms {
        // TODO(veluca): apply local transforms.
    }

    Ok(())
}
