// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use super::channel::{decode_modular_channel, decode_modular_channel_i16};
use crate::{
    bit_reader::BitReader,
    entropy_coding::decode::SymbolReader,
    error::{Error, Result},
    frame::modular::{
        ModularChannel, ModularChannelI16, Tree, transforms::apply::meta_apply_local_transforms,
    },
    headers::{JxlHeader, modular::GroupHeader},
};

// This function will decode a header and apply local transforms if a header is not given.
// The intended use of passing a header is for the DcGlobal section.
pub fn decode_modular_subbitstream(
    buffers: Vec<&mut ModularChannel>,
    stream_id: usize,
    header: Option<GroupHeader>,
    global_tree: &Option<Tree>,
    br: &mut BitReader,
) -> Result<()> {
    // Skip decoding if all grids are zero-sized.
    let is_empty = buffers
        .iter()
        .all(|buffer| matches!(buffer.data.size(), (0, _) | (_, 0)));
    if is_empty {
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
        let num_local_samples = buffers
            .iter()
            .map(|buf| {
                let (width, height) = buf.channel_info().size;
                width * height
            })
            .sum::<usize>();
        let size_limit = (1024 + num_local_samples).min(1 << 20);
        Some(Tree::read(br, size_limit)?)
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

    reader.check_final_state(&tree.histograms, br)?;

    drop(buffers);

    for step in transform_steps.iter().rev() {
        step.local_apply(&mut buffer_storage)?;
    }

    Ok(())
}

/// Decode modular data directly to i16 buffers.
/// This is used when there are no transforms and bit depth <= 16.
/// IMPORTANT: The header must already have been read and must have no transforms.
/// The caller is responsible for ensuring this is only called in valid scenarios.
pub fn decode_modular_subbitstream_i16(
    mut buffers: Vec<&mut ModularChannelI16>,
    stream_id: usize,
    header: &GroupHeader,
    global_tree: &Option<Tree>,
    br: &mut BitReader,
) -> Result<()> {
    // Skip decoding if all grids are zero-sized.
    let is_empty = buffers
        .iter()
        .all(|buffer| matches!(buffer.data.size(), (0, _) | (_, 0)));
    if is_empty {
        return Ok(());
    }

    // i16 path does not support transforms
    debug_assert!(header.transforms.is_empty());

    if header.use_global_tree && global_tree.is_none() {
        return Err(Error::NoGlobalTree);
    }
    let local_tree = if !header.use_global_tree {
        let num_local_samples = buffers
            .iter()
            .map(|buf| {
                let (width, height) = buf.data.size();
                width * height
            })
            .sum::<usize>();
        let size_limit = (1024 + num_local_samples).min(1 << 20);
        Some(Tree::read(br, size_limit)?)
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
        .map(|info| info.data.size().0)
        .max()
        .unwrap_or(0);
    let mut reader = SymbolReader::new(&tree.histograms, br, Some(image_width))?;

    for i in 0..buffers.len() {
        decode_modular_channel_i16(&mut buffers, i, stream_id, header, tree, &mut reader, br)?;
    }

    reader.check_final_state(&tree.histograms, br)?;

    Ok(())
}
