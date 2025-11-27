// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use super::channel::decode_modular_channel;
use crate::{
    bit_reader::BitReader,
    entropy_coding::decode::SymbolReader,
    error::{Error, Result},
    frame::modular::{ModularChannel, Tree, transforms::apply::meta_apply_local_transforms},
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
    let header = if let Some(header) = header {
        header
    } else {
        GroupHeader::read(br)?
    };

    if !header.transforms.is_empty() {
        let mut transform_steps = vec![];
        let mut buffer_storage = vec![];
        let (mut buffers, steps) =
            meta_apply_local_transforms(buffers, &mut buffer_storage, &header)?;
        transform_steps = steps;

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
    } else {
        let mut buffers = buffers;
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
    }

    Ok(())
}
