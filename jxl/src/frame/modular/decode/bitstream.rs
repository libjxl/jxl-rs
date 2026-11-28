// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use super::channel::decode_modular_channel;
use crate::bit_reader::BitReader;
use crate::entropy_coding::decode::{Codes, SymbolReader, unpack_signed};
use crate::entropy_coding::hybrid_uint::HybridUint;
use crate::error::{Error, Result};
use crate::frame::modular::predict::clamped_gradient;
use crate::frame::modular::transforms::apply_local::meta_apply_local_transforms;
use crate::frame::modular::tree::TreeNode;
use crate::frame::modular::{ModularChannel, ModularStorage, Predictor, ScratchSpace, Tree};
use crate::headers::JxlHeader;
use crate::headers::modular::GroupHeader;
use crate::image::{BufferRecycler, ImageRectMut, SignedSample};

// If we have at least this many bits still available to read,
// we can be sure that none of the reads up to this point read garbage.
const DECODE_SAFETY_MARGIN: usize = 32;

#[inline]
fn apply_row_run<T: SignedSample>(
    row: &mut [T],
    row_top: &[T],
    rle_sym: i32,
    mut left: i32,
    mut topleft: i32,
) -> (i32, i32) {
    debug_assert!(!row_top.is_empty());
    if rle_sym == 0 && left == topleft {
        row.copy_from_slice(row_top);
        let last = row_top[row_top.len() - 1].to_i32();
        return (last, last);
    }
    let sym64 = rle_sym as i64;
    for (&top_sample, p) in row_top.iter().zip(row.iter_mut()) {
        let top = top_sample.to_i32();
        let val = (clamped_gradient(left as i64, top as i64, topleft as i64) + sym64) as i32;
        *p = T::from_i32(val);
        left = val;
        topleft = top;
    }
    (left, topleft)
}

fn decode_fast_lossless_channel<T: SignedSample, const FAST_CONF: bool>(
    channel: &mut ModularChannel,
    tree: &Tree,
    cluster: usize,
    br: &mut BitReader,
    rle_len: &mut usize,
    rle_sym: &mut i32,
) {
    let Codes::Huffman(codes) = tree.histograms.codes() else {
        unreachable!();
    };
    let table = codes.table(cluster);
    let min_sym = tree.histograms.lz77_params().min_symbol.unwrap();
    let min_len = tree.histograms.lz77_params().min_length.unwrap() as usize;
    let sym_conf = tree.histograms.uint(cluster);
    let lz_conf = tree.histograms.lz77_length_uint();

    let mut buf = ImageRectMut::<T>::from_raw(channel.data.as_rect_mut());
    let (w, h) = buf.size();
    let (mut cur_rle_len, mut cur_rle_sym) = (*rle_len, *rle_sym);

    let mut last = 0i32;
    for p in &mut buf.row(0)[..w] {
        if cur_rle_len > 0 {
            cur_rle_len -= 1;
        } else {
            let sym = table.read(br);
            if let Some(token) = sym.checked_sub(min_sym) {
                cur_rle_len = (lz_conf.read(token, br) as usize).wrapping_add(min_len - 1);
            } else {
                cur_rle_sym = unpack_signed(sym_conf.read(sym, br));
            }
        }
        last = last.wrapping_add(cur_rle_sym);
        *p = T::from_i32(last);
    }

    for y in 1..h {
        let [row, row_top] = buf.distinct_rows_mut([y, y - 1]);
        let (row, row_top) = (&mut row[..w], &row_top[..w]);
        let mut left = row_top[0].to_i32();
        let mut topleft = left;
        let mut x = 0;

        if cur_rle_len > 0 {
            let run = cur_rle_len.min(w);
            cur_rle_len -= run;
            (left, topleft) =
                apply_row_run(&mut row[..run], &row_top[..run], cur_rle_sym, left, topleft);
            x = run;
        }

        while x < w {
            let sym = table.read(br);
            if let Some(token) = sym.checked_sub(min_sym) {
                let count = if FAST_CONF {
                    HybridUint::read_config_n00::<4>(token, br)
                } else {
                    lz_conf.read(token, br)
                } as usize;
                let total = count.wrapping_add(min_len - 1);
                let extra = total.min(w - 1 - x);
                cur_rle_len = total - extra;
                let run = 1 + extra;
                (left, topleft) = apply_row_run(
                    &mut row[x..x + run],
                    &row_top[x..x + run],
                    cur_rle_sym,
                    left,
                    topleft,
                );
                x += run;
            } else {
                let raw = if FAST_CONF {
                    HybridUint::read_config_n00::<0>(sym, br)
                } else {
                    sym_conf.read(sym, br)
                };
                cur_rle_sym = unpack_signed(raw);
                let top = row_top[x].to_i32();
                let val = (clamped_gradient(left as i64, top as i64, topleft as i64)
                    + cur_rle_sym as i64) as i32;
                row[x] = T::from_i32(val);
                left = val;
                topleft = top;
                x += 1;
            }
        }
    }

    (*rle_len, *rle_sym) = (cur_rle_len, cur_rle_sym);
}

fn can_decode_fast_lossless(tree: &Tree) -> bool {
    if !tree.nodes.iter().all(|n| {
        matches!(
            n,
            TreeNode::Split { property: 0, .. }
                | TreeNode::Leaf {
                    predictor: Predictor::Gradient,
                    offset: 0,
                    multiplier: 1,
                    ..
                }
        )
    }) {
        return false;
    }

    if !tree.histograms.is_rle() {
        return false;
    }

    matches!(tree.histograms.codes(), Codes::Huffman(_))
}

#[inline(never)]
fn decode_fast_lossless(
    buffers: Vec<&mut ModularChannel>,
    tree: &Tree,
    br: &mut BitReader,
    partial_decoded_buffers: Option<&mut usize>,
    storage: ModularStorage,
) -> Result<()> {
    let mut rle_len: usize = 0;
    let mut rle_sym = 0;

    let mut last_safe_buf = 0;
    for (c, buf) in buffers.into_iter().enumerate() {
        let (w, h) = buf.size(storage);
        if w == 0 || h == 0 {
            continue;
        }
        let TreeNode::Leaf { id, .. } = tree.walk(&[c as i32]) else {
            unreachable!();
        };
        if br.total_bits_available() >= DECODE_SAFETY_MARGIN {
            last_safe_buf = c;
        }

        let cluster = tree.histograms.map_context_to_cluster(id as usize);
        let fast_conf = tree.histograms.uint(cluster).is_config_n00(0)
            && tree.histograms.lz77_length_uint().is_config_n00(4);
        let decode_fn = match (storage, fast_conf) {
            (ModularStorage::I16, true) => decode_fast_lossless_channel::<i16, true>,
            (ModularStorage::I16, false) => decode_fast_lossless_channel::<i16, false>,
            (ModularStorage::I32, _) => decode_fast_lossless_channel::<i32, false>,
        };
        decode_fn(buf, tree, cluster, br, &mut rle_len, &mut rle_sym);

        if let Err(e) = br.check_for_error() {
            if let Some(p) = partial_decoded_buffers {
                *p = last_safe_buf;
            }
            return Err(e);
        }
    }

    Ok(())
}

// This function will decode a header and apply local transforms if a header is not given.
// The intended use of passing a header is for the DcGlobal section.
#[allow(clippy::too_many_arguments)]
pub(in crate::frame::modular) fn decode_modular_subbitstream(
    buffers: Vec<&mut ModularChannel>,
    storage: ModularStorage,
    stream_id: usize,
    header: Option<GroupHeader>,
    global_tree: &Option<Tree>,
    br: &mut BitReader,
    partial_decoded_buffers: Option<&mut usize>,
    scratch_space: &mut ScratchSpace,
    recycler: &BufferRecycler,
    level5_limits: bool,
) -> Result<()> {
    // Skip decoding if all grids are zero-sized.
    let is_empty = buffers
        .iter()
        .all(|buffer| matches!(buffer.size(storage), (0, _) | (_, 0)));
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
                (new_bufs, transform_steps) = meta_apply_local_transforms(
                    buffers,
                    &mut buffer_storage,
                    &h,
                    storage,
                    recycler,
                    level5_limits,
                )?;
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
                let (width, height) = buf.channel_info(storage).size;
                width * height
            })
            .sum::<usize>();
        let size_limit = (1024 + num_local_samples).min(1 << 20);
        Some(Tree::read(
            br,
            size_limit,
            level5_limits,
            "local modular tree nodes",
            &mut scratch_space.lz77_window,
        )?)
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
        .map(|info| info.channel_info(storage).size.0)
        .max()
        .unwrap_or(0);

    if can_decode_fast_lossless(tree) {
        decode_fast_lossless(buffers, tree, br, partial_decoded_buffers, storage)?
    } else {
        let mut reader = SymbolReader::new(
            &tree.histograms,
            br,
            Some(image_width),
            &mut scratch_space.lz77_window,
        )?;

        let mut last_safe_buf = 0;
        for i in 0..buffers.len() {
            // Keep channel numbering stable, but skip actually decoding empty channels.
            // This matches libjxl, which continues the loop without renumbering.
            let (w, h) = buffers[i].size(storage);
            if w == 0 || h == 0 {
                continue;
            }
            if br.total_bits_available() >= DECODE_SAFETY_MARGIN {
                last_safe_buf = i;
            }
            if let Err(e) = decode_modular_channel(
                &mut buffers,
                i,
                stream_id,
                &header,
                tree,
                &mut reader,
                br,
                storage,
                scratch_space,
            ) {
                reader.reclaim_lz77_scratch(&mut scratch_space.lz77_window);
                if let Some(p) = partial_decoded_buffers {
                    *p = last_safe_buf;
                }
                return Err(e);
            }
        }

        reader.reclaim_lz77_scratch(&mut scratch_space.lz77_window);
        reader.check_final_state(&tree.histograms, br)?;

        drop(buffers);
    }

    for step in transform_steps.iter().rev() {
        step.local_apply(&mut buffer_storage, scratch_space, storage, recycler)?;
    }

    for buf in buffer_storage {
        buf.recycle(recycler);
    }

    Ok(())
}
