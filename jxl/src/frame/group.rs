// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    bit_reader::BitReader,
    error::{Error, Result},
    frame::{block_context_map::*, transform_map::*, HfGlobalState, HfMetadata, LfGlobalState},
    headers::frame_header::FrameHeader,
    image::Image,
    util::tracing_wrappers::*,
    util::CeilLog2,
    BLOCK_SIZE,
};

fn predict_num_nonzeros(nzeros_map: &Image<u32>, bx: usize, by: usize) -> usize {
    if bx == 0 {
        if by == 0 {
            32
        } else {
            nzeros_map.as_rect().row(by - 1)[0] as usize
        }
    } else if by == 0 {
        nzeros_map.as_rect().row(by)[bx - 1] as usize
    } else {
        (nzeros_map.as_rect().row(by - 1)[bx] + nzeros_map.as_rect().row(by)[bx - 1]).div_ceil(2)
            as usize
    }
}

pub fn decode_vardct_group(
    group: usize,
    pass: usize,
    frame_header: &FrameHeader,
    lf_global: &mut LfGlobalState,
    hf_global: &HfGlobalState,
    hf_meta: &HfMetadata,
    br: &mut BitReader,
) -> Result<(), Error> {
    let num_histo_bits = hf_global.num_histograms.ceil_log2();
    let histogram_index: usize = br.read(num_histo_bits as usize)? as usize;
    debug!(?histogram_index);
    let mut reader = hf_global.passes[pass].histograms.make_reader(br)?;
    let block_rect = frame_header.block_group_rect(group);
    debug!(?block_rect);
    let transform_map = hf_meta.transform_map.as_rect();
    let transform_map_rect = transform_map.rect(block_rect)?;
    let raw_quant_map = hf_meta.raw_quant_map.as_rect();
    let raw_quant_map_rect = raw_quant_map.rect(block_rect)?;
    let mut num_nzeros: [Image<u32>; 3] = [
        Image::new(block_rect.size)?,
        Image::new(block_rect.size)?,
        Image::new(block_rect.size)?,
    ];
    let block_context_map = lf_global.block_context_map.as_mut().unwrap();
    if block_context_map.num_lf_contexts > 1 {
        todo!("Unsupported block context map");
    }
    let context_offset = histogram_index * block_context_map.num_ac_contexts();
    for by in 0..block_rect.size.1 {
        for bx in 0..block_rect.size.0 {
            let raw_quant = raw_quant_map_rect.row(by)[bx] as u32;
            let raw_transform_id = transform_map_rect.row(by)[bx];
            let transform_id = raw_transform_id & 127;
            let is_first_block = raw_transform_id >= 128;
            if !is_first_block {
                continue;
            }
            let transform_type = HfTransformType::from_usize(transform_id as usize)?;
            let cx = covered_blocks_x(transform_type) as usize;
            let cy = covered_blocks_y(transform_type) as usize;
            let shape_id = block_shape_id(transform_type) as usize;
            let num_blocks = cx * cy;
            let block_size = num_blocks * BLOCK_SIZE;
            for c in [1, 0, 2] {
                trace!(
                    "Decoding block ({},{}) channel {} with {}x{} block transform {} (shape id {})",
                    bx,
                    by,
                    c,
                    cx,
                    cy,
                    transform_id,
                    shape_id
                );
                let predicted_nzeros = predict_num_nonzeros(&num_nzeros[c], bx, by);
                let block_context = block_context_map.block_context(0, raw_quant, shape_id, c);
                let nonzero_context = block_context_map
                    .nonzero_context(predicted_nzeros, block_context)
                    + context_offset;
                let mut nonzeros = reader.read(br, nonzero_context)? as usize;
                if nonzeros + num_blocks > block_size {
                    return Err(Error::InvalidNumNonZeros(nonzeros, num_blocks));
                }
                for iy in 0..cy {
                    for ix in 0..cx {
                        num_nzeros[c].as_rect_mut().row(by + iy)[bx + ix] =
                            nonzeros.div_ceil(num_blocks) as u32;
                    }
                }
                let histo_offset =
                    block_context_map.zero_density_context_offset(block_context) + context_offset;
                let mut prev = if nonzeros > block_size / 16 { 0 } else { 1 };
                for k in num_blocks..block_size {
                    if nonzeros == 0 {
                        break;
                    }
                    let ctx = histo_offset + zero_density_context(nonzeros, k, num_blocks, prev);
                    let coeff = reader.read_signed(br, ctx)?;
                    prev = if coeff != 0 { 1 } else { 0 };
                    nonzeros -= prev;
                }
                if nonzeros != 0 {
                    return Err(Error::EndOfBlockResidualNonZeros(nonzeros));
                }
            }
        }
    }
    reader.check_final_state()?;
    // TODO(szabadka): Add dequantization, chroma from luma, inverse dct and call render pipeline.
    Ok(())
}
