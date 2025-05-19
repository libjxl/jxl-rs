// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    bit_reader::BitReader,
    error::{Error, Result},
    frame::{
        block_context_map::*, coeff_order, transform_map::*, HfGlobalState, HfMetadata,
        LfGlobalState,
    },
    headers::frame_header::FrameHeader,
    image::{Image, Rect},
    util::tracing_wrappers::*,
    util::CeilLog2,
    var_dct::transform::*,
    BLOCK_DIM, BLOCK_SIZE,
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

#[allow(dead_code)]
fn adjust_quant_bias(c: usize, quant_i: i32, biases: &[f32; 4]) -> f32 {
    match quant_i {
        0 => 0.0,
        1 => biases[c],
        -1 => -biases[c],
        _ => {
            let quant = quant_i as f32;
            quant - biases[3] / quant
        }
    }
}

#[allow(dead_code, clippy::too_many_arguments)]
fn dequant_lane(
    scaled_dequant_x: f32,
    scaled_dequant_y: f32,
    scaled_dequant_b: f32,
    dequant_matrices: &[f32],
    size: usize,
    k: usize,
    x_cc_mul: f32,
    b_cc_mul: f32,
    biases: &[f32; 4],
    qblock: &[&[f32]; 3],
    block: &mut [f32],
) {
    let x_mul = dequant_matrices[k] * scaled_dequant_x;
    let y_mul = dequant_matrices[size + k] * scaled_dequant_y;
    let b_mul = dequant_matrices[2 * size + k] * scaled_dequant_b;

    let quantized_x_int = qblock[0][k] as i32;
    let quantized_y_int = qblock[1][k] as i32;
    let quantized_b_int = qblock[2][k] as i32;

    let dequant_x_cc = adjust_quant_bias(0, quantized_x_int, biases) * x_mul;
    let dequant_y = adjust_quant_bias(1, quantized_y_int, biases) * y_mul;
    let dequant_b_cc = adjust_quant_bias(2, quantized_b_int, biases) * b_mul;

    let dequant_x = x_cc_mul * dequant_y + dequant_x_cc;
    let dequant_b = b_cc_mul * dequant_y + dequant_b_cc;
    block[k] = dequant_x;
    block[size + k] = dequant_y;
    block[2 * size + k] = dequant_b;
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub fn decode_vardct_group(
    group: usize,
    pass: usize,
    frame_header: &FrameHeader,
    lf_global: &mut LfGlobalState,
    hf_global: &mut HfGlobalState,
    hf_meta: &HfMetadata,
    quant_lf: &Image<u8>,
    on_output: &mut dyn FnMut(usize, usize, &Image<f32>) -> Result<()>,
    br: &mut BitReader,
) -> Result<(), Error> {
    let num_histo_bits = hf_global.num_histograms.ceil_log2();
    let histogram_index: usize = br.read(num_histo_bits as usize)? as usize;
    debug!(?histogram_index);
    let mut reader = hf_global.passes[pass].histograms.make_reader(br)?;
    let block_group_rect = frame_header.block_group_rect(group);
    let group_size = (
        block_group_rect.size.0 * BLOCK_DIM,
        block_group_rect.size.1 * BLOCK_DIM,
    );
    let mut pixels: [Image<f32>; 3] = [
        Image::<f32>::new(group_size)?,
        Image::<f32>::new(group_size)?,
        Image::<f32>::new(group_size)?,
    ];
    debug!(?block_group_rect);
    let transform_map = hf_meta.transform_map.as_rect();
    let transform_map_rect = transform_map.rect(block_group_rect)?;
    let raw_quant_map = hf_meta.raw_quant_map.as_rect();
    let raw_quant_map_rect = raw_quant_map.rect(block_group_rect)?;
    let mut num_nzeros: [Image<u32>; 3] = [
        Image::new(block_group_rect.size)?,
        Image::new(block_group_rect.size)?,
        Image::new(block_group_rect.size)?,
    ];
    let quant_lf_rect = quant_lf.as_rect().rect(block_group_rect)?;
    let block_context_map = lf_global.block_context_map.as_mut().unwrap();
    let context_offset = histogram_index * block_context_map.num_ac_contexts();
    let mut coeffs_storage;
    let mut hf_coefficients_rect;
    let coeffs = match hf_global.hf_coefficients.as_mut() {
        Some(hf_coefficients) => {
            hf_coefficients_rect = hf_coefficients.as_rect_mut();
            let row = hf_coefficients_rect.row(group);
            if pass == 0 {
                row.fill(0);
            }
            row
        }
        None => {
            coeffs_storage = vec![0; FrameHeader::GROUP_DIM * FrameHeader::GROUP_DIM];
            coeffs_storage.as_mut_slice()
        }
    };
    let mut transform_buffer: [Vec<f32>; 3] = [
        vec![0.0; MAX_COEFF_AREA],
        vec![0.0; MAX_COEFF_AREA],
        vec![0.0; MAX_COEFF_AREA],
    ];
    for by in 0..block_group_rect.size.1 {
        for bx in 0..block_group_rect.size.0 {
            let raw_quant = raw_quant_map_rect.row(by)[bx] as u32;
            let quant_lf = quant_lf_rect.row(by)[bx] as usize;
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
            let block_size = (cx * BLOCK_DIM, cy * BLOCK_DIM);
            let block_rect = Rect {
                origin: (bx * BLOCK_DIM, by * BLOCK_DIM),
                size: block_size,
            };
            let num_blocks = cx * cy;
            let num_coeffs = num_blocks * BLOCK_SIZE;
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
                let block_context =
                    block_context_map.block_context(quant_lf, raw_quant, shape_id, c);
                let nonzero_context = block_context_map
                    .nonzero_context(predicted_nzeros, block_context)
                    + context_offset;
                let mut nonzeros = reader.read(br, nonzero_context)? as usize;
                trace!(
                    "block ({bx},{by},{c}) predicted_nzeros: {predicted_nzeros} \
			nzero_ctx: {nonzero_context} (offset: {context_offset}) \
			nzeros: {nonzeros}"
                );
                if nonzeros + num_blocks > num_coeffs {
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
                let mut prev = if nonzeros > num_coeffs / 16 { 0 } else { 1 };
                for k in num_blocks..num_coeffs {
                    if nonzeros == 0 {
                        break;
                    }
                    let ctx = histo_offset + zero_density_context(nonzeros, k, num_blocks, prev);
                    let coeff = reader.read_signed(br, ctx)?;
                    prev = if coeff != 0 { 1 } else { 0 };
                    nonzeros -= prev;
                    let order_type = coeff_order::ORDER_LUT[transform_id as usize];
                    let coeff_index =
                        hf_global.passes[pass].coeff_orders[order_type * 3 + c][k] as usize;
                    coeffs[coeff_index] = coeff;
                }
                if nonzeros != 0 {
                    return Err(Error::EndOfBlockResidualNonZeros(nonzeros));
                }
            }
            // TODO(szabadka): Fill in transform_buffer with dequantized coefficients.
            // TODO(szabadka): Apply chroma-from-luma on the transform_buffer.
            // TODO(szabadka): Fill in cx x cy corner of transform_buffer with DCT of LF coeffs.
            for c in [1, 0, 2] {
                transform_to_pixels(transform_type, &mut transform_buffer[c])?;
                let mut output = pixels[c].as_rect_mut();
                let mut output_rect = output.rect(block_rect)?;
                for i in 0..block_rect.size.1 {
                    let offset = i * MAX_BLOCK_DIM;
                    output_rect
                        .row(i)
                        .copy_from_slice(&transform_buffer[c][offset..offset + block_rect.size.0]);
                }
            }
        }
    }
    reader.check_final_state()?;
    for c in [0, 1, 2] {
        on_output(c, group, &pixels[c])?;
    }
    Ok(())
}
