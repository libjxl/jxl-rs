// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use num_traits::Float;

use crate::{
    BLOCK_DIM, BLOCK_SIZE,
    bit_reader::BitReader,
    entropy_coding::decode::SymbolReader,
    error::{Error, Result},
    frame::{
        HfGlobalState, HfMetadata, LfGlobalState, block_context_map::*,
        color_correlation_map::COLOR_TILE_DIM_IN_BLOCKS, quant_weights::DequantMatrices,
        transform_map::*,
    },
    headers::frame_header::FrameHeader,
    image::{Image, ImageRect, Rect},
    util::{CeilLog2, tracing_wrappers::*},
    var_dct::{
        dct::{DCT1D, DCT1DImpl, compute_scaled_dct},
        dct_scales::{DctResampleScales, HasDctResampleScales, dct_total_resample_scale},
        transform::*,
    },
};

// Computes the lowest-frequency ROWSxCOLS-sized square in output, which is a
// DCT_ROWS*DCT_COLS-sized DCT block, by doing a ROWS*COLS DCT on the input
// block.
fn reinterpreting_dct<
    const DCT_ROWS: usize,
    const DCT_COLS: usize,
    const ROWS: usize,
    const COLS: usize,
>(
    input: &ImageRect<f32>,
    output: &mut [f32],
    output_stride: usize,
    block: &mut [f32],
) where
    DctResampleScales<ROWS, DCT_ROWS>: HasDctResampleScales<ROWS>,
    DctResampleScales<COLS, DCT_COLS>: HasDctResampleScales<COLS>,
    DCT1DImpl<ROWS>: DCT1D,
    DCT1DImpl<COLS>: DCT1D,
{
    let mut dct_input = [[0.0; COLS]; ROWS];
    #[allow(clippy::needless_range_loop)]
    for y in 0..ROWS {
        dct_input[y].copy_from_slice(&input.row(y)[0..COLS]);
    }
    compute_scaled_dct::<ROWS, COLS>(dct_input, block);
    if ROWS < COLS {
        for y in 0..ROWS {
            for x in 0..COLS {
                output[y * output_stride + x] = block[y * COLS + x]
                    * dct_total_resample_scale::<ROWS, DCT_ROWS>(y)
                    * dct_total_resample_scale::<COLS, DCT_COLS>(x);
            }
        }
    } else {
        for y in 0..COLS {
            for x in 0..ROWS {
                output[y * output_stride + x] = block[y * ROWS + x]
                    * dct_total_resample_scale::<COLS, DCT_COLS>(y)
                    * dct_total_resample_scale::<ROWS, DCT_ROWS>(x);
            }
        }
    }
}

fn lowest_frequencies_from_lf(
    hf_type: HfTransformType,
    lf: &ImageRect<f32>,
    llf: &mut [f32],
    scratch: &mut [f32],
) {
    match hf_type {
        HfTransformType::DCT16X8 => {
            reinterpreting_dct::<
                /*DCT_ROWS=*/ { 2 * BLOCK_DIM },
                /*DCT_COLS=*/ BLOCK_DIM,
                /*ROWS=*/ 2,
                /*COLS=*/ 1,
            >(lf, llf, 2 * BLOCK_DIM, scratch);
        }
        HfTransformType::DCT8X16 => {
            reinterpreting_dct::<
                /*DCT_ROWS=*/ BLOCK_DIM,
                /*DCT_COLS=*/ { 2 * BLOCK_DIM },
                /*ROWS=*/ 1,
                /*COLS=*/ 2,
            >(lf, llf, 2 * BLOCK_DIM, scratch);
        }
        HfTransformType::DCT16X16 => {
            reinterpreting_dct::<
                /*DCT_ROWS=*/ { 2 * BLOCK_DIM },
                /*DCT_COLS=*/ { 2 * BLOCK_DIM },
                /*ROWS=*/ 2,
                /*COLS=*/ 2,
            >(lf, llf, 2 * BLOCK_DIM, scratch);
        }
        HfTransformType::DCT32X8 => {
            reinterpreting_dct::<
                /*DCT_ROWS=*/ { 4 * BLOCK_DIM },
                /*DCT_COLS=*/ BLOCK_DIM,
                /*ROWS=*/ 4,
                /*COLS=*/ 1,
            >(lf, llf, 4 * BLOCK_DIM, scratch);
        }
        HfTransformType::DCT8X32 => {
            reinterpreting_dct::<
                /*DCT_ROWS=*/ BLOCK_DIM,
                /*DCT_COLS=*/ { 4 * BLOCK_DIM },
                /*ROWS=*/ 1,
                /*COLS=*/ 4,
            >(lf, llf, 4 * BLOCK_DIM, scratch);
        }
        HfTransformType::DCT32X16 => {
            reinterpreting_dct::<
                /*DCT_ROWS=*/ { 4 * BLOCK_DIM },
                /*DCT_COLS=*/ { 2 * BLOCK_DIM },
                /*ROWS=*/ 4,
                /*COLS=*/ 2,
            >(lf, llf, 4 * BLOCK_DIM, scratch);
        }
        HfTransformType::DCT16X32 => {
            reinterpreting_dct::<
                /*DCT_ROWS=*/ { 2 * BLOCK_DIM },
                /*DCT_COLS=*/ { 4 * BLOCK_DIM },
                /*ROWS=*/ 2,
                /*COLS=*/ 4,
            >(lf, llf, 4 * BLOCK_DIM, scratch);
        }
        HfTransformType::DCT32X32 => {
            reinterpreting_dct::<
                /*DCT_ROWS=*/ { 4 * BLOCK_DIM },
                /*DCT_COLS=*/ { 4 * BLOCK_DIM },
                /*ROWS=*/ 4,
                /*COLS=*/ 4,
            >(lf, llf, 4 * BLOCK_DIM, scratch);
        }
        HfTransformType::DCT64X32 => {
            reinterpreting_dct::<
                /*DCT_ROWS=*/ { 8 * BLOCK_DIM },
                /*DCT_COLS=*/ { 4 * BLOCK_DIM },
                /*ROWS=*/ 8,
                /*COLS=*/ 4,
            >(lf, llf, 8 * BLOCK_DIM, scratch);
        }
        HfTransformType::DCT32X64 => {
            reinterpreting_dct::<
                /*DCT_ROWS=*/ { 4 * BLOCK_DIM },
                /*DCT_COLS=*/ { 8 * BLOCK_DIM },
                /*ROWS=*/ 4,
                /*COLS=*/ 8,
            >(lf, llf, 8 * BLOCK_DIM, scratch);
        }
        HfTransformType::DCT64X64 => {
            reinterpreting_dct::<
                /*DCT_ROWS=*/ { 8 * BLOCK_DIM },
                /*DCT_COLS=*/ { 8 * BLOCK_DIM },
                /*ROWS=*/ 8,
                /*COLS=*/ 8,
            >(lf, llf, 8 * BLOCK_DIM, scratch);
        }
        HfTransformType::DCT128X64 => {
            reinterpreting_dct::<
                /*DCT_ROWS=*/ { 16 * BLOCK_DIM },
                /*DCT_COLS=*/ { 8 * BLOCK_DIM },
                /*ROWS=*/ 16,
                /*COLS=*/ 8,
            >(lf, llf, 16 * BLOCK_DIM, scratch);
        }
        HfTransformType::DCT64X128 => {
            reinterpreting_dct::<
                /*DCT_ROWS=*/ { 8 * BLOCK_DIM },
                /*DCT_COLS=*/ { 16 * BLOCK_DIM },
                /*ROWS=*/ 8,
                /*COLS=*/ 16,
            >(lf, llf, 16 * BLOCK_DIM, scratch);
        }
        HfTransformType::DCT128X128 => {
            reinterpreting_dct::<
                /*DCT_ROWS=*/ { 16 * BLOCK_DIM },
                /*DCT_COLS=*/ { 16 * BLOCK_DIM },
                /*ROWS=*/ 16,
                /*COLS=*/ 16,
            >(lf, llf, 16 * BLOCK_DIM, scratch);
        }
        HfTransformType::DCT256X128 => {
            reinterpreting_dct::<
                /*DCT_ROWS=*/ { 32 * BLOCK_DIM },
                /*DCT_COLS=*/ { 16 * BLOCK_DIM },
                /*ROWS=*/ 32,
                /*COLS=*/ 16,
            >(lf, llf, 32 * BLOCK_DIM, scratch);
        }
        HfTransformType::DCT128X256 => {
            reinterpreting_dct::<
                /*DCT_ROWS=*/ { 16 * BLOCK_DIM },
                /*DCT_COLS=*/ { 32 * BLOCK_DIM },
                /*ROWS=*/ 16,
                /*COLS=*/ 32,
            >(lf, llf, 32 * BLOCK_DIM, scratch);
        }
        HfTransformType::DCT256X256 => {
            reinterpreting_dct::<
                /*DCT_ROWS=*/ { 32 * BLOCK_DIM },
                /*DCT_COLS=*/ { 32 * BLOCK_DIM },
                /*ROWS=*/ 32,
                /*COLS=*/ 32,
            >(lf, llf, 32 * BLOCK_DIM, scratch);
        }
        HfTransformType::DCT
        | HfTransformType::DCT2X2
        | HfTransformType::DCT4X4
        | HfTransformType::DCT4X8
        | HfTransformType::DCT8X4
        | HfTransformType::AFV0
        | HfTransformType::AFV1
        | HfTransformType::AFV2
        | HfTransformType::AFV3
        | HfTransformType::IDENTITY => {
            llf[0] = lf.row(0)[0];
        }
    }
}

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

#[allow(clippy::too_many_arguments)]
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
    qblock: &[&[i32]; 3],
    block: &mut [Vec<f32>; 3],
) {
    let x_mul = dequant_matrices[k] * scaled_dequant_x;
    let y_mul = dequant_matrices[size + k] * scaled_dequant_y;
    let b_mul = dequant_matrices[2 * size + k] * scaled_dequant_b;

    let quantized_x = qblock[0][k];
    let quantized_y = qblock[1][k];
    let quantized_b = qblock[2][k];

    let dequant_x_cc = adjust_quant_bias(0, quantized_x, biases) * x_mul;
    let dequant_y = adjust_quant_bias(1, quantized_y, biases) * y_mul;
    let dequant_b_cc = adjust_quant_bias(2, quantized_b, biases) * b_mul;

    let dequant_x = x_cc_mul * dequant_y + dequant_x_cc;
    let dequant_b = b_cc_mul * dequant_y + dequant_b_cc;
    block[0][k] = dequant_x;
    block[1][k] = dequant_y;
    block[2][k] = dequant_b;
}

#[allow(clippy::too_many_arguments)]
fn dequant_block(
    hf_type: HfTransformType,
    inv_global_scale: f32,
    quant: u32,
    x_dm_multiplier: f32,
    b_dm_multiplier: f32,
    x_cc_mul: f32,
    b_cc_mul: f32,
    size: usize,
    dequant_matrices: &DequantMatrices,
    covered_blocks: usize,
    lf: &Option<[ImageRect<f32>; 3]>,
    biases: &[f32; 4],
    qblock: &[&[i32]; 3],
    block: &mut [Vec<f32>; 3],
    scratch: &mut [f32],
) {
    let scaled_dequant_y = inv_global_scale / (quant as f32);

    let scaled_dequant_x = scaled_dequant_y * x_dm_multiplier;
    let scaled_dequant_b = scaled_dequant_y * b_dm_multiplier;

    let matrices = dequant_matrices.matrix(hf_type, 0);

    for k in 0..covered_blocks * BLOCK_SIZE {
        dequant_lane(
            scaled_dequant_x,
            scaled_dequant_y,
            scaled_dequant_b,
            matrices,
            size,
            k,
            x_cc_mul,
            b_cc_mul,
            biases,
            qblock,
            block,
        );
    }
    if let Some(lf) = lf.as_ref() {
        for c in 0..3 {
            lowest_frequencies_from_lf(hf_type, &lf[c], &mut block[c], scratch);
        }
    }
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
    lf_image: &Option<[Image<f32>; 3]>,
    quant_lf: &Image<u8>,
    quant_biases: &[f32; 4],
    br: &mut BitReader,
) -> Result<[Image<f32>; 3], Error> {
    let x_dm_multiplier = (1.0 / (1.25)).powf(frame_header.x_qm_scale as f32 - 2.0);
    let b_dm_multiplier = (1.0 / (1.25)).powf(frame_header.b_qm_scale as f32 - 2.0);

    let num_histo_bits = hf_global.num_histograms.ceil_log2();
    let histogram_index: usize = br.read(num_histo_bits as usize)? as usize;
    debug!(?histogram_index);
    let mut reader = SymbolReader::new(&hf_global.passes[pass].histograms, br, None)?;
    let block_group_rect = frame_header.block_group_rect(group);
    let group_size = (
        block_group_rect.size.0 * BLOCK_DIM,
        block_group_rect.size.1 * BLOCK_DIM,
    );
    let mut pixels: [Image<f32>; 3] = [
        Image::new((
            group_size.0 >> frame_header.hshift(0),
            group_size.1 >> frame_header.vshift(0),
        ))?,
        Image::new((
            group_size.0 >> frame_header.hshift(1),
            group_size.1 >> frame_header.vshift(1),
        ))?,
        Image::new((
            group_size.0 >> frame_header.hshift(2),
            group_size.1 >> frame_header.vshift(2),
        ))?,
    ];
    debug!(?block_group_rect);
    let max_block_size = HfTransformType::VALUES
        .iter()
        .filter(|&transform_type| (hf_meta.used_hf_types & (1 << *transform_type as u32)) != 0)
        .map(|&transform_type| {
            BLOCK_SIZE
                * covered_blocks_x(transform_type) as usize
                * covered_blocks_y(transform_type) as usize
        })
        .max()
        .unwrap_or(0);
    let mut scratch = vec![0.0; max_block_size];
    let color_correlation_params = lf_global.color_correlation_params.as_ref().unwrap();
    let cmap_rect = Rect {
        origin: (
            block_group_rect.origin.0 / COLOR_TILE_DIM_IN_BLOCKS,
            block_group_rect.origin.1 / COLOR_TILE_DIM_IN_BLOCKS,
        ),
        size: (
            block_group_rect.size.0.div_ceil(COLOR_TILE_DIM_IN_BLOCKS),
            block_group_rect.size.1.div_ceil(COLOR_TILE_DIM_IN_BLOCKS),
        ),
    };
    let quant_params = lf_global.quant_params.as_ref().unwrap();
    let inv_global_scale = quant_params.inv_global_scale();
    let ytox_map = hf_meta.ytox_map.as_rect();
    let ytox_map_rect = ytox_map.rect(cmap_rect)?;
    let ytob_map = hf_meta.ytob_map.as_rect();
    let ytob_map_rect = ytob_map.rect(cmap_rect)?;
    let transform_map = hf_meta.transform_map.as_rect();
    let transform_map_rect = transform_map.rect(block_group_rect)?;
    let raw_quant_map = hf_meta.raw_quant_map.as_rect();
    let raw_quant_map_rect = raw_quant_map.rect(block_group_rect)?;
    let mut num_nzeros: [Image<u32>; 3] = [
        Image::new((
            block_group_rect.size.0 >> frame_header.hshift(0),
            block_group_rect.size.1 >> frame_header.vshift(0),
        ))?,
        Image::new((
            block_group_rect.size.0 >> frame_header.hshift(1),
            block_group_rect.size.1 >> frame_header.vshift(1),
        ))?,
        Image::new((
            block_group_rect.size.0 >> frame_header.hshift(2),
            block_group_rect.size.1 >> frame_header.vshift(2),
        ))?,
    ];
    let quant_lf_rect = quant_lf.as_rect().rect(block_group_rect)?;
    let block_context_map = lf_global.block_context_map.as_mut().unwrap();
    let context_offset = histogram_index * block_context_map.num_ac_contexts();
    let mut coeffs_storage;
    let mut hf_coefficients_rects;
    let coeffs = match hf_global.hf_coefficients.as_mut() {
        Some(hf_coefficients) => {
            hf_coefficients_rects = (
                hf_coefficients.0.as_rect_mut(),
                hf_coefficients.1.as_rect_mut(),
                hf_coefficients.2.as_rect_mut(),
            );
            [
                hf_coefficients_rects.0.row(group),
                hf_coefficients_rects.1.row(group),
                hf_coefficients_rects.2.row(group),
            ]
        }
        None => {
            coeffs_storage = vec![0; 3 * FrameHeader::GROUP_DIM * FrameHeader::GROUP_DIM];
            let (coeffs_x, coeffs_y_b) =
                coeffs_storage.split_at_mut(FrameHeader::GROUP_DIM * FrameHeader::GROUP_DIM);
            let (coeffs_y, coeffs_b) =
                coeffs_y_b.split_at_mut(FrameHeader::GROUP_DIM * FrameHeader::GROUP_DIM);
            [coeffs_x, coeffs_y, coeffs_b]
        }
    };
    let shift_for_pass = if pass < frame_header.passes.shift.len() {
        frame_header.passes.shift[pass]
    } else {
        0
    };
    let mut coeffs_offset = 0;
    let mut transform_buffer: [Vec<f32>; 3] = [
        vec![0.0; MAX_COEFF_AREA],
        vec![0.0; MAX_COEFF_AREA],
        vec![0.0; MAX_COEFF_AREA],
    ];

    let hshift = [
        frame_header.hshift(0),
        frame_header.hshift(1),
        frame_header.hshift(2),
    ];
    let vshift = [
        frame_header.vshift(0),
        frame_header.vshift(1),
        frame_header.vshift(2),
    ];
    let lf = match lf_image.as_ref() {
        None => None,
        Some(lf_planes) => {
            let r: [Rect; 3] = core::array::from_fn(|i| Rect {
                origin: (
                    block_group_rect.origin.0 >> hshift[i],
                    block_group_rect.origin.1 >> vshift[i],
                ),
                size: (
                    block_group_rect.size.0 >> hshift[i],
                    block_group_rect.size.1 >> vshift[i],
                ),
            });

            let [lf_x, lf_y, lf_b] = lf_planes.each_ref();
            Some([
                lf_x.as_rect().rect(r[0])?,
                lf_y.as_rect().rect(r[1])?,
                lf_b.as_rect().rect(r[2])?,
            ])
        }
    };
    for by in 0..block_group_rect.size.1 {
        let sby = [by >> vshift[0], by >> vshift[1], by >> vshift[2]];
        let ty = by / COLOR_TILE_DIM_IN_BLOCKS;

        let row_cmap_x = ytox_map_rect.row(ty);
        let row_cmap_b = ytob_map_rect.row(ty);

        for bx in 0..block_group_rect.size.0 {
            let sbx = [bx >> hshift[0], bx >> hshift[1], bx >> hshift[2]];
            let tx = bx / COLOR_TILE_DIM_IN_BLOCKS;
            let x_cc_mul = color_correlation_params.y_to_x(row_cmap_x[tx] as i32);
            let b_cc_mul = color_correlation_params.y_to_b(row_cmap_b[tx] as i32);
            let raw_quant = raw_quant_map_rect.row(by)[bx] as u32;
            let quant_lf = quant_lf_rect.row(by)[bx] as usize;
            let raw_transform_id = transform_map_rect.row(by)[bx];
            let transform_id = raw_transform_id & 127;
            let is_first_block = raw_transform_id >= 128;
            if !is_first_block {
                continue;
            }
            let lf_rects = match lf.as_ref() {
                None => None,
                Some(lf) => {
                    let [lf_x, lf_y, lf_b] = lf.each_ref();
                    Some([
                        lf_x.rect(Rect {
                            origin: (sbx[0], sby[0]),
                            size: (lf_x.size().0 - sbx[0], lf_x.size().1 - sby[0]),
                        })?,
                        lf_y.rect(Rect {
                            origin: (sbx[1], sby[1]),
                            size: (lf_y.size().0 - sbx[1], lf_y.size().1 - sby[1]),
                        })?,
                        lf_b.rect(Rect {
                            origin: (sbx[2], sby[2]),
                            size: (lf_b.size().0 - sbx[2], lf_b.size().1 - sby[2]),
                        })?,
                    ])
                }
            };

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
                if (sbx[c] << hshift[c]) != bx || (sby[c] << vshift[c] != by) {
                    continue;
                }
                trace!(
                    "Decoding block ({},{}) channel {} with {}x{} block transform {} (shape id {})",
                    sbx[c], sby[c], c, cx, cy, transform_id, shape_id
                );
                let predicted_nzeros = predict_num_nonzeros(&num_nzeros[c], sbx[c], sby[c]);
                let block_context =
                    block_context_map.block_context(quant_lf, raw_quant, shape_id, c);
                let nonzero_context = block_context_map
                    .nonzero_context(predicted_nzeros, block_context)
                    + context_offset;
                let mut nonzeros =
                    reader.read_unsigned(&hf_global.passes[pass].histograms, br, nonzero_context)?
                        as usize;
                trace!(
                    "block ({},{},{c}) predicted_nzeros: {predicted_nzeros} \
                       nzero_ctx: {nonzero_context} (offset: {context_offset}) \
                       nzeros: {nonzeros}",
                    sbx[c], sby[c]
                );
                if nonzeros + num_blocks > num_coeffs {
                    return Err(Error::InvalidNumNonZeros(nonzeros, num_blocks));
                }
                for iy in 0..cy {
                    for ix in 0..cx {
                        num_nzeros[c].as_rect_mut().row(sby[c] + iy)[sbx[c] + ix] =
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
                    let coeff = reader.read_signed(&hf_global.passes[pass].histograms, br, ctx)?
                        << shift_for_pass;
                    prev = if coeff != 0 { 1 } else { 0 };
                    nonzeros -= prev;
                    let coeff_index =
                        hf_global.passes[pass].coeff_orders[shape_id * 3 + c][k] as usize;
                    coeffs[c][coeffs_offset + coeff_index] += coeff;
                }
                if nonzeros != 0 {
                    return Err(Error::EndOfBlockResidualNonZeros(nonzeros));
                }
            }
            let qblock = [
                &coeffs[0][coeffs_offset..],
                &coeffs[1][coeffs_offset..],
                &coeffs[2][coeffs_offset..],
            ];
            dequant_block(
                transform_type,
                inv_global_scale,
                raw_quant,
                x_dm_multiplier,
                b_dm_multiplier,
                x_cc_mul,
                b_cc_mul,
                num_coeffs,
                &hf_global.dequant_matrices,
                num_blocks,
                &lf_rects,
                quant_biases,
                &qblock,
                &mut transform_buffer,
                &mut scratch,
            );
            for c in [1, 0, 2] {
                if (sbx[c] << hshift[c]) != bx || (sby[c] << vshift[c] != by) {
                    continue;
                }
                transform_to_pixels(transform_type, &mut transform_buffer[c])?;
                let mut output = pixels[c].as_rect_mut();
                let downsampled_rect = Rect {
                    origin: (
                        block_rect.origin.0 >> hshift[c],
                        block_rect.origin.1 >> vshift[c],
                    ),
                    size: block_rect.size,
                };
                let mut output_rect = output.rect(downsampled_rect)?;
                for i in 0..downsampled_rect.size.1 {
                    let offset = i * downsampled_rect.size.0;
                    output_rect.row(i).copy_from_slice(
                        &transform_buffer[c][offset..offset + downsampled_rect.size.0],
                    );
                }
            }
            coeffs_offset += num_coeffs;
        }
    }
    reader.check_final_state(&hf_global.passes[pass].histograms)?;
    Ok(pixels)
}
