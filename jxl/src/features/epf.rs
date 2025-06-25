// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    SIGMA_PADDING,
    error::Result,
    frame::{HfMetadata, LfGlobalState, transform_map::*},
    headers::frame_header::{Encoding, FrameHeader},
    image::Image,
};

pub fn create_sigma_image(
    frame_header: &FrameHeader,
    lf_global: &LfGlobalState,
    hf_meta: &Option<HfMetadata>,
) -> Result<Image<f32>> {
    let size_blocks = frame_header.size_blocks();
    let rf = &frame_header.restoration_filter;
    let sigma_xsize = size_blocks.0 + 2 * SIGMA_PADDING;
    let sigma_ysize = size_blocks.1 + 2 * SIGMA_PADDING;
    let mut sigma_image = Image::<f32>::new((sigma_xsize, sigma_ysize))?;
    #[allow(clippy::excessive_precision)]
    const INV_SIGMA_NUM: f32 = -1.1715728752538099024;
    if frame_header.encoding == Encoding::VarDCT {
        let hf_meta = hf_meta.as_ref().unwrap();
        let raw_quant_map = hf_meta.raw_quant_map.as_rect();
        let transform_map = hf_meta.transform_map.as_rect();
        let quant_params = lf_global.quant_params.as_ref().unwrap();
        let quant_scale = 1.0 / quant_params.inv_global_scale();
        let epf_map = hf_meta.epf_map.as_rect();
        let mut sigma_rect = sigma_image.as_rect_mut();
        for by in 0..size_blocks.1 {
            let sby = SIGMA_PADDING + by;
            for bx in 0..size_blocks.0 {
                let sbx = SIGMA_PADDING + bx;
                let raw_quant = raw_quant_map.row(by)[bx];
                let raw_transform_id = transform_map.row(by)[bx];
                let transform_id = raw_transform_id & 127;
                let is_first_block = raw_transform_id >= 128;
                if !is_first_block {
                    continue;
                }
                let transform_type = HfTransformType::from_usize(transform_id as usize)?;
                let cx = covered_blocks_x(transform_type) as usize;
                let cy = covered_blocks_y(transform_type) as usize;
                let sigma_quant =
                    rf.epf_quant_mul / (quant_scale * raw_quant as f32 * INV_SIGMA_NUM);
                for iy in 0..cy {
                    for ix in 0..cx {
                        let sharpness = epf_map.row(by + iy)[bx + ix] as usize;
                        let sigma = (sigma_quant * rf.epf_sharp_lut[sharpness]).min(-1e-4);
                        sigma_rect.row(sby + iy)[sbx + ix] = 1.0 / sigma;
                    }
                }
            }
            sigma_rect.row(sby)[SIGMA_PADDING - 1] = sigma_rect.row(sby)[SIGMA_PADDING];
            sigma_rect.row(sby)[SIGMA_PADDING + size_blocks.0] =
                sigma_rect.row(sby)[SIGMA_PADDING + size_blocks.0 - 1];
        }
        for bx in 0..sigma_xsize {
            sigma_rect.row(SIGMA_PADDING - 1)[bx] = sigma_rect.row(SIGMA_PADDING)[bx];
            sigma_rect.row(SIGMA_PADDING + size_blocks.1)[bx] =
                sigma_rect.row(SIGMA_PADDING + size_blocks.1 - 1)[bx];
        }
    } else {
        // TODO(szabadka): Instead of allocating an image, return an enum with image and f32
        // variants.
        let sigma = INV_SIGMA_NUM / rf.epf_sigma_for_modular;
        sigma_image.fill(sigma);
    }
    Ok(sigma_image)
}
