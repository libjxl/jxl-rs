// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO(firsching): remove once we use this!
#![allow(dead_code)]

use crate::{
    bit_reader::BitReader,
    entropy_coding::decode::Histograms,
    error::{Error, Result},
    image::Image,
    util::tracing_wrappers::*,
};

// TODO(firsching): move to some common place?
const MAX_NUM_REFERENCE_FRAMES: u32 = 4;

// Context numbers as specified in Section C.4.5, Listing C.2:
const NUM_REF_PATCH_CONTEXT: usize = 0;
const REFERENCE_FRAME_CONTEXT: usize = 1;
const PATCH_SIZE_CONTEXT: usize = 2;
const PATCH_REFERENCE_POSITION_CONTEXT: usize = 3;
const PATCH_POSITION_CONTEXT: usize = 4;
const PATCH_BLEND_MODE_CONTEXT: usize = 5;
const PATCH_OFFSET_CONTEXT: usize = 6;
const PATCH_COUNT_CONTEXT: usize = 7;
const PATCH_ALPHA_CHANNEL_CONTEXT: usize = 8;
const PATCH_CLAMP_CONTEXT: usize = 9;
const NUM_PATCH_DICTIONARY_CONTEXTS: usize = 10;

// Blend modes
// The new values are the old ones. Useful to skip some channels.
const PATCH_BLEND_MODE_NONE: u8 = 0;
// The new values (in the crop) replace the old ones: sample = new
const PATCH_BLEND_MODE_REPLACE: u8 = 1;
// The new values (in the crop) get added to the old ones: sample = old + new
const PATCH_BLEND_MODE_ADD: u8 = 2;
// The new values (in the crop) get multiplied by the old ones:
// sample = old * new
// This blend mode is only supported if BlendColorSpace is kEncoded. The
// range of the new value matters for multiplication purposes, and its
// nominal range of 0..1 is computed the same way as this is done for the
// alpha values in kBlend and kAlphaWeightedAdd.
const PATCH_BLEND_MODE_MUL: u8 = 3;
// The new values (in the crop) replace the old ones if alpha>0:
// For first alpha channel:
// alpha = old + new * (1 - old)
// For other channels if !alpha_associated:
// sample = ((1 - new_alpha) * old * old_alpha + new_alpha * new) / alpha
// For other channels if alpha_associated:
// sample = (1 - new_alpha) * old + new
// The alpha formula applies to the alpha used for the division in the other
// channels formula, and applies to the alpha channel itself if its
// blend_channel value matches itself.
// If using kBlendAbove, new is the patch and old is the original image; if
// using kBlendBelow, the meaning is inverted.
const PATCH_BLEND_MODE_BLEND_ABOVE: u8 = 4;
const PATCH_BLEND_MODE_BLEND_BELOW: u8 = 5;
// The new values (in the crop) are added to the old ones if alpha>0:
// For first alpha channel: sample = sample = old + new * (1 - old)
// For other channels: sample = old + alpha * new
const PATCH_BLEND_MODE_ALPHA_WEIGHTED_ADD_ABOVE: u8 = 6;
const PATCH_BLEND_MODE_ALPHA_WEIGHTED_ADD_BELOW: u8 = 7;
const PATCH_BLEND_MODE_NUM_BLEND_MODES: u8 = 8;

#[derive(Debug, Clone, Copy)]
struct PatchBlending {
    mode: u8,
    alpha_channel: u32,
    clamp: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct PatchReferencePosition {
    // Not using `ref` like in the spec here, because it is a keyword.
    reference: u8,
    x0: usize,
    y0: usize,
    xsize: usize,
    ysize: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PatchPosition {
    x: usize,
    y: usize,
    ref_pos_idx: usize,
}

#[derive(Debug, Default)]
pub struct PatchesDictionary {
    pub reference_frames: [Option<Box<Image<u8>>>; 4],
    pub positions: Vec<PatchPosition>,
    ref_positions: Vec<PatchReferencePosition>,
    blendings: Vec<PatchBlending>,
    blendings_stride: u32,
}

impl PatchesDictionary {
    #[instrument(level = "debug", skip(br), ret, err)]
    pub fn read(
        br: &mut BitReader,
        xsize: u32,
        ysize: u32,
        num_extra_channels: u32,
        reference_frames: [Option<Box<Image<u8>>>; 4],
    ) -> Result<PatchesDictionary> {
        let blendings_stride = num_extra_channels + 1;
        trace!(pos = br.total_bits_read());
        let patches_histograms = Histograms::decode(NUM_PATCH_DICTIONARY_CONTEXTS, br, true)?;
        let mut patches_reader = patches_histograms.make_reader(br)?;
        let num_ref_patch = 1 + patches_reader.read(br, NUM_REF_PATCH_CONTEXT)?;
        let num_pixels = xsize * ysize;
        let max_ref_patches = 1024 + num_pixels / 4;
        let max_patches = max_ref_patches * 4;
        let _max_blending_infos = max_patches * 4;
        if num_ref_patch > max_ref_patches {
            return Err(Error::PatchesTooMany(num_ref_patch, max_ref_patches));
        }
        let mut total_patches = 0;
        let mut next_size = 1;
        let mut positions: Vec<PatchPosition> = Vec::new();
        let mut blendings = Vec::new();
        let mut ref_positions: Vec<PatchReferencePosition> =
            Vec::with_capacity(num_ref_patch as usize);
        for _ in 0..num_ref_patch {
            let reference = patches_reader.read(br, REFERENCE_FRAME_CONTEXT)?;
            if reference >= MAX_NUM_REFERENCE_FRAMES {
                return Err(Error::PatchesRefTooLarge(
                    reference,
                    MAX_NUM_REFERENCE_FRAMES,
                ));
            }
            // TODO: add check if we are after xyb color transform?
            // let image = reference_frames[ref_positions[0].reference as usize];
            // add checks that use that image here?
            // TODO: fill in correct numbers here
            let x0 = patches_reader.read(br, PATCH_REFERENCE_POSITION_CONTEXT)? as usize;
            let y0 = patches_reader.read(br, PATCH_REFERENCE_POSITION_CONTEXT)? as usize;
            let ref_pos_xsize = patches_reader.read(br, PATCH_SIZE_CONTEXT)? as usize + 1;
            let ref_pos_ysize = patches_reader.read(br, PATCH_SIZE_CONTEXT)? as usize + 1;

            // TODO: add check : ref_pos.x0 + ref_pos.xsize > ib.xsize()
            // TODO: add check : ref_pos.y0 + ref_pos.ysize > ib.ysize()

            let id_count = patches_reader.read(br, PATCH_COUNT_CONTEXT)? + 1;
            if id_count > max_patches + 1 {
                return Err(Error::PatchesTooMany(id_count, max_patches));
            }
            total_patches += id_count;

            if total_patches > max_patches {
                return Err(Error::PatchesTooMany(total_patches, max_patches));
            }

            if next_size < total_patches {
                next_size *= 2;
                next_size = std::cmp::min(next_size, max_patches);
            }
            positions.reserve(next_size as usize);
            blendings.reserve(next_size as usize * PATCH_BLEND_MODE_NUM_BLEND_MODES as usize);

            for _ in 0..id_count {
                let mut pos = PatchPosition {
                    x: 0,
                    y: 0,
                    ref_pos_idx: ref_positions.len(),
                };
                if positions.is_empty() {
                    // Read initial position
                    pos.x = patches_reader.read(br, PATCH_POSITION_CONTEXT)? as usize;
                    pos.y = patches_reader.read(br, PATCH_POSITION_CONTEXT)? as usize;
                } else {
                    // Read offsets and calculate new position
                    let delta_x = patches_reader.read_signed(br, PATCH_OFFSET_CONTEXT)?;
                    if delta_x < 0 && (-delta_x as usize) > positions.last().unwrap().x {
                        return Err(Error::PatchesInvalidDelta(
                            "x".to_string(),
                            positions.last().unwrap().x,
                            delta_x,
                        ));
                    }
                    pos.x = (positions.last().unwrap().x as i32 + delta_x) as usize;

                    let delta_y = patches_reader.read_signed(br, PATCH_OFFSET_CONTEXT)?;
                    if delta_y < 0 && (-delta_y as usize) > positions.last().unwrap().y {
                        return Err(Error::PatchesInvalidDelta(
                            "y".to_string(),
                            positions.last().unwrap().y,
                            delta_y,
                        ));
                    }
                    pos.y = (positions.last().unwrap().y as i32 + delta_y) as usize;
                }

                if pos.x + ref_pos_xsize > xsize as usize {
                    return Err(Error::PatchesOutOfBounds(
                        "x".to_string(),
                        pos.x,
                        ref_pos_xsize,
                        xsize,
                    ));
                }
                if pos.y + ref_pos_ysize > ysize as usize {
                    return Err(Error::PatchesOutOfBounds(
                        "y".to_string(),
                        pos.y,
                        ref_pos_ysize,
                        ysize,
                    ));
                }

                let mut alpha_channel = 0;
                let mut clamp = false;
                for _ in 0..blendings_stride {
                    let blend_mode = patches_reader.read(br, PATCH_BLEND_MODE_CONTEXT)? as u8;
                    if blend_mode >= PATCH_BLEND_MODE_NUM_BLEND_MODES {
                        return Err(Error::PatchesInvalidBlendMode(
                            blend_mode,
                            PATCH_BLEND_MODE_NUM_BLEND_MODES,
                        ));
                    }

                    if blend_mode == PATCH_BLEND_MODE_BLEND_ABOVE
                        || blend_mode == PATCH_BLEND_MODE_BLEND_BELOW
                    {
                        alpha_channel =
                            patches_reader.read(br, PATCH_ALPHA_CHANNEL_CONTEXT)? as u32;
                        if alpha_channel >= PATCH_BLEND_MODE_NUM_BLEND_MODES as u32 {
                            return Err(Error::PatchesInvalidAlphaChannel(
                                alpha_channel,
                                num_extra_channels,
                            ));
                        }
                    }

                    if blend_mode == PATCH_BLEND_MODE_ADD || blend_mode == PATCH_BLEND_MODE_MUL {
                        clamp = patches_reader.read(br, PATCH_CLAMP_CONTEXT)? != 0;
                    }
                    blendings.push(PatchBlending {
                        mode: blend_mode,
                        alpha_channel,
                        clamp,
                    });
                }
                positions.push(pos);
            }

            ref_positions.push(PatchReferencePosition {
                reference: 0,
                x0,
                y0,
                xsize: ref_pos_xsize,
                ysize: ref_pos_ysize,
            })
        }
        Ok(PatchesDictionary {
            reference_frames,
            positions,
            blendings,
            ref_positions,
            blendings_stride,
        })
    }
}
