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
    util::tracing_wrappers::*,
};

// TODO(firsching): move to some common place?
const MAX_NUM_REFERENCE_FRAMES : u32 = 4;

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
    pub positions: Vec<PatchPosition>,
    pub ref_positions: Vec<PatchReferencePosition>,
    blendings: Vec<PatchBlending>,
    num_patches: Vec<usize>,
    sorted_patches_y0: Vec<(usize, usize)>,
    sorted_patches_y1: Vec<(usize, usize)>,
}

impl PatchesDictionary {
    #[instrument(level = "debug", skip(br), ret, err)]
    pub fn read(br: &mut BitReader, xsize: u32, ysize: u32) -> Result<PatchesDictionary> {
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

        let mut ref_positions: Vec<PatchReferencePosition> = Vec::with_capacity(num_ref_patch as usize);
        for _ in 0..num_ref_patch {
            let reference = patches_reader.read(br, REFERENCE_FRAME_CONTEXT)?;
            if reference >= MAX_NUM_REFERENCE_FRAMES {
                return Err(Error::PatchesRefTooLarge(reference, MAX_NUM_REFERENCE_FRAMES));
            }

            // TODO: fill in correct numbers here
            ref_positions.push(PatchReferencePosition {reference:0, x0:0, y0:0, xsize:0, ysize: 0})

        }

        todo!("implement patch decoding")
    }
}
