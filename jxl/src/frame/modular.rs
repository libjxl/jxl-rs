// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::fmt::Debug;

use crate::{
    bit_reader::BitReader,
    error::{Error, Result},
    headers::{
        extra_channels::ExtraChannelInfo, frame_header::FrameHeader, modular::GroupHeader,
        JxlHeader,
    },
    util::{tracing_wrappers::*, CeilLog2},
};

mod predict;
mod transforms;
mod tree;

pub use predict::Predictor;
use transforms::TransformStep;
pub use tree::Tree;

#[derive(Clone, PartialEq, Eq)]
struct ChannelInfo {
    size: (usize, usize),
    shift: Option<(usize, usize)>, // None for meta-channels
}

impl Debug for ChannelInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}x{}", self.size.0, self.size.1)?;
        if let Some(shift) = self.shift {
            write!(f, "(shift {},{})", shift.0, shift.1)
        } else {
            write!(f, "(meta)")
        }
    }
}

impl ChannelInfo {
    fn is_meta(&self) -> bool {
        self.shift.is_none()
    }

    fn is_meta_or_small(&self, group_dim: usize) -> bool {
        self.is_meta() || (self.size.0 <= group_dim && self.size.1 <= group_dim)
    }

    fn is_shift_in_range(&self, min: usize, max: usize) -> bool {
        assert!(min <= max);
        self.shift.is_some_and(|(a, b)| {
            let shift = a.min(b);
            min <= shift && shift <= max
        })
    }
}

#[allow(dead_code)]
#[derive(Debug)]
struct ModularBufferInfo {
    info: ChannelInfo,
    // Only accurate for output and coded channels.
    channel_id: usize,
    is_output: bool,
    is_coded: bool,
    description: String,
}

/// A modular image is a sequence of channels to which one or more transforms might have been
/// applied. We represent a modular image as a list of buffers, some of which are coded in the
/// bitstream; other buffers are obtained as the output of one of the transformation steps.
/// Some buffers are marked as `output`: those are the buffers corresponding to the pre-transform
/// image channels.
/// The buffers are internally divided in grids, matching the sizes of the groups they are coded
/// in (with appropriate shifts), or the size of the data produced by applying the appropriate
/// transforms to each of the groups in the input of the transforms.
#[allow(dead_code)]
#[derive(Debug)]
pub struct FullModularImage {
    buffer_info: Vec<ModularBufferInfo>,
    transform_steps: Vec<TransformStep>,
    // List of buffer indices of the channels of the modular image encoded in each kind of section.
    // In order, LfGlobal, LfGroup, HfGroup(pass 0), ..., HfGroup(last pass).
    section_modular_images: Vec<Vec<usize>>,
}

impl FullModularImage {
    #[instrument(level = "debug", skip_all, ret)]
    pub fn read(
        frame_header: &FrameHeader,
        modular_color_channels: usize,
        extra_channel_info: &[ExtraChannelInfo],
        global_tree: &Option<Tree>,
        br: &mut BitReader,
    ) -> Result<Self> {
        let mut channels = vec![];
        for c in 0..modular_color_channels {
            let shift = (frame_header.hshift(c), frame_header.vshift(c));
            let size = (frame_header.width as usize, frame_header.height as usize);
            channels.push(ChannelInfo {
                size: (size.0.div_ceil(1 << shift.0), size.1.div_ceil(1 << shift.1)),
                shift: Some(shift),
            });
        }

        for info in extra_channel_info {
            let shift = info
                .dim_shift()
                .checked_sub(frame_header.upsampling.ceil_log2())
                .expect("ec_upsampling >= upsampling should be checked in frame header")
                as usize;
            let size = frame_header.size_upsampled();
            let size = (size.0 >> info.dim_shift(), size.1 >> info.dim_shift());
            channels.push(ChannelInfo {
                size,
                shift: Some((shift, shift)),
            });
        }

        trace!("reading modular header");
        let header = GroupHeader::read(br)?;

        if header.use_global_tree && global_tree.is_none() {
            return Err(Error::NoGlobalTree);
        }

        let (buffer_info, transform_steps) =
            transforms::meta_apply_transforms(&channels, &header.transforms)?;

        // Assign each (channel, group) pair present in the bitstream to the section in which it will be decoded.
        let mut section_modular_images: Vec<Vec<usize>> = vec![];

        section_modular_images.push(
            buffer_info
                .iter()
                .enumerate()
                .filter(|x| x.1.is_coded)
                .filter(|x| x.1.info.is_meta_or_small(frame_header.group_dim()))
                .map(|x| x.0)
                .collect(),
        );

        section_modular_images.push(
            buffer_info
                .iter()
                .enumerate()
                .filter(|x| x.1.is_coded)
                .filter(|x| !x.1.info.is_meta_or_small(frame_header.group_dim()))
                .filter(|x| x.1.info.is_shift_in_range(3, usize::MAX))
                .map(|x| x.0)
                .collect(),
        );

        for pass in 0..frame_header.passes.num_passes as usize {
            let (min_shift, max_shift) = frame_header.passes.downsampling_bracket(pass);
            section_modular_images.push(
                buffer_info
                    .iter()
                    .enumerate()
                    .filter(|x| x.1.is_coded)
                    .filter(|x| !x.1.info.is_meta_or_small(frame_header.group_dim()))
                    .filter(|x| x.1.info.is_shift_in_range(min_shift, max_shift))
                    .map(|x| x.0)
                    .collect(),
            );
        }

        // Ensure that the channel list in each group is sorted by actual channel ID.
        for list in section_modular_images.iter_mut() {
            list.sort_by_key(|x| buffer_info[*x].channel_id);
        }

        // TODO(veluca93): prepare grids, grid sizes and dependency counts for the various grids.

        // TODO(veluca93): read global channels

        Ok(FullModularImage {
            buffer_info,
            transform_steps,
            section_modular_images,
        })
    }
}
