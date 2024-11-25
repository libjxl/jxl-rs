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
    image::Image,
    util::tracing_wrappers::*,
    util::CeilLog2,
};

mod predict;
mod transforms;
mod tree;

pub use predict::Predictor;
use transforms::Transform;
pub use tree::Tree;

#[allow(dead_code)]
#[derive(Debug)]
struct ChannelInfo {
    size: (usize, usize),
    shift: Option<(isize, isize)>, // None for meta-channels
}

#[allow(dead_code)]
#[derive(Debug)]
struct MetaInfo {
    channels: Vec<ChannelInfo>,
    transforms: Vec<Transform>,
}

pub struct FullModularImage {
    // TODO: decoding graph for processing global transforms
    meta_info: MetaInfo,
    global_channels: Vec<Image<i32>>,
}

impl Debug for FullModularImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[info: {:?}, global channel sizes: {:?}]",
            self.meta_info,
            self.global_channels
                .iter()
                .map(Image::size)
                .collect::<Vec<_>>()
        )
    }
}

impl FullModularImage {
    #[instrument(level = "debug", skip_all, ret)]
    pub fn read(
        header: &FrameHeader,
        modular_color_channels: usize,
        extra_channel_info: &[ExtraChannelInfo],
        global_tree: &Option<Tree>,
        br: &mut BitReader,
    ) -> Result<Self> {
        let mut channels = vec![];
        for c in 0..modular_color_channels {
            let shift = (header.hshift(c) as isize, header.vshift(c) as isize);
            let size = (header.width as usize, header.height as usize);
            channels.push(ChannelInfo {
                size: (size.0.div_ceil(1 << shift.0), size.1.div_ceil(1 << shift.1)),
                shift: Some(shift),
            });
        }

        for info in extra_channel_info {
            let shift = info.dim_shift() as isize - header.upsampling.ceil_log2() as isize;
            let size = header.size_upsampled();
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

        let meta_info = MetaInfo {
            transforms: header
                .transforms
                .iter()
                .map(|x| Transform::from_bitstream(x, 0, &channels))
                .filter(|x| !x.is_noop())
                .collect(),
            channels,
        };

        // TODO(veluca93): meta-apply transforms

        Ok(FullModularImage {
            meta_info,
            global_channels: vec![], // TODO(veluca93): read global channels
        })
    }
}
