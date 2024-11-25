// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

pub mod bit_depth;
pub mod color_encoding;
pub mod encodings;
pub mod extra_channels;
pub mod frame_header;
pub mod image_metadata;
pub mod modular;
pub mod permutation;
pub mod size;
pub mod transform_data;

use crate::{bit_reader::BitReader, error::Error, headers::encodings::*};
use frame_header::FrameHeaderNonserialized;
use jxl_macros::UnconditionalCoder;

pub use image_metadata::*;
pub use size::Size;
pub use transform_data::*;

#[derive(UnconditionalCoder, Debug, Clone)]
pub struct FileHeader {
    #[allow(dead_code)]
    signature: Signature,
    pub size: Size,
    pub image_metadata: ImageMetadata,
    #[nonserialized(xyb_encoded : image_metadata.xyb_encoded)]
    pub transform_data: CustomTransformData,
}

pub trait JxlHeader
where
    Self: Sized,
{
    fn read(br: &mut BitReader) -> Result<Self, Error>;
}

impl<T> JxlHeader for T
where
    T: UnconditionalCoder<()>,
    T::Nonserialized: Default,
{
    fn read(br: &mut BitReader) -> Result<Self, Error> {
        Self::read_unconditional(&(), br, &T::Nonserialized::default())
    }
}

impl FileHeader {
    pub fn frame_header_nonserialized(&self) -> FrameHeaderNonserialized {
        let have_timecode = match self.image_metadata.animation {
            Some(ref animation) => animation.have_timecodes,
            None => false,
        };
        FrameHeaderNonserialized {
            xyb_encoded: self.image_metadata.xyb_encoded,
            num_extra_channels: self.image_metadata.extra_channel_info.len() as u32,
            extra_channel_info: self.image_metadata.extra_channel_info.clone(),
            have_animation: self.image_metadata.animation.is_some(),
            have_timecode,
            img_width: self.size.xsize(),
            img_height: self.size.ysize(),
        }
    }
}
