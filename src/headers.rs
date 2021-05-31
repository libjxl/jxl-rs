// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

extern crate jxl_headers_derive;

use jxl_headers_derive::UnconditionalCoder;

pub mod bit_depth;
pub mod color_encoding;
pub mod encodings;
pub mod extra_channels;
pub mod image_metadata;
pub mod size;

use crate::bit_reader::BitReader;
use crate::error::Error;
use crate::headers::encodings::Empty;
use crate::headers::encodings::UnconditionalCoder;

pub use image_metadata::*;
pub use size::Size;

#[derive(UnconditionalCoder, Debug)]
pub struct FileHeaders {
    #[allow(dead_code)]
    signature: Signature,
    pub size: Size,
    pub image_metadata: ImageMetadata,
    // transform_data: CustomTransformData,
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
