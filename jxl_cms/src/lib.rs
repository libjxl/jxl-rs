// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

pub mod lcms2;

use jxl::api::JxlColorProfile;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("lcms2 failed to parse input ICC: {0}")]
    Lcms2InputParseError(String),
    #[error("lcms2 failed to parse output ICC: {0}")]
    Lcms2OutputParseError(String),
    #[error("lcms2 failed to create transform: {0}")]
    Lcms2TransformError(String),
    #[error("Cannot create ICC for input profile")]
    InputIccError,
    #[error("Cannot create ICC for output profile")]
    OutputIccError,
    #[error("Output buffer too small: expected {0}, got {1}")]
    OutputBufferTooSmall(usize, usize),
}

pub type Result<T> = std::result::Result<T, Error>;

pub trait JxlCmsTransformer {
    /// Runs a single transform. The buffers each contain `num_pixels` x `num_channels` interleaved
    /// floating point (0..1) samples, where `num_channels` is the number of color channels of
    /// their respective color profiles. For CMYK data, 0 represents the maximum amount of ink
    /// while 1 represents no ink.
    fn do_transform(&mut self, input: &[f32], output: &mut [f32]) -> Result<()>;
}

pub trait JxlCms {
    /// Initializes `n` transforms (different transforms might be used in parallel) to
    /// convert from color space `input` to colorspace `output`, assuming an intensity of 1.0 for
    /// non-absolute luminance colorspaces of `intensity_target`.
    /// It is an error to not return `n` transforms.
    /// Returns the number of channels the ICC outputs, and the transforms.
    fn initialize_transforms(
        &self,
        n: usize,
        max_pixels_per_transform: usize,
        input: JxlColorProfile,
        output: JxlColorProfile,
        intensity_target: f32,
    ) -> Result<(usize, Vec<Box<dyn JxlCmsTransformer + Send>>)>;
}
