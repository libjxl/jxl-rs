// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(unused_variables)]
// #![warn(missing_docs)]

use std::{borrow::Cow, marker::PhantomData, mem::MaybeUninit, ops::DerefMut};

use crate::{
    error::Result,
    headers::{color_encoding::ColorEncoding, frame_header::FrameHeader},
};

mod input;
mod signature;

pub use input::*;
pub use signature::*;

/// This type represents the return value of a function that reads input from a bitstream. The
/// variant `Complete` indicates that the operation was completed successfully, and its return
/// value is available. The variant `NeedsMoreInput` indicates that more input is needed, and the
/// function should be called again. This variant comes with a `size_hint`, representing an
/// estimate of the number of additional bytes needed, and a `fallback`, representing additional
/// information that might be needed to call the function again (i.e. because it takes a decoder
/// object by value).
#[derive(Debug, PartialEq)]
pub enum ProcessingResult<T, U> {
    Complete { result: T },
    NeedsMoreInput { size_hint: usize, fallback: U },
}
pub enum JxlColorProfile {
    Icc(Vec<u8>),
    // TODO(veluca): this should probably not be the raw header representation.
    Internal(ColorEncoding),
}

impl JxlColorProfile {
    pub fn as_icc(&self) -> Cow<Vec<u8>> {
        match self {
            Self::Icc(x) => Cow::Borrowed(x),
            Self::Internal(e) => todo!(),
        }
    }
}

// TODO: do we want/need to return errors from here?
pub trait JxlCmsTransformer {
    /// Runs a single transform. The buffers each contain `num_pixels` x `num_channels` interleaved
    /// floating point (0..1) samples, where `num_channels` is the number of color channels of
    /// their respective color profiles. For CMYK data, 0 represents the maximum amount of ink
    /// while 1 represents no ink.
    fn do_transform(&mut self, input: &[f32], output: &mut [f32]);

    /// Runs a single transform in-place. The buffer contains `num_pixels` x `num_channels`
    /// interleaved floating point (0..1) samples, where `num_channels` is the number of color
    /// channels of the input and output color profiles. For CMYK data, 0 represents the maximum
    /// amount of ink while 1 represents no ink.
    fn do_transform_inplace(&mut self, inout: &mut [f32]);
}

pub trait JxlCms {
    /// Parses an ICC profile, returning a ColorEncoding and whether the ICC profile represents a
    /// CMYK profile.
    fn parse_icc(&mut self, icc: &[u8]) -> Result<(ColorEncoding, bool)>;

    /// Initializes `n` transforms (different transforms might be used in parallel) to
    /// convert from color space `input` to colorspace `output`, assuming an intensity of 1.0 for
    /// non-absolute luminance colorspaces of `intensity_target`.
    /// It is an error to not return `n` transforms.
    fn initialize_transforms(
        &mut self,
        n: usize,
        max_pixels_per_transform: usize,
        input: JxlColorProfile,
        output: JxlColorProfile,
        intensity_target: f32,
    ) -> Result<Vec<Box<dyn JxlCmsTransformer>>>;
}

pub enum JxlProgressiveMode {
    /// Renders all pixels in every call to Process.
    Eager,
    /// Renders pixels once passes are completed.
    Pass,
    /// Renders pixels only once the final frame is ready.
    FullFrame,
}

#[non_exhaustive]
pub struct JxlDecoderOptions {
    pub adjust_orientation: bool,
    pub unpremultiply_alpha: bool,
    pub render_spot_colors: bool,
    pub coalescing: bool,
    pub desired_intensity_target: Option<f32>,
    pub skip_preview: bool,
    pub progressive_mode: JxlProgressiveMode,
}

impl Default for JxlDecoderOptions {
    fn default() -> Self {
        Self {
            adjust_orientation: true,
            unpremultiply_alpha: false,
            render_spot_colors: true,
            coalescing: true,
            skip_preview: false,
            desired_intensity_target: None,
            progressive_mode: JxlProgressiveMode::Pass,
        }
    }
}

mod states {
    pub trait JxlState {}
    pub struct Initialized;
    pub struct WithImageInfo;
    pub struct WithFrameInfo;
    impl JxlState for Initialized {}
    impl JxlState for WithImageInfo {}
    impl JxlState for WithFrameInfo {}
}

use states::*;

#[allow(dead_code)]
struct JxlDecoderInner {
    // TODO more fields
    options: JxlDecoderOptions,
    cms: Option<Box<dyn JxlCms>>,
}

// Q: do we plan to add support for box decoding?
// If we do, one way is to take a callback &[u8; 4] -> Box<dyn Write>.

pub struct JxlDecoder<State: JxlState> {
    inner: JxlDecoderInner,
    _state: PhantomData<State>,
}

impl<S: JxlState> JxlDecoder<S> {
    fn wrap_inner(inner: JxlDecoderInner) -> Self {
        Self {
            inner,
            _state: PhantomData,
        }
    }

    /// Resets entirely a decoder, producing a new decoder with the same settings.
    /// This is faster than creating a new decoder in some cases.
    pub fn reset(self) -> JxlDecoder<Initialized> {
        JxlDecoder::wrap_inner(self.inner)
    }

    /// Rewinds a decoder to the start of the file, allowing past frames to be displayed again.
    pub fn rewind(self) -> JxlDecoder<Initialized> {
        JxlDecoder::wrap_inner(self.inner)
    }
}

impl JxlDecoder<Initialized> {
    pub fn new(options: JxlDecoderOptions) -> Self {
        Self::wrap_inner(JxlDecoderInner { options, cms: None })
    }

    pub fn new_with_cms(options: JxlDecoderOptions, cms: impl JxlCms + 'static) -> Self {
        Self::wrap_inner(JxlDecoderInner {
            options,
            cms: Some(Box::new(cms)),
        })
    }

    pub fn process(
        self,
        input: &mut impl JxlBitstreamInput,
    ) -> Result<ProcessingResult<JxlDecoder<WithImageInfo>, Self>> {
        todo!()
    }
}

pub enum JxlColorType {
    Grayscale,
    GrayscaleAlpha,
    Rgb,
    Rgba,
    Bgr,
    Bgra,
}

impl JxlColorType {
    pub fn samples_per_pixel(&self) -> usize {
        match self {
            Self::Grayscale => 1,
            Self::GrayscaleAlpha => 2,
            Self::Rgb | Self::Bgr => 3,
            Self::Rgba | Self::Bgra => 4,
        }
    }
}

pub enum Endianness {
    LittleEndian,
    BigEndian,
}

pub enum JxlDataFormat {
    U8 {
        bit_depth: u8,
    },
    U16 {
        endianness: Endianness,
        bit_depth: u8,
    },
    F16 {
        endianness: Endianness,
    },
    F32 {
        endianness: Endianness,
    },
}

impl JxlDataFormat {
    pub fn bytes_per_sample(&self) -> usize {
        match self {
            Self::U8 { .. } => 1,
            Self::U16 { .. } | Self::F16 { .. } => 2,
            Self::F32 { .. } => 4,
        }
    }
}

pub struct JxlPixelFormat {
    pub color_type: JxlColorType,
    // None -> ignore
    pub color_data_format: Option<JxlDataFormat>,
    pub extra_channel_format: Vec<Option<JxlDataFormat>>,
}

pub struct JxlBasicInfo {
    // TODO: fields (including for extra channels, including their names)
}

impl JxlDecoder<WithImageInfo> {
    /// Skip the next `count` frames.
    pub fn skip_frames(
        self,
        input: &mut impl JxlBitstreamInput,
        count: usize,
    ) -> Result<ProcessingResult<Self, Self>> {
        todo!()
    }

    /// Obtains the image's basic information.
    pub fn basic_info(&self) -> &JxlBasicInfo {
        todo!()
    }

    /// Retrieves the file's color profile.
    pub fn embedded_color_profile(&self) -> &JxlColorProfile {
        todo!()
    }

    /// Specifies the preferred color profile to be used for outputting data.
    /// Same semantics as JxlDecoderSetOutputColorProfile.
    pub fn set_output_color_profile(&mut self, profile: &JxlColorProfile) -> Result<()> {
        todo!()
    }

    pub fn current_pixel_format(&self) -> &JxlPixelFormat {
        todo!()
    }

    pub fn set_pixel_format(&mut self, pixel_format: JxlPixelFormat) {
        todo!()
    }

    pub fn process(
        self,
        input: &mut impl JxlBitstreamInput,
    ) -> Result<ProcessingResult<JxlDecoder<WithFrameInfo>, Self>> {
        todo!()
    }
}

// TODO: implement this for &mut [u8] and &mut [&mut [u8]] (and the corresponding MaybeUninit
// variants).
pub trait JxlOutputBuffer {
    #[allow(unsafe_code)]
    /// # Safety
    /// The returned buffers must not be populated with uninit data.
    unsafe fn get_row_buffers(
        &mut self,
        shape: (usize, usize),
        bytes_per_sample: usize,
    ) -> impl DerefMut<Target = [&mut [MaybeUninit<u8>]]>;
}

impl JxlDecoder<WithFrameInfo> {
    /// Skip the current frame.
    pub fn skip_frame(
        self,
        input: &mut impl JxlBitstreamInput,
    ) -> Result<ProcessingResult<JxlDecoder<WithImageInfo>, Self>> {
        todo!()
    }

    // TODO: don't use the raw bitstream type; include name and extra channel blend info.
    pub fn frame_header(&self) -> &FrameHeader {
        todo!()
    }

    /// Number of passes we have full data for.
    pub fn num_completed_passes(&self) -> usize {
        todo!()
    }

    /// Draws all the pixels we have data for.
    pub fn flush_pixels(&mut self) -> Result<()> {
        todo!()
    }

    /// Guarantees to populate exactly the appropriate part of the buffers.
    /// Wants one buffer for each non-ignored pixel type, i.e. color channels and each extra channel.
    pub fn process(
        self,
        input: &mut impl JxlBitstreamInput,
        buffers: &mut [impl JxlOutputBuffer], // TODO: figure out if dyn is better.
    ) -> Result<ProcessingResult<JxlDecoder<WithImageInfo>, Self>> {
        todo!()
    }
}
