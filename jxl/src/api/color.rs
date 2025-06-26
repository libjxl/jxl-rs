// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::borrow::Cow;

use crate::{
    error::Result,
    headers::{
        color_encoding::{
            ColorEncoding, ColorSpace, CustomXY, Primaries, RenderingIntent, TransferFunction,
            WhitePoint,
        },
        encodings::Empty,
    },
};

#[derive(Clone)]
pub enum JxlWhitePoint {
    D65,
    E,
    DCI,
    Chromaticity { wx: f32, wy: f32 },
}

#[derive(Clone)]
pub enum JxlPrimaries {
    SRGB,
    BT2100,
    P3,
    Chromaticities {
        rx: f32,
        ry: f32,
        gx: f32,
        gy: f32,
        bx: f32,
        by: f32,
    },
}

#[derive(Clone)]
pub enum JxlTransferFunction {
    BT709,
    Linear,
    SRGB,
    PQ,
    DCI,
    HLG,
    Gamma(f32),
}

#[derive(Clone)]
pub enum JxlColorEncoding {
    RgbColorSpace {
        white_point: JxlWhitePoint,
        primaries: JxlPrimaries,
        transfer_function: JxlTransferFunction,
        rendering_intent: RenderingIntent,
    },
    GrayscaleColorSpace {
        white_point: JxlWhitePoint,
        transfer_function: JxlTransferFunction,
        rendering_intent: RenderingIntent,
    },
    XYB {
        rendering_intent: RenderingIntent,
    },
}

impl JxlColorEncoding {
    pub fn to_internal(&self) -> Result<ColorEncoding> {
        let white_point = match self {
            Self::RgbColorSpace { white_point, .. }
            | Self::GrayscaleColorSpace { white_point, .. } => white_point,
            // The white point won't actually be used; any value should do.
            Self::XYB { .. } => &JxlWhitePoint::D65,
        };
        let internal_white_point = match white_point {
            JxlWhitePoint::D65 => WhitePoint::D65,
            JxlWhitePoint::E => WhitePoint::E,
            JxlWhitePoint::DCI => WhitePoint::DCI,
            JxlWhitePoint::Chromaticity { .. } => WhitePoint::Custom,
        };
        let white_point_chromaticity = {
            let (x, y) = if let JxlWhitePoint::Chromaticity { wx, wy } = white_point {
                (*wx, *wy)
            } else {
                internal_white_point.to_xy_coords(None)?
            };
            CustomXY::from_f32_coords(x, y)
        };

        let internal_primaries = match self {
            Self::RgbColorSpace { primaries, .. } => match primaries {
                JxlPrimaries::SRGB => Primaries::SRGB,
                JxlPrimaries::BT2100 => Primaries::BT2100,
                JxlPrimaries::P3 => Primaries::P3,
                JxlPrimaries::Chromaticities { .. } => Primaries::Custom,
            },
            // Likewise, this value won't be used.
            Self::GrayscaleColorSpace { .. } | Self::XYB { .. } => Primaries::Custom,
        };
        let primaries_chromaticities = if let Self::RgbColorSpace {
            primaries:
                JxlPrimaries::Chromaticities {
                    rx,
                    ry,
                    gx,
                    gy,
                    bx,
                    by,
                },
            ..
        } = self
        {
            [
                CustomXY::from_f32_coords(*rx, *ry),
                CustomXY::from_f32_coords(*gx, *gy),
                CustomXY::from_f32_coords(*bx, *by),
            ]
        } else {
            [
                CustomXY::default(&Empty {}),
                CustomXY::default(&Empty {}),
                CustomXY::default(&Empty {}),
            ]
        };

        let mut gamma = None;
        let internal_transfer_function = match self {
            Self::RgbColorSpace {
                transfer_function, ..
            }
            | Self::GrayscaleColorSpace {
                transfer_function, ..
            } => match transfer_function {
                JxlTransferFunction::BT709 => TransferFunction::BT709,
                JxlTransferFunction::Linear => TransferFunction::Linear,
                JxlTransferFunction::SRGB => TransferFunction::SRGB,
                JxlTransferFunction::PQ => TransferFunction::PQ,
                JxlTransferFunction::DCI => TransferFunction::DCI,
                JxlTransferFunction::HLG => TransferFunction::HLG,
                JxlTransferFunction::Gamma(g) => {
                    gamma = Some((g * 10_000_000.0).round() as u32);
                    // Whereas JxlColorEncoding has a gamma only if it doesn't
                    // have another transfer function, ColorEncoding has a
                    // transfer function only if it doesn't have a gamma.
                    TransferFunction::Unknown
                }
            },
            Self::XYB { .. } => {
                gamma = Some(3333333);
                TransferFunction::Unknown
            }
        };

        let mut result = ColorEncoding::default(&Empty {});
        result.want_icc = false;
        result.color_space = match self {
            Self::RgbColorSpace { .. } => ColorSpace::RGB,
            Self::GrayscaleColorSpace { .. } => ColorSpace::Gray,
            Self::XYB { .. } => ColorSpace::XYB,
        };
        result.white_point = internal_white_point;
        result.white = white_point_chromaticity;
        result.primaries = internal_primaries;
        result.custom_primaries = primaries_chromaticities;
        result.tf.have_gamma = gamma.is_some();
        if let Some(g) = gamma {
            result.tf.gamma = g;
        }
        result.tf.transfer_function = internal_transfer_function;
        result.rendering_intent = match self {
            Self::RgbColorSpace {
                rendering_intent, ..
            }
            | Self::GrayscaleColorSpace {
                rendering_intent, ..
            }
            | Self::XYB { rendering_intent } => *rendering_intent,
        };
        Ok(result)
    }
}

#[derive(Clone)]
pub enum JxlColorProfile {
    Icc(Vec<u8>),
    Simple(JxlColorEncoding),
}

impl JxlColorProfile {
    pub fn as_icc(&self) -> Cow<Vec<u8>> {
        match self {
            Self::Icc(x) => Cow::Borrowed(x),
            Self::Simple(encoding) => Cow::Owned(
                encoding
                    .to_internal()
                    .unwrap()
                    .maybe_create_profile()
                    .unwrap()
                    .unwrap(),
            ),
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
