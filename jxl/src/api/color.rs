// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::borrow::Cow;

use crate::{
    error::{Error, Result},
    headers::{
        color_encoding::{
            ColorEncoding, ColorSpace, CustomXY, Primaries, RenderingIntent, TransferFunction,
            WhitePoint,
        },
        encodings::Empty,
    },
};

#[derive(Clone, Debug, PartialEq)]
pub enum JxlWhitePoint {
    D65,
    E,
    DCI,
    Chromaticity { wx: f32, wy: f32 },
}

#[derive(Clone, Debug, PartialEq)]
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

#[derive(Clone, Debug, PartialEq)]
pub enum JxlTransferFunction {
    BT709,
    Linear,
    SRGB,
    PQ,
    DCI,
    HLG,
    Gamma(f32),
}

#[derive(Clone, Debug, PartialEq)]
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

    pub fn from_internal(internal: &ColorEncoding) -> Result<Self> {
        let rendering_intent = internal.rendering_intent;
        if internal.color_space == ColorSpace::XYB {
            return Ok(Self::XYB { rendering_intent });
        }

        let white_point = match internal.white_point {
            WhitePoint::D65 => JxlWhitePoint::D65,
            WhitePoint::E => JxlWhitePoint::E,
            WhitePoint::DCI => JxlWhitePoint::DCI,
            WhitePoint::Custom => {
                let (wx, wy) = internal.white.as_f32_coords();
                JxlWhitePoint::Chromaticity { wx, wy }
            }
        };
        let transfer_function = if internal.tf.have_gamma {
            JxlTransferFunction::Gamma(internal.tf.gamma())
        } else {
            match internal.tf.transfer_function {
                TransferFunction::BT709 => JxlTransferFunction::BT709,
                TransferFunction::Linear => JxlTransferFunction::Linear,
                TransferFunction::SRGB => JxlTransferFunction::SRGB,
                TransferFunction::PQ => JxlTransferFunction::SRGB,
                TransferFunction::DCI => JxlTransferFunction::DCI,
                TransferFunction::HLG => JxlTransferFunction::HLG,
                TransferFunction::Unknown => {
                    return Err(Error::InvalidColorEncoding);
                }
            }
        };

        if internal.color_space == ColorSpace::Gray {
            return Ok(Self::GrayscaleColorSpace {
                white_point,
                transfer_function,
                rendering_intent,
            });
        }

        let primaries = match internal.primaries {
            Primaries::SRGB => JxlPrimaries::SRGB,
            Primaries::BT2100 => JxlPrimaries::BT2100,
            Primaries::P3 => JxlPrimaries::P3,
            Primaries::Custom => {
                let (rx, ry) = internal.custom_primaries[0].as_f32_coords();
                let (gx, gy) = internal.custom_primaries[1].as_f32_coords();
                let (bx, by) = internal.custom_primaries[2].as_f32_coords();
                JxlPrimaries::Chromaticities {
                    rx,
                    ry,
                    gx,
                    gy,
                    bx,
                    by,
                }
            }
        };

        match internal.color_space {
            ColorSpace::Gray | ColorSpace::XYB => unreachable!(),
            ColorSpace::RGB => Ok(Self::RgbColorSpace {
                white_point,
                primaries,
                transfer_function,
                rendering_intent,
            }),
            ColorSpace::Unknown => Err(Error::InvalidColorSpace),
        }
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

#[cfg(test)]
mod test {
    use super::{JxlColorEncoding, JxlPrimaries, JxlTransferFunction, JxlWhitePoint};
    use crate::headers::color_encoding::RenderingIntent;

    #[test]
    fn test_roundtrip() {
        for input in [
            JxlColorEncoding::RgbColorSpace {
                white_point: JxlWhitePoint::D65,
                primaries: JxlPrimaries::P3,
                transfer_function: JxlTransferFunction::HLG,
                rendering_intent: RenderingIntent::Relative,
            },
            JxlColorEncoding::GrayscaleColorSpace {
                white_point: JxlWhitePoint::Chromaticity {
                    wx: 0.3457,
                    wy: 0.3585,
                },
                transfer_function: JxlTransferFunction::Linear,
                rendering_intent: RenderingIntent::Absolute,
            },
            JxlColorEncoding::XYB {
                rendering_intent: RenderingIntent::Perceptual,
            },
        ] {
            let internal = input.to_internal().unwrap();
            let reconstructed = JxlColorEncoding::from_internal(&internal).unwrap();

            assert_eq!(input, reconstructed);

            let internal_again = reconstructed.to_internal().unwrap();

            assert_eq!(
                internal.get_color_encoding_description(),
                internal_again.get_color_encoding_description()
            );
        }
    }
}
