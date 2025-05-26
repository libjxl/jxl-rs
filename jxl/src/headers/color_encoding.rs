// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{bit_reader::BitReader, error::Error, headers::encodings::*};
use jxl_macros::UnconditionalCoder;
use num_derive::FromPrimitive;

use lcms2::{CIExyY, CIExyYTRIPLE, Intent, Profile, ToneCurve};

#[allow(clippy::upper_case_acronyms)]
#[derive(UnconditionalCoder, Copy, Clone, PartialEq, Debug, FromPrimitive)]
pub enum ColorSpace {
    RGB,
    Gray,
    XYB,
    Unknown,
}

#[allow(clippy::upper_case_acronyms)]
#[derive(UnconditionalCoder, Copy, Clone, PartialEq, Debug, FromPrimitive)]
pub enum WhitePoint {
    D65 = 1,
    Custom = 2,
    E = 10,
    DCI = 11,
}

impl WhitePoint {
    pub fn as_lcms_white_point(&self, custom_xy_for_custom_case: &CustomXY) -> CIExyY {
        let (x_val, y_val) = match self {
            WhitePoint::Custom => (
                custom_xy_for_custom_case.x as f64 / 1_000_000.0,
                custom_xy_for_custom_case.y as f64 / 1_000_000.0,
            ),
            WhitePoint::D65 => (0.3127, 0.3290),
            // From https://ieeexplore.ieee.org/document/7290729 C.2 page 11
            WhitePoint::DCI => (0.314, 0.351),
            // Equal energy illuminant
            WhitePoint::E => (1.0 / 3.0, 1.0 / 3.0),
        };
        // Y is 1.0 for chromaticity of white points
        CIExyY {
            x: x_val,
            y: y_val,
            Y: 1.0,
        }
    }
}

#[allow(clippy::upper_case_acronyms)]
#[derive(UnconditionalCoder, Copy, Clone, PartialEq, Debug, FromPrimitive)]
pub enum Primaries {
    SRGB = 1,
    Custom = 2,
    BT2100 = 9,
    P3 = 11,
}

impl Primaries {
    pub fn as_lcms_primaries(
        &self,
        custom_primaries_for_custom_case: &[CustomXY; 3],
    ) -> CIExyYTRIPLE {
        match self {
            Primaries::Custom => {
                let r_xy = &custom_primaries_for_custom_case[0];
                let g_xy = &custom_primaries_for_custom_case[1];
                let b_xy = &custom_primaries_for_custom_case[2];
                CIExyYTRIPLE {
                    Red: CIExyY {
                        x: r_xy.x as f64 / 1_000_000.0,
                        y: r_xy.y as f64 / 1_000_000.0,
                        Y: 1.0,
                    },
                    Green: CIExyY {
                        x: g_xy.x as f64 / 1_000_000.0,
                        y: g_xy.y as f64 / 1_000_000.0,
                        Y: 1.0,
                    },
                    Blue: CIExyY {
                        x: b_xy.x as f64 / 1_000_000.0,
                        y: b_xy.y as f64 / 1_000_000.0,
                        Y: 1.0,
                    },
                }
            }
            Primaries::SRGB => CIExyYTRIPLE {
                Red: CIExyY {
                    x: 0.639998686,
                    y: 0.330010138,
                    Y: 1.0,
                },
                Green: CIExyY {
                    x: 0.300003784,
                    y: 0.600003357,
                    Y: 1.0,
                },
                Blue: CIExyY {
                    x: 0.150002046,
                    y: 0.059997204,
                    Y: 1.0,
                },
            },
            Primaries::BT2100 => CIExyYTRIPLE {
                // Corresponds to k2100 (Rec. BT.2020/BT.2100 primaries)
                Red: CIExyY {
                    x: 0.708,
                    y: 0.292,
                    Y: 1.0,
                },
                Green: CIExyY {
                    x: 0.170,
                    y: 0.797,
                    Y: 1.0,
                },
                Blue: CIExyY {
                    x: 0.131,
                    y: 0.046,
                    Y: 1.0,
                },
            },
            Primaries::P3 => CIExyYTRIPLE {
                // Display P3 / DCI-P3 primaries
                Red: CIExyY {
                    x: 0.680,
                    y: 0.320,
                    Y: 1.0,
                },
                Green: CIExyY {
                    x: 0.265,
                    y: 0.690,
                    Y: 1.0,
                },
                Blue: CIExyY {
                    x: 0.150,
                    y: 0.060,
                    Y: 1.0,
                },
            },
        }
    }
}

#[allow(clippy::upper_case_acronyms)]
#[derive(UnconditionalCoder, Copy, Clone, PartialEq, Debug, FromPrimitive)]
pub enum TransferFunction {
    BT709 = 1,
    Unknown = 2,
    Linear = 8,
    SRGB = 13,
    PQ = 16,
    DCI = 17,
    HLG = 18,
}

#[derive(UnconditionalCoder, Copy, Clone, PartialEq, Debug, FromPrimitive)]
pub enum RenderingIntent {
    Perceptual = 0,
    Relative,
    Saturation,
    Absolute,
}

impl RenderingIntent {
    pub fn as_lcms_intent(&self) -> Intent {
        match self {
            RenderingIntent::Perceptual => Intent::Perceptual,
            RenderingIntent::Relative => Intent::RelativeColorimetric,
            RenderingIntent::Saturation => Intent::Saturation,
            RenderingIntent::Absolute => Intent::AbsoluteColorimetric,
        }
    }
}

#[derive(UnconditionalCoder, Debug, Clone)]
pub struct CustomXY {
    #[default(0)]
    #[coder(u2S(Bits(19), Bits(19) + 524288, Bits(20) + 1048576, Bits(21) + 2097152))]
    pub x: i32,
    #[default(0)]
    #[coder(u2S(Bits(19), Bits(19) + 524288, Bits(20) + 1048576, Bits(21) + 2097152))]
    pub y: i32,
}

pub struct CustomTransferFunctionNonserialized {
    color_space: ColorSpace,
}

#[derive(UnconditionalCoder, Debug, Clone)]
#[nonserialized(CustomTransferFunctionNonserialized)]
#[validate]
pub struct CustomTransferFunction {
    #[condition(nonserialized.color_space != ColorSpace::XYB)]
    #[default(false)]
    pub have_gamma: bool,
    #[condition(have_gamma)]
    #[default(3333333)] // XYB gamma
    #[coder(Bits(24))]
    pub gamma: u32,
    #[condition(!have_gamma && nonserialized.color_space != ColorSpace::XYB)]
    #[default(TransferFunction::SRGB)]
    pub transfer_function: TransferFunction,
}

impl CustomTransferFunction {
    #[cfg(test)]
    pub fn empty() -> CustomTransferFunction {
        CustomTransferFunction {
            have_gamma: false,
            gamma: 0,
            transfer_function: TransferFunction::Unknown,
        }
    }
    pub fn gamma(&self) -> f32 {
        assert!(self.have_gamma);
        self.gamma as f32 * 0.0000001
    }

    pub fn check(&self, _: &CustomTransferFunctionNonserialized) -> Result<(), Error> {
        if self.have_gamma {
            let gamma = self.gamma();
            if gamma > 1.0 || gamma * 8192.0 < 1.0 {
                Err(Error::InvalidGamma(gamma))
            } else {
                Ok(())
            }
        } else {
            Ok(())
        }
    }

    // ToneCurve::new expects f64.
    fn gamma_as_f64(&self) -> f64 {
        assert!(self.have_gamma);
        self.gamma as f64 * 0.0000001
    }

    /// Converts this CustomTransferFunction to an lcms2::ToneCurve.
    pub fn as_lcms_tone_curve(&self) -> Result<ToneCurve, Error> {
        if self.have_gamma {
            Ok(ToneCurve::new(self.gamma_as_f64()))
        } else {
            match self.transfer_function {
                TransferFunction::SRGB => {
                    // ICC Parametric Curve Type 4 (sRGB EOTF)
                    // Parameters: [gamma, a, b, c, d]
                    let params = [
                        2.4,           // gamma
                        1.0 / 1.055,   // a
                        0.055 / 1.055, // b
                        1.0 / 12.92,   // c
                        0.04045,       // d
                    ];
                    ToneCurve::new_parametric(4, &params).map_err(Error::LcmsError)
                }
                TransferFunction::Linear => Ok(ToneCurve::new(1.0)),
                TransferFunction::BT709 => {
                    // Also uses an sRGB-like parametric curve structure (ICC Type 4)
                    // Parameters: [gamma, a, b, c, d]
                    let params = [
                        1.0 / 0.45,    // gamma (approx 2.222)
                        1.0 / 1.099,   // a
                        0.099 / 1.099, // b
                        1.0 / 4.5,     // c
                        0.081,         // d
                    ];
                    ToneCurve::new_parametric(4, &params).map_err(Error::LcmsError)
                }
                TransferFunction::DCI => {
                    // Pure gamma 2.6
                    Ok(ToneCurve::new(2.6))
                }
                TransferFunction::PQ => {
                    // TODO: check with Sami
                    // ICC Parametric Curve Type 5 (SMPTE ST 2084 PQ EOTF)
                    // Parameters are ignored by the ICC spec.
                    // The lcms2 wrapper requires 7 parameters for type 5.
                    let params = [0.0; 7]; // Dummy parameters
                    ToneCurve::new_parametric(5, &params).map_err(Error::LcmsError)
                }
                TransferFunction::HLG => {
                    // TODO: check with Sami
                    // ICC Parametric Curve Type 7 (ARIB STD-B67 HLG EOTF)
                    // Parameters are ignored by the ICC spec.
                    // The lcms2 wrapper requires 5 parameters for type 7.
                    let params = [0.0; 5]; // Dummy parameters
                    ToneCurve::new_parametric(7, &params).map_err(Error::LcmsError)
                }
                TransferFunction::Unknown => Err(Error::TransferFunctionUnknown),
            }
        }
    }
}

#[derive(UnconditionalCoder, Debug, Clone)]
#[validate]
pub struct ColorEncoding {
    #[all_default]
    // TODO(firsching): remove once we use this!
    #[allow(dead_code)]
    all_default: bool,
    #[default(false)]
    pub want_icc: bool,
    #[default(ColorSpace::RGB)]
    pub color_space: ColorSpace,
    #[condition(!want_icc && color_space != ColorSpace::XYB)]
    #[default(WhitePoint::D65)]
    pub white_point: WhitePoint,
    // TODO(veluca): can this be merged in the enum?
    #[condition(white_point == WhitePoint::Custom)]
    #[default(CustomXY::default(&field_nonserialized))]
    pub white: CustomXY,
    #[condition(!want_icc && color_space != ColorSpace::XYB && color_space != ColorSpace::Gray)]
    #[default(Primaries::SRGB)]
    pub primaries: Primaries,
    #[condition(primaries == Primaries::Custom)]
    #[default([CustomXY::default(&field_nonserialized), CustomXY::default(&field_nonserialized), CustomXY::default(&field_nonserialized)])]
    pub custom_primaries: [CustomXY; 3],
    #[condition(!want_icc)]
    #[default(CustomTransferFunction::default(&field_nonserialized))]
    #[nonserialized(color_space: color_space)]
    pub tf: CustomTransferFunction,
    #[condition(!want_icc)]
    #[default(RenderingIntent::Relative)]
    pub rendering_intent: RenderingIntent,
}

impl ColorEncoding {
    pub fn check(&self, _: &Empty) -> Result<(), Error> {
        if !self.want_icc
            && (self.color_space == ColorSpace::Unknown
                || self.tf.transfer_function == TransferFunction::Unknown)
        {
            Err(Error::InvalidColorEncoding)
        } else {
            Ok(())
        }
    }
    pub fn whitepoint_as_lcms(&self) -> CIExyY {
        self.white_point.as_lcms_white_point(&self.white)
    }

    pub fn primaries_as_lcms(&self) -> CIExyYTRIPLE {
        self.primaries.as_lcms_primaries(&self.custom_primaries)
    }

    pub fn maybe_create_profile(&self) -> Result<Option<Vec<u8>>, Error> {
        // TODO can reuse `check` above? or at least simplify logic/dedup somehow?
        if self.color_space == ColorSpace::Unknown
            || self.tf.transfer_function == TransferFunction::Unknown
        {
            return Ok(None);
        }
        if !matches!(
            self.color_space,
            ColorSpace::RGB | ColorSpace::Gray | ColorSpace::XYB
        ) {
            return Err(Error::InvalidColorSpace);
        }

        if self.color_space == ColorSpace::XYB
            && self.rendering_intent != RenderingIntent::Perceptual
        {
            return Err(Error::InvalidRenderingIntent);
        }

        match self.color_space {
            ColorSpace::RGB => {
                // transfer_function: &[&ToneCurve],
                let tone_curve = &self.tf.as_lcms_tone_curve()?;
                let mut profile = Profile::new_rgb(
                    &self.whitepoint_as_lcms(),
                    &self.primaries_as_lcms(),
                    // check if we really want repeat the tone curve here.
                    &[tone_curve, tone_curve, tone_curve],
                )
                .map_err(Error::LcmsError)?;
                // TODO add args:  rendering intent
                profile.set_header_rendering_intent(self.rendering_intent.as_lcms_intent());
                let icc_data = profile.icc().map_err(Error::LcmsError)?;
                Ok(Some(icc_data))
            }
            ColorSpace::Gray => {
                let tone_curve = &self.tf.as_lcms_tone_curve()?;
                let mut profile = Profile::new_gray(&self.whitepoint_as_lcms(), tone_curve)
                    .map_err(Error::LcmsError)?;
                profile.set_header_rendering_intent(self.rendering_intent.as_lcms_intent());
                let icc_data = profile.icc().map_err(Error::LcmsError)?;
                Ok(Some(icc_data))
            }
            ColorSpace::XYB => unimplemented!(),
            _ => unreachable!(),
        }
    }
}
