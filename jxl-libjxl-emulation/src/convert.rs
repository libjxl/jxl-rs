// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use jxl::api::{
    ExtraChannel, JxlBasicInfo as RsBasicInfo, JxlBitDepth as RsBitDepth,
    JxlColorEncoding as RsColorEncoding, JxlExtraChannel, JxlPrimaries as RsPrimaries,
    JxlTransferFunction as RsTransferFunction, JxlWhitePoint as RsWhitePoint, Orientation,
    RenderingIntent as RsRenderingIntent,
};

use crate::types::{
    JXL_CHANNEL_ALPHA, JXL_CHANNEL_BLACK, JXL_CHANNEL_CFA, JXL_CHANNEL_DEPTH, JXL_CHANNEL_OPTIONAL,
    JXL_CHANNEL_RESERVED0, JXL_CHANNEL_RESERVED1, JXL_CHANNEL_RESERVED2, JXL_CHANNEL_RESERVED3,
    JXL_CHANNEL_RESERVED4, JXL_CHANNEL_RESERVED5, JXL_CHANNEL_RESERVED6, JXL_CHANNEL_RESERVED7,
    JXL_CHANNEL_SELECTION_MASK, JXL_CHANNEL_SPOT_COLOR, JXL_CHANNEL_THERMAL, JXL_CHANNEL_UNKNOWN,
    JXL_COLOR_SPACE_GRAY, JXL_COLOR_SPACE_RGB, JXL_COLOR_SPACE_XYB, JXL_FALSE,
    JXL_ORIENT_ANTI_TRANSPOSE, JXL_ORIENT_FLIP_HORIZONTAL, JXL_ORIENT_FLIP_VERTICAL,
    JXL_ORIENT_IDENTITY, JXL_ORIENT_ROTATE_90_CCW, JXL_ORIENT_ROTATE_90_CW, JXL_ORIENT_ROTATE_180,
    JXL_ORIENT_TRANSPOSE, JXL_PRIMARIES_2100, JXL_PRIMARIES_CUSTOM, JXL_PRIMARIES_P3,
    JXL_PRIMARIES_SRGB, JXL_RENDERING_INTENT_ABSOLUTE, JXL_RENDERING_INTENT_PERCEPTUAL,
    JXL_RENDERING_INTENT_RELATIVE, JXL_RENDERING_INTENT_SATURATION, JXL_TRANSFER_FUNCTION_709,
    JXL_TRANSFER_FUNCTION_DCI, JXL_TRANSFER_FUNCTION_GAMMA, JXL_TRANSFER_FUNCTION_HLG,
    JXL_TRANSFER_FUNCTION_LINEAR, JXL_TRANSFER_FUNCTION_PQ, JXL_TRANSFER_FUNCTION_SRGB, JXL_TRUE,
    JXL_WHITE_POINT_CUSTOM, JXL_WHITE_POINT_D65, JXL_WHITE_POINT_DCI, JXL_WHITE_POINT_E,
    JxlAnimationHeader, JxlBasicInfo, JxlColorEncoding, JxlExtraChannelInfo, JxlExtraChannelType,
    JxlOrientation, JxlPreviewHeader, JxlPrimaries, JxlRenderingIntent, JxlTransferFunction,
    JxlWhitePoint,
};

pub fn convert_orientation(o: Orientation) -> JxlOrientation {
    match o {
        Orientation::Identity => JXL_ORIENT_IDENTITY,
        Orientation::FlipHorizontal => JXL_ORIENT_FLIP_HORIZONTAL,
        Orientation::Rotate180 => JXL_ORIENT_ROTATE_180,
        Orientation::FlipVertical => JXL_ORIENT_FLIP_VERTICAL,
        Orientation::Transpose => JXL_ORIENT_TRANSPOSE,
        Orientation::Rotate90Cw => JXL_ORIENT_ROTATE_90_CW,
        Orientation::AntiTranspose => JXL_ORIENT_ANTI_TRANSPOSE,
        Orientation::Rotate90Ccw => JXL_ORIENT_ROTATE_90_CCW,
    }
}

pub fn convert_extra_channel_type(ec: ExtraChannel) -> JxlExtraChannelType {
    use ExtraChannel::*;
    match ec {
        Alpha => JXL_CHANNEL_ALPHA,
        Depth => JXL_CHANNEL_DEPTH,
        SpotColor => JXL_CHANNEL_SPOT_COLOR,
        SelectionMask => JXL_CHANNEL_SELECTION_MASK,
        Black => JXL_CHANNEL_BLACK,
        CFA => JXL_CHANNEL_CFA,
        Thermal => JXL_CHANNEL_THERMAL,
        Reserved0 => JXL_CHANNEL_RESERVED0,
        Reserved1 => JXL_CHANNEL_RESERVED1,
        Reserved2 => JXL_CHANNEL_RESERVED2,
        Reserved3 => JXL_CHANNEL_RESERVED3,
        Reserved4 => JXL_CHANNEL_RESERVED4,
        Reserved5 => JXL_CHANNEL_RESERVED5,
        Reserved6 => JXL_CHANNEL_RESERVED6,
        Reserved7 => JXL_CHANNEL_RESERVED7,
        Unknown => JXL_CHANNEL_UNKNOWN,
        Optional => JXL_CHANNEL_OPTIONAL,
    }
}

pub fn convert_extra_channel_info(ec: &JxlExtraChannel) -> JxlExtraChannelInfo {
    let (bits_per_sample, exponent_bits_per_sample) = match ec.bit_depth {
        RsBitDepth::Int { bits_per_sample } => (bits_per_sample, 0),
        RsBitDepth::Float {
            bits_per_sample,
            exponent_bits_per_sample,
        } => (bits_per_sample, exponent_bits_per_sample),
    };
    JxlExtraChannelInfo {
        type_: convert_extra_channel_type(ec.ec_type),
        bits_per_sample,
        exponent_bits_per_sample,
        dim_shift: ec.dim_shift,
        name_length: ec.name.chars().count() as u32,
        alpha_premultiplied: if ec.alpha_associated {
            JXL_TRUE
        } else {
            JXL_FALSE
        },
        spot_color: ec.spot_color.unwrap_or([0.0; 4]),
        cfa_channel: ec.cfa_channel.unwrap_or(0),
    }
}

pub fn convert_basic_info(
    rs_info: &RsBasicInfo,
    have_container: bool,
    is_gray: bool,
    keep_orientation: bool,
    desired_intensity_target: f32,
) -> JxlBasicInfo {
    let (bits_per_sample, exponent_bits_per_sample) = match rs_info.bit_depth {
        RsBitDepth::Int { bits_per_sample } => (bits_per_sample, 0),
        RsBitDepth::Float {
            bits_per_sample,
            exponent_bits_per_sample,
        } => (bits_per_sample, exponent_bits_per_sample),
    };
    let first_alpha = rs_info
        .extra_channels
        .iter()
        .find(|ec| ec.ec_type == ExtraChannel::Alpha);
    let (alpha_bits, alpha_exponent_bits, alpha_premultiplied) = if let Some(a) = first_alpha {
        let (ab, aeb) = match a.bit_depth {
            RsBitDepth::Int { bits_per_sample } => (bits_per_sample, 0),
            RsBitDepth::Float {
                bits_per_sample,
                exponent_bits_per_sample,
            } => (bits_per_sample, exponent_bits_per_sample),
        };
        (
            ab,
            aeb,
            if a.alpha_associated {
                JXL_TRUE
            } else {
                JXL_FALSE
            },
        )
    } else {
        (0, 0, JXL_FALSE)
    };
    let (have_preview, preview) = if let Some((px, py)) = rs_info.preview_size {
        (
            JXL_TRUE,
            JxlPreviewHeader {
                xsize: px as u32,
                ysize: py as u32,
            },
        )
    } else {
        (JXL_FALSE, JxlPreviewHeader { xsize: 0, ysize: 0 })
    };
    let (have_animation, animation) = if let Some(anim) = &rs_info.animation {
        (
            JXL_TRUE,
            JxlAnimationHeader {
                tps_numerator: anim.tps_numerator,
                tps_denominator: anim.tps_denominator,
                num_loops: anim.num_loops,
                have_timecodes: if anim.have_timecodes {
                    JXL_TRUE
                } else {
                    JXL_FALSE
                },
            },
        )
    } else {
        (
            JXL_FALSE,
            JxlAnimationHeader {
                tps_numerator: 0,
                tps_denominator: 0,
                num_loops: 0,
                have_timecodes: JXL_FALSE,
            },
        )
    };

    let orientation = if keep_orientation {
        convert_orientation(rs_info.orientation)
    } else {
        JXL_ORIENT_IDENTITY
    };
    let intensity_target = if desired_intensity_target > 0.0 {
        desired_intensity_target
    } else {
        rs_info.tone_mapping.intensity_target
    };

    JxlBasicInfo {
        have_container: if have_container { JXL_TRUE } else { JXL_FALSE },
        xsize: rs_info.size.0 as u32,
        ysize: rs_info.size.1 as u32,
        bits_per_sample,
        exponent_bits_per_sample,
        intensity_target,
        min_nits: rs_info.tone_mapping.min_nits,
        relative_to_max_display: if rs_info.tone_mapping.relative_to_max_display {
            JXL_TRUE
        } else {
            JXL_FALSE
        },
        linear_below: rs_info.tone_mapping.linear_below,
        uses_original_profile: if rs_info.uses_original_profile {
            JXL_TRUE
        } else {
            JXL_FALSE
        },
        have_preview,
        have_animation,
        orientation,
        num_color_channels: if is_gray { 1 } else { 3 },
        num_extra_channels: rs_info.extra_channels.len() as u32,
        alpha_bits,
        alpha_exponent_bits,
        alpha_premultiplied,
        preview,
        animation,
        intrinsic_xsize: rs_info.size.0 as u32,
        intrinsic_ysize: rs_info.size.1 as u32,
        padding: [0u8; 100],
    }
}

pub fn to_rs_color_encoding(enc: &JxlColorEncoding) -> Option<RsColorEncoding> {
    let convert_wp = |wp: JxlWhitePoint, xy: [f64; 2]| -> Option<RsWhitePoint> {
        match wp {
            JXL_WHITE_POINT_D65 => Some(RsWhitePoint::D65),
            JXL_WHITE_POINT_E => Some(RsWhitePoint::E),
            JXL_WHITE_POINT_DCI => Some(RsWhitePoint::DCI),
            JXL_WHITE_POINT_CUSTOM => Some(RsWhitePoint::Chromaticity {
                wx: xy[0] as f32,
                wy: xy[1] as f32,
            }),
            _ => None,
        }
    };
    let convert_prim =
        |prim: JxlPrimaries, r: [f64; 2], g: [f64; 2], b: [f64; 2]| -> Option<RsPrimaries> {
            match prim {
                JXL_PRIMARIES_SRGB => Some(RsPrimaries::SRGB),
                JXL_PRIMARIES_2100 => Some(RsPrimaries::BT2100),
                JXL_PRIMARIES_P3 => Some(RsPrimaries::P3),
                JXL_PRIMARIES_CUSTOM => Some(RsPrimaries::Chromaticities {
                    rx: r[0] as f32,
                    ry: r[1] as f32,
                    gx: g[0] as f32,
                    gy: g[1] as f32,
                    bx: b[0] as f32,
                    by: b[1] as f32,
                }),
                _ => None,
            }
        };
    let convert_tf = |tf: JxlTransferFunction, gamma: f64| -> Option<RsTransferFunction> {
        match tf {
            JXL_TRANSFER_FUNCTION_709 => Some(RsTransferFunction::BT709),
            JXL_TRANSFER_FUNCTION_LINEAR => Some(RsTransferFunction::Linear),
            JXL_TRANSFER_FUNCTION_SRGB => Some(RsTransferFunction::SRGB),
            JXL_TRANSFER_FUNCTION_PQ => Some(RsTransferFunction::PQ),
            JXL_TRANSFER_FUNCTION_DCI => Some(RsTransferFunction::DCI),
            JXL_TRANSFER_FUNCTION_HLG => Some(RsTransferFunction::HLG),
            JXL_TRANSFER_FUNCTION_GAMMA => {
                if gamma > 0.0 && gamma <= 1.0 && gamma.is_finite() {
                    Some(RsTransferFunction::Gamma(gamma as f32))
                } else {
                    None
                }
            }
            _ => None,
        }
    };
    let convert_intent = |ri: JxlRenderingIntent| -> Option<RsRenderingIntent> {
        match ri {
            JXL_RENDERING_INTENT_PERCEPTUAL => Some(RsRenderingIntent::Perceptual),
            JXL_RENDERING_INTENT_RELATIVE => Some(RsRenderingIntent::Relative),
            JXL_RENDERING_INTENT_SATURATION => Some(RsRenderingIntent::Saturation),
            JXL_RENDERING_INTENT_ABSOLUTE => Some(RsRenderingIntent::Absolute),
            _ => None,
        }
    };

    let rendering_intent = convert_intent(enc.rendering_intent)?;
    match enc.color_space {
        JXL_COLOR_SPACE_RGB => Some(RsColorEncoding::RgbColorSpace {
            white_point: convert_wp(enc.white_point, enc.white_point_xy)?,
            primaries: convert_prim(
                enc.primaries,
                enc.primaries_red_xy,
                enc.primaries_green_xy,
                enc.primaries_blue_xy,
            )?,
            transfer_function: convert_tf(enc.transfer_function, enc.gamma)?,
            rendering_intent,
        }),
        JXL_COLOR_SPACE_GRAY => Some(RsColorEncoding::GrayscaleColorSpace {
            white_point: convert_wp(enc.white_point, enc.white_point_xy)?,
            transfer_function: convert_tf(enc.transfer_function, enc.gamma)?,
            rendering_intent,
        }),
        JXL_COLOR_SPACE_XYB => Some(RsColorEncoding::XYB { rendering_intent }),
        _ => None,
    }
}

pub fn convert_color_encoding(enc: &RsColorEncoding) -> JxlColorEncoding {
    let convert_wp = |wp: &RsWhitePoint| -> (JxlWhitePoint, [f64; 2]) {
        let (x, y) = wp.to_xy_coords();
        let kind = match wp {
            RsWhitePoint::D65 => JXL_WHITE_POINT_D65,
            RsWhitePoint::E => JXL_WHITE_POINT_E,
            RsWhitePoint::DCI => JXL_WHITE_POINT_DCI,
            RsWhitePoint::Chromaticity { .. } => JXL_WHITE_POINT_CUSTOM,
        };
        (kind, [x as f64, y as f64])
    };
    let convert_prim = |prim: &RsPrimaries| -> (JxlPrimaries, [f64; 2], [f64; 2], [f64; 2]) {
        let [r, g, b] = prim.to_xy_coords();
        let kind = match prim {
            RsPrimaries::SRGB => JXL_PRIMARIES_SRGB,
            RsPrimaries::BT2100 => JXL_PRIMARIES_2100,
            RsPrimaries::P3 => JXL_PRIMARIES_P3,
            RsPrimaries::Chromaticities { .. } => JXL_PRIMARIES_CUSTOM,
        };
        (
            kind,
            [r.0 as f64, r.1 as f64],
            [g.0 as f64, g.1 as f64],
            [b.0 as f64, b.1 as f64],
        )
    };
    let convert_tf = |tf: &RsTransferFunction| -> (JxlTransferFunction, f64) {
        match tf {
            RsTransferFunction::BT709 => (JXL_TRANSFER_FUNCTION_709, 0.0),
            RsTransferFunction::Linear => (JXL_TRANSFER_FUNCTION_LINEAR, 0.0),
            RsTransferFunction::SRGB => (JXL_TRANSFER_FUNCTION_SRGB, 0.0),
            RsTransferFunction::PQ => (JXL_TRANSFER_FUNCTION_PQ, 0.0),
            RsTransferFunction::DCI => (JXL_TRANSFER_FUNCTION_DCI, 0.0),
            RsTransferFunction::HLG => (JXL_TRANSFER_FUNCTION_HLG, 0.0),
            RsTransferFunction::Gamma(g) => (JXL_TRANSFER_FUNCTION_GAMMA, *g as f64),
        }
    };
    let convert_intent = |ri: &RsRenderingIntent| -> JxlRenderingIntent {
        match ri {
            RsRenderingIntent::Perceptual => JXL_RENDERING_INTENT_PERCEPTUAL,
            RsRenderingIntent::Relative => JXL_RENDERING_INTENT_RELATIVE,
            RsRenderingIntent::Saturation => JXL_RENDERING_INTENT_SATURATION,
            RsRenderingIntent::Absolute => JXL_RENDERING_INTENT_ABSOLUTE,
        }
    };

    match enc {
        RsColorEncoding::RgbColorSpace {
            white_point,
            primaries,
            transfer_function,
            rendering_intent,
        } => {
            let (wp, wp_xy) = convert_wp(white_point);
            let (prim, r_xy, g_xy, b_xy) = convert_prim(primaries);
            let (tf, gamma) = convert_tf(transfer_function);
            JxlColorEncoding {
                color_space: JXL_COLOR_SPACE_RGB,
                white_point: wp,
                white_point_xy: wp_xy,
                primaries: prim,
                primaries_red_xy: r_xy,
                primaries_green_xy: g_xy,
                primaries_blue_xy: b_xy,
                transfer_function: tf,
                gamma,
                rendering_intent: convert_intent(rendering_intent),
            }
        }
        RsColorEncoding::GrayscaleColorSpace {
            white_point,
            transfer_function,
            rendering_intent,
        } => {
            let (wp, wp_xy) = convert_wp(white_point);
            let (tf, gamma) = convert_tf(transfer_function);
            JxlColorEncoding {
                color_space: JXL_COLOR_SPACE_GRAY,
                white_point: wp,
                white_point_xy: wp_xy,
                primaries: JXL_PRIMARIES_SRGB,
                primaries_red_xy: [0.0; 2],
                primaries_green_xy: [0.0; 2],
                primaries_blue_xy: [0.0; 2],
                transfer_function: tf,
                gamma,
                rendering_intent: convert_intent(rendering_intent),
            }
        }
        RsColorEncoding::XYB { rendering_intent } => JxlColorEncoding {
            color_space: JXL_COLOR_SPACE_XYB,
            white_point: JXL_WHITE_POINT_D65,
            white_point_xy: [0.3127, 0.3290],
            primaries: JXL_PRIMARIES_SRGB,
            primaries_red_xy: [0.0; 2],
            primaries_green_xy: [0.0; 2],
            primaries_blue_xy: [0.0; 2],
            transfer_function: JXL_TRANSFER_FUNCTION_GAMMA,
            gamma: 1.0 / 3.0,
            rendering_intent: convert_intent(rendering_intent),
        },
    }
}
