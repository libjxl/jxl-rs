// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(clippy::excessive_precision)]

use crate::{
    bit_reader::BitReader,
    error::Error,
    headers::{encodings::*, extra_channels::ExtraChannelInfo},
};

use jxl_macros::UnconditionalCoder;
use num_derive::FromPrimitive;

#[derive(UnconditionalCoder, Copy, Clone, PartialEq, Debug, FromPrimitive)]
enum FrameType {
    RegularFrame = 0,
    LFFrame = 1,
    ReferenceOnly = 2,
    SkipProgressive = 3,
}

#[derive(UnconditionalCoder, Copy, Clone, PartialEq, Debug, FromPrimitive)]
enum Encoding {
    VarDCT = 0,
    Modular = 1,
}

struct Flags;

#[allow(dead_code)]
impl Flags {
    pub const ENABLE_NOISE: u64 = 1;
    pub const ENABLE_PATCHES: u64 = 2;
    pub const ENABLE_SPLINES: u64 = 0x10;
    pub const USE_LF_FRAME: u64 = 0x20;
    pub const SKIP_ADAPTIVE_LF_SMOOTHING: u64 = 0x80;
}

#[derive(UnconditionalCoder, Debug, PartialEq)]
struct Passes {
    #[coder(u2S(1, 2, 3, Bits(3) + 4))]
    #[default(1)]
    num_passes: u32,

    #[coder(u2S(0, 1, 2, Bits(1) + 3))]
    #[default(0)]
    #[condition(num_passes != 1)]
    num_ds: u32,

    #[size_coder(explicit(num_passes - 1))]
    #[coder(Bits(2))]
    #[default_element(0)]
    #[condition(num_passes != 1)]
    shift: Vec<u32>,

    #[size_coder(explicit(num_ds))]
    #[coder(u2S(1, 2, 4, 8))]
    #[default_element(1)]
    #[condition(num_passes != 1)]
    downsample: Vec<u32>,

    #[size_coder(explicit(num_ds))]
    #[coder(u2S(0, 1, 2, Bits(3)))]
    #[default_element(0)]
    #[condition(num_passes != 1)]
    last_pass: Vec<u32>,
}

#[derive(UnconditionalCoder, Copy, Clone, PartialEq, Debug, FromPrimitive)]
enum BlendingMode {
    Replace = 0,
    Add = 1,
    Blend = 2,
    AlphaWeightedAdd = 3,
    Mul = 4,
}

struct BlendingInfoNonserialized {
    num_extra_channels: u32,
    have_crop: bool,
    x0: i32,
    y0: i32,
    width: u32,
    height: u32,
    img_width: u32,
    img_height: u32,
}

#[derive(UnconditionalCoder, Debug, PartialEq, Clone)]
#[nonserialized(BlendingInfoNonserialized)]
struct BlendingInfo {
    #[coder(u2S(0, 1, 2, Bits(2) + 3))]
    #[default(BlendingMode::Replace)]
    mode: BlendingMode,

    /* Spec: "Let multi_extra be true if and only if and the number of extra channels is at least two."
    libjxl condition is num_extra_channels > 0 */
    #[coder(u2S(0, 1, 2, Bits(3) + 3))]
    #[default(0)]
    #[condition(nonserialized.num_extra_channels > 0 &&
        (mode == BlendingMode::Blend || mode == BlendingMode::AlphaWeightedAdd))]
    alpha_channel: u32,

    #[default(false)]
    #[condition(nonserialized.num_extra_channels > 0 &&
        (mode == BlendingMode::Blend || mode == BlendingMode::AlphaWeightedAdd || mode == BlendingMode::Mul))]
    clamp: bool,

    #[coder(u2S(0, 1, 2, 3))]
    #[default(0)]
    // TODO(TomasKralCZ): figure out a way of extracting this huge condition into separate variables
    /* Let full_frame be true if and only if have_crop is false or if the
    frame area given by width and height and offsets x0 and y0 completely covers the image area. */
    #[condition(mode != BlendingMode::Replace && !(!nonserialized.have_crop ||
        (nonserialized.x0 == 0 && nonserialized.y0 == 0 &&
        nonserialized.width as i64 + nonserialized.x0 as i64 >= nonserialized.img_width as i64 &&
        nonserialized.height as i64 + nonserialized.y0 as i64 >= nonserialized.img_height as i64)))]
    source: u32,
}

struct RestorationFilterNonserialized {
    encoding: Encoding,
}

#[derive(UnconditionalCoder, Debug, PartialEq)]
#[nonserialized(RestorationFilterNonserialized)]
struct RestorationFilter {
    // all_default isn't mentioned in the spec, but libjxl has it
    #[all_default]
    #[default(true)]
    all_default: bool,

    #[default(true)]
    gab: bool,

    #[default(false)]
    #[condition(gab)]
    gab_custom: bool,

    #[default(0.115169525)]
    #[condition(gab_custom)]
    gab_x_weight1: f32,

    #[default(0.061248592)]
    #[condition(gab_custom)]
    gab_x_weight2: f32,

    #[default(0.115169525)]
    #[condition(gab_custom)]
    gab_y_weight1: f32,

    #[default(0.061248592)]
    #[condition(gab_custom)]
    gab_y_weight2: f32,

    #[default(0.115169525)]
    #[condition(gab_custom)]
    gab_b_weight1: f32,

    #[default(0.061248592)]
    #[condition(gab_custom)]
    gab_b_weight2: f32,

    #[coder(Bits(2))]
    #[default(2)]
    epf_iters: u32,

    #[default(false)]
    #[condition(epf_iters > 0 && nonserialized.encoding == Encoding::VarDCT)]
    epf_sharp_custom: bool,

    #[default([0.0, 1.0 / 7.0, 2.0 / 7.0, 3.0 / 7.0, 4.0 / 7.0, 5.0 / 7.0, 6.0 / 7.0, 1.0])]
    #[condition(epf_sharp_custom)]
    epf_sharp_lut: [f32; 8],

    #[default(false)]
    #[condition(epf_iters > 0)]
    epf_weight_custom: bool,

    #[default([40.0, 5.0, 3.5])]
    #[condition(epf_weight_custom)]
    epf_channel_scale: [f32; 3],

    #[default(0.45)]
    #[condition(epf_weight_custom)]
    epf_pass1_zeroflush: f32,

    #[default(0.6)]
    #[condition(epf_weight_custom)]
    epf_pass2_zeroflush: f32,

    #[default(false)]
    #[condition(epf_iters > 0)]
    epf_sigma_custom: bool,

    #[default(0.46)]
    #[condition(epf_sigma_custom && nonserialized.encoding == Encoding::VarDCT)]
    epf_quant_mul: f32,

    #[default(0.9)]
    #[condition(epf_sigma_custom)]
    epf_pass0_sigma_scale: f32,

    #[default(6.5)]
    #[condition(epf_sigma_custom)]
    epf_pass2_sigma_scale: f32,

    #[default(2.0 / 3.0)]
    #[condition(epf_sigma_custom)]
    epf_border_sad_mul: f32,

    #[default(1.0)]
    #[condition(epf_iters > 0 && nonserialized.encoding == Encoding::Modular)]
    epf_sigma_for_modular: f32,

    #[default(Extensions::default())]
    extensions: Extensions,
}

pub struct FrameHeaderNonserialized {
    pub xyb_encoded: bool,
    pub num_extra_channels: u32,
    pub extra_channel_info: Vec<ExtraChannelInfo>,
    pub have_animation: bool,
    pub have_timecode: bool,
    pub img_width: u32,
    pub img_height: u32,
}

#[derive(UnconditionalCoder, Debug, PartialEq)]
#[nonserialized(FrameHeaderNonserialized)]
#[aligned]
#[validate]
pub struct FrameHeader {
    #[all_default]
    #[default(true)]
    all_default: bool,

    #[default(FrameType::RegularFrame)]
    frame_type: FrameType,

    #[coder(Bits(1))]
    #[default(Encoding::VarDCT)]
    encoding: Encoding,

    #[default(0)]
    flags: u64,

    #[default(false)]
    #[condition(!nonserialized.xyb_encoded)]
    do_ycbcr: bool,

    #[coder(Bits(2))]
    #[default([0, 0, 0])]
    #[condition(do_ycbcr && flags & Flags::USE_LF_FRAME == 0)]
    jpeg_upsampling: [u32; 3],

    #[coder(u2S(1, 2, 4, 8))]
    #[default(1)]
    #[condition(flags & Flags::USE_LF_FRAME == 0)]
    upsampling: u32,

    #[size_coder(explicit(nonserialized.num_extra_channels))]
    #[coder(u2S(1, 2, 4, 8))]
    #[default_element(1)]
    #[condition(flags & Flags::USE_LF_FRAME == 0)]
    ec_upsampling: Vec<u32>,

    #[coder(Bits(2))]
    #[default(1)]
    #[condition(encoding == Encoding::Modular)]
    group_size_shift: u32,

    #[coder(Bits(3))]
    #[default(3)]
    #[condition(encoding == Encoding::VarDCT && nonserialized.xyb_encoded)]
    x_qm_scale: u32,

    #[coder(Bits(3))]
    #[default(2)]
    #[condition(encoding == Encoding::VarDCT && nonserialized.xyb_encoded)]
    b_qm_scale: u32,

    #[condition(frame_type != FrameType::ReferenceOnly)]
    #[default(Passes::default())]
    passes: Passes,

    #[coder(u2S(1, 2, 3, 4))]
    #[default(0)]
    #[condition(frame_type == FrameType::LFFrame)]
    lf_level: u32,

    #[default(false)]
    #[condition(frame_type != FrameType::LFFrame)]
    have_crop: bool,

    #[coder(u2S(Bits(8), Bits(11) + 256, Bits(14) + 2304, Bits(30) + 18688))]
    #[default(0)]
    #[condition(have_crop && frame_type != FrameType::ReferenceOnly)]
    x0: i32,

    #[coder(u2S(Bits(8), Bits(11) + 256, Bits(14) + 2304, Bits(30) + 18688))]
    #[default(0)]
    #[condition(have_crop && frame_type != FrameType::ReferenceOnly)]
    y0: i32,

    #[coder(u2S(Bits(8), Bits(11) + 256, Bits(14) + 2304, Bits(30) + 18688))]
    #[default(0)]
    #[condition(have_crop)]
    width: u32,

    #[coder(u2S(Bits(8), Bits(11) + 256, Bits(14) + 2304, Bits(30) + 18688))]
    #[default(0)]
    #[condition(have_crop)]
    height: u32,

    /* "normal_frame" denotes the condition !all_default
    && (frame_type == kRegularFrame || frame_type == kSkipProgressive) */
    #[default(BlendingInfo::default())]
    #[condition(frame_type == FrameType::RegularFrame || frame_type == FrameType::SkipProgressive)]
    #[nonserialized(num_extra_channels : nonserialized.num_extra_channels,
        have_crop : have_crop, x0: x0, y0: y0, width: width, height: height,
        img_width: nonserialized.img_width, img_height: nonserialized.img_height)]
    blending_info: BlendingInfo,

    #[size_coder(explicit(nonserialized.num_extra_channels))]
    #[default_element(BlendingInfo::default())]
    #[nonserialized(num_extra_channels : nonserialized.num_extra_channels,
        have_crop : have_crop, x0: x0, y0: y0, width: width, height: height,
        img_width: nonserialized.img_width, img_height: nonserialized.img_height)]
    ec_blending_info: Vec<BlendingInfo>,

    #[coder(u2S(0, 1, Bits(8), Bits(32)))]
    #[default(0)]
    #[condition((frame_type == FrameType::RegularFrame ||
        frame_type == FrameType::SkipProgressive) && nonserialized.have_animation)]
    duration: u32,

    #[coder(Bits(32))]
    #[default(0)]
    #[condition((frame_type == FrameType::RegularFrame ||
        frame_type == FrameType::SkipProgressive) && nonserialized.have_timecode)]
    timecode: u32,

    #[default(frame_type == FrameType::RegularFrame)]
    #[condition(frame_type == FrameType::RegularFrame || frame_type == FrameType::SkipProgressive)]
    pub is_last: bool,

    #[coder(Bits(2))]
    #[default(0)]
    #[condition(frame_type != FrameType::LFFrame && !is_last)]
    save_as_reference: u32,

    // TODO(TomasKralCZ): figure out a way of extracting this huge default into separate variables
    /* The default value for save_before_ct is d_sbct = !(full_frame && (frame_type ==
    kRegularFrame || frame_type == kSkipProgressive) && blending_info.mode ==
    kReplace && (duration == 0 || save_as_reference != 0) && !is_last). */
    #[default(!(!have_crop || (x0 == 0 && y0 == 0 &&
        width as i64 + x0 as i64 >= nonserialized.img_width as i64 &&
        height as i64 + y0 as i64 >= nonserialized.img_height as i64))
        && (frame_type == FrameType::RegularFrame || frame_type == FrameType::SkipProgressive) &&
        blending_info.mode == BlendingMode::Replace && (duration == 0 || save_as_reference != 0) && !is_last)]
    #[condition(frame_type != FrameType::LFFrame)]
    save_before_ct: bool,

    name: String,

    #[default(RestorationFilter::default())]
    #[nonserialized(encoding : encoding)]
    restoration_filter: RestorationFilter,

    #[default(Extensions::default())]
    extensions: Extensions,
}

impl FrameHeader {
    fn check(&self, nonserialized: &FrameHeaderNonserialized) -> Result<(), Error> {
        if self.upsampling > 1 {
            if let Some((info, upsampling)) = nonserialized
                .extra_channel_info
                .iter()
                .zip(&self.ec_upsampling)
                .find(|(info, ec_upsampling)| {
                    ((*ec_upsampling << info.dim_shift()) < self.upsampling)
                        || (**ec_upsampling > 8)
                })
            {
                return Err(Error::InvalidEcUpsampling(
                    self.upsampling,
                    info.dim_shift(),
                    *upsampling,
                ));
            }
        }

        if self.passes.num_ds >= self.passes.num_passes {
            return Err(Error::NumPassesTooLarge(
                self.passes.num_ds,
                self.passes.num_passes,
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod test_frame_header {
    use super::*;
    use crate::util::test::test_utils::read_test_file;
    use crate::{
        bit_reader::BitReader,
        container::ContainerParser,
        headers::{FileHeaders, JxlHeader},
    };

    fn test_frame_header(image: &[u8], correct_frame_header: FrameHeader) {
        let codestream = ContainerParser::collect_codestream(image).unwrap();
        let mut br = BitReader::new(&codestream);
        let fh = FileHeaders::read(&mut br).unwrap();

        let have_timecode = match fh.image_metadata.animation {
            Some(ref a) => a.have_timecodes,
            None => false,
        };
        let frame_header = FrameHeader::read_unconditional(
            &(),
            &mut br,
            &FrameHeaderNonserialized {
                xyb_encoded: fh.image_metadata.xyb_encoded,
                num_extra_channels: fh.image_metadata.extra_channel_info.len() as u32,
                extra_channel_info: fh.image_metadata.extra_channel_info,
                have_animation: fh.image_metadata.animation.is_some(),
                have_timecode,
                img_width: fh.size.xsize(),
                img_height: fh.size.ysize(),
            },
        )
        .unwrap();

        assert_eq!(correct_frame_header, frame_header);
    }

    #[test]
    fn test_basic() {
        let bytes = read_test_file("basic.jxl").unwrap();
        test_frame_header(
            &bytes,
            // TODO(TomasKralCZ): It would be nice to use struct update syntax like this:
            // FrameHeader {
            //     fields that we actually care about
            //     ..FrameHeader::default_from_nonserialized(&nonserialized)
            // }
            FrameHeader {
                all_default: false,
                frame_type: FrameType::RegularFrame,
                encoding: Encoding::VarDCT,
                flags: 0,
                do_ycbcr: false,
                jpeg_upsampling: [0, 0, 0],
                upsampling: 1,
                ec_upsampling: vec![],
                group_size_shift: 1,
                x_qm_scale: 3,
                b_qm_scale: 2,
                passes: Passes::default(),
                lf_level: 0,
                have_crop: false,
                x0: 0,
                y0: 0,
                width: 0,
                height: 0,
                blending_info: BlendingInfo::default(),
                ec_blending_info: vec![],
                duration: 0,
                timecode: 0,
                is_last: true,
                save_as_reference: 0,
                save_before_ct: false,
                name: String::from(""),
                restoration_filter: RestorationFilter::default(),
                extensions: Extensions::default(),
            },
        );
    }

    #[test]
    fn test_extra_channel() {
        let bytes = read_test_file("extra_channels.jxl").unwrap();
        test_frame_header(
            &bytes,
            FrameHeader {
                all_default: false,
                frame_type: FrameType::RegularFrame,
                encoding: Encoding::Modular,
                flags: 0,
                do_ycbcr: false,
                jpeg_upsampling: [0, 0, 0],
                upsampling: 1,
                ec_upsampling: vec![1],
                group_size_shift: 2,
                // libjxl x_qm_scale = 2, but condition is false (should be 3 according to the draft)
                x_qm_scale: 3,
                b_qm_scale: 2,
                passes: Passes::default(),
                lf_level: 0,
                have_crop: false,
                x0: 0,
                y0: 0,
                width: 0,
                height: 0,
                blending_info: BlendingInfo::default(),
                ec_blending_info: vec![BlendingInfo::default()],
                duration: 0,
                timecode: 0,
                is_last: true,
                save_as_reference: 0,
                save_before_ct: false,
                name: String::from(""),
                restoration_filter: RestorationFilter {
                    all_default: false,
                    gab: false,
                    epf_iters: 0,
                    ..RestorationFilter::default()
                },
                extensions: Extensions::default(),
            },
        );
    }

    #[test]
    fn test_has_permutation() {
        let bytes = read_test_file("has_permutation.jxl").unwrap();
        test_frame_header(
            &bytes,
            FrameHeader {
                all_default: false,
                frame_type: FrameType::RegularFrame,
                encoding: Encoding::VarDCT,
                flags: 0,
                do_ycbcr: false,
                jpeg_upsampling: [0, 0, 0],
                upsampling: 1,
                ec_upsampling: vec![],
                group_size_shift: 1,
                x_qm_scale: 2,
                b_qm_scale: 2,
                passes: Passes::default(),
                lf_level: 0,
                have_crop: false,
                x0: 0,
                y0: 0,
                width: 0,
                height: 0,
                blending_info: BlendingInfo::default(),
                ec_blending_info: vec![],
                duration: 0,
                timecode: 0,
                is_last: true,
                save_as_reference: 0,
                save_before_ct: false,
                name: String::from(""),
                restoration_filter: RestorationFilter::default(),
                extensions: Extensions::default(),
            },
        );
    }
}
