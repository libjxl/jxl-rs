// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(clippy::excessive_precision)]

use crate::{
    bit_reader::BitReader,
    error::Error,
    headers::{encodings::*, extra_channels::ExtraChannelInfo},
    util::FloorLog2,
};

use jxl_macros::UnconditionalCoder;
use num_derive::FromPrimitive;

use super::{permutation::Permutation, Animation};

#[derive(UnconditionalCoder, Copy, Clone, PartialEq, Debug, FromPrimitive)]
enum FrameType {
    RegularFrame = 0,
    LFFrame = 1,
    ReferenceOnly = 2,
    SkipProgressive = 3,
}

#[derive(UnconditionalCoder, Copy, Clone, PartialEq, Debug, FromPrimitive)]
pub enum Encoding {
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
pub struct Passes {
    #[coder(u2S(1, 2, 3, Bits(3) + 4))]
    #[default(1)]
    pub num_passes: u32,

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

impl Passes {
    pub fn downsampling_bracket(&self, pass: usize) -> (usize, usize) {
        let mut max_shift = 2;
        let mut min_shift = 3;
        for i in 0..pass + 1 {
            for j in 0..self.num_ds as usize {
                min_shift = self.downsample[j].floor_log2();
            }
            if i + 1 == self.num_passes as usize {
                min_shift = 0
            }
            if i != pass {
                max_shift = min_shift - 1;
            }
        }
        (min_shift as usize, max_shift as usize)
    }
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

pub struct TocNonserialized {
    pub num_entries: u32,
}

pub struct PermutationNonserialized {
    pub num_entries: u32,
    pub permuted: bool,
}

#[derive(UnconditionalCoder, Debug, PartialEq)]
#[nonserialized(TocNonserialized)]
pub struct Toc {
    #[default(false)]
    pub permuted: bool,

    // Here we don't use `condition(permuted)`, because `jump_to_byte_boundary` needs to be executed in both cases
    #[default(Permutation::default())]
    #[nonserialized(num_entries: nonserialized.num_entries, permuted: permuted)]
    pub permutation: Permutation,

    #[coder(u2S(Bits(10), Bits(14) + 1024, Bits(22) + 17408, Bits(30) + 4211712))]
    #[size_coder(explicit(nonserialized.num_entries))]
    pub entries: Vec<u32>,
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

const H_SHIFT: [usize; 4] = [0, 1, 1, 0];
const V_SHIFT: [usize; 4] = [0, 1, 0, 1];

fn compute_jpeg_shift(jpeg_upsampling: &[u32], shift_table: &[usize]) -> u32 {
    jpeg_upsampling
        .iter()
        .map(|&ch| shift_table[ch as usize])
        .max()
        .unwrap_or(0) as u32
}

#[derive(UnconditionalCoder, Debug, PartialEq)]
#[nonserialized(FrameHeaderNonserialized)]
#[aligned]
#[validate]
pub struct FrameHeader {
    #[all_default]
    all_default: bool,

    #[coder(Bits(2))]
    #[default(FrameType::RegularFrame)]
    frame_type: FrameType,

    #[coder(Bits(1))]
    #[default(Encoding::VarDCT)]
    pub encoding: Encoding,

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
    pub upsampling: u32,

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
    pub passes: Passes,

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
    pub x0: i32,

    #[coder(u2S(Bits(8), Bits(11) + 256, Bits(14) + 2304, Bits(30) + 18688))]
    #[default(0)]
    #[condition(have_crop && frame_type != FrameType::ReferenceOnly)]
    pub y0: i32,

    #[coder(u2S(Bits(8), Bits(11) + 256, Bits(14) + 2304, Bits(30) + 18688))]
    #[default(0)]
    #[condition(have_crop)]
    frame_width: u32,

    #[coder(u2S(Bits(8), Bits(11) + 256, Bits(14) + 2304, Bits(30) + 18688))]
    #[default(0)]
    #[condition(have_crop)]
    frame_height: u32,

    /* "normal_frame" denotes the condition !all_default
    && (frame_type == kRegularFrame || frame_type == kSkipProgressive) */
    #[default(BlendingInfo::default())]
    #[condition(frame_type == FrameType::RegularFrame || frame_type == FrameType::SkipProgressive)]
    #[nonserialized(num_extra_channels : nonserialized.num_extra_channels,
        have_crop : have_crop, x0: x0, y0: y0, width: frame_width, height: frame_height,
        img_width: nonserialized.img_width, img_height: nonserialized.img_height)]
    blending_info: BlendingInfo,

    #[size_coder(explicit(nonserialized.num_extra_channels))]
    #[default_element(BlendingInfo::default())]
    #[nonserialized(num_extra_channels : nonserialized.num_extra_channels,
        have_crop : have_crop, x0: x0, y0: y0, width: frame_width, height: frame_height,
        img_width: nonserialized.img_width, img_height: nonserialized.img_height)]
    ec_blending_info: Vec<BlendingInfo>,

    #[coder(u2S(0, 1, Bits(8), Bits(32)))]
    #[default(0)]
    #[condition((frame_type == FrameType::RegularFrame ||
        frame_type == FrameType::SkipProgressive) && nonserialized.have_animation)]
    pub duration: u32,

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
    pub save_as_reference: u32,

    // The following 3 fields are not actually serialized, but just used as variables to help with
    // defining later conditions.
    #[default(!is_last && frame_type != FrameType::LFFrame && (duration == 0 || save_as_reference != 0))]
    #[condition(false)]
    pub can_be_referenced: bool,

    #[default(!have_crop || frame_width >= nonserialized.img_width && frame_height >= nonserialized.img_height && x0 == 0 && y0 == 0)]
    #[condition(false)]
    full_frame: bool,

    #[default(can_be_referenced && blending_info.mode == BlendingMode::Replace && full_frame &&
              (frame_type == FrameType::RegularFrame || frame_type == FrameType::SkipProgressive))]
    #[condition(false)]
    save_before_ct_def_false: bool,

    #[default(frame_type == FrameType::LFFrame)]
    #[condition(frame_type == FrameType::ReferenceOnly || save_before_ct_def_false)]
    pub save_before_ct: bool,

    name: String,

    #[default(RestorationFilter::default())]
    #[nonserialized(encoding : encoding)]
    restoration_filter: RestorationFilter,

    #[default(Extensions::default())]
    extensions: Extensions,

    #[coder(Bits(0))]
    #[default(if frame_width == 0 { nonserialized.img_width } else { frame_width })]
    #[condition(false)]
    pub width: u32,

    #[coder(Bits(0))]
    #[default(if frame_height == 0 { nonserialized.img_height } else { frame_height })]
    #[condition(false)]
    pub height: u32,

    #[coder(Bits(0))]
    #[default(compute_jpeg_shift(&jpeg_upsampling, &H_SHIFT))]
    #[condition(false)]
    pub maxhs: u32,

    #[coder(Bits(0))]
    #[default(compute_jpeg_shift(&jpeg_upsampling, &V_SHIFT))]
    #[condition(false)]
    pub maxvs: u32,
}

// TODO(firsching): remove once we use this!
#[allow(dead_code)]
impl FrameHeader {
    const GROUP_DIM: usize = 256;
    const BLOCK_DIM: usize = 8;

    pub fn group_dim(&self) -> usize {
        (Self::GROUP_DIM >> 1) << self.group_size_shift
    }
    pub fn lf_group_dim(&self) -> usize {
        self.group_dim() * Self::BLOCK_DIM
    }

    pub fn num_groups(&self) -> usize {
        self.size_groups().0 * self.size_groups().1
    }

    pub fn num_lf_groups(&self) -> usize {
        self.size_lf_groups().0 * self.size_lf_groups().1
    }

    pub fn num_toc_entries(&self) -> usize {
        let num_groups = self.num_groups();
        let num_dc_groups = self.num_lf_groups();

        if num_groups == 1 && self.passes.num_passes == 1 {
            1
        } else {
            2 + num_dc_groups + num_groups * self.passes.num_passes as usize
        }
    }

    pub fn duration(&self, animation: &Animation) -> f64 {
        (self.duration as f64) * 1000.0 * (animation.tps_denominator as f64)
            / (animation.tps_numerator as f64)
    }

    pub fn has_patches(&self) -> bool {
        self.flags & Flags::ENABLE_PATCHES != 0
    }

    pub fn has_noise(&self) -> bool {
        self.flags & Flags::ENABLE_NOISE != 0
    }

    pub fn has_splines(&self) -> bool {
        self.flags & Flags::ENABLE_SPLINES != 0
    }
    pub fn raw_hshift(&self, c: usize) -> usize {
        H_SHIFT[self.jpeg_upsampling[c] as usize]
    }
    pub fn hshift(&self, c: usize) -> usize {
        (self.maxhs as usize) - self.raw_hshift(c)
    }
    pub fn raw_vshift(&self, c: usize) -> usize {
        V_SHIFT[self.jpeg_upsampling[c] as usize]
    }
    pub fn vshift(&self, c: usize) -> usize {
        (self.maxvs as usize) - self.raw_vshift(c)
    }
    pub fn is444(&self) -> bool {
        self.hshift(0) == 0 && self.vshift(0) == 0 &&  // Cb
        self.hshift(2) == 0 && self.vshift(2) == 0 &&  // Cr
        self.hshift(1) == 0 && self.vshift(1) == 0 // Y
    }
    pub fn is420(&self) -> bool {
        self.hshift(0) == 1 && self.vshift(0) == 1 &&  // Cb
        self.hshift(2) == 1 && self.vshift(2) == 1 &&  // Cr
        self.hshift(1) == 0 && self.vshift(1) == 0 // Y
    }
    pub fn is422(&self) -> bool {
        self.hshift(0) == 1 && self.vshift(0) == 0 &&  // Cb
        self.hshift(2) == 1 && self.vshift(2) == 0 &&  // Cr
        self.hshift(1) == 0 && self.vshift(1) == 0 // Y
    }
    pub fn is440(&self) -> bool {
        self.hshift(0) == 0 && self.vshift(0) == 1 &&  // Cb
        self.hshift(2) == 0 && self.vshift(2) == 1 &&  // Cr
        self.hshift(1) == 0 && self.vshift(1) == 0 // Y
    }

    /// The dimensions of this frame, as coded in the codestream, excluding padding pixels.
    pub fn size(&self) -> (usize, usize) {
        (
            (self.width as usize).div_ceil(self.upsampling as usize),
            (self.height as usize).div_ceil(self.upsampling as usize),
        )
    }

    /// The dimensions of this frame, as coded in the codestream, in 8x8 blocks.
    pub fn size_blocks(&self) -> (usize, usize) {
        (
            self.size().0.div_ceil(Self::BLOCK_DIM << self.maxhs) << self.maxhs,
            self.size().1.div_ceil(Self::BLOCK_DIM << self.maxvs) << self.maxvs,
        )
    }

    /// The dimensions of this frame, as coded in the codestream but including padding pixels.
    pub fn size_padded(&self) -> (usize, usize) {
        if self.encoding == Encoding::Modular {
            self.size()
        } else {
            (
                self.size_blocks().0 * Self::BLOCK_DIM,
                self.size_blocks().1 * Self::BLOCK_DIM,
            )
        }
    }

    /// The dimensions of this frame, after upsampling.
    pub fn size_upsampled(&self) -> (usize, usize) {
        (self.width as usize, self.height as usize)
    }

    /// The dimensions of this frame, in groups.
    pub fn size_groups(&self) -> (usize, usize) {
        (
            self.size().0.div_ceil(self.group_dim()),
            self.size().1.div_ceil(self.group_dim()),
        )
    }

    /// The dimensions of this frame, in LF groups.
    pub fn size_lf_groups(&self) -> (usize, usize) {
        (
            self.size_blocks().0.div_ceil(self.group_dim()),
            self.size_blocks().1.div_ceil(self.group_dim()),
        )
    }

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

        if !self.save_before_ct && !self.full_frame && self.frame_type == FrameType::ReferenceOnly {
            return Err(Error::NonPatchReferenceWithCrop);
        }
        if !self.is444()
            && ((self.flags & Flags::SKIP_ADAPTIVE_LF_SMOOTHING) != 0)
            && self.encoding == Encoding::VarDCT
        {
            return Err(Error::Non444ChromaSubsampling);
        }
        Ok(())
    }
}

#[cfg(test)]
mod test_frame_header {
    use super::*;
    use crate::{
        bit_reader::BitReader,
        container::ContainerParser,
        headers::{FileHeader, JxlHeader},
    };

    fn read_frame_header_and_toc(image: &[u8]) -> Result<(FrameHeader, Toc), Error> {
        let codestream = ContainerParser::collect_codestream(image).unwrap();
        let mut br = BitReader::new(&codestream);
        let file_header = FileHeader::read(&mut br).unwrap();

        let frame_header = FrameHeader::read_unconditional(
            &(),
            &mut br,
            &file_header.frame_header_nonserialized(),
        )
        .unwrap();
        let num_toc_entries = frame_header.num_toc_entries();
        let toc = Toc::read_unconditional(
            &(),
            &mut br,
            &TocNonserialized {
                num_entries: num_toc_entries as u32,
            },
        )
        .unwrap();
        Ok((frame_header, toc))
    }

    #[test]
    fn test_basic() {
        let (frame_header, toc) =
            read_frame_header_and_toc(include_bytes!("../../resources/test/basic.jxl")).unwrap();
        assert_eq!(frame_header.frame_type, FrameType::RegularFrame);
        assert_eq!(frame_header.encoding, Encoding::VarDCT);
        assert_eq!(frame_header.flags, 0);
        assert_eq!(frame_header.upsampling, 1);
        assert_eq!(frame_header.x_qm_scale, 2);
        assert_eq!(frame_header.b_qm_scale, 2);
        assert!(!frame_header.have_crop);
        assert!(!frame_header.save_before_ct);
        assert_eq!(frame_header.name, String::from(""));
        assert_eq!(frame_header.restoration_filter.epf_iters, 1);
        assert_eq!(
            toc,
            Toc {
                permuted: false,
                permutation: Permutation::default(),
                entries: [53].to_vec(),
            }
        )
    }

    #[test]
    fn test_extra_channel() {
        let frame_header =
            read_frame_header_and_toc(include_bytes!("../../resources/test/extra_channels.jxl"))
                .unwrap()
                .0;
        assert_eq!(frame_header.frame_type, FrameType::RegularFrame);
        assert_eq!(frame_header.encoding, Encoding::Modular);
        assert_eq!(frame_header.flags, 0);
        assert_eq!(frame_header.upsampling, 1);
        assert_eq!(frame_header.ec_upsampling, vec![1]);
        // libjxl x_qm_scale = 2, but condition is false (should be 3 according to the draft)
        // Doesn't actually matter since this is modular mode and the value doesn't get used.
        assert_eq!(frame_header.x_qm_scale, 3);
        assert_eq!(frame_header.b_qm_scale, 2);
        assert!(!frame_header.have_crop);
        assert!(!frame_header.save_before_ct);
        assert_eq!(frame_header.name, String::from(""));
        assert_eq!(frame_header.restoration_filter.epf_iters, 0);
        assert!(!frame_header.restoration_filter.gab);
    }

    #[test]
    fn test_has_permutation() {
        let (frame_header, toc) =
            read_frame_header_and_toc(include_bytes!("../../resources/test/has_permutation.jxl"))
                .unwrap();
        assert_eq!(frame_header.frame_type, FrameType::RegularFrame);
        assert_eq!(frame_header.encoding, Encoding::VarDCT);
        assert_eq!(frame_header.flags, 0);
        assert_eq!(frame_header.upsampling, 1);
        assert_eq!(frame_header.x_qm_scale, 3);
        assert_eq!(frame_header.b_qm_scale, 2);
        assert!(!frame_header.have_crop);
        assert!(!frame_header.save_before_ct);
        assert_eq!(frame_header.name, String::from(""));
        assert_eq!(frame_header.restoration_filter.epf_iters, 1);
        assert_eq!(
            toc,
            Toc {
                permuted: true,
                permutation: Permutation(vec![
                    0u32, 1, 42, 48, 2, 3, 4, 5, 6, 7, 8, 9, 43, 10, 11, 12, 13, 14, 15, 16, 17,
                    44, 18, 19, 20, 21, 22, 23, 24, 25, 45, 26, 27, 28, 29, 30, 31, 32, 33, 46, 34,
                    35, 36, 37, 38, 39, 40, 41, 47,
                ]),
                entries: vec![
                    155, 992, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 9, 9, 9, 9, 9, 9, 9,
                    9, 9, 9, 9, 9, 9, 9, 9, 9, 5, 5, 5, 5, 5, 5, 5, 5, 697, 5, 5, 5, 5, 5, 60,
                ],
            },
        )
    }
}
