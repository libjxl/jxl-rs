// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::ffi::c_void;

use jxl::api::{
    AlphaMode, Event, JxlBasicInfo as RsBasicInfo, JxlBitDepth as RsBitDepth, JxlBitstreamInput,
    JxlColorEncoding as RsColorEncoding, JxlColorProfile as RsColorProfile, JxlColorType,
    JxlDataFormat, JxlDecoder as RsDecoder, JxlDecoderOptions, JxlOutputBuffer,
    JxlParallelRunner as RsParallelRunner, JxlPixelFormat as RsPixelFormat,
    JxlPrimaries as RsPrimaries, JxlSignature as RsSignature,
    JxlTransferFunction as RsTransferFunction, JxlWhitePoint as RsWhitePoint, Orientation,
    ProfileLevel, RenderingIntent as RsRenderingIntent, check_signature,
};

use crate::parallel_runner::CParallelRunner;
use crate::pixel_buffer::{ImageOutTarget, build_rs_pixel_format, compute_buffer_size};
use crate::types::{
    JXL_BIT_DEPTH_FROM_PIXEL_FORMAT, JXL_BLEND_REPLACE, JXL_CHANNEL_ALPHA, JXL_CHANNEL_BLACK,
    JXL_CHANNEL_CFA, JXL_CHANNEL_DEPTH, JXL_CHANNEL_OPTIONAL, JXL_CHANNEL_RESERVED0,
    JXL_CHANNEL_RESERVED1, JXL_CHANNEL_RESERVED2, JXL_CHANNEL_RESERVED3, JXL_CHANNEL_RESERVED4,
    JXL_CHANNEL_RESERVED5, JXL_CHANNEL_RESERVED6, JXL_CHANNEL_RESERVED7,
    JXL_CHANNEL_SELECTION_MASK, JXL_CHANNEL_SPOT_COLOR, JXL_CHANNEL_THERMAL, JXL_CHANNEL_UNKNOWN,
    JXL_COLOR_PROFILE_TARGET_DATA, JXL_COLOR_PROFILE_TARGET_ORIGINAL, JXL_COLOR_SPACE_GRAY,
    JXL_COLOR_SPACE_RGB, JXL_COLOR_SPACE_XYB, JXL_DEC_BASIC_INFO, JXL_DEC_BOX,
    JXL_DEC_BOX_COMPLETE, JXL_DEC_COLOR_ENCODING, JXL_DEC_ERROR, JXL_DEC_FRAME,
    JXL_DEC_FULL_IMAGE, JXL_DEC_NEED_IMAGE_OUT_BUFFER, JXL_DEC_NEED_MORE_INPUT,
    JXL_DEC_NEED_PREVIEW_OUT_BUFFER, JXL_DEC_PREVIEW_IMAGE, JXL_DEC_SUCCESS, JXL_FALSE, JXL_ORIENT_ANTI_TRANSPOSE, JXL_ORIENT_FLIP_HORIZONTAL, JXL_ORIENT_FLIP_VERTICAL, JXL_ORIENT_IDENTITY,
    JXL_ORIENT_ROTATE_90_CCW, JXL_ORIENT_ROTATE_90_CW, JXL_ORIENT_ROTATE_180, JXL_ORIENT_TRANSPOSE,
    JXL_PRIMARIES_2100, JXL_PRIMARIES_CUSTOM, JXL_PRIMARIES_P3, JXL_PRIMARIES_SRGB,
    JXL_RENDERING_INTENT_ABSOLUTE, JXL_RENDERING_INTENT_PERCEPTUAL, JXL_RENDERING_INTENT_RELATIVE,
    JXL_RENDERING_INTENT_SATURATION, JXL_TRANSFER_FUNCTION_709, JXL_TRANSFER_FUNCTION_DCI,
    JXL_TRANSFER_FUNCTION_GAMMA, JXL_TRANSFER_FUNCTION_HLG, JXL_TRANSFER_FUNCTION_LINEAR,
    JXL_TRANSFER_FUNCTION_PQ, JXL_TRANSFER_FUNCTION_SRGB, JXL_TRUE, JXL_WHITE_POINT_CUSTOM,
    JXL_WHITE_POINT_D65, JXL_WHITE_POINT_DCI, JXL_WHITE_POINT_E, JxlAnimationHeader, JxlBasicInfo,
    JxlBitDepth, JxlBlendInfo, JxlColorEncoding, JxlColorProfileTarget, JxlDecoderStatus,
    JxlExtraChannelInfo, JxlExtraChannelType, JxlFrameHeader, JxlImageOutCallback,
    JxlImageOutDestroyCallback, JxlImageOutInitCallback, JxlImageOutRunCallback, JxlLayerInfo,
    JxlOrientation, JxlPixelFormat, JxlPreviewHeader, JxlPrimaries,
    JxlRenderingIntent, JxlTransferFunction, JxlWhitePoint,
};

const CODESTREAM_EVENTS_MASK: i32 = (JXL_DEC_BASIC_INFO.0 as i32)
    | (JXL_DEC_COLOR_ENCODING.0 as i32)
    | (JXL_DEC_PREVIEW_IMAGE.0 as i32)
    | (JXL_DEC_FRAME.0 as i32)
    | (JXL_DEC_FULL_IMAGE.0 as i32);

const SUPPORTED_EVENTS_MASK: i32 =
    CODESTREAM_EVENTS_MASK | (JXL_DEC_BOX.0 as i32) | (JXL_DEC_BOX_COMPLETE.0 as i32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stage {
    BeforeBasicInfo,
    EmitBasicInfo,
    EmitColorEncoding,
    PreviewNeedOutBuffer,
    PreviewFrameHeader,
    PreviewDecode,
    MainFrameHeader,
    MainEmitFrame,
    MainNeedOutBuffer,
    MainDecode,
    Done,
}

struct TrackingInput<'a> {
    next_in: &'a mut *const u8,
    avail_in: &'a mut usize,
}

impl JxlBitstreamInput for TrackingInput<'_> {
    fn available_bytes(&mut self) -> std::io::Result<usize> {
        Ok(*self.avail_in)
    }

    fn read(&mut self, bufs: &mut [std::io::IoSliceMut<'_>]) -> std::io::Result<usize> {
        let mut total_written = 0;
        for buf in bufs.iter_mut() {
            if *self.avail_in == 0 || self.next_in.is_null() {
                break;
            }
            let to_copy = buf.len().min(*self.avail_in);
            // SAFETY: `self.next_in` was provided by the caller via `JxlDecoderSetInput` with at
            // least `self.avail_in` readable bytes.
            let slice = unsafe { std::slice::from_raw_parts(*self.next_in, to_copy) };
            buf[..to_copy].copy_from_slice(slice);
            // SAFETY: Advancing `self.next_in` by `to_copy <= self.avail_in` stays within bounds.
            *self.next_in = unsafe { self.next_in.add(to_copy) };
            *self.avail_in -= to_copy;
            total_written += to_copy;
            if to_copy < buf.len() {
                break;
            }
        }
        Ok(total_written)
    }

    fn skip(&mut self, bytes: usize) -> std::io::Result<usize> {
        if *self.avail_in == 0 || self.next_in.is_null() {
            return Ok(0);
        }
        let to_skip = bytes.min(*self.avail_in);
        // SAFETY: Advancing `self.next_in` by `to_skip <= self.avail_in` stays within bounds.
        *self.next_in = unsafe { self.next_in.add(to_skip) };
        *self.avail_in -= to_skip;
        Ok(to_skip)
    }
}

pub struct JxlDecoder {
    rs_decoder: Option<RsDecoder>,
    stage: Stage,
    process_started: bool,

    // Input state
    next_in: *const u8,
    avail_in: usize,
    input_is_set: bool,
    input_closed: bool,
    have_container: Option<bool>,
    signature_probe: Vec<u8>,
    basic_info_size_hint: usize,

    // Subscribed events
    events_wanted: i32,

    // Options
    keep_orientation: bool,
    render_spotcolors: bool,
    coalescing: bool,
    unpremultiply_alpha: bool,
    desired_intensity_target: f32,
    output_bit_depth: JxlBitDepth,
    parallel_runner: Option<CParallelRunner>,

    // Decoded state
    preview_done: bool,
    main_frames_started: usize,
    main_decode_started: bool,
    frames_to_skip: usize,
    skip_current_frame: bool,
    has_more_frames: bool,

    // Output targets
    preview_out_target: Option<ImageOutTarget>,
    image_out_target: Option<ImageOutTarget>,
    extra_channel_targets: Vec<Option<ImageOutTarget>>,
}

impl Default for JxlDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl JxlDecoder {
    pub fn new() -> Self {
        Self {
            rs_decoder: None,
            stage: Stage::BeforeBasicInfo,
            process_started: false,
            next_in: std::ptr::null(),
            avail_in: 0,
            input_is_set: false,
            input_closed: false,
            have_container: None,
            signature_probe: Vec::new(),
            basic_info_size_hint: 64,
            events_wanted: 0,
            keep_orientation: false,
            render_spotcolors: true,
            coalescing: true,
            unpremultiply_alpha: false,
            desired_intensity_target: 0.0,
            output_bit_depth: JxlBitDepth {
                type_: JXL_BIT_DEPTH_FROM_PIXEL_FORMAT,
                bits_per_sample: 0,
                exponent_bits_per_sample: 0,
            },
            parallel_runner: None,
            preview_done: false,
            main_frames_started: 0,
            main_decode_started: false,
            frames_to_skip: 0,
            skip_current_frame: false,
            has_more_frames: true,
            preview_out_target: None,
            image_out_target: None,
            extra_channel_targets: Vec::new(),
        }
    }

    /// Resets the decoder completely back to initial creation state (`JxlDecoderReset`).
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Rewinds the decoder to the beginning of the file while preserving settings and subscribed
    /// events (`JxlDecoderRewind`).
    pub fn rewind(&mut self) {
        self.rs_decoder = None;
        self.stage = Stage::BeforeBasicInfo;
        self.process_started = false;
        self.next_in = std::ptr::null();
        self.avail_in = 0;
        self.input_is_set = false;
        self.input_closed = false;
        self.have_container = None;
        self.signature_probe.clear();
        self.basic_info_size_hint = 64;
        self.preview_done = false;
        self.main_frames_started = 0;
        self.main_decode_started = false;
        self.frames_to_skip = 0;
        self.skip_current_frame = false;
        self.has_more_frames = true;
        self.preview_out_target = None;
        if matches!(self.image_out_target, Some(ImageOutTarget::Buffer { .. })) {
            self.image_out_target = None;
        }
        self.extra_channel_targets.clear();
    }

    pub fn size_hint_basic_info(&self) -> usize {
        if self.has_basic_info() {
            0
        } else {
            self.basic_info_size_hint
        }
    }

    pub fn subscribe_events(&mut self, events_wanted: i32) -> JxlDecoderStatus {
        if self.process_started {
            return JXL_DEC_ERROR;
        }
        if (events_wanted & !SUPPORTED_EVENTS_MASK) != 0 {
            return JXL_DEC_ERROR;
        }
        let combined = self.events_wanted | events_wanted;
        if combined != 0 && (combined & CODESTREAM_EVENTS_MASK) == 0 {
            return JXL_DEC_ERROR;
        }
        self.events_wanted = combined;
        JXL_DEC_SUCCESS
    }

    pub fn set_keep_orientation(&mut self, keep_orientation: bool) -> JxlDecoderStatus {
        if self.process_started {
            return JXL_DEC_ERROR;
        }
        self.keep_orientation = keep_orientation;
        JXL_DEC_SUCCESS
    }

    pub fn set_unpremultiply_alpha(&mut self, unpremultiply_alpha: bool) -> JxlDecoderStatus {
        if self.process_started {
            return JXL_DEC_ERROR;
        }
        self.unpremultiply_alpha = unpremultiply_alpha;
        JXL_DEC_SUCCESS
    }

    pub fn set_render_spotcolors(&mut self, render_spotcolors: bool) -> JxlDecoderStatus {
        if self.process_started {
            return JXL_DEC_ERROR;
        }
        self.render_spotcolors = render_spotcolors;
        JXL_DEC_SUCCESS
    }

    pub fn set_coalescing(&mut self, coalescing: bool) -> JxlDecoderStatus {
        if self.process_started || !coalescing {
            return JXL_DEC_ERROR;
        }
        self.coalescing = coalescing;
        JXL_DEC_SUCCESS
    }

    pub fn set_desired_intensity_target(
        &mut self,
        desired_intensity_target: f32,
    ) -> JxlDecoderStatus {
        if self.process_started {
            return JXL_DEC_ERROR;
        }
        if desired_intensity_target < 0.0 || !desired_intensity_target.is_finite() {
            return JXL_DEC_ERROR;
        }
        self.desired_intensity_target = desired_intensity_target;
        JXL_DEC_SUCCESS
    }

    pub fn set_output_bit_depth(&mut self, bit_depth: &JxlBitDepth) -> JxlDecoderStatus {
        if self.main_decode_started {
            return JXL_DEC_ERROR;
        }
        self.output_bit_depth = *bit_depth;
        if let Some(fmt) = self.desired_main_rs_pixel_format()
            && let Some(dec) = self.rs_decoder.as_mut()
            && dec.set_pixel_format(fmt).is_err()
        {
            return JXL_DEC_ERROR;
        }
        JXL_DEC_SUCCESS
    }

    pub fn set_decompress_boxes(&mut self) -> JxlDecoderStatus {
        if self.process_started {
            return JXL_DEC_ERROR;
        }
        JXL_DEC_SUCCESS
    }

    pub fn set_parallel_runner(&mut self, runner: Option<CParallelRunner>) -> JxlDecoderStatus {
        if self.process_started {
            return JXL_DEC_ERROR;
        }
        self.parallel_runner = runner;
        JXL_DEC_SUCCESS
    }

    pub(crate) fn set_input(&mut self, data: *const u8, size: usize) -> JxlDecoderStatus {
        if (data.is_null() && size > 0) || self.input_is_set || self.input_closed {
            return JXL_DEC_ERROR;
        }
        if self.have_container.is_none() && size > 0 && !data.is_null() {
            let need = 16usize.saturating_sub(self.signature_probe.len()).min(size);
            if need > 0 {
                // SAFETY: `data` is non-null and has at least `need <= size` readable bytes.
                let slice = unsafe { std::slice::from_raw_parts(data, need) };
                self.signature_probe.extend_from_slice(slice);
            }
            match check_signature(&self.signature_probe) {
                RsSignature::Container => self.have_container = Some(true),
                RsSignature::Codestream => self.have_container = Some(false),
                _ => {}
            }
        }
        self.next_in = data;
        self.avail_in = size;
        self.input_is_set = true;
        JXL_DEC_SUCCESS
    }

    pub fn release_input(&mut self) -> usize {
        let rem = self.avail_in;
        self.next_in = std::ptr::null();
        self.avail_in = 0;
        self.input_is_set = false;
        rem
    }

    pub fn close_input(&mut self) {
        self.input_closed = true;
        if let Some(dec) = self.rs_decoder.as_mut() {
            dec.close_input();
        }
    }

    pub(crate) fn has_basic_info(&self) -> bool {
        self.rs_decoder
            .as_ref()
            .and_then(|d| d.basic_info())
            .is_some()
    }

    fn build_options(&self, skip_preview: bool) -> JxlDecoderOptions {
        let mut options = JxlDecoderOptions::default();
        options.adjust_orientation = !self.keep_orientation;
        options.render_spot_colors = self.render_spotcolors;
        options.coalescing = self.coalescing;
        options.skip_preview = skip_preview;
        options.desired_intensity_target =
            (self.desired_intensity_target > 0.0).then_some(self.desired_intensity_target);
        options.alpha_mode = if self.unpremultiply_alpha {
            AlphaMode::Unpremultiplied
        } else {
            AlphaMode::AsIs
        };
        options.max_profile_level = ProfileLevel::Main10;
        options
    }

    fn wants_preview(&self) -> bool {
        (self.events_wanted & (JXL_DEC_PREVIEW_IMAGE.0 as i32)) != 0
    }

    fn wants_any_after_color_encoding(&self) -> bool {
        let mask = (JXL_DEC_PREVIEW_IMAGE.0 as i32)
            | (JXL_DEC_FRAME.0 as i32)
            | (JXL_DEC_FULL_IMAGE.0 as i32);
        (self.events_wanted & mask) != 0
    }

    fn wants_main_frames(&self) -> bool {
        let mask = (JXL_DEC_FRAME.0 as i32) | (JXL_DEC_FULL_IMAGE.0 as i32);
        (self.events_wanted & mask) != 0
    }

    fn wants_frame_pixels(&self) -> bool {
        (self.events_wanted & (JXL_DEC_FULL_IMAGE.0 as i32)) != 0
    }

    fn is_gray(&self) -> bool {
        self.rs_decoder
            .as_ref()
            .and_then(|d| d.embedded_color_profile())
            .is_some_and(|p| p.channels() == 1)
    }

    fn default_u8_pixel_format(&self) -> Option<RsPixelFormat> {
        let basic = self.rs_decoder.as_ref()?.basic_info()?;
        let num_ec = basic.extra_channels.len();
        let color_type = if self.is_gray() {
            JxlColorType::Grayscale
        } else {
            JxlColorType::Rgb
        };
        Some(RsPixelFormat {
            color_type,
            color_data_format: Some(JxlDataFormat::U8 { bit_depth: 8 }),
            extra_channel_format: vec![None; num_ec],
        })
    }

    fn desired_main_rs_pixel_format(&self) -> Option<RsPixelFormat> {
        let basic = self.rs_decoder.as_ref()?.basic_info()?;
        let codestream_bd = &basic.bit_depth;
        let is_gray = self.is_gray();
        let color_fmt = self.image_out_target.as_ref().map(|t| *t.format());
        let ec_fmts: Vec<(Option<JxlPixelFormat>, RsBitDepth)> = self
            .extra_channel_targets
            .iter()
            .zip(basic.extra_channels.iter())
            .map(|(t, ec)| {
                (
                    t.as_ref().map(|target| *target.format()),
                    ec.bit_depth.clone(),
                )
            })
            .collect();
        build_rs_pixel_format(
            color_fmt.as_ref(),
            &ec_fmts,
            &self.output_bit_depth,
            codestream_bd,
            is_gray,
        )
    }

    fn desired_preview_rs_pixel_format(&self) -> Option<RsPixelFormat> {
        let basic = self.rs_decoder.as_ref()?.basic_info()?;
        let codestream_bd = &basic.bit_depth;
        let is_gray = self.is_gray();
        let color_fmt = self.preview_out_target.as_ref().map(|t| *t.format());
        let ec_fmts: Vec<(Option<JxlPixelFormat>, RsBitDepth)> = basic
            .extra_channels
            .iter()
            .map(|ec| (None, ec.bit_depth.clone()))
            .collect();
        build_rs_pixel_format(
            color_fmt.as_ref(),
            &ec_fmts,
            &self.output_bit_depth,
            codestream_bd,
            is_gray,
        )
    }

    fn apply_main_pixel_format(&mut self) -> bool {
        let Some(desired_format) = self.desired_main_rs_pixel_format() else {
            return false;
        };
        self.rs_decoder
            .as_mut()
            .is_some_and(|d| d.set_pixel_format(desired_format).is_ok())
    }

    fn has_any_main_output_target(&self) -> bool {
        self.image_out_target.is_some() || self.extra_channel_targets.iter().any(|t| t.is_some())
    }

    pub fn process_input(&mut self) -> JxlDecoderStatus {
        self.process_started = true;
        if self.rs_decoder.is_none() {
            let skip_preview = !self.wants_preview();
            let dec = RsDecoder::new(self.build_options(skip_preview));
            self.rs_decoder = Some(dec);
            if self.input_closed {
                self.rs_decoder.as_mut().unwrap().close_input();
            }
        }

        loop {
            match self.stage {
                Stage::BeforeBasicInfo => {
                    let dec = self.rs_decoder.as_mut().unwrap();
                    let mut input = TrackingInput {
                        next_in: &mut self.next_in,
                        avail_in: &mut self.avail_in,
                    };
                    let mut runner = self.parallel_runner;
                    let runner_dyn = runner.as_mut().map(|r| r as &mut dyn RsParallelRunner);
                    match dec.process(&mut input, None, runner_dyn) {
                        Ok(Event::NeedMoreInput { size_hint }) => {
                            self.basic_info_size_hint = size_hint;
                            return JXL_DEC_NEED_MORE_INPUT;
                        }
                        Ok(Event::BasicInfo) => {
                            self.basic_info_size_hint = 0;
                            let num_ec = dec
                                .basic_info()
                                .map(|b| b.extra_channels.len())
                                .unwrap_or(0);
                            self.extra_channel_targets = (0..num_ec).map(|_| None).collect();
                            if let Some(default_fmt) = self.default_u8_pixel_format() {
                                let _ = self
                                    .rs_decoder
                                    .as_mut()
                                    .unwrap()
                                    .set_pixel_format(default_fmt);
                            }
                            self.stage = Stage::EmitBasicInfo;
                        }
                        _ => return JXL_DEC_ERROR,
                    }
                }

                Stage::EmitBasicInfo => {
                    self.stage = Stage::EmitColorEncoding;
                    if (self.events_wanted & (JXL_DEC_BASIC_INFO.0 as i32)) != 0 {
                        return JXL_DEC_BASIC_INFO;
                    }
                }

                Stage::EmitColorEncoding => {
                    let has_preview = self
                        .rs_decoder
                        .as_ref()
                        .and_then(|d| d.basic_info())
                        .is_some_and(|b| b.preview_size.is_some());
                    if self.wants_preview() && has_preview {
                        self.stage = Stage::PreviewNeedOutBuffer;
                    } else if self.wants_main_frames() {
                        self.stage = Stage::MainFrameHeader;
                    } else {
                        self.stage = Stage::Done;
                    }
                    if (self.events_wanted & (JXL_DEC_COLOR_ENCODING.0 as i32)) != 0 {
                        return JXL_DEC_COLOR_ENCODING;
                    }
                    if !self.wants_any_after_color_encoding() {
                        self.stage = Stage::Done;
                        return JXL_DEC_SUCCESS;
                    }
                }

                Stage::PreviewNeedOutBuffer => {
                    if self.preview_out_target.is_none() {
                        return JXL_DEC_NEED_PREVIEW_OUT_BUFFER;
                    }
                    let Some(preview_fmt) = self.desired_preview_rs_pixel_format() else {
                        return JXL_DEC_ERROR;
                    };
                    if self
                        .rs_decoder
                        .as_mut()
                        .unwrap()
                        .set_pixel_format(preview_fmt)
                        .is_err()
                    {
                        return JXL_DEC_ERROR;
                    }
                    self.stage = Stage::PreviewFrameHeader;
                }

                Stage::PreviewFrameHeader => {
                    let dec = self.rs_decoder.as_mut().unwrap();
                    let mut input = TrackingInput {
                        next_in: &mut self.next_in,
                        avail_in: &mut self.avail_in,
                    };
                    let mut runner = self.parallel_runner;
                    let runner_dyn = runner.as_mut().map(|r| r as &mut dyn RsParallelRunner);
                    match dec.process(&mut input, None, runner_dyn) {
                        Ok(Event::NeedMoreInput { .. }) => return JXL_DEC_NEED_MORE_INPUT,
                        Ok(Event::FrameHeader) => {
                            self.stage = Stage::PreviewDecode;
                        }
                        _ => return JXL_DEC_ERROR,
                    }
                }

                Stage::PreviewDecode => {
                    let Some((pw, ph)) = self
                        .rs_decoder
                        .as_ref()
                        .and_then(|d| d.basic_info())
                        .and_then(|b| b.preview_size)
                    else {
                        return JXL_DEC_ERROR;
                    };
                    let Some(target) = self.preview_out_target.as_mut() else {
                        return JXL_DEC_ERROR;
                    };
                    let Some(buf) = target.make_output_buffer(pw, ph) else {
                        return JXL_DEC_ERROR;
                    };
                    let mut buffers = [buf];
                    let dec = self.rs_decoder.as_mut().unwrap();
                    let mut input = TrackingInput {
                        next_in: &mut self.next_in,
                        avail_in: &mut self.avail_in,
                    };
                    let mut runner = self.parallel_runner;
                    let runner_dyn = runner.as_mut().map(|r| r as &mut dyn RsParallelRunner);
                    let event = dec.process(&mut input, Some(&mut buffers), runner_dyn);
                    match event {
                        Ok(Event::NeedMoreInput { .. }) => return JXL_DEC_NEED_MORE_INPUT,
                        Ok(Event::FrameComplete { has_more_frames }) => {
                            self.preview_out_target = None;
                            self.preview_done = true;
                            self.has_more_frames = has_more_frames;
                            if self.wants_main_frames() && has_more_frames {
                                self.stage = Stage::MainFrameHeader;
                            } else {
                                self.stage = Stage::Done;
                            }
                            return JXL_DEC_PREVIEW_IMAGE;
                        }
                        _ => return JXL_DEC_ERROR,
                    }
                }

                Stage::MainFrameHeader => {
                    if !self.wants_main_frames() || !self.has_more_frames {
                        self.stage = Stage::Done;
                        return JXL_DEC_SUCCESS;
                    }
                    if self.main_frames_started == 0
                        && self.has_any_main_output_target()
                        && let Some(fmt) = self.desired_main_rs_pixel_format()
                    {
                        let _ = self.rs_decoder.as_mut().unwrap().set_pixel_format(fmt);
                    }
                    let dec = self.rs_decoder.as_mut().unwrap();
                    let mut input = TrackingInput {
                        next_in: &mut self.next_in,
                        avail_in: &mut self.avail_in,
                    };
                    let mut runner = self.parallel_runner;
                    let runner_dyn = runner.as_mut().map(|r| r as &mut dyn RsParallelRunner);
                    match dec.process(&mut input, None, runner_dyn) {
                        Ok(Event::NeedMoreInput { .. }) => return JXL_DEC_NEED_MORE_INPUT,
                        Ok(Event::FrameHeader) => {
                            self.main_frames_started += 1;
                            if self.frames_to_skip > 0 {
                                self.frames_to_skip -= 1;
                                self.skip_current_frame = true;
                                self.stage = Stage::MainDecode;
                            } else {
                                self.skip_current_frame = false;
                                self.stage = Stage::MainEmitFrame;
                            }
                        }
                        Ok(Event::Complete) => {
                            self.stage = Stage::Done;
                            return JXL_DEC_SUCCESS;
                        }
                        _ => return JXL_DEC_ERROR,
                    }
                }

                Stage::MainEmitFrame => {
                    if self.wants_frame_pixels() {
                        self.stage = Stage::MainNeedOutBuffer;
                    } else {
                        self.skip_current_frame = true;
                        self.stage = Stage::MainDecode;
                    }
                    if (self.events_wanted & (JXL_DEC_FRAME.0 as i32)) != 0 {
                        return JXL_DEC_FRAME;
                    }
                }

                Stage::MainNeedOutBuffer => {
                    if self.skip_current_frame || !self.wants_frame_pixels() {
                        self.stage = Stage::MainDecode;
                        continue;
                    }
                    if self.image_out_target.is_none() {
                        return JXL_DEC_NEED_IMAGE_OUT_BUFFER;
                    }
                    if !self.apply_main_pixel_format() {
                        return JXL_DEC_ERROR;
                    }
                    self.stage = Stage::MainDecode;
                }

                Stage::MainDecode => {
                    if self.skip_current_frame || !self.wants_frame_pixels() {
                        self.main_decode_started = true;
                        let dec = self.rs_decoder.as_mut().unwrap();
                        let mut input = TrackingInput {
                            next_in: &mut self.next_in,
                            avail_in: &mut self.avail_in,
                        };
                        let mut runner = self.parallel_runner;
                        let runner_dyn = runner.as_mut().map(|r| r as &mut dyn RsParallelRunner);
                        match dec.process(&mut input, None, runner_dyn) {
                            Ok(Event::NeedMoreInput { .. }) => return JXL_DEC_NEED_MORE_INPUT,
                            Ok(Event::FrameComplete { has_more_frames }) => {
                                self.has_more_frames = has_more_frames;
                                self.skip_current_frame = false;
                                self.main_decode_started = false;
                                if has_more_frames {
                                    self.stage = Stage::MainFrameHeader;
                                } else {
                                    self.stage = Stage::Done;
                                    return JXL_DEC_SUCCESS;
                                }
                            }
                            Ok(Event::Complete) => {
                                self.stage = Stage::Done;
                                return JXL_DEC_SUCCESS;
                            }
                            _ => return JXL_DEC_ERROR,
                        }
                        continue;
                    }

                    if !self.apply_main_pixel_format() {
                        return JXL_DEC_ERROR;
                    }

                    let Some((width, height)) = self
                        .rs_decoder
                        .as_ref()
                        .and_then(|d| d.basic_info())
                        .map(|b| b.size)
                    else {
                        return JXL_DEC_ERROR;
                    };

                    let mut output_buffers: Vec<JxlOutputBuffer<'_>> = Vec::new();
                    if let Some(color_target) = self.image_out_target.as_mut() {
                        let Some(buf) = color_target.make_output_buffer(width, height) else {
                            return JXL_DEC_ERROR;
                        };
                        output_buffers.push(buf);
                    }
                    for ec_target in self.extra_channel_targets.iter_mut().flatten() {
                        let Some(buf) = ec_target.make_output_buffer(width, height) else {
                            return JXL_DEC_ERROR;
                        };
                        output_buffers.push(buf);
                    }

                    self.main_decode_started = true;
                    let dec = self.rs_decoder.as_mut().unwrap();
                    let mut input = TrackingInput {
                        next_in: &mut self.next_in,
                        avail_in: &mut self.avail_in,
                    };
                    let mut runner = self.parallel_runner;
                    let runner_dyn = runner.as_mut().map(|r| r as &mut dyn RsParallelRunner);
                    let event = dec.process(&mut input, Some(&mut output_buffers), runner_dyn);
                    drop(output_buffers);

                    match event {
                        Ok(Event::NeedMoreInput { .. }) => return JXL_DEC_NEED_MORE_INPUT,
                        Ok(Event::FrameComplete { has_more_frames }) => {
                            self.main_decode_started = false;
                            if let Some(color_target) = self.image_out_target.as_ref()
                                && !color_target.finalize_output(width, height)
                            {
                                return JXL_DEC_ERROR;
                            }
                            if matches!(self.image_out_target, Some(ImageOutTarget::Buffer { .. }))
                            {
                                self.image_out_target = None;
                            }
                            for ec_target in &mut self.extra_channel_targets {
                                *ec_target = None;
                            }
                            self.has_more_frames = has_more_frames;
                            if has_more_frames {
                                self.stage = Stage::MainFrameHeader;
                            } else {
                                self.stage = Stage::Done;
                            }
                            if (self.events_wanted & (JXL_DEC_FULL_IMAGE.0 as i32)) != 0 {
                                return JXL_DEC_FULL_IMAGE;
                            }
                            if !has_more_frames {
                                return JXL_DEC_SUCCESS;
                            }
                        }
                        _ => return JXL_DEC_ERROR,
                    }
                }

                Stage::Done => {
                    return JXL_DEC_SUCCESS;
                }
            }
        }
    }

    pub fn skip_frames(&mut self, amount: usize) {
        self.frames_to_skip = self.frames_to_skip.saturating_add(amount);
    }

    pub fn skip_current_frame(&mut self) -> JxlDecoderStatus {
        match self.stage {
            Stage::MainEmitFrame | Stage::MainNeedOutBuffer | Stage::MainDecode => {
                self.skip_current_frame = true;
                JXL_DEC_SUCCESS
            }
            _ => JXL_DEC_ERROR,
        }
    }

    pub fn flush_image(&mut self) -> JxlDecoderStatus {
        if self.stage != Stage::MainDecode || !self.has_any_main_output_target() {
            return JXL_DEC_ERROR;
        }
        if matches!(
            self.image_out_target,
            Some(
                ImageOutTarget::SimpleCallback { .. }
                    | ImageOutTarget::MultithreadedCallback { .. }
            )
        ) {
            return JXL_DEC_ERROR;
        }
        let Some((width, height)) = self
            .rs_decoder
            .as_ref()
            .and_then(|d| d.basic_info())
            .map(|b| b.size)
        else {
            return JXL_DEC_ERROR;
        };
        let mut output_buffers: Vec<JxlOutputBuffer<'_>> = Vec::new();
        if let Some(color_target) = self.image_out_target.as_mut() {
            let Some(buf) = color_target.make_output_buffer(width, height) else {
                return JXL_DEC_ERROR;
            };
            output_buffers.push(buf);
        }
        for ec_target in self.extra_channel_targets.iter_mut().flatten() {
            let Some(buf) = ec_target.make_output_buffer(width, height) else {
                return JXL_DEC_ERROR;
            };
            output_buffers.push(buf);
        }
        let dec = self.rs_decoder.as_mut().unwrap();
        let mut runner = self.parallel_runner;
        let runner_dyn = runner.as_mut().map(|r| r as &mut dyn RsParallelRunner);
        let ok = dec.flush_pixels(&mut output_buffers, runner_dyn).is_ok();
        drop(output_buffers);
        if !ok {
            return JXL_DEC_ERROR;
        }
        JXL_DEC_SUCCESS
    }

    pub(crate) fn get_basic_info(&self) -> Option<JxlBasicInfo> {
        let rs_info = self.rs_decoder.as_ref().and_then(|d| d.basic_info())?;
        Some(convert_basic_info(
            rs_info,
            self.have_container.unwrap_or(false),
            self.is_gray(),
            self.keep_orientation,
            self.desired_intensity_target,
        ))
    }

    pub(crate) fn get_extra_channel_info(&self, index: usize) -> Option<JxlExtraChannelInfo> {
        let rs_info = self.rs_decoder.as_ref().and_then(|d| d.basic_info())?;
        let ec = rs_info.extra_channels.get(index)?;
        let (bits_per_sample, exponent_bits_per_sample) = match ec.bit_depth {
            RsBitDepth::Int { bits_per_sample } => (bits_per_sample, 0),
            RsBitDepth::Float {
                bits_per_sample,
                exponent_bits_per_sample,
            } => (bits_per_sample, exponent_bits_per_sample),
        };
        Some(JxlExtraChannelInfo {
            type_: convert_extra_channel_type(ec.ec_type),
            bits_per_sample,
            exponent_bits_per_sample,
            dim_shift: ec.dim_shift,
            name_length: ec.name.len() as u32,
            alpha_premultiplied: if ec.alpha_associated {
                JXL_TRUE
            } else {
                JXL_FALSE
            },
            spot_color: ec.spot_color.unwrap_or([0.0; 4]),
            cfa_channel: ec.cfa_channel.unwrap_or(0),
        })
    }

    pub(crate) fn get_extra_channel_name(
        &self,
        index: usize,
        buffer: *mut u8,
        size: usize,
    ) -> JxlDecoderStatus {
        let Some(rs_info) = self.rs_decoder.as_ref().and_then(|d| d.basic_info()) else {
            return JXL_DEC_ERROR;
        };
        let Some(ec) = rs_info.extra_channels.get(index) else {
            return JXL_DEC_ERROR;
        };
        let bytes = ec.name.as_bytes();
        if buffer.is_null() || size <= bytes.len() {
            return JXL_DEC_ERROR;
        }
        // SAFETY: `buffer` is non-null and valid for at least `size > bytes.len()` writable bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), buffer, bytes.len());
            *buffer.add(bytes.len()) = 0;
        }
        JXL_DEC_SUCCESS
    }

    fn select_color_profile(&self, target: JxlColorProfileTarget) -> Option<&RsColorProfile> {
        let dec = self.rs_decoder.as_ref()?;
        match target {
            JXL_COLOR_PROFILE_TARGET_ORIGINAL => dec.embedded_color_profile(),
            JXL_COLOR_PROFILE_TARGET_DATA => dec.output_color_profile(),
            _ => None,
        }
    }

    pub(crate) fn get_color_as_encoded_profile(
        &self,
        target: JxlColorProfileTarget,
    ) -> Option<JxlColorEncoding> {
        let profile = self.select_color_profile(target)?;
        match profile {
            RsColorProfile::Simple(enc) => Some(convert_color_encoding(enc)),
            RsColorProfile::Icc(_) => None,
        }
    }

    pub(crate) fn get_icc_profile_size(&self, target: JxlColorProfileTarget) -> Option<usize> {
        let profile = self.select_color_profile(target)?;
        let icc = profile.try_as_icc()?;
        Some(icc.len())
    }

    pub(crate) fn get_color_as_icc_profile(
        &self,
        target: JxlColorProfileTarget,
        icc_profile: *mut u8,
        size: usize,
    ) -> JxlDecoderStatus {
        if icc_profile.is_null() {
            return JXL_DEC_ERROR;
        }
        let Some(profile) = self.select_color_profile(target) else {
            return JXL_DEC_ERROR;
        };
        let Some(icc) = profile.try_as_icc() else {
            return JXL_DEC_ERROR;
        };
        if size < icc.len() {
            return JXL_DEC_ERROR;
        }
        // SAFETY: `icc_profile` is non-null and valid for at least `size >= icc.len()` bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(icc.as_ptr(), icc_profile, icc.len());
        }
        JXL_DEC_SUCCESS
    }

    pub fn set_preferred_color_profile(
        &mut self,
        color_encoding: &JxlColorEncoding,
    ) -> JxlDecoderStatus {
        if self.main_frames_started > 0 || self.preview_done {
            return JXL_DEC_ERROR;
        }
        let Some(rs_enc) = to_rs_color_encoding(color_encoding) else {
            return JXL_DEC_ERROR;
        };
        let Some(dec) = self.rs_decoder.as_mut() else {
            return JXL_DEC_ERROR;
        };
        let profile = RsColorProfile::Simple(rs_enc);
        if dec.set_output_color_profile(profile).is_ok() {
            return JXL_DEC_SUCCESS;
        }
        JXL_DEC_ERROR
    }

    pub fn preview_out_buffer_size(&self, format: &JxlPixelFormat) -> Option<usize> {
        let rs_info = self.rs_decoder.as_ref()?.basic_info()?;
        if !self.is_gray() && format.num_channels < 3 {
            return None;
        }
        let (pw, ph) = rs_info.preview_size?;
        compute_buffer_size(pw, ph, format)
    }

    pub fn set_preview_out_buffer(
        &mut self,
        format: &JxlPixelFormat,
        buffer: *mut c_void,
        size: usize,
    ) -> JxlDecoderStatus {
        let Some(min_size) = self.preview_out_buffer_size(format) else {
            return JXL_DEC_ERROR;
        };
        if buffer.is_null() || size < min_size {
            return JXL_DEC_ERROR;
        }
        self.preview_out_target = Some(ImageOutTarget::Buffer {
            ptr: buffer.cast::<u8>(),
            size,
            format: *format,
        });
        JXL_DEC_SUCCESS
    }

    pub fn image_out_buffer_size(&self, format: &JxlPixelFormat) -> Option<usize> {
        let rs_info = self.rs_decoder.as_ref()?.basic_info()?;
        if !self.is_gray() && format.num_channels < 3 {
            return None;
        }
        let (w, h) = rs_info.size;
        compute_buffer_size(w, h, format)
    }

    pub fn set_image_out_buffer(
        &mut self,
        format: &JxlPixelFormat,
        buffer: *mut c_void,
        size: usize,
    ) -> JxlDecoderStatus {
        let Some(min_size) = self.image_out_buffer_size(format) else {
            return JXL_DEC_ERROR;
        };
        if buffer.is_null() || size < min_size {
            return JXL_DEC_ERROR;
        }
        self.image_out_target = Some(ImageOutTarget::Buffer {
            ptr: buffer.cast::<u8>(),
            size,
            format: *format,
        });
        // Update output color profile immediately if still before first frame header.
        if let Some(fmt) = self.desired_main_rs_pixel_format()
            && let Some(dec) = self.rs_decoder.as_mut()
        {
            let _ = dec.set_pixel_format(fmt);
        }
        JXL_DEC_SUCCESS
    }

    pub fn set_image_out_callback(
        &mut self,
        format: &JxlPixelFormat,
        callback: JxlImageOutCallback,
        opaque: *mut c_void,
    ) -> JxlDecoderStatus {
        if self.has_basic_info() && !self.is_gray() && format.num_channels < 3 {
            return JXL_DEC_ERROR;
        }
        let Some(target) = ImageOutTarget::new_simple_callback(*format, callback, opaque) else {
            return JXL_DEC_ERROR;
        };
        if compute_buffer_size(1, 1, format).is_none() {
            return JXL_DEC_ERROR;
        }
        self.image_out_target = Some(target);
        if let Some(fmt) = self.desired_main_rs_pixel_format()
            && let Some(dec) = self.rs_decoder.as_mut()
        {
            let _ = dec.set_pixel_format(fmt);
        }
        JXL_DEC_SUCCESS
    }

    pub fn set_multithreaded_image_out_callback(
        &mut self,
        format: &JxlPixelFormat,
        init_callback: JxlImageOutInitCallback,
        run_callback: JxlImageOutRunCallback,
        destroy_callback: JxlImageOutDestroyCallback,
        init_opaque: *mut c_void,
    ) -> JxlDecoderStatus {
        if self.has_basic_info() && !self.is_gray() && format.num_channels < 3 {
            return JXL_DEC_ERROR;
        }
        let Some(target) = ImageOutTarget::new_multithreaded_callback(
            *format,
            init_callback,
            run_callback,
            destroy_callback,
            init_opaque,
        ) else {
            return JXL_DEC_ERROR;
        };
        if compute_buffer_size(1, 1, format).is_none() {
            return JXL_DEC_ERROR;
        }
        self.image_out_target = Some(target);
        if let Some(fmt) = self.desired_main_rs_pixel_format()
            && let Some(dec) = self.rs_decoder.as_mut()
        {
            let _ = dec.set_pixel_format(fmt);
        }
        JXL_DEC_SUCCESS
    }

    pub fn extra_channel_buffer_size(&self, format: &JxlPixelFormat, index: u32) -> Option<usize> {
        let rs_info = self.rs_decoder.as_ref()?.basic_info()?;
        if (index as usize) >= rs_info.extra_channels.len() {
            return None;
        }
        let mut ec_format = *format;
        ec_format.num_channels = 1;
        let (w, h) = rs_info.size;
        compute_buffer_size(w, h, &ec_format)
    }

    pub fn set_extra_channel_buffer(
        &mut self,
        format: &JxlPixelFormat,
        buffer: *mut c_void,
        size: usize,
        index: u32,
    ) -> JxlDecoderStatus {
        let Some(min_size) = self.extra_channel_buffer_size(format, index) else {
            return JXL_DEC_ERROR;
        };
        if buffer.is_null() || size < min_size {
            return JXL_DEC_ERROR;
        }
        let mut ec_format = *format;
        ec_format.num_channels = 1;
        let idx = index as usize;
        if idx >= self.extra_channel_targets.len() {
            return JXL_DEC_ERROR;
        }
        self.extra_channel_targets[idx] = Some(ImageOutTarget::Buffer {
            ptr: buffer.cast::<u8>(),
            size,
            format: ec_format,
        });
        if let Some(fmt) = self.desired_main_rs_pixel_format()
            && let Some(dec) = self.rs_decoder.as_mut()
        {
            let _ = dec.set_pixel_format(fmt);
        }
        JXL_DEC_SUCCESS
    }

    pub(crate) fn get_frame_header(&self) -> Option<JxlFrameHeader> {
        let dec = self.rs_decoder.as_ref()?;
        let fh = dec.frame_header()?;
        let duration_ticks = dec
            .scanned_frames()
            .last()
            .map(|f| f.duration_ticks)
            .unwrap_or(0);
        let is_last = dec
            .scanned_frames()
            .last()
            .map(|f| f.is_last)
            .unwrap_or(!self.has_more_frames);
        Some(JxlFrameHeader {
            duration: duration_ticks,
            timecode: 0,
            name_length: fh.name.len() as u32,
            is_last: if is_last { JXL_TRUE } else { JXL_FALSE },
            layer_info: JxlLayerInfo {
                have_crop: JXL_FALSE,
                crop_x0: 0,
                crop_y0: 0,
                xsize: fh.size.0 as u32,
                ysize: fh.size.1 as u32,
                blend_info: JxlBlendInfo {
                    blendmode: JXL_BLEND_REPLACE,
                    source: 0,
                    alpha: 0,
                    clamp: JXL_FALSE,
                },
                save_as_reference: 0,
            },
        })
    }

    pub(crate) fn get_frame_name(&self, name: *mut u8, size: usize) -> JxlDecoderStatus {
        if name.is_null() || size == 0 {
            return JXL_DEC_ERROR;
        }
        let Some(dec) = self.rs_decoder.as_ref() else {
            return JXL_DEC_ERROR;
        };
        let Some(fh) = dec.frame_header() else {
            return JXL_DEC_ERROR;
        };
        let bytes = fh.name.as_bytes();
        if size <= bytes.len() {
            return JXL_DEC_ERROR;
        }
        // SAFETY: `name` is non-null and valid for at least `size > bytes.len()` writable bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), name, bytes.len());
            *name.add(bytes.len()) = 0;
        }
        JXL_DEC_SUCCESS
    }
}

fn convert_orientation(o: Orientation) -> JxlOrientation {
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

fn convert_extra_channel_type(ec: jxl::api::ExtraChannel) -> JxlExtraChannelType {
    use jxl::api::ExtraChannel::*;
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

fn convert_basic_info(
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
        .find(|ec| ec.ec_type == jxl::api::ExtraChannel::Alpha);
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
                if gamma <= 0.0 || !gamma.is_finite() {
                    return None;
                }
                Some(RsTransferFunction::Gamma(gamma as f32))
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
