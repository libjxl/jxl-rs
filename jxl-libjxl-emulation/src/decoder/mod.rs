// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

pub mod boxes;
pub mod input;
pub mod output;

use jxl::api::{
    AlphaMode, Event, JxlBasicInfo as RsBasicInfo, JxlColorEncoding as RsColorEncoding,
    JxlColorProfile as RsColorProfile, JxlColorType, JxlDataFormat, JxlDecoder as RsDecoder,
    JxlDecoderOptions, JxlOutputBuffer, JxlParallelRunner as RsParallelRunner,
    JxlPixelFormat as RsPixelFormat,
};

use self::boxes::{AUX_BOX_TYPES, BoxStreamer};
use self::input::InputCursor;
use self::output::FrameOutput;
use crate::parallel_runner::CParallelRunner;
use crate::types::{
    JXL_COLOR_PROFILE_TARGET_DATA, JXL_COLOR_PROFILE_TARGET_ORIGINAL, JXL_DEC_BASIC_INFO,
    JXL_DEC_BOX, JXL_DEC_BOX_COMPLETE, JXL_DEC_COLOR_ENCODING, JXL_DEC_ERROR, JXL_DEC_FRAME,
    JXL_DEC_FRAME_PROGRESSION, JXL_DEC_FULL_IMAGE, JXL_DEC_JPEG_RECONSTRUCTION,
    JXL_DEC_NEED_IMAGE_OUT_BUFFER, JXL_DEC_NEED_MORE_INPUT, JXL_DEC_PREVIEW_IMAGE, JXL_DEC_SUCCESS,
    JxlCmsInterface, JxlColorProfileTarget, JxlDecoderStatus,
};

pub(crate) const ALL_INFORMATIVE_EVENTS_MASK: i32 = (JXL_DEC_BASIC_INFO.0 as i32)
    | (JXL_DEC_COLOR_ENCODING.0 as i32)
    | (JXL_DEC_PREVIEW_IMAGE.0 as i32)
    | (JXL_DEC_FRAME.0 as i32)
    | (JXL_DEC_FULL_IMAGE.0 as i32)
    | (JXL_DEC_JPEG_RECONSTRUCTION.0 as i32)
    | (JXL_DEC_BOX.0 as i32)
    | (JXL_DEC_FRAME_PROGRESSION.0 as i32)
    | (JXL_DEC_BOX_COMPLETE.0 as i32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum Stage {
    #[default]
    Headers,
    EmitBasicInfo,
    EmitColorEncoding,
    FrameHeader,
    EmitFrame,
    NeedImageOutBuffer,
    DecodeFrame,
    TrailingBoxes,
    Done,
}

/// Settings preserved across `JxlDecoderRewind`.
pub(crate) struct DecoderSettings {
    pub(crate) events_wanted: i32,
    pub(crate) keep_orientation: bool,
    pub(crate) render_spotcolors: bool,
    pub(crate) coalescing: bool,
    pub(crate) unpremultiply_alpha: bool,
    pub(crate) decompress_boxes: bool,
    pub(crate) desired_intensity_target: f32,
    pub(crate) cms: Option<JxlCmsInterface>,
    pub(crate) parallel_runner: Option<CParallelRunner>,
}

impl Default for DecoderSettings {
    fn default() -> Self {
        Self {
            events_wanted: 0,
            keep_orientation: false,
            render_spotcolors: true,
            coalescing: true,
            unpremultiply_alpha: false,
            decompress_boxes: false,
            desired_intensity_target: 0.0,
            cms: None,
            parallel_runner: None,
        }
    }
}

/// Per-stream decode state reset on `JxlDecoderRewind`.
pub(crate) struct DecodeSession {
    pub(crate) rs_decoder: Option<RsDecoder>,
    pub(crate) stage: Stage,
    pub(crate) process_started: bool,
    pub(crate) is_gray: bool,
    pub(crate) frames_to_skip: usize,
    pub(crate) skip_current_frame: bool,
    pub(crate) can_skip_current_frame: bool,
    pub(crate) has_more_frames: bool,
    pub(crate) mid_frame_decode: bool,
    pub(crate) input: InputCursor,
    pub(crate) boxes: BoxStreamer,
    pub(crate) output: FrameOutput,
}

impl Default for DecodeSession {
    fn default() -> Self {
        Self {
            rs_decoder: None,
            stage: Stage::Headers,
            process_started: false,
            is_gray: false,
            frames_to_skip: 0,
            skip_current_frame: false,
            can_skip_current_frame: false,
            has_more_frames: true,
            mid_frame_decode: false,
            input: InputCursor::default(),
            boxes: BoxStreamer::default(),
            output: FrameOutput::default(),
        }
    }
}

pub struct JxlDecoder {
    pub(crate) settings: DecoderSettings,
    pub(crate) session: DecodeSession,
}

impl Default for JxlDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl JxlDecoder {
    pub fn new() -> Self {
        Self {
            settings: DecoderSettings::default(),
            session: DecodeSession::default(),
        }
    }

    pub(crate) fn rs_basic_info(&self) -> Option<&RsBasicInfo> {
        self.session.rs_decoder.as_ref()?.basic_info()
    }

    pub(crate) fn has_basic_info(&self) -> bool {
        self.rs_basic_info().is_some()
    }

    pub(crate) fn select_color_profile(
        &self,
        target: JxlColorProfileTarget,
    ) -> Option<&RsColorProfile> {
        let dec = self.session.rs_decoder.as_ref()?;
        match target {
            JXL_COLOR_PROFILE_TARGET_ORIGINAL => dec.embedded_color_profile(),
            JXL_COLOR_PROFILE_TARGET_DATA => dec.output_color_profile(),
            _ => None,
        }
    }

    pub(crate) fn refresh_pixel_format(&mut self) -> bool {
        if self.session.mid_frame_decode {
            return true;
        }
        let Some(dec) = self.session.rs_decoder.as_mut() else {
            return false;
        };
        let Some(codestream_bd) = dec.basic_info().map(|b| b.bit_depth.clone()) else {
            return false;
        };
        let Some(fmt) = self
            .session
            .output
            .desired_rs_pixel_format(&codestream_bd, self.session.is_gray)
        else {
            return false;
        };
        dec.set_pixel_format(fmt).is_ok()
    }

    fn wants_boxes(&self) -> bool {
        (self.settings.events_wanted & (JXL_DEC_BOX.0 as i32)) != 0
    }

    fn wants_pixels(&self) -> bool {
        (self.settings.events_wanted & (JXL_DEC_FULL_IMAGE.0 as i32)) != 0
    }

    fn wants_frames_or_boxes(&self) -> bool {
        let mask =
            (JXL_DEC_FRAME.0 as i32) | (JXL_DEC_FULL_IMAGE.0 as i32) | (JXL_DEC_BOX.0 as i32);
        (self.settings.events_wanted & mask) != 0
    }

    fn build_options(&self) -> JxlDecoderOptions {
        let mut options = JxlDecoderOptions::default();
        options.adjust_orientation = !self.settings.keep_orientation;
        options.render_spot_colors = self.settings.render_spotcolors;
        options.coalescing = self.settings.coalescing;
        options.skip_preview = true;
        options.scan_frames_only = !self.wants_pixels();
        options.desired_intensity_target = (self.settings.desired_intensity_target > 0.0)
            .then_some(self.settings.desired_intensity_target);
        options.alpha_mode = if self.settings.unpremultiply_alpha {
            AlphaMode::Unpremultiplied
        } else {
            AlphaMode::AsIs
        };
        if self.wants_boxes() {
            options.request_aux_boxes = AUX_BOX_TYPES.to_vec();
        }
        options
    }

    fn step_rs_decoder(
        &mut self,
        buffers: Option<&mut [JxlOutputBuffer<'_>]>,
    ) -> Result<Event, jxl::error::Error> {
        let mut slice = self.session.input.as_slice();
        let prev_len = slice.len();
        let dec = self.session.rs_decoder.as_mut().unwrap();
        let mut runner = self.settings.parallel_runner;
        let runner_dyn = runner.as_mut().map(|r| r as &mut dyn RsParallelRunner);
        let res = dec.process(&mut slice, buffers, runner_dyn);
        let consumed = prev_len - slice.len();
        let parsed = dec.parsed_bytes();
        self.session.input.record_rs_process(consumed, &res, parsed);
        res
    }

    fn collect_new_aux_boxes(&mut self) -> bool {
        if !self.wants_boxes() {
            return true;
        }
        let Some(dec) = self.session.rs_decoder.as_ref() else {
            return true;
        };
        self.session
            .boxes
            .collect_from(dec, self.settings.decompress_boxes)
    }

    pub(crate) fn process_input(&mut self) -> JxlDecoderStatus {
        self.session.process_started = true;
        if self.session.rs_decoder.is_none() {
            let mut dec = RsDecoder::new(self.build_options());
            if self.session.input.is_closed() {
                dec.close_input();
            }
            self.session.rs_decoder = Some(dec);
        }

        loop {
            if let Some(status) = self.session.boxes.step(self.settings.events_wanted) {
                return status;
            }

            match self.session.stage {
                Stage::Headers => match self.step_rs_decoder(None) {
                    Ok(Event::NeedMoreInput { .. }) => return JXL_DEC_NEED_MORE_INPUT,
                    Ok(Event::BasicInfo) => {
                        let dec = self.session.rs_decoder.as_ref().unwrap();
                        let num_ec = dec
                            .basic_info()
                            .map(|b| b.extra_channels.len())
                            .unwrap_or(0);
                        let uses_original =
                            dec.basic_info().is_some_and(|b| b.uses_original_profile);
                        let can_output_embedded = dec
                            .embedded_color_profile()
                            .is_some_and(|p| p.can_output_to());
                        let is_gray = matches!(
                            dec.current_pixel_format().map(|f| f.color_type),
                            Some(JxlColorType::Grayscale | JxlColorType::GrayscaleAlpha)
                        );
                        self.session.is_gray = is_gray;
                        self.session.output.init_extra_channels(num_ec);
                        let color_type = if is_gray {
                            JxlColorType::Grayscale
                        } else {
                            JxlColorType::Rgb
                        };
                        let default_fmt = RsPixelFormat {
                            color_type,
                            color_data_format: Some(JxlDataFormat::U8 { bit_depth: 8 }),
                            extra_channel_format: vec![None; num_ec],
                        };
                        let rs_dec = self.session.rs_decoder.as_mut().unwrap();
                        let _ = rs_dec.set_pixel_format(default_fmt);
                        // Match libjxl (`OutputEncodingInfo::SetFromMetadata`): for XYB images
                        // whose embedded profile cannot be output directly (e.g. ICC), default
                        // `JXL_COLOR_PROFILE_TARGET_DATA` to Linear sRGB regardless of pixel type.
                        if !uses_original && !can_output_embedded {
                            let _ = rs_dec.set_output_color_profile(RsColorProfile::Simple(
                                RsColorEncoding::linear_srgb(is_gray),
                            ));
                        }
                        if !self.collect_new_aux_boxes() {
                            return JXL_DEC_ERROR;
                        }
                        self.session.stage = Stage::EmitBasicInfo;
                    }
                    _ => return JXL_DEC_ERROR,
                },

                Stage::EmitBasicInfo => {
                    self.session.stage = Stage::EmitColorEncoding;
                    if (self.settings.events_wanted & (JXL_DEC_BASIC_INFO.0 as i32)) != 0 {
                        return JXL_DEC_BASIC_INFO;
                    }
                }

                Stage::EmitColorEncoding => {
                    if self.wants_frames_or_boxes() {
                        self.session.stage = Stage::FrameHeader;
                    } else {
                        self.session.stage = Stage::Done;
                    }
                    if (self.settings.events_wanted & (JXL_DEC_COLOR_ENCODING.0 as i32)) != 0 {
                        return JXL_DEC_COLOR_ENCODING;
                    }
                }

                Stage::FrameHeader => {
                    if !self.session.has_more_frames {
                        self.session.stage = Stage::TrailingBoxes;
                        continue;
                    }
                    self.session.mid_frame_decode = false;
                    if self.session.output.has_targets() {
                        let _ = self.refresh_pixel_format();
                    }
                    match self.step_rs_decoder(None) {
                        Ok(Event::NeedMoreInput { .. }) => return JXL_DEC_NEED_MORE_INPUT,
                        Ok(Event::FrameHeader) => {
                            if !self.collect_new_aux_boxes() {
                                return JXL_DEC_ERROR;
                            }
                            if self.session.frames_to_skip > 0 {
                                self.session.frames_to_skip -= 1;
                                self.session.skip_current_frame = true;
                                self.session.stage = Stage::DecodeFrame;
                            } else {
                                self.session.skip_current_frame = false;
                                self.session.stage = Stage::EmitFrame;
                            }
                        }
                        Ok(Event::Complete) => {
                            self.session.boxes.trailing_parsed = true;
                            if !self.collect_new_aux_boxes() {
                                return JXL_DEC_ERROR;
                            }
                            self.session.stage = Stage::TrailingBoxes;
                        }
                        _ => return JXL_DEC_ERROR,
                    }
                }

                Stage::EmitFrame => {
                    if self.wants_pixels() && !self.session.skip_current_frame {
                        self.session.stage = Stage::NeedImageOutBuffer;
                    } else {
                        self.session.skip_current_frame = true;
                        self.session.stage = Stage::DecodeFrame;
                    }
                    if (self.settings.events_wanted & (JXL_DEC_FRAME.0 as i32)) != 0 {
                        return JXL_DEC_FRAME;
                    }
                }

                Stage::NeedImageOutBuffer => {
                    if self.session.skip_current_frame || !self.wants_pixels() {
                        self.session.stage = Stage::DecodeFrame;
                        continue;
                    }
                    if self.session.output.image_out_target.is_none() {
                        self.session.can_skip_current_frame = true;
                        return JXL_DEC_NEED_IMAGE_OUT_BUFFER;
                    }
                    if !self.refresh_pixel_format() {
                        return JXL_DEC_ERROR;
                    }
                    self.session.stage = Stage::DecodeFrame;
                }

                Stage::DecodeFrame => {
                    if self.session.skip_current_frame || !self.wants_pixels() {
                        self.session.mid_frame_decode = true;
                        match self.step_rs_decoder(None) {
                            Ok(Event::NeedMoreInput { .. }) => return JXL_DEC_NEED_MORE_INPUT,
                            Ok(Event::FrameComplete { has_more_frames }) => {
                                self.session.mid_frame_decode = false;
                                self.session.skip_current_frame = false;
                                self.session.output.clear_targets();
                                self.session.has_more_frames = has_more_frames;
                                if !self.collect_new_aux_boxes() {
                                    return JXL_DEC_ERROR;
                                }
                                self.session.stage = if has_more_frames {
                                    Stage::FrameHeader
                                } else {
                                    Stage::TrailingBoxes
                                };
                            }
                            Ok(Event::Complete) => {
                                self.session.mid_frame_decode = false;
                                self.session.skip_current_frame = false;
                                self.session.output.clear_targets();
                                self.session.boxes.trailing_parsed = true;
                                if !self.collect_new_aux_boxes() {
                                    return JXL_DEC_ERROR;
                                }
                                self.session.stage = Stage::TrailingBoxes;
                            }
                            _ => return JXL_DEC_ERROR,
                        }
                        continue;
                    }

                    if !self.refresh_pixel_format() {
                        return JXL_DEC_ERROR;
                    }
                    let Some((width, height)) = self.rs_basic_info().map(|b| b.size) else {
                        return JXL_DEC_ERROR;
                    };

                    let mut runner = self.settings.parallel_runner;
                    let runner_dyn = runner.as_mut().map(|r| r as &mut dyn RsParallelRunner);
                    let DecodeSession {
                        rs_decoder,
                        output,
                        input,
                        ..
                    } = &mut self.session;
                    let mut slice = input.as_slice();
                    let prev_len = slice.len();
                    let Some(mut output_buffers) = output.make_output_buffers(width, height) else {
                        return JXL_DEC_ERROR;
                    };
                    let dec = rs_decoder.as_mut().unwrap();
                    let event = dec.process(&mut slice, Some(&mut output_buffers), runner_dyn);
                    let consumed = prev_len - slice.len();
                    let parsed = dec.parsed_bytes();
                    drop(output_buffers);
                    self.session
                        .input
                        .record_rs_process(consumed, &event, parsed);
                    self.session.mid_frame_decode = true;
                    self.session.can_skip_current_frame = true;

                    match event {
                        Ok(Event::NeedMoreInput { .. }) => return JXL_DEC_NEED_MORE_INPUT,
                        Ok(Event::FrameComplete { has_more_frames }) => {
                            self.session.mid_frame_decode = false;
                            self.session.can_skip_current_frame = false;
                            if !self.session.output.finalize_all(width) {
                                return JXL_DEC_ERROR;
                            }
                            self.session.output.clear_targets();
                            self.session.has_more_frames = has_more_frames;
                            if !self.collect_new_aux_boxes() {
                                return JXL_DEC_ERROR;
                            }
                            self.session.stage = if has_more_frames {
                                Stage::FrameHeader
                            } else {
                                Stage::TrailingBoxes
                            };
                            if (self.settings.events_wanted & (JXL_DEC_FULL_IMAGE.0 as i32)) != 0 {
                                return JXL_DEC_FULL_IMAGE;
                            }
                        }
                        _ => return JXL_DEC_ERROR,
                    }
                }

                Stage::TrailingBoxes => {
                    if !self.wants_boxes() {
                        self.session.stage = Stage::Done;
                        continue;
                    }
                    if !self.session.boxes.trailing_parsed {
                        match self.step_rs_decoder(None) {
                            Ok(Event::NeedMoreInput { .. }) => return JXL_DEC_NEED_MORE_INPUT,
                            Ok(Event::Complete) => {
                                self.session.boxes.trailing_parsed = true;
                                if !self.collect_new_aux_boxes() {
                                    return JXL_DEC_ERROR;
                                }
                            }
                            _ => return JXL_DEC_ERROR,
                        }
                    }
                    let has_infinite_trailing = self
                        .session
                        .rs_decoder
                        .as_ref()
                        .and_then(|d| d.trailing_box())
                        .is_some();
                    if has_infinite_trailing {
                        self.session
                            .input
                            .drain_into(&mut self.session.boxes.trailing_box_extra);
                        if !self.session.input.is_closed() {
                            return JXL_DEC_NEED_MORE_INPUT;
                        }
                        let tb = self
                            .session
                            .rs_decoder
                            .as_ref()
                            .and_then(|d| d.trailing_box())
                            .unwrap();
                        if !self
                            .session
                            .boxes
                            .enqueue_trailing_box(tb, self.settings.decompress_boxes)
                        {
                            return JXL_DEC_ERROR;
                        }
                    }
                    self.session.stage = Stage::Done;
                }

                Stage::Done => return JXL_DEC_SUCCESS,
            }
        }
    }
}
