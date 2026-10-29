// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![deny(unsafe_code)]

use std::collections::VecDeque;

use jxl::api::{
    AlphaMode, Event, JxlAuxBox, JxlAuxBoxType, JxlBasicInfo as RsBasicInfo,
    JxlColorProfile as RsColorProfile, JxlColorType, JxlDataFormat, JxlDecoder as RsDecoder,
    JxlDecoderOptions, JxlOutputBuffer, JxlParallelRunner as RsParallelRunner,
    JxlPixelFormat as RsPixelFormat, ProfileLevel,
};
use jxl::error::Result;

use crate::conversions::{to_rs_color_type, to_rs_data_format};
use crate::parallel_runner::RayonParallelRunner;
use crate::pixel_buffer::{
    BoxOutBuffer, ImageOutBuffer, ImageOutTarget, InputBuffer, compute_buffer_size,
};
use crate::types::{
    JXL_BIT_DEPTH_FROM_PIXEL_FORMAT, JXL_BLEND_REPLACE, JXL_COLOR_PROFILE_TARGET_DATA,
    JXL_COLOR_PROFILE_TARGET_ORIGINAL, JXL_DEC_BASIC_INFO, JXL_DEC_BOX, JXL_DEC_BOX_COMPLETE,
    JXL_DEC_BOX_NEED_MORE_OUTPUT, JXL_DEC_COLOR_ENCODING, JXL_DEC_ERROR, JXL_DEC_FRAME,
    JXL_DEC_FULL_IMAGE, JXL_DEC_NEED_IMAGE_OUT_BUFFER, JXL_DEC_NEED_MORE_INPUT,
    JXL_DEC_NEED_PREVIEW_OUT_BUFFER, JXL_DEC_PREVIEW_IMAGE, JXL_DEC_SUCCESS, JXL_FALSE, JXL_TRUE,
    JxlBitDepth, JxlBlendInfo, JxlColorProfileTarget, JxlDecoderStatus, JxlFrameHeader,
    JxlLayerInfo, JxlPixelFormat,
};

pub(crate) const CODESTREAM_EVENTS_MASK: i32 = (JXL_DEC_BASIC_INFO.0 as i32)
    | (JXL_DEC_COLOR_ENCODING.0 as i32)
    | (JXL_DEC_PREVIEW_IMAGE.0 as i32)
    | (JXL_DEC_FRAME.0 as i32)
    | (JXL_DEC_FULL_IMAGE.0 as i32);

pub(crate) const TOP_LEVEL_EVENTS_MASK: i32 = CODESTREAM_EVENTS_MASK | (JXL_DEC_BOX.0 as i32);

pub(crate) const SUPPORTED_EVENTS_MASK: i32 =
    TOP_LEVEL_EVENTS_MASK | (JXL_DEC_BOX_COMPLETE.0 as i32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum Stage {
    #[default]
    BeforeBasicInfo,
    EmitColorEncoding,
    PreviewNeedOutBuffer,
    PreviewDecode,
    MainFrameHeader,
    MainNeedOutBuffer,
    MainDecode,
    TrailingBoxes,
    BoxOutput,
    Done,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum PendingBoxSource {
    Finite(JxlAuxBoxType, usize),
    Trailing,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PendingBox {
    pub(crate) source: PendingBoxSource,
    pub(crate) box_type: [u8; 4],
    pub(crate) is_compressed: bool,
    pub(crate) contents_size: u64,
    pub(crate) raw_size: u64,
}

pub(crate) struct ActiveBox {
    pub(crate) pending: PendingBox,
    pub(crate) payload: Option<Vec<u8>>,
    pub(crate) written: usize,
    pub(crate) buffer_set: bool,
}

#[derive(Default)]
pub(crate) struct StreamState {
    pub(crate) stage: Stage,
    pub(crate) process_started: bool,
    pub(crate) input: Option<InputBuffer>,
    pub(crate) input_had_bytes: bool,
    pub(crate) held_back_byte: bool,
    pub(crate) can_continue: bool,
    pub(crate) input_closed: bool,
    pub(crate) have_container: Option<bool>,
    pub(crate) signature_probe: Vec<u8>,
    pub(crate) basic_info_size_hint: Option<usize>,
    pub(crate) frame_flushed: bool,
    pub(crate) current_frame_header: Option<(JxlFrameHeader, String)>,
    pub(crate) preview_out_target: Option<ImageOutBuffer>,
    pub(crate) image_out_target: Option<ImageOutTarget>,
    pub(crate) extra_channel_targets: Vec<Option<ImageOutBuffer>>,
    pub(crate) queued_exif_count: usize,
    pub(crate) queued_xmp_count: usize,
    pub(crate) queued_trailing: bool,
    pub(crate) trailing_complete: bool,
    pub(crate) trailing_box_extra: Vec<u8>,
    pub(crate) pending_boxes: VecDeque<PendingBox>,
    pub(crate) current_box: Option<ActiveBox>,
    pub(crate) box_out_buffer: Option<BoxOutBuffer>,
}

pub struct JxlDecoder {
    pub(crate) rs_decoder: RsDecoder,
    pub(crate) stream: StreamState,
    pub(crate) events_wanted: i32,
    pub(crate) parallel: bool,
    pub(crate) keep_orientation: bool,
    pub(crate) render_spotcolors: bool,
    pub(crate) unpremultiply_alpha: bool,
    pub(crate) decompress_boxes: bool,
    pub(crate) output_bit_depth: JxlBitDepth,
}

impl Default for JxlDecoder {
    fn default() -> Self {
        Self::new()
    }
}

fn step_decoder(
    dec: &mut RsDecoder,
    input: &mut Option<InputBuffer>,
    buffers: Option<&mut [JxlOutputBuffer<'_>]>,
    parallel: bool,
) -> Result<Event> {
    let mut runner = RayonParallelRunner;
    let runner_opt = parallel.then_some(&mut runner as &mut dyn RsParallelRunner);
    match input.as_mut() {
        Some(inp) => {
            let mut slice = inp.as_slice();
            let initial_len = slice.len();
            let res = dec.process(&mut slice, buffers, runner_opt);
            inp.consume(initial_len - slice.len());
            res
        }
        None => {
            let mut empty: &[u8] = &[];
            dec.process(&mut empty, buffers, runner_opt)
        }
    }
}

pub(crate) fn make_main_output_buffers<'a>(
    width: usize,
    height: usize,
    image_out_target: &'a mut Option<ImageOutTarget>,
    extra_channel_targets: &'a mut [Option<ImageOutBuffer>],
) -> Option<Vec<JxlOutputBuffer<'a>>> {
    let mut output_buffers = Vec::new();
    if let Some(color_target) = image_out_target.as_mut() {
        output_buffers.push(color_target.make_output_buffer(width, height)?);
    }
    for ec_target in extra_channel_targets.iter_mut().flatten() {
        output_buffers.push(ec_target.make_output_buffer(width, height)?);
    }
    Some(output_buffers)
}

impl JxlDecoder {
    pub(crate) fn new() -> Self {
        Self {
            rs_decoder: RsDecoder::new(JxlDecoderOptions::default()),
            stream: StreamState::default(),
            events_wanted: 0,
            parallel: false,
            keep_orientation: false,
            render_spotcolors: true,
            unpremultiply_alpha: false,
            decompress_boxes: false,
            output_bit_depth: JxlBitDepth {
                type_: JXL_BIT_DEPTH_FROM_PIXEL_FORMAT,
                bits_per_sample: 0,
                exponent_bits_per_sample: 0,
            },
        }
    }

    pub(crate) fn set_before_process(&mut self, f: impl FnOnce(&mut Self)) -> JxlDecoderStatus {
        if self.stream.process_started {
            return JXL_DEC_ERROR;
        }
        f(self);
        JXL_DEC_SUCCESS
    }

    pub(crate) fn basic_info_ref(&self) -> Option<&RsBasicInfo> {
        self.rs_decoder.basic_info()
    }

    pub(crate) fn has_basic_info(&self) -> bool {
        self.basic_info_ref().is_some()
    }

    fn build_options(&self, skip_preview: bool) -> JxlDecoderOptions {
        let mut options = JxlDecoderOptions::default();
        options.adjust_orientation = !self.keep_orientation;
        options.render_spot_colors = self.render_spotcolors;
        options.skip_preview = skip_preview;
        options.alpha_mode = if self.unpremultiply_alpha {
            AlphaMode::Unpremultiplied
        } else {
            AlphaMode::AsIs
        };
        if self.wants_boxes() {
            options.request_aux_boxes = vec![JxlAuxBoxType::EXIF, JxlAuxBoxType::XMP];
        }
        if !self.wants_preview() && !self.wants_main_frames() {
            options.scan_frames_only = true;
        }
        options.max_profile_level = ProfileLevel::Main10;
        options
    }

    fn wants_preview(&self) -> bool {
        (self.events_wanted & (JXL_DEC_PREVIEW_IMAGE.0 as i32)) != 0
    }

    fn wants_boxes(&self) -> bool {
        (self.events_wanted & (JXL_DEC_BOX.0 as i32)) != 0
    }

    fn after_frames_stage(&self) -> Stage {
        if self.wants_boxes() && self.stream.have_container != Some(false) {
            Stage::TrailingBoxes
        } else {
            Stage::Done
        }
    }

    fn wants_main_frames(&self) -> bool {
        let mask = (JXL_DEC_FRAME.0 as i32) | (JXL_DEC_FULL_IMAGE.0 as i32);
        (self.events_wanted & mask) != 0
    }

    fn wants_frame_pixels(&self) -> bool {
        (self.events_wanted & (JXL_DEC_FULL_IMAGE.0 as i32)) != 0
            || self.has_any_main_output_target()
    }

    pub(crate) fn can_set_color_profile(&self) -> bool {
        let Some(info) = self.basic_info_ref() else {
            return false;
        };
        if self.wants_preview() && info.preview_size.is_some() {
            matches!(
                self.stream.stage,
                Stage::EmitColorEncoding | Stage::PreviewNeedOutBuffer
            )
        } else {
            self.rs_decoder.scanned_frames().is_empty()
        }
    }

    fn queue_new_boxes(&mut self) -> bool {
        let mut added = false;
        for (ty, count) in [
            (JxlAuxBoxType::EXIF, &mut self.stream.queued_exif_count),
            (JxlAuxBoxType::XMP, &mut self.stream.queued_xmp_count),
        ] {
            let boxes = self.rs_decoder.aux_boxes(ty);
            while *count < boxes.len() {
                let b = &boxes[*count];
                let extra = if b.is_compressed() { 4 } else { 0 };
                let header_size = if b.has_xlbox() { 16 } else { 8 };
                let contents_size = (b.raw_data().len() + extra) as u64;
                self.stream.pending_boxes.push_back(PendingBox {
                    source: PendingBoxSource::Finite(ty, *count),
                    box_type: ty.0,
                    is_compressed: b.is_compressed(),
                    contents_size,
                    raw_size: contents_size + header_size,
                });
                *count += 1;
                added = true;
            }
        }
        if self.stream.trailing_complete
            && !self.stream.queued_trailing
            && let Some(tb) = self.rs_decoder.trailing_box()
        {
            if let Some(inp) = self.stream.input.as_mut() {
                let rem = inp.as_slice();
                if !rem.is_empty() {
                    self.stream.trailing_box_extra.extend_from_slice(rem);
                    let len = rem.len();
                    inp.consume(len);
                }
            }
            if self.stream.input_closed {
                self.stream.queued_trailing = true;
                self.stream.pending_boxes.push_back(PendingBox {
                    source: PendingBoxSource::Trailing,
                    box_type: tb.box_type().0,
                    is_compressed: tb.is_compressed(),
                    contents_size: 0,
                    raw_size: 0,
                });
                added = true;
            }
        }
        added
    }

    fn capture_current_frame_header(&self) -> Option<(JxlFrameHeader, String)> {
        let fh = self.rs_decoder.frame_header()?;
        let scanned = self.rs_decoder.scanned_frames().last();
        let duration = scanned.map_or(0, |f| f.duration_ticks);
        let timecode = scanned.map_or(0, |f| f.timecode);
        let is_last = scanned.is_some_and(|f| f.is_last);
        let header = JxlFrameHeader {
            duration,
            timecode,
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
        };
        Some((header, fh.name))
    }

    pub(crate) fn is_gray(&self) -> bool {
        self.rs_decoder
            .embedded_color_profile()
            .is_some_and(|p| p.channels() == 1)
    }

    fn desired_rs_pixel_format(&self, preview: bool) -> Option<RsPixelFormat> {
        let basic = self.basic_info_ref()?;
        let is_gray = self.is_gray();
        let color_format = if preview {
            self.stream
                .preview_out_target
                .as_ref()
                .map(ImageOutBuffer::format)
        } else {
            self.stream
                .image_out_target
                .as_ref()
                .map(ImageOutTarget::format)
        };
        let (color_type, color_data_format) = if let Some(cf) = color_format {
            let ct = to_rs_color_type(cf.num_channels, is_gray)?;
            let df = to_rs_data_format(&cf, &self.output_bit_depth, &basic.bit_depth)?;
            (ct, Some(df))
        } else {
            let ct = if is_gray {
                JxlColorType::Grayscale
            } else {
                JxlColorType::Rgb
            };
            (ct, Some(JxlDataFormat::U8 { bit_depth: 8 }))
        };

        let mut extra_channel_format = Vec::with_capacity(basic.extra_channels.len());
        for (i, ec) in basic.extra_channels.iter().enumerate() {
            let target = (!preview)
                .then(|| self.stream.extra_channel_targets.get(i)?.as_ref())
                .flatten();
            let df = match target {
                Some(t) => Some(to_rs_data_format(
                    &t.format(),
                    &self.output_bit_depth,
                    &ec.bit_depth,
                )?),
                None => None,
            };
            extra_channel_format.push(df);
        }

        Some(RsPixelFormat {
            color_type,
            color_data_format,
            extra_channel_format,
        })
    }

    pub(crate) fn apply_main_pixel_format(&mut self) -> bool {
        let Some(desired_format) = self.desired_rs_pixel_format(false) else {
            return false;
        };
        self.rs_decoder.set_pixel_format(desired_format).is_ok()
    }

    pub(crate) fn has_any_main_output_target(&self) -> bool {
        self.stream.image_out_target.is_some()
            || self
                .stream
                .extra_channel_targets
                .iter()
                .any(|t| t.is_some())
    }

    pub(crate) fn select_color_profile(
        &self,
        target: JxlColorProfileTarget,
    ) -> Option<&RsColorProfile> {
        match target {
            JXL_COLOR_PROFILE_TARGET_ORIGINAL => self.rs_decoder.embedded_color_profile(),
            JXL_COLOR_PROFILE_TARGET_DATA => self.rs_decoder.output_color_profile(),
            _ => None,
        }
    }

    pub(crate) fn color_buffer_size(
        &self,
        size: (usize, usize),
        format: &JxlPixelFormat,
    ) -> Option<usize> {
        to_rs_color_type(format.num_channels, self.is_gray())?;
        compute_buffer_size(size.0, size.1, format)
    }

    pub(crate) fn process_input(&mut self) -> JxlDecoderStatus {
        let status = self.process_input_inner();
        self.stream.can_continue = self.stream.stage != Stage::Done
            && !matches!(
                status,
                JXL_DEC_NEED_MORE_INPUT | JXL_DEC_SUCCESS | JXL_DEC_ERROR
            );
        status
    }

    fn process_input_inner(&mut self) -> JxlDecoderStatus {
        if !self.stream.process_started {
            self.stream.process_started = true;
            let opts = self.build_options(!self.wants_preview());
            self.rs_decoder.set_options(opts);
        }

        loop {
            match self.stream.stage {
                Stage::BeforeBasicInfo => {
                    match step_decoder(
                        &mut self.rs_decoder,
                        &mut self.stream.input,
                        None,
                        self.parallel,
                    ) {
                        Ok(Event::NeedMoreInput { size_hint }) => {
                            self.stream.basic_info_size_hint = Some(size_hint);
                            return JXL_DEC_NEED_MORE_INPUT;
                        }
                        Ok(Event::BasicInfo) => {
                            self.stream.basic_info_size_hint = Some(0);
                            let num_ec =
                                self.basic_info_ref().map_or(0, |b| b.extra_channels.len());
                            self.stream.extra_channel_targets.clear();
                            self.stream
                                .extra_channel_targets
                                .resize_with(num_ec, || None);
                            let _ = self.apply_main_pixel_format();
                            self.stream.stage = Stage::EmitColorEncoding;
                            if (self.events_wanted & (JXL_DEC_BASIC_INFO.0 as i32)) != 0 {
                                return JXL_DEC_BASIC_INFO;
                            }
                        }
                        _ => return JXL_DEC_ERROR,
                    }
                }

                Stage::EmitColorEncoding => {
                    let has_preview = self
                        .basic_info_ref()
                        .is_some_and(|b| b.preview_size.is_some());
                    self.stream.stage = if self.wants_preview() && has_preview {
                        Stage::PreviewNeedOutBuffer
                    } else if self.wants_main_frames() {
                        Stage::MainFrameHeader
                    } else {
                        self.after_frames_stage()
                    };
                    if (self.events_wanted & (JXL_DEC_COLOR_ENCODING.0 as i32)) != 0 {
                        return JXL_DEC_COLOR_ENCODING;
                    }
                }

                Stage::PreviewNeedOutBuffer => {
                    if self.stream.preview_out_target.is_none() {
                        return JXL_DEC_NEED_PREVIEW_OUT_BUFFER;
                    }
                    let Some(preview_fmt) = self.desired_rs_pixel_format(true) else {
                        return JXL_DEC_ERROR;
                    };
                    if self.rs_decoder.set_pixel_format(preview_fmt).is_err() {
                        return JXL_DEC_ERROR;
                    }
                    self.stream.stage = Stage::PreviewDecode;
                }

                Stage::PreviewDecode => {
                    let Some((pw, ph)) = self.basic_info_ref().and_then(|b| b.preview_size) else {
                        return JXL_DEC_ERROR;
                    };
                    let Some(target) = self.stream.preview_out_target.as_mut() else {
                        return JXL_DEC_ERROR;
                    };
                    let Some(buf) = target.make_output_buffer(pw, ph) else {
                        return JXL_DEC_ERROR;
                    };
                    let mut buffers = [buf];
                    let event = step_decoder(
                        &mut self.rs_decoder,
                        &mut self.stream.input,
                        Some(&mut buffers),
                        self.parallel,
                    );
                    match event {
                        Ok(Event::NeedMoreInput { .. }) => return JXL_DEC_NEED_MORE_INPUT,
                        Ok(Event::FrameHeader) => continue,
                        Ok(Event::FrameComplete { has_more_frames }) => {
                            self.stream.preview_out_target = None;
                            let _ = self.apply_main_pixel_format();
                            self.stream.stage = if self.wants_main_frames() && has_more_frames {
                                Stage::MainFrameHeader
                            } else {
                                self.after_frames_stage()
                            };
                            return JXL_DEC_PREVIEW_IMAGE;
                        }
                        _ => return JXL_DEC_ERROR,
                    }
                }

                Stage::MainFrameHeader => {
                    self.stream.current_frame_header = None;
                    self.stream.frame_flushed = false;
                    match step_decoder(
                        &mut self.rs_decoder,
                        &mut self.stream.input,
                        None,
                        self.parallel,
                    ) {
                        Ok(Event::NeedMoreInput { .. }) => return JXL_DEC_NEED_MORE_INPUT,
                        Ok(Event::FrameHeader) => {
                            self.stream.current_frame_header = self.capture_current_frame_header();
                            self.stream.stage = Stage::MainNeedOutBuffer;
                            if (self.events_wanted & (JXL_DEC_FRAME.0 as i32)) != 0 {
                                return JXL_DEC_FRAME;
                            }
                        }
                        Ok(Event::Complete) => {
                            self.stream.trailing_complete = true;
                            self.stream.stage = self.after_frames_stage();
                        }
                        _ => return JXL_DEC_ERROR,
                    }
                }

                Stage::MainNeedOutBuffer => {
                    if !self.wants_frame_pixels() {
                        self.stream.stage = Stage::MainDecode;
                        continue;
                    }
                    if self.stream.image_out_target.is_none() {
                        return JXL_DEC_NEED_IMAGE_OUT_BUFFER;
                    }
                    if !self.apply_main_pixel_format() {
                        return JXL_DEC_ERROR;
                    }
                    self.stream.stage = Stage::MainDecode;
                }

                Stage::MainDecode => {
                    let skipping = self.stream.image_out_target.is_none();
                    let img_size = self.basic_info_ref().map(|b| b.size);
                    let mut output_buffers = if skipping {
                        None
                    } else {
                        let Some((width, height)) = img_size else {
                            return JXL_DEC_ERROR;
                        };
                        let Some(bufs) = make_main_output_buffers(
                            width,
                            height,
                            &mut self.stream.image_out_target,
                            &mut self.stream.extra_channel_targets,
                        ) else {
                            return JXL_DEC_ERROR;
                        };
                        Some(bufs)
                    };

                    let event = step_decoder(
                        &mut self.rs_decoder,
                        &mut self.stream.input,
                        output_buffers.as_deref_mut(),
                        self.parallel,
                    );
                    drop(output_buffers);

                    match event {
                        Ok(Event::NeedMoreInput { .. }) => return JXL_DEC_NEED_MORE_INPUT,
                        Ok(Event::FrameComplete { has_more_frames }) => {
                            let emit_ok = match (self.stream.image_out_target.as_ref(), img_size) {
                                (Some(target), Some((width, height))) => {
                                    target.emit_callbacks(width, height, self.parallel)
                                }
                                _ => true,
                            };
                            self.stream.frame_flushed = false;
                            self.stream.image_out_target = None;
                            self.stream.extra_channel_targets.fill_with(|| None);
                            if !emit_ok {
                                return JXL_DEC_ERROR;
                            }
                            self.stream.stage = if has_more_frames {
                                Stage::MainFrameHeader
                            } else {
                                self.after_frames_stage()
                            };
                            if !skipping
                                && (self.events_wanted & (JXL_DEC_FULL_IMAGE.0 as i32)) != 0
                            {
                                return JXL_DEC_FULL_IMAGE;
                            }
                        }
                        Ok(Event::Complete) if skipping => {
                            self.stream.trailing_complete = true;
                            self.stream.stage = self.after_frames_stage();
                        }
                        _ => return JXL_DEC_ERROR,
                    }
                }

                Stage::TrailingBoxes => {
                    self.queue_new_boxes();
                    if let Some(pending) = self.stream.pending_boxes.pop_front() {
                        self.stream.current_box = Some(ActiveBox {
                            pending,
                            payload: None,
                            written: 0,
                            buffer_set: false,
                        });
                        self.stream.stage = Stage::BoxOutput;
                        return JXL_DEC_BOX;
                    }
                    if self.stream.trailing_complete {
                        if self.rs_decoder.trailing_box().is_some() && !self.stream.queued_trailing
                        {
                            return JXL_DEC_NEED_MORE_INPUT;
                        }
                        self.stream.stage = Stage::Done;
                        return JXL_DEC_SUCCESS;
                    }
                    match step_decoder(
                        &mut self.rs_decoder,
                        &mut self.stream.input,
                        None,
                        self.parallel,
                    ) {
                        Ok(Event::FrameHeader) | Ok(Event::FrameComplete { .. }) => continue,
                        Ok(Event::Complete) => {
                            self.stream.trailing_complete = true;
                        }
                        Ok(Event::NeedMoreInput { .. }) => {
                            if !self.queue_new_boxes() {
                                return JXL_DEC_NEED_MORE_INPUT;
                            }
                        }
                        _ => return JXL_DEC_ERROR,
                    }
                }

                Stage::BoxOutput => {
                    let Some(active) = self.stream.current_box.as_mut() else {
                        return JXL_DEC_ERROR;
                    };
                    if !active.buffer_set {
                        self.stream.stage = Stage::TrailingBoxes;
                        continue;
                    }
                    let Some(out_buf) = self.stream.box_out_buffer.as_mut() else {
                        self.stream.stage = Stage::TrailingBoxes;
                        continue;
                    };
                    if active.payload.is_none() {
                        let (aux_box, trailing): (&JxlAuxBox, &[u8]) = match active.pending.source {
                            PendingBoxSource::Finite(ty, idx) => {
                                (&self.rs_decoder.aux_boxes(ty)[idx], &[])
                            }
                            PendingBoxSource::Trailing => (
                                self.rs_decoder.trailing_box().unwrap(),
                                &self.stream.trailing_box_extra,
                            ),
                        };
                        let data = if self.decompress_boxes && aux_box.is_compressed() {
                            #[cfg(feature = "brotli")]
                            {
                                let Ok(cow) = aux_box.data(trailing) else {
                                    return JXL_DEC_ERROR;
                                };
                                cow.into_owned()
                            }
                            #[cfg(not(feature = "brotli"))]
                            {
                                return JXL_DEC_ERROR;
                            }
                        } else {
                            let prefix: &[u8] = if aux_box.is_compressed() {
                                &active.pending.box_type
                            } else {
                                &[]
                            };
                            let mut v = Vec::with_capacity(
                                prefix.len() + aux_box.raw_data().len() + trailing.len(),
                            );
                            v.extend_from_slice(prefix);
                            v.extend_from_slice(aux_box.raw_data());
                            v.extend_from_slice(trailing);
                            v
                        };
                        active.payload = Some(data);
                    }
                    let payload = active.payload.as_ref().unwrap();
                    active.written += out_buf.write(&payload[active.written..]);
                    if active.written < payload.len() {
                        return JXL_DEC_BOX_NEED_MORE_OUTPUT;
                    }
                    self.stream.stage = Stage::TrailingBoxes;
                    if (self.events_wanted & (JXL_DEC_BOX_COMPLETE.0 as i32)) != 0 {
                        return JXL_DEC_BOX_COMPLETE;
                    }
                }

                Stage::Done => return JXL_DEC_SUCCESS,
            }
        }
    }
}
