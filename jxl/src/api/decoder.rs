// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::marker::PhantomData;

use states::*;

use super::{
    JxlBasicInfo, JxlBitstreamInput, JxlColorProfile, JxlDecoderInner, JxlDecoderOptions,
    JxlOutputBuffer, JxlPixelFormat, ProcessingResult, TocEntry,
};
use crate::api::{BoxParserCheckpoint, JxlFrameHeader, JxlParallelRunner};
use crate::error::Result;
#[cfg(test)]
use crate::{frame::Frame, headers::FileHeader};

pub mod states {
    pub trait JxlState {}
    pub struct Initialized;
    pub struct WithImageInfo;
    pub struct WithFrameInfo;
    impl JxlState for Initialized {}
    impl JxlState for WithImageInfo {}
    impl JxlState for WithFrameInfo {}
}

// Q: do we plan to add support for box decoding?
// If we do, one way is to take a callback &[u8; 4] -> Box<dyn Write>.

/// High level API using the typestate pattern to forbid invalid usage.
pub struct JxlDecoder<State: JxlState> {
    inner: Box<JxlDecoderInner>,
    _state: PhantomData<State>,
}

#[cfg(test)]
pub type FrameCallback = dyn FnMut(&FileHeader, &Frame, usize) -> Result<()>;

/// Information about a single visible frame discovered while decoding.
#[derive(Debug, Clone)]
pub struct VisibleFrameInfo {
    /// Zero-based index among visible frames.
    pub index: usize,
    /// Duration in milliseconds (0 for still images or the last frame).
    pub duration_ms: f64,
    /// Duration in raw ticks from the animation header.
    pub duration_ticks: u32,
    /// Byte offset of this frame's header in the input file.
    pub file_offset: u64,
    /// Whether this is the last frame in the codestream.
    pub is_last: bool,
    /// Whether this frame is a seek-keyframe for visible-frame playback.
    ///
    /// This is equivalent to `seek_target.visible_frames_to_skip == 0`.
    pub is_keyframe: bool,
    /// Precomputed seek inputs for this visible frame.
    pub seek_target: VisibleFrameSeekTarget,
    /// Frame name, if any.
    pub name: String,
}

/// Computed seek inputs for a target visible frame.
#[derive(Debug, Clone, Copy)]
pub struct VisibleFrameSeekTarget {
    /// File byte offset to start feeding input from.
    pub decode_start_file_offset: u64,
    /// State of the box parser at the file offset we want to seek to.
    /// Pass this to [`JxlDecoder::start_new_frame`].
    pub box_parser_checkpoint: BoxParserCheckpoint,
    /// Number of visible frames to skip after seek-start before decoding the
    /// requested target frame.
    pub visible_frames_to_skip: usize,
}

impl<S: JxlState> JxlDecoder<S> {
    fn wrap_inner(inner: Box<JxlDecoderInner>) -> Self {
        Self {
            inner,
            _state: PhantomData,
        }
    }

    /// Sets a callback that processes all frames by calling `callback(frame, frame_index)`.
    #[cfg(test)]
    pub fn set_frame_callback(&mut self, callback: Box<FrameCallback>) {
        self.inner.set_frame_callback(callback);
    }

    /// Returns visible frame info entries collected so far.
    ///
    /// When `JxlDecoderOptions::scan_frames_only` is enabled this is the
    /// primary output of decoding.
    pub fn scanned_frames(&self) -> &[VisibleFrameInfo] {
        self.inner.scanned_frames()
    }

    fn map_inner_processing_result<SuccessState: JxlState>(
        self,
        inner_result: ProcessingResult<(), ()>,
    ) -> ProcessingResult<JxlDecoder<SuccessState>, Self> {
        match inner_result {
            ProcessingResult::Complete { .. } => ProcessingResult::Complete {
                result: JxlDecoder::wrap_inner(self.inner),
            },
            ProcessingResult::NeedsMoreInput { size_hint, .. } => {
                ProcessingResult::NeedsMoreInput {
                    size_hint,
                    fallback: self,
                }
            }
        }
    }
}

impl JxlDecoder<Initialized> {
    pub fn new(options: JxlDecoderOptions) -> Self {
        Self::wrap_inner(Box::new(JxlDecoderInner::new(options)))
    }

    pub fn process(
        mut self,
        input: &mut impl JxlBitstreamInput,
        parallel_runner: Option<&mut dyn JxlParallelRunner>,
    ) -> Result<ProcessingResult<JxlDecoder<WithImageInfo>, Self>> {
        let inner_result = self.inner.process(input, None, parallel_runner)?;
        Ok(self.map_inner_processing_result(inner_result))
    }
}

impl JxlDecoder<WithImageInfo> {
    /// Obtains the image's basic information.
    pub fn basic_info(&self) -> &JxlBasicInfo {
        self.inner.basic_info().unwrap()
    }

    /// Retrieves the file's color profile.
    pub fn embedded_color_profile(&self) -> &JxlColorProfile {
        self.inner.embedded_color_profile().unwrap()
    }

    /// Retrieves the current output color profile.
    pub fn output_color_profile(&self) -> &JxlColorProfile {
        self.inner.output_color_profile().unwrap()
    }

    /// Retrieves the current pixel format for output buffers.
    pub fn current_pixel_format(&self) -> &JxlPixelFormat {
        self.inner.current_pixel_format().unwrap()
    }

    /// Specifies pixel format for output buffers.
    ///
    /// Setting this may also change output color profile in some cases, if the profile was not set
    /// manually before.
    ///
    /// The pixel format can only be changed before the first frame header is
    /// decoded, i.e. right after basic info becomes available; afterwards
    /// this returns an error (except when the format does not change).
    pub fn set_pixel_format(&mut self, pixel_format: JxlPixelFormat) -> Result<()> {
        self.inner.set_pixel_format(pixel_format)
    }

    pub fn process(
        mut self,
        input: &mut impl JxlBitstreamInput,
        parallel_runner: Option<&mut dyn JxlParallelRunner>,
    ) -> Result<ProcessingResult<JxlDecoder<WithFrameInfo>, Self>> {
        let inner_result = self.inner.process(input, None, parallel_runner)?;
        Ok(self.map_inner_processing_result(inner_result))
    }

    /// Draws all the pixels we have data for. This is useful for i.e. previewing LF frames.
    ///
    /// Returns `true` if any new pixels were written to `buffers` since the
    /// previous call to `flush_pixels`; `false` if nothing new was rendered.
    ///
    /// Note: see `process` for alignment requirements for the buffer data.
    pub fn flush_pixels(
        &mut self,
        buffers: &mut [JxlOutputBuffer<'_>],
        parallel_runner: Option<&mut dyn JxlParallelRunner>,
    ) -> Result<bool> {
        self.inner.flush_pixels(buffers, parallel_runner)
    }

    pub fn has_more_frames(&self) -> bool {
        self.inner.has_more_frames()
    }

    /// Returns the total length of the JPEG XL file, once decoding is finished.
    /// This is needed because the decoder might over-consume bytes from the
    /// provided input stream in some cases.
    pub fn file_length(&self) -> Option<u64> {
        self.inner.file_length()
    }

    /// Resets frame-level state to prepare for decoding a new frame.
    ///
    /// After seeking the first time, scanned frame information will no longer
    /// be updated. If you seek before having completed decoding once, the scanned
    /// frames might be incomplete.
    ///
    /// After calling this, provide raw file input starting from
    /// `seek_target.decode_start_file_offset`.
    pub fn start_new_frame(&mut self, seek_target: VisibleFrameSeekTarget) {
        self.inner.start_new_frame(seek_target);
    }

    #[cfg(test)]
    pub(crate) fn set_use_simple_pipeline(&mut self, u: bool) {
        self.inner.set_use_simple_pipeline(u);
    }
}

impl JxlDecoder<WithFrameInfo> {
    /// Skip the current frame without decoding pixels.
    ///
    /// This reads section data from the input to advance past the frame, but
    /// does not render pixels. Reference frames that may be needed by later
    /// frames are still decoded internally.
    ///
    /// For efficient frame seeking in animations, enable
    /// `JxlDecoderOptions::scan_frames_only` and use
    /// [`scanned_frames`](JxlDecoder::scanned_frames), then
    /// [`start_new_frame`](JxlDecoder::start_new_frame) to jump directly to a
    /// target frame.
    pub fn skip_frame(
        mut self,
        input: &mut impl JxlBitstreamInput,
    ) -> Result<ProcessingResult<JxlDecoder<WithImageInfo>, Self>> {
        let inner_result = self.inner.process(input, None, None)?;
        Ok(self.map_inner_processing_result(inner_result))
    }

    pub fn frame_header(&self) -> JxlFrameHeader {
        self.inner.frame_header().unwrap()
    }

    /// Number of passes we have full data for in the current frame.
    ///
    /// A pass counts once every group has decoded it. The value only grows
    /// while a frame is being fed, which makes it a cheap trigger for
    /// progressive consumers: re-render when it increases.
    ///
    /// (Re-added downstream of upstream v0.6.0, which dropped it: Hikaru's
    /// progressive medical-image streaming drives its re-flush loop off this.)
    pub fn num_completed_passes(&self) -> usize {
        self.inner.num_completed_passes()
    }

    /// Returns the number of TOC entries in the current frame.
    ///
    /// The TOC (table of contents) describes the byte layout of frame
    /// sections. Use [`toc_entry`](Self::toc_entry) to get details about
    /// each entry. For single-group frames (small images) this is 1; for
    /// multi-group frames it is `2 + num_lf_groups + num_passes * num_groups`.
    ///
    /// Returns 0 if no frame is currently being decoded (e.g. under
    /// [`JxlDecoderOptions::scan_frames_only`]).
    pub fn toc_num_entries(&self) -> usize {
        self.inner.toc_num_entries().unwrap_or(0)
    }

    /// Returns the TOC entry at the given index, or `None` if out of bounds.
    ///
    /// # TOC layout
    ///
    /// Entries are returned in **bitstream (physical) order**. For
    /// single-group frames there is one [`TocGroupKind::All`] entry. For
    /// multi-group frames the *spec* order is:
    /// - index 0: [`TocGroupKind::LfGlobal`]
    /// - indices `1..=num_lf_groups`: [`TocGroupKind::LfGroup`]
    /// - index `1 + num_lf_groups`: [`TocGroupKind::HfGlobal`]
    /// - the rest: [`TocGroupKind::GroupPass`], pass-major
    ///
    /// When the frame's TOC is permuted, the physical order differs; the
    /// returned `kind` is resolved through the permutation, so it always
    /// describes the section actually stored at that physical position.
    ///
    /// The entry `offset` is relative to
    /// [`frame_data_offset`](Self::frame_data_offset). Designed for
    /// progressive-streaming use cases that need section byte boundaries
    /// without fully decoding the frame.
    ///
    /// [`TocGroupKind::All`]: crate::api::TocGroupKind::All
    /// [`TocGroupKind::LfGlobal`]: crate::api::TocGroupKind::LfGlobal
    /// [`TocGroupKind::LfGroup`]: crate::api::TocGroupKind::LfGroup
    /// [`TocGroupKind::HfGlobal`]: crate::api::TocGroupKind::HfGlobal
    /// [`TocGroupKind::GroupPass`]: crate::api::TocGroupKind::GroupPass
    pub fn toc_entry(&self, index: usize) -> Option<TocEntry> {
        self.inner.toc_entry(index)
    }

    /// Returns the total size of frame section data in bytes.
    ///
    /// This is the sum of all TOC entry sizes — the amount of section data
    /// needed to fully decode the frame (not counting the frame header/TOC).
    pub fn frame_data_size(&self) -> u64 {
        self.inner.frame_data_size().unwrap_or(0)
    }

    /// Returns the byte offset, from the start of the *input* (file-absolute,
    /// so it includes any ISOBMFF container overhead), at which the current
    /// frame's TOC-described section data begins — immediately after the frame
    /// header and TOC.
    ///
    /// [`toc_entry`](Self::toc_entry) offsets are relative to this. Add the two
    /// to locate a section in the original file bytes — useful for
    /// progressive-streaming consumers that slice the raw bitstream into
    /// prefixes.
    pub fn frame_data_offset(&self) -> u64 {
        self.inner.frame_data_offset().unwrap_or(0)
    }

    /// Draws all the pixels we have data for.
    ///
    /// Returns `true` if any new pixels were written to `buffers` since the
    /// previous call to `flush_pixels`; `false` if nothing new was rendered.
    ///
    /// Note: see `process` for alignment requirements for the buffer data.
    pub fn flush_pixels(
        &mut self,
        buffers: &mut [JxlOutputBuffer<'_>],
        parallel_runner: Option<&mut dyn JxlParallelRunner>,
    ) -> Result<bool> {
        self.inner.flush_pixels(buffers, parallel_runner)
    }

    /// Guarantees to populate exactly the appropriate part of the buffers.
    /// Wants one buffer for each non-ignored pixel type, i.e. color channels and each extra channel.
    ///
    /// Note: the data in `buffers` should have alignment requirements that are compatible with the
    /// requested pixel format. This means that, if we are asking for 2-byte or 4-byte output (i.e.
    /// u16/f16 and f32 respectively), each row in the provided buffers must be aligned to 2 or 4
    /// bytes respectively. If that is not the case, the library may panic.
    pub fn process<In: JxlBitstreamInput>(
        mut self,
        input: &mut In,
        buffers: &mut [JxlOutputBuffer<'_>],
        parallel_runner: Option<&mut dyn JxlParallelRunner>,
    ) -> Result<ProcessingResult<JxlDecoder<WithImageInfo>, Self>> {
        let inner_result = self.inner.process(input, Some(buffers), parallel_runner)?;
        Ok(self.map_inner_processing_result(inner_result))
    }
}
