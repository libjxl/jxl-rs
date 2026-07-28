// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use super::{
    JxlBasicInfo, JxlBitstreamInput, JxlColorProfile, JxlDecoderInner, JxlDecoderOptions,
    JxlOutputBuffer, JxlPixelFormat, ProcessingResult, TocEntry,
};
use crate::{
    api::{BoxParserCheckpoint, JxlFrameHeader},
    error::Result,
};
#[cfg(test)]
use crate::{frame::Frame, headers::FileHeader};
use states::*;
use std::marker::PhantomData;

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
    ) -> Result<ProcessingResult<JxlDecoder<WithImageInfo>, Self>> {
        let inner_result = self.inner.process(input, None)?;
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
    pub fn set_pixel_format(&mut self, pixel_format: JxlPixelFormat) {
        self.inner.set_pixel_format(pixel_format);
    }

    pub fn process(
        mut self,
        input: &mut impl JxlBitstreamInput,
    ) -> Result<ProcessingResult<JxlDecoder<WithFrameInfo>, Self>> {
        let inner_result = self.inner.process(input, None)?;
        Ok(self.map_inner_processing_result(inner_result))
    }

    /// Draws all the pixels we have data for. This is useful for i.e. previewing LF frames.
    ///
    /// Returns `true` if any new pixels were written to `buffers` since the
    /// previous call to `flush_pixels`; `false` if nothing new was rendered.
    ///
    /// Note: see `process` for alignment requirements for the buffer data.
    pub fn flush_pixels(&mut self, buffers: &mut [JxlOutputBuffer<'_>]) -> Result<bool> {
        self.inner.flush_pixels(buffers)
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
        let inner_result = self.inner.process(input, None)?;
        Ok(self.map_inner_processing_result(inner_result))
    }

    pub fn frame_header(&self) -> JxlFrameHeader {
        self.inner.frame_header().unwrap()
    }

    /// Returns the minimum number of completed passes across the current
    /// frame's groups. For single-pass frames this is 0 or 1; for progressive
    /// frames it climbs as passes arrive. Used to drive progressive rendering.
    pub fn num_completed_passes(&self) -> usize {
        self.inner.num_completed_passes().unwrap()
    }

    /// Returns the number of TOC entries in the current frame.
    ///
    /// The TOC (table of contents) describes the byte layout of frame
    /// sections. Use [`toc_entry`](Self::toc_entry) to get details about
    /// each entry. For single-group frames (small images) this is 1; for
    /// multi-group frames it is `2 + num_lf_groups + num_passes * num_groups`.
    pub fn toc_num_entries(&self) -> usize {
        self.inner.toc_num_entries().unwrap()
    }

    /// Returns the TOC entry at the given index, or `None` if out of bounds.
    ///
    /// # TOC layout
    ///
    /// For single-group frames there is one [`TocGroupKind::All`] entry.
    /// For multi-group frames the order is:
    /// - index 0: [`TocGroupKind::LfGlobal`]
    /// - indices `1..=num_lf_groups`: [`TocGroupKind::LfGroup`]
    /// - index `1 + num_lf_groups`: [`TocGroupKind::HfGlobal`]
    /// - the rest: [`TocGroupKind::GroupPass`], pass-major
    ///
    /// The entry `offset` is relative to the start of frame data (after the
    /// frame header) in the original (un-permuted) layout. Designed for
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
    /// needed to fully decode the frame (not counting the frame header).
    pub fn frame_data_size(&self) -> u64 {
        self.inner.frame_data_size().unwrap()
    }

    /// Returns the byte offset, from the start of the input (file-absolute;
    /// includes any ISOBMFF container), at which the current frame's
    /// TOC-described section data begins (immediately after the frame header).
    ///
    /// [`toc_entry`](Self::toc_entry) offsets are relative to this position.
    /// Add the two to locate a section in the original codestream bytes —
    /// useful for progressive-streaming consumers that slice the raw
    /// bitstream into prefixes.
    pub fn frame_data_offset(&self) -> u64 {
        self.inner.frame_data_offset().unwrap()
    }

    /// Draws all the pixels we have data for.
    ///
    /// Returns `true` if any new pixels were written to `buffers` since the
    /// previous call to `flush_pixels`; `false` if nothing new was rendered.
    ///
    /// Note: see `process` for alignment requirements for the buffer data.
    pub fn flush_pixels(&mut self, buffers: &mut [JxlOutputBuffer<'_>]) -> Result<bool> {
        self.inner.flush_pixels(buffers)
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
    ) -> Result<ProcessingResult<JxlDecoder<WithImageInfo>, Self>> {
        let inner_result = self.inner.process(input, Some(buffers))?;
        Ok(self.map_inner_processing_result(inner_result))
    }
}

// ---- TOC API tests ---------------------------------------------------------

#[cfg(test)]
mod toc_api_tests {
    use super::states::{self, WithFrameInfo};
    use crate::api::{JxlDecoder, JxlDecoderOptions, ProcessingResult, TocEntry, TocGroupKind};

    /// Drive a fresh decoder over `file` to the `WithFrameInfo` state, where
    /// the TOC API is available.
    fn decode_to_frame_info(file: &[u8]) -> JxlDecoder<WithFrameInfo> {
        let options = JxlDecoderOptions::default();
        let mut decoder = JxlDecoder::<states::Initialized>::new(options);
        let mut input = file;
        let mut with_info = loop {
            match decoder.process(&mut input).unwrap() {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { fallback, .. } => decoder = fallback,
            }
        };
        loop {
            match with_info.process(&mut input).unwrap() {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { fallback, .. } => with_info = fallback,
            }
        }
    }

    fn collect_toc(d: &JxlDecoder<WithFrameInfo>) -> Vec<TocEntry> {
        (0..d.toc_num_entries())
            .map(|i| d.toc_entry(i).expect("in-range TOC entry"))
            .collect()
    }

    /// Invariants that must hold for any valid frame's TOC, derived purely
    /// from the entries themselves (no external reference decoder).
    fn assert_toc_invariants(d: &JxlDecoder<WithFrameInfo>) {
        let entries = collect_toc(d);
        assert!(!entries.is_empty(), "a frame always has >= 1 TOC entry");

        // Offsets are relative to frame_data_offset, start at 0, and are the
        // running sum of preceding section sizes (contiguous layout).
        let mut acc = 0u64;
        for (i, e) in entries.iter().enumerate() {
            assert_eq!(e.offset, acc, "entry {i} offset is not contiguous");
            acc += e.size as u64;
        }
        // frame_data_size is the total of all section sizes.
        assert_eq!(d.frame_data_size(), acc, "frame_data_size != sum of sizes");
        // Section data starts after the codestream header + frame header + TOC.
        assert!(d.frame_data_offset() > 0, "frame_data_offset should be > 0");

        // The multiset of kinds must be exactly the JPEG XL section layout —
        // this validates the (possibly permuted) bitstream-order -> spec-kind
        // mapping is internally consistent.
        if entries.len() == 1 {
            assert_eq!(entries[0].kind, TocGroupKind::All);
            return;
        }
        let mut lf_globals = 0;
        let mut hf_globals = 0;
        let mut lf_group_idxs = Vec::new();
        let mut group_passes = Vec::new();
        for e in &entries {
            match e.kind {
                TocGroupKind::All => panic!("All kind in a multi-entry frame"),
                TocGroupKind::LfGlobal => lf_globals += 1,
                TocGroupKind::HfGlobal => hf_globals += 1,
                TocGroupKind::LfGroup(i) => lf_group_idxs.push(i),
                TocGroupKind::GroupPass {
                    pass_idx,
                    group_idx,
                } => group_passes.push((pass_idx, group_idx)),
            }
        }
        assert_eq!(lf_globals, 1, "exactly one LfGlobal expected");
        assert_eq!(hf_globals, 1, "exactly one HfGlobal expected");

        // LfGroup indices form a complete 0..num_lf_groups set.
        lf_group_idxs.sort_unstable();
        for (i, idx) in lf_group_idxs.iter().enumerate() {
            assert_eq!(*idx as usize, i, "LfGroup indices not a contiguous 0..n");
        }

        // GroupPass (pass, group) pairs form a complete pass x group grid.
        let num_groups = group_passes.iter().map(|&(_, g)| g).max().unwrap_or(0) as usize + 1;
        let num_passes = group_passes.len() / num_groups.max(1);
        assert_eq!(
            group_passes.len(),
            num_groups * num_passes,
            "GroupPass count is not num_groups * num_passes"
        );
        let mut seen = std::collections::HashSet::new();
        for &(p, g) in &group_passes {
            assert!(seen.insert((p, g)), "duplicate GroupPass ({p}, {g})");
            assert!((g as usize) < num_groups, "group_idx out of range");
            assert!((p as usize) < num_passes, "pass_idx out of range");
        }
    }

    #[test]
    fn test_toc_invariants_basic() {
        let file = std::fs::read("resources/test/basic.jxl").unwrap();
        assert_toc_invariants(&decode_to_frame_info(&file));
    }

    #[test]
    fn test_toc_invariants_multigroup() {
        // A multi-group image exercises the LfGroup / HfGlobal / GroupPass
        // layout rather than the single "All" entry.
        let file = std::fs::read("resources/test/multiple_lf_420.jxl").unwrap();
        assert_toc_invariants(&decode_to_frame_info(&file));
    }

    #[test]
    fn test_toc_invariants_permuted() {
        // has_permutation.jxl has a permuted TOC; the invariants check that the
        // bitstream-order -> spec-kind mapping survives the permutation.
        let file = std::fs::read("resources/test/has_permutation.jxl").unwrap();
        assert_toc_invariants(&decode_to_frame_info(&file));
    }

    #[test]
    fn test_toc_permutation_container_consistency() {
        // The bare and containerised variants of the same permuted image must
        // report identical TOC structure (kinds, indices, sizes, relative
        // offsets) and frame_data_size. Only frame_data_offset differs — the
        // containerised file's section data starts later by the box overhead.
        let bare = std::fs::read("resources/test/has_permutation.jxl").unwrap();
        let cont = std::fs::read("resources/test/has_permutation_with_container.jxl").unwrap();

        let db = decode_to_frame_info(&bare);
        let dc = decode_to_frame_info(&cont);

        let eb = collect_toc(&db);
        let ec = collect_toc(&dc);

        assert_eq!(eb.len(), ec.len(), "entry count differs bare vs container");
        for (i, (b, c)) in eb.iter().zip(ec.iter()).enumerate() {
            assert_eq!(b.kind, c.kind, "entry {i} kind differs");
            assert_eq!(b.offset, c.offset, "entry {i} relative offset differs");
            assert_eq!(b.size, c.size, "entry {i} size differs");
        }
        assert_eq!(
            db.frame_data_size(),
            dc.frame_data_size(),
            "frame_data_size differs bare vs container"
        );
        assert!(
            dc.frame_data_offset() > db.frame_data_offset(),
            "containerised frame_data_offset ({}) should exceed bare ({}) by the \
             container box overhead",
            dc.frame_data_offset(),
            db.frame_data_offset(),
        );
    }

    /// Drive a decoder to `WithFrameInfo` by feeding `chunk_size` bytes at a
    /// time, exercising the incremental `OutOfBounds` parse path (frame header
    /// + TOC split across multiple `process()` calls).
    fn decode_to_frame_info_chunked(file: &[u8], chunk_size: usize) -> JxlDecoder<WithFrameInfo> {
        let options = JxlDecoderOptions::default();
        let mut decoder = JxlDecoder::<states::Initialized>::new(options);
        let mut remaining = file;
        let mut window = &remaining[0..0];
        // Stage 1: Initialized -> WithImageInfo.
        let mut with_info = loop {
            window = &remaining[..(window.len() + chunk_size).min(remaining.len())];
            let before = window.len();
            let res = decoder.process(&mut window).unwrap();
            remaining = &remaining[(before - window.len())..];
            match res {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { fallback, .. } => {
                    assert!(!remaining.is_empty(), "ran out of input before image info");
                    decoder = fallback;
                }
            }
        };
        // Stage 2: WithImageInfo -> WithFrameInfo (this is where the TOC is
        // parsed; small chunks force the OutOfBounds retry path).
        loop {
            window = &remaining[..(window.len() + chunk_size).min(remaining.len())];
            let before = window.len();
            let res = with_info.process(&mut window).unwrap();
            remaining = &remaining[(before - window.len())..];
            match res {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { fallback, .. } => {
                    assert!(!remaining.is_empty(), "ran out of input before frame info");
                    with_info = fallback;
                }
            }
        }
    }

    #[test]
    fn test_toc_offset_chunked_matches_single_shot() {
        // Regression: feeding a frame in tiny chunks splits frame-header + TOC
        // parsing across multiple process() calls (the OutOfBounds path). The
        // resulting frame_data_offset / frame_data_size / TOC entries MUST
        // match the single-shot feed — otherwise the section-data byte anchor
        // undercounts and every progressive split boundary shifts.
        for name in [
            "resources/test/basic.jxl",
            "resources/test/multiple_lf_420.jxl",
            "resources/test/has_permutation.jxl",
            "resources/test/has_permutation_with_container.jxl",
        ] {
            let file = std::fs::read(name).unwrap();
            let single = decode_to_frame_info(&file);
            let single_offset = single.frame_data_offset();
            let single_size = single.frame_data_size();
            let single_entries = collect_toc(&single);

            // 1 byte at a time is the most aggressive split; also test a few
            // small sizes to vary where the boundary lands.
            for chunk in [1usize, 3, 7, 17] {
                let chunked = decode_to_frame_info_chunked(&file, chunk);
                assert_eq!(
                    chunked.frame_data_offset(),
                    single_offset,
                    "{name}: frame_data_offset differs at chunk_size={chunk} \
                     (chunked={}, single={single_offset})",
                    chunked.frame_data_offset(),
                );
                assert_eq!(
                    chunked.frame_data_size(),
                    single_size,
                    "{name}: frame_data_size differs at chunk_size={chunk}"
                );
                let chunked_entries = collect_toc(&chunked);
                assert_eq!(
                    chunked_entries.len(),
                    single_entries.len(),
                    "{name}: TOC entry count differs at chunk_size={chunk}"
                );
                for (i, (a, b)) in chunked_entries
                    .iter()
                    .zip(single_entries.iter())
                    .enumerate()
                {
                    assert_eq!(
                        a.kind, b.kind,
                        "{name}: entry {i} kind differs at chunk={chunk}"
                    );
                    assert_eq!(
                        a.offset, b.offset,
                        "{name}: entry {i} offset differs at chunk={chunk}"
                    );
                    assert_eq!(
                        a.size, b.size,
                        "{name}: entry {i} size differs at chunk={chunk}"
                    );
                }
            }
        }
    }
}
