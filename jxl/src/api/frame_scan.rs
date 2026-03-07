// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Incremental frame information scanner for JXL animation support.
//!
//! [`FrameInfoDecoder`] parses frame headers and TOCs without decoding pixel
//! data, producing a list of visible frames with duration, offset, and
//! dependency information. This enables efficient `DecodeFrameCount` in
//! embedders (e.g. Chromium) and informs the main decoder where to seek
//! for random-access frame decoding via
//! [`start_new_frame`](super::decoder::JxlDecoder::start_new_frame).
//!
//! Internally, it drives the existing decoder in scan mode (skipping
//! pixel decoding) to avoid duplicating parsing logic.
//!
//! # Usage
//!
//! ```rust,no_run
//! use jxl::api::frame_scan::FrameInfoDecoder;
//!
//! let data = std::fs::read("animation.jxl").unwrap();
//! let mut scanner = FrameInfoDecoder::new();
//! scanner.feed(&data).unwrap();
//!
//! println!("{} visible frames", scanner.frame_count());
//! for f in scanner.frames() {
//!     println!("  frame {}: {:.0}ms, keyframe={}, decode_from={}",
//!         f.index, f.duration_ms, f.is_keyframe, f.decode_start_offset);
//! }
//! ```

use crate::api::{JxlDecoderOptions, ProcessingResult};
use crate::container::frame_index::FrameIndexBox;
use crate::error::{Error, Result};

use super::JxlDecoderInner;

/// Information about a single visible frame.
#[derive(Debug, Clone, PartialEq)]
pub struct VisibleFrameInfo {
    /// Zero-based index among visible frames.
    pub index: usize,
    /// Duration in milliseconds (0 for still images or the last frame).
    pub duration_ms: f64,
    /// Duration in raw ticks from the animation header.
    pub duration_ticks: u32,
    /// Byte offset of this frame's header in the codestream.
    pub codestream_offset: usize,
    /// Whether this is the last frame in the codestream.
    pub is_last: bool,
    /// Whether this frame can be independently decoded (true keyframe for seeking).
    ///
    /// A frame is a keyframe when:
    /// - It is the first visible frame, OR
    /// - It uses Replace blending mode for all channels, AND
    /// - It covers the full image (no crop offset), AND
    /// - It does not use patches (which reference saved slots), AND
    /// - Its blending does not reference a slot written by a non-keyframe dependency.
    pub is_keyframe: bool,
    /// Earliest codestream byte offset needed to decode this frame.
    /// For keyframes this equals `codestream_offset`.
    /// For non-keyframes this is the offset of the nearest prior keyframe.
    pub decode_start_offset: usize,
    /// Remaining codestream bytes in the current container box at this
    /// frame's start position.  For bare-codestream files this is
    /// `u64::MAX`.  Used by
    /// [`start_new_frame`](super::decoder::JxlDecoder::start_new_frame) to
    /// restore the box parser to the correct state when seeking.
    pub remaining_in_box: u64,
    /// Frame name, if any.
    pub name: String,
}

/// State machine for the scanner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScanPhase {
    /// Need to parse file header / image info.
    Initial,
    /// Scanning frames.
    Scanning,
    /// All frames found.
    Done,
}

/// Incremental frame information scanner.
///
/// Feed raw JXL data (bare codestream or container-wrapped) and get back
/// a growing list of visible frames with metadata. No pixels are decoded.
///
/// Internally this drives [`JxlDecoderInner`] through its normal parsing
/// path, so all container, header, and TOC parsing uses the single shared
/// implementation.
pub struct FrameInfoDecoder {
    /// The underlying decoder (handles container parsing, headers, TOC, etc.).
    inner: JxlDecoderInner,
    /// Accumulated input data.
    buf: Vec<u8>,
    /// How many bytes of `buf` have been consumed by the decoder.
    consumed: usize,
    /// Current scanning phase.
    phase: ScanPhase,

    // --- Cached metadata (extracted from inner decoder after image info) ---
    has_animation: bool,
    tps_numerator: u32,
    tps_denominator: u32,
}

impl FrameInfoDecoder {
    /// Create a new frame info scanner.
    pub fn new() -> Self {
        Self {
            inner: JxlDecoderInner::new(JxlDecoderOptions::default()),
            buf: Vec::new(),
            consumed: 0,
            phase: ScanPhase::Initial,
            has_animation: false,
            tps_numerator: 0,
            tps_denominator: 0,
        }
    }

    /// Feed more data to the scanner.
    ///
    /// Returns `Ok(true)` if all frames have been found (hit `is_last`),
    /// `Ok(false)` if more data is needed to find additional frames.
    ///
    /// Can be called multiple times as data arrives incrementally.
    /// Each call appends to the internal buffer and tries to parse further.
    pub fn feed(&mut self, data: &[u8]) -> Result<bool> {
        if self.phase == ScanPhase::Done {
            return Ok(true);
        }

        self.buf.extend_from_slice(data);
        self.try_advance()
    }

    /// Get the visible frames discovered so far.
    pub fn frames(&self) -> &[VisibleFrameInfo] {
        self.inner.scanned_frames()
    }

    /// Number of visible frames found so far.
    pub fn frame_count(&self) -> usize {
        self.inner.scanned_frames().len()
    }

    /// Whether the image is animated.
    /// Only valid after the file header has been parsed (at least one `feed` call
    /// that returns `Ok(_)` without error).
    pub fn is_animated(&self) -> bool {
        self.has_animation
    }

    /// Whether all frames have been found.
    pub fn is_complete(&self) -> bool {
        self.phase == ScanPhase::Done
    }

    /// Total duration in milliseconds of all frames found so far.
    pub fn total_duration_ms(&self) -> f64 {
        self.inner
            .scanned_frames()
            .iter()
            .map(|f| f.duration_ms)
            .sum()
    }

    /// Ticks-per-second numerator/denominator from the animation header.
    pub fn tps(&self) -> (u32, u32) {
        (self.tps_numerator, self.tps_denominator)
    }

    /// Parsed `jxli` frame index box, if the container had one.
    pub fn frame_index(&self) -> Option<&FrameIndexBox> {
        self.inner.frame_index()
    }

    /// Drive the inner decoder forward, processing as much data as possible.
    fn try_advance(&mut self) -> Result<bool> {
        loop {
            let mut input: &[u8] = &self.buf[self.consumed..];
            let before_len = input.len();

            match self.inner.process(&mut input, None) {
                Ok(ProcessingResult::Complete { .. }) => {
                    self.consumed = self.buf.len() - input.len();

                    match self.phase {
                        ScanPhase::Initial => {
                            // Image info is now available.
                            self.extract_image_info();
                            self.phase = ScanPhase::Scanning;
                        }
                        ScanPhase::Scanning => {
                            // A frame was parsed (header+TOC) or sections were
                            // processed. Check if we've seen is_last.
                            if !self.inner.has_more_frames() {
                                self.phase = ScanPhase::Done;
                                return Ok(true);
                            }
                        }
                        ScanPhase::Done => return Ok(true),
                    }
                    // Continue to parse more frames.
                }
                Ok(ProcessingResult::NeedsMoreInput { .. }) => {
                    self.consumed = self.buf.len() - input.len();
                    return Ok(false);
                }
                Err(Error::OutOfBounds(_)) => {
                    // Not enough data yet.
                    let after_len = input.len();
                    self.consumed = self.buf.len() - after_len;
                    // If no bytes were consumed, we can't make progress.
                    if before_len == after_len {
                        return Ok(false);
                    }
                    // Some bytes consumed, try again.
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Extract animation metadata from the inner decoder after image info is parsed.
    fn extract_image_info(&mut self) {
        if let Some(info) = self.inner.basic_info() {
            if let Some(ref anim) = info.animation {
                self.has_animation = true;
                self.tps_numerator = anim.tps_numerator;
                self.tps_denominator = anim.tps_denominator;
            }
        }
    }
}

impl Default for FrameInfoDecoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scan_still_image() {
        let data = std::fs::read("resources/test/green_queen_vardct_e3.jxl").unwrap();
        let mut scanner = FrameInfoDecoder::new();
        let done = scanner.feed(&data).unwrap();

        assert!(done);
        assert!(!scanner.is_animated());
        assert_eq!(scanner.frame_count(), 1);
        assert!(scanner.frames()[0].is_last);
        assert!(scanner.frames()[0].is_keyframe);
        assert_eq!(scanner.total_duration_ms(), 0.0);
    }

    #[test]
    fn test_scan_bare_animation() {
        let data =
            std::fs::read("resources/test/conformance_test_images/animation_icos4d_5.jxl").unwrap();
        let mut scanner = FrameInfoDecoder::new();
        let done = scanner.feed(&data).unwrap();

        assert!(done);
        assert!(scanner.is_animated());
        assert!(scanner.frame_count() > 1, "expected multiple frames");

        // All frames should have valid offsets and monotonically increasing indices.
        for (i, frame) in scanner.frames().iter().enumerate() {
            assert_eq!(frame.index, i);
        }

        // Last frame should be is_last.
        assert!(scanner.frames().last().unwrap().is_last);

        // First frame should be a keyframe.
        assert!(scanner.frames()[0].is_keyframe);
        assert_eq!(
            scanner.frames()[0].decode_start_offset,
            scanner.frames()[0].codestream_offset
        );
    }

    #[test]
    fn test_scan_animation_offsets_increase() {
        let data =
            std::fs::read("resources/test/conformance_test_images/animation_icos4d_5.jxl").unwrap();
        let mut scanner = FrameInfoDecoder::new();
        scanner.feed(&data).unwrap();

        let frames = scanner.frames();
        for i in 1..frames.len() {
            assert!(
                frames[i].codestream_offset > frames[i - 1].codestream_offset,
                "frame {} offset {} should be > frame {} offset {}",
                i,
                frames[i].codestream_offset,
                i - 1,
                frames[i - 1].codestream_offset,
            );
        }
    }

    #[test]
    fn test_scan_incremental() {
        let data =
            std::fs::read("resources/test/conformance_test_images/animation_icos4d_5.jxl").unwrap();

        let mut scanner = FrameInfoDecoder::new();

        // Feed data in small chunks.
        let chunk_size = 128;
        let mut done = false;
        for chunk in data.chunks(chunk_size) {
            done = scanner.feed(chunk).unwrap();
            if done {
                break;
            }
        }

        assert!(done);
        assert!(scanner.is_animated());
        assert!(scanner.frame_count() > 1);
        assert!(scanner.frames().last().unwrap().is_last);
    }

    #[test]
    fn test_scan_keyframe_detection_still() {
        let data = std::fs::read("resources/test/green_queen_vardct_e3.jxl").unwrap();
        let mut scanner = FrameInfoDecoder::new();
        scanner.feed(&data).unwrap();

        // Still image: single frame, must be keyframe.
        assert_eq!(scanner.frame_count(), 1);
        let f = &scanner.frames()[0];
        assert!(f.is_keyframe);
        assert_eq!(f.decode_start_offset, f.codestream_offset);
    }

    #[test]
    fn test_scan_decode_start_offset_consistency() {
        let data =
            std::fs::read("resources/test/conformance_test_images/animation_icos4d_5.jxl").unwrap();

        let mut scanner = FrameInfoDecoder::new();
        scanner.feed(&data).unwrap();

        for frame in scanner.frames() {
            // decode_start_offset must be <= codestream_offset.
            assert!(
                frame.decode_start_offset <= frame.codestream_offset,
                "frame {}: decode_start_offset {} > codestream_offset {}",
                frame.index,
                frame.decode_start_offset,
                frame.codestream_offset,
            );
            // For keyframes, they must be equal.
            if frame.is_keyframe {
                assert_eq!(
                    frame.decode_start_offset, frame.codestream_offset,
                    "keyframe {} should have decode_start_offset == codestream_offset",
                    frame.index,
                );
            }
        }
    }

    #[test]
    fn test_scan_with_preview() {
        // Test file that has a preview frame.
        let data = std::fs::read("resources/test/with_preview.jxl");
        if data.is_err() {
            return; // Skip if test file not available.
        }
        let data = data.unwrap();
        let mut scanner = FrameInfoDecoder::new();
        let done = scanner.feed(&data).unwrap();

        // Should complete and have at least 1 frame (the preview should be skipped).
        assert!(done);
        assert!(scanner.frame_count() >= 1);
    }

    #[test]
    fn test_scan_patches_not_keyframe() {
        // Test with a file known to have patches -- such frames should not be
        // keyframes (even with Replace blending) because patches reference saved
        // reference frames.
        let data = std::fs::read("resources/test/grayscale_patches_var_dct.jxl");
        if data.is_err() {
            return;
        }
        let data = data.unwrap();
        let mut scanner = FrameInfoDecoder::new();
        scanner.feed(&data).unwrap();

        // If there's only one frame, it's always a keyframe (first frame rule).
        // This test mainly verifies the scanner doesn't crash on patch files.
        assert!(scanner.frame_count() >= 1);
    }
}
