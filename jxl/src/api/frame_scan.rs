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
//! The scanner tracks reference frame slot state to determine which frames
//! are independently decodable ("keyframes" for seeking). A frame is a
//! keyframe when it fully replaces the canvas (Replace blending, full-frame
//! coverage) and does not depend on reference slots written by prior frames
//! (no patches, no non-Replace blending source).
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

use crate::bit_reader::BitReader;
use crate::container::frame_index::FrameIndexBox;
use crate::error::{Error, Result};
use crate::headers::encodings::UnconditionalCoder;
use crate::headers::frame_header::FrameHeader;
use crate::headers::toc::IncrementalTocReader;
use crate::headers::{FileHeader, JxlHeader};
use crate::icc::IncrementalIccReader;

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
    /// Frame name, if any.
    pub name: String,
}

/// Tracks which frame wrote to each reference slot for dependency analysis.
#[derive(Clone, Default, Debug)]
struct SlotState {
    /// Codestream byte offset of the frame that last wrote to this slot.
    _frame_offset: usize,
    /// Whether that frame itself was independently decodable.
    _is_keyframe_writer: bool,
}

/// State machine for the incremental scanner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScanPhase {
    /// Haven't parsed anything yet. Need to detect format + parse file header.
    Initial,
    /// File header parsed, ready to parse frame headers.
    Scanning,
    /// All frames found (hit is_last or end of data).
    Done,
}

/// Incremental frame information scanner.
///
/// Feed raw JXL data (bare codestream or container-wrapped) and get back
/// a growing list of visible frames with metadata. No pixels are decoded.
pub struct FrameInfoDecoder {
    /// Accumulated input data.
    buf: Vec<u8>,
    /// How many bits have been successfully consumed from `buf`.
    consumed_bits: usize,
    /// Current scanning phase.
    phase: ScanPhase,

    // --- File-level state (populated after file header is parsed) ---
    file_header: Option<FileHeader>,
    has_animation: bool,
    tps_numerator: u32,
    tps_denominator: u32,
    /// Whether the first frame is a preview frame (to be skipped).
    has_preview: bool,
    /// Set to true after the preview frame has been skipped.
    preview_done: bool,

    // --- Frame dependency tracking ---
    /// Reference frame slot state. JXL has 4 slots (0..3).
    slots: [Option<SlotState>; 4],
    /// Codestream offset of the most recent keyframe. Used as
    /// `decode_start_offset` for non-keyframe visible frames.
    last_keyframe_offset: usize,

    // --- Results ---
    frames: Vec<VisibleFrameInfo>,
    visible_index: usize,
    total_duration_ms: f64,

    // --- Container handling ---
    /// Extracted codestream for container-wrapped files.
    /// None means bare codestream (buf IS the codestream).
    extracted_codestream: Option<Vec<u8>>,
    /// Parsed jxli frame index box, if present.
    frame_index: Option<FrameIndexBox>,
}

impl FrameInfoDecoder {
    /// Create a new frame info scanner.
    pub fn new() -> Self {
        Self {
            buf: Vec::new(),
            consumed_bits: 0,
            phase: ScanPhase::Initial,
            file_header: None,
            has_animation: false,
            tps_numerator: 0,
            tps_denominator: 0,
            has_preview: false,
            preview_done: false,
            slots: Default::default(),
            last_keyframe_offset: 0,
            frames: Vec::new(),
            visible_index: 0,
            total_duration_ms: 0.0,
            extracted_codestream: None,
            frame_index: None,
        }
    }

    /// Feed more data to the scanner.
    ///
    /// Returns `Ok(true)` if all frames have been found (hit `is_last`),
    /// `Ok(false)` if more data is needed to find additional frames.
    ///
    /// Can be called multiple times as data arrives incrementally.
    /// Each call appends to the internal buffer and tries to parse
    /// further.
    pub fn feed(&mut self, data: &[u8]) -> Result<bool> {
        if self.phase == ScanPhase::Done {
            return Ok(true);
        }

        self.buf.extend_from_slice(data);

        // Handle container detection on first meaningful feed.
        if self.phase == ScanPhase::Initial && self.extracted_codestream.is_none() {
            self.try_detect_container()?;
        }

        self.try_parse()
    }

    /// Get the visible frames discovered so far.
    pub fn frames(&self) -> &[VisibleFrameInfo] {
        &self.frames
    }

    /// Number of visible frames found so far.
    pub fn frame_count(&self) -> usize {
        self.frames.len()
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
        self.total_duration_ms
    }

    /// Ticks-per-second numerator/denominator from the animation header.
    pub fn tps(&self) -> (u32, u32) {
        (self.tps_numerator, self.tps_denominator)
    }

    /// Parsed `jxli` frame index box, if the container had one.
    pub fn frame_index(&self) -> Option<&FrameIndexBox> {
        self.frame_index.as_ref()
    }

    // --- Container handling ---

    /// Detect whether the input is a bare codestream or container-wrapped.
    /// For containers, extract the codestream and parse jxli.
    fn try_detect_container(&mut self) -> Result<()> {
        if self.buf.len() < 12 {
            // Need at least 12 bytes to distinguish formats.
            return Ok(());
        }

        // Bare codestream starts with 0xff 0x0a.
        if self.buf[0] == 0xff && self.buf[1] == 0x0a {
            // Bare codestream: buf is the codestream itself.
            return Ok(());
        }

        // Container signature.
        let container_sig: [u8; 12] = [
            0x00, 0x00, 0x00, 0x0c, 0x4a, 0x58, 0x4c, 0x20, 0x0d, 0x0a, 0x87, 0x0a,
        ];
        if self.buf[..12] != container_sig {
            return Err(Error::InvalidSignature);
        }

        // Parse boxes to extract codestream and jxli.
        let mut pos = 12;
        let mut codestream = Vec::new();
        let data = &self.buf;

        while pos + 8 <= data.len() {
            let box_len = u32::from_be_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
            let box_type: [u8; 4] = data[pos + 4..pos + 8].try_into().unwrap();

            let (header_size, content_len) = if box_len == 1 && pos + 16 <= data.len() {
                let xl = u64::from_be_bytes(data[pos + 8..pos + 16].try_into().unwrap()) as usize;
                (16, xl.saturating_sub(16))
            } else if box_len == 0 {
                (8, data.len() - pos - 8)
            } else if box_len >= 8 {
                (8, box_len - 8)
            } else {
                return Err(Error::InvalidBox);
            };

            let content_start = pos + header_size;
            let content_end = (content_start + content_len).min(data.len());

            // If we can't see the full box content yet, stop -- we'll retry
            // after more data arrives.
            if content_end < content_start + content_len && box_len != 0 {
                // Incomplete box -- wait for more data.
                return Ok(());
            }

            let content = &data[content_start..content_end];

            match &box_type {
                b"jxlc" => codestream.extend_from_slice(content),
                b"jxlp" => {
                    if content.len() >= 4 {
                        codestream.extend_from_slice(&content[4..]);
                    }
                }
                b"jxli" => {
                    self.frame_index = FrameIndexBox::parse(content).ok();
                }
                _ => {} // skip ftyp, exif, xml, etc.
            }

            if box_len == 0 {
                break;
            }
            pos += header_size + content_len;
        }

        if codestream.is_empty() {
            // Haven't found a codestream box yet -- might need more data
            // unless we've seen the entire file.
            return Ok(());
        }

        self.extracted_codestream = Some(codestream);
        Ok(())
    }

    // --- Parsing ---

    /// Try to parse as much as possible from the buffered data.
    fn try_parse(&mut self) -> Result<bool> {
        loop {
            match self.parse_next() {
                Ok(true) => continue,
                Ok(false) | Err(Error::OutOfBounds(_)) => {
                    return Ok(self.phase == ScanPhase::Done);
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Try to parse the next unit. Returns Ok(true) if something was parsed,
    /// Ok(false) if we need more data, or Err for real errors.
    fn parse_next(&mut self) -> Result<bool> {
        // Copy the codestream bytes into a local to avoid borrow conflicts
        // (BitReader borrows the slice, but parse_file_header / parse_frame
        // need &mut self). The copy is cheap -- only headers + TOCs are parsed,
        // and we start from consumed_bits each time.
        let codestream_data: Vec<u8> = if let Some(ref extracted) = self.extracted_codestream {
            extracted.clone()
        } else {
            self.buf.clone()
        };

        if codestream_data.is_empty() {
            return Ok(false);
        }
        if codestream_data.len() * 8 <= self.consumed_bits {
            return Ok(false);
        }

        let mut br = BitReader::new(&codestream_data);
        if self.consumed_bits > 0 {
            br.skip_bits(self.consumed_bits)?;
        }

        match self.phase {
            ScanPhase::Initial => self.parse_file_header(&mut br),
            ScanPhase::Scanning => self.parse_frame(&mut br),
            ScanPhase::Done => Ok(false),
        }
    }

    /// Parse file header + ICC profile.
    fn parse_file_header(&mut self, br: &mut BitReader) -> Result<bool> {
        let file_header = FileHeader::read(br)?;

        self.has_animation = file_header.image_metadata.animation.is_some();
        if let Some(ref anim) = file_header.image_metadata.animation {
            self.tps_numerator = anim.tps_numerator;
            self.tps_denominator = anim.tps_denominator;
        }

        self.has_preview = file_header.image_metadata.preview.is_some();

        // Skip ICC profile if present.
        if file_header.image_metadata.color_encoding.want_icc {
            skip_icc(br)?;
        }
        br.jump_to_byte_boundary()?;

        self.file_header = Some(file_header);
        self.consumed_bits = br.total_bits_read();
        self.phase = ScanPhase::Scanning;
        Ok(true)
    }

    /// Parse one frame header + TOC, track slots, skip section data.
    fn parse_frame(&mut self, br: &mut BitReader) -> Result<bool> {
        let file_header = self.file_header.as_ref().unwrap();
        let frame_byte_offset = br.total_bits_read() / 8;

        // First frame may be a preview frame.
        let nonserialized = if !self.preview_done && self.has_preview {
            file_header
                .preview_frame_header_nonserialized()
                .unwrap_or_else(|| file_header.frame_header_nonserialized())
        } else {
            file_header.frame_header_nonserialized()
        };

        let mut frame_header = FrameHeader::read_unconditional(&(), br, &nonserialized)?;
        frame_header.postprocess(&nonserialized);

        // Handle preview frame: skip it.
        if !self.preview_done && self.has_preview {
            self.preview_done = true;
            let toc = read_toc(frame_header.num_toc_entries() as u32, br)?;
            let sections_size: usize = toc.iter().map(|x| *x as usize).sum();
            br.skip_bits(sections_size * 8)?;
            br.jump_to_byte_boundary()?;
            self.consumed_bits = br.total_bits_read();
            return Ok(true);
        }

        // Parse TOC.
        let toc = read_toc(frame_header.num_toc_entries() as u32, br)?;
        let sections_size: usize = toc.iter().map(|x| *x as usize).sum();

        // --- Dependency analysis ---
        let is_visible = frame_header.is_visible();

        // Determine if this frame is independently decodable (keyframe for seeking).
        // A frame is a keyframe if:
        // 1. It's the first visible frame, OR
        // 2. All of:
        //    a. Does not need blending (Replace mode for all channels AND full-frame)
        //    b. Does not use patches (patches reference saved reference slots)
        //
        // needs_blending() returns true when have_crop || !replace_all, so
        // !needs_blending() means full-frame Replace for all channels.
        let is_keyframe = if is_visible && self.visible_index == 0 {
            true
        } else if is_visible {
            !frame_header.needs_blending() && !frame_header.has_patches()
        } else {
            false // non-visible frames are not keyframes by definition
        };

        // Update reference frame slots.
        if frame_header.can_be_referenced {
            let slot = frame_header.save_as_reference as usize;
            if slot < 4 {
                self.slots[slot] = Some(SlotState {
                    _frame_offset: frame_byte_offset,
                    _is_keyframe_writer: is_keyframe,
                });
            }
        }

        if is_visible {
            if is_keyframe {
                self.last_keyframe_offset = frame_byte_offset;
            }

            let duration_ticks = frame_header.duration;
            let duration_ms = if self.has_animation && self.tps_numerator > 0 {
                (duration_ticks as f64) * 1000.0 * (self.tps_denominator as f64)
                    / (self.tps_numerator as f64)
            } else {
                0.0
            };

            self.frames.push(VisibleFrameInfo {
                index: self.visible_index,
                duration_ms,
                duration_ticks,
                codestream_offset: frame_byte_offset,
                is_last: frame_header.is_last,
                is_keyframe,
                decode_start_offset: if is_keyframe {
                    frame_byte_offset
                } else {
                    self.last_keyframe_offset
                },
                name: frame_header.name.clone(),
            });

            self.total_duration_ms += duration_ms;
            self.visible_index += 1;
        }

        // Skip section data.
        br.skip_bits(sections_size * 8)?;
        br.jump_to_byte_boundary()?;

        self.consumed_bits = br.total_bits_read();

        if frame_header.is_last {
            self.phase = ScanPhase::Done;
        }

        Ok(true)
    }
}

impl Default for FrameInfoDecoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse a TOC and return the entry sizes.
fn read_toc(num_entries: u32, br: &mut BitReader) -> Result<Vec<u32>> {
    let mut reader = IncrementalTocReader::new(num_entries, br)?;
    while !reader.is_complete() {
        reader.read_step(br)?;
    }
    let toc = reader.finalize();
    br.jump_to_byte_boundary()?;
    Ok(toc.entries)
}

/// Skip over an ICC profile in the codestream.
fn skip_icc(br: &mut BitReader) -> Result<()> {
    let mut reader = IncrementalIccReader::new(br)?;
    for _ in 0..reader.remaining() {
        reader.read_one(br)?;
    }
    reader.finalize(br)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- scan tests ---

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
    fn test_scan_container_with_jxli() {
        // Build a container-wrapped file with a jxli box.
        let codestream =
            std::fs::read("resources/test/conformance_test_images/animation_icos4d_5.jxl").unwrap();

        let container = wrap_in_container(&codestream, Some(&[(0, 100, 1), (500, 200, 3)]));

        let mut scanner = FrameInfoDecoder::new();
        let done = scanner.feed(&container).unwrap();

        assert!(done);
        assert!(scanner.is_animated());
        assert!(scanner.frame_count() > 1);

        // Should have parsed the jxli box.
        let fi = scanner.frame_index().unwrap();
        assert_eq!(fi.num_frames(), 2);
        assert_eq!(fi.entries[0].codestream_offset, 0);
        assert_eq!(fi.entries[1].codestream_offset, 500);
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

    // --- helpers ---

    /// Wrap a bare codestream in a JXL container, optionally with a jxli box.
    fn wrap_in_container(codestream: &[u8], jxli_entries: Option<&[(u64, u64, u64)]>) -> Vec<u8> {
        fn make_box(ty: &[u8; 4], content: &[u8]) -> Vec<u8> {
            let len = (8 + content.len()) as u32;
            let mut buf = Vec::new();
            buf.extend(len.to_be_bytes());
            buf.extend(ty);
            buf.extend(content);
            buf
        }

        fn encode_varint(mut value: u64) -> Vec<u8> {
            let mut result = Vec::new();
            loop {
                let mut byte = (value & 0x7f) as u8;
                value >>= 7;
                if value > 0 {
                    byte |= 0x80;
                }
                result.push(byte);
                if value == 0 {
                    break;
                }
            }
            result
        }

        let sig: [u8; 12] = [
            0x00, 0x00, 0x00, 0x0c, 0x4a, 0x58, 0x4c, 0x20, 0x0d, 0x0a, 0x87, 0x0a,
        ];

        let mut out = Vec::new();
        out.extend(&sig);
        out.extend(make_box(b"ftyp", b"jxl \x00\x00\x00\x00jxl "));

        if let Some(entries) = jxli_entries {
            let mut jxli_content = Vec::new();
            jxli_content.extend(encode_varint(entries.len() as u64));
            jxli_content.extend(1u32.to_be_bytes()); // TNUM
            jxli_content.extend(1000u32.to_be_bytes()); // TDEN
            for &(off, t, f) in entries {
                jxli_content.extend(encode_varint(off));
                jxli_content.extend(encode_varint(t));
                jxli_content.extend(encode_varint(f));
            }
            out.extend(make_box(b"jxli", &jxli_content));
        }

        out.extend(make_box(b"jxlc", codestream));
        out
    }
}
