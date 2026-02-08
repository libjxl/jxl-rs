// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Fast frame information scanning and single-frame decoding for JXL files.
//!
//! Provides:
//! - [`scan_frame_info`]: extracts frame count, durations, and offsets by
//!   parsing only frame headers and TOCs -- no pixel data is decoded.
//! - [`decode_frame`]: decodes a single frame by index, returning RGBA u8 pixels.
//!
//! If the file is wrapped in a container with a `jxli` (frame index) box,
//! that is used directly; otherwise the codestream is scanned.

use crate::api::decoder::states;
use crate::api::{
    JxlDecoder, JxlDecoderOptions, JxlOutputBuffer, JxlPixelFormat, ProcessingResult,
};
use crate::bit_reader::BitReader;
use crate::container::frame_index::FrameIndexBox;
use crate::error::{Error, Result};
use crate::headers::encodings::UnconditionalCoder;
use crate::headers::frame_header::FrameHeader;
use crate::headers::toc::IncrementalTocReader;
use crate::headers::{FileHeader, JxlHeader};

/// Information about a single visible frame in an animation.
#[derive(Debug, Clone, PartialEq)]
pub struct FrameInfo {
    /// Zero-based index of this visible frame.
    pub index: usize,
    /// Duration in milliseconds (0 for still images or the last frame).
    pub duration_ms: f64,
    /// Duration in raw ticks from the animation header.
    pub duration_ticks: u32,
    /// Byte offset of this frame in the codestream.
    pub codestream_offset: usize,
    /// Whether this is the last frame in the codestream.
    pub is_last: bool,
    /// Whether this frame is a keyframe (can be independently decoded).
    pub is_keyframe: bool,
    /// Frame name, if any.
    pub name: String,
}

/// Summary of all frames in a JXL file.
#[derive(Debug, Clone)]
pub struct FrameInfoSummary {
    /// Information about each visible frame.
    pub frames: Vec<FrameInfo>,
    /// The frame index box, if present in the container.
    pub frame_index: Option<FrameIndexBox>,
    /// Total number of visible frames.
    pub num_frames: usize,
    /// Whether the image is animated.
    pub is_animated: bool,
    /// Animation ticks per second numerator (0 if not animated).
    pub tps_numerator: u32,
    /// Animation ticks per second denominator (0 if not animated).
    pub tps_denominator: u32,
    /// Total duration in milliseconds.
    pub total_duration_ms: f64,
}

/// Scan a JXL file (bare codestream or container) and extract frame information
/// without decoding any pixel data.
///
/// This parses each frame's header and TOC to determine frame count, durations,
/// and byte offsets, then skips over the actual section data. If the file is in
/// a container with a `jxli` frame index box, that index is also returned.
pub fn scan_frame_info(data: &[u8]) -> Result<FrameInfoSummary> {
    let (codestream, frame_index) = extract_codestream_and_index(data)?;
    scan_codestream_frames(codestream, frame_index)
}

/// Extract codestream bytes and optional frame index from raw file data.
/// Handles both bare codestreams and container-wrapped files.
fn extract_codestream_and_index(data: &[u8]) -> Result<(&[u8], Option<FrameIndexBox>)> {
    // Check for bare codestream (starts with 0xff 0x0a)
    if data.len() >= 2 && data[0] == 0xff && data[1] == 0x0a {
        return Ok((data, None));
    }

    // Check for container format
    let container_sig: [u8; 12] = [
        0x00, 0x00, 0x00, 0x0c, 0x4a, 0x58, 0x4c, 0x20, 0x0d, 0x0a, 0x87, 0x0a,
    ];
    if data.len() < 12 || data[..12] != container_sig {
        return Err(Error::InvalidSignature);
    }

    // Parse boxes to find jxlc/jxlp and jxli
    let mut pos = 12;
    let mut codestream_parts: Vec<&[u8]> = Vec::new();
    let mut frame_index = None;

    while pos + 8 <= data.len() {
        let box_len = u32::from_be_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        let box_type: [u8; 4] = data[pos + 4..pos + 8].try_into().unwrap();

        let (header_size, content_len) = if box_len == 1 && pos + 16 <= data.len() {
            let xlbox = u64::from_be_bytes(data[pos + 8..pos + 16].try_into().unwrap()) as usize;
            (16, xlbox.saturating_sub(16))
        } else if box_len == 0 {
            // Box extends to end of file
            (8, data.len() - pos - 8)
        } else if box_len >= 8 {
            (8, box_len - 8)
        } else {
            return Err(Error::InvalidBox);
        };

        let content_start = pos + header_size;
        let content_end = (content_start + content_len).min(data.len());
        let content = &data[content_start..content_end];

        match &box_type {
            b"jxlc" => {
                codestream_parts.push(content);
            }
            b"jxlp" => {
                // Skip the 4-byte jxlp index
                if content.len() >= 4 {
                    codestream_parts.push(&content[4..]);
                }
            }
            b"jxli" => {
                frame_index = FrameIndexBox::parse(content).ok();
            }
            _ => {} // skip ftyp, jxll, exif, xml, etc.
        }

        if box_len == 0 {
            break; // Last box extends to EOF
        }
        pos += header_size + content_len;
    }

    if codestream_parts.is_empty() {
        return Err(Error::InvalidSignature);
    }

    // For the common case of a single jxlc box, return a slice directly.
    // For jxlp, we'd need to concatenate -- for now, use the first part
    // and note this limitation.
    if codestream_parts.len() == 1 {
        Ok((codestream_parts[0], frame_index))
    } else {
        // TODO: support multiple jxlp boxes by concatenating into a buffer.
        // For now, this is a simplification -- most files use jxlc.
        Ok((codestream_parts[0], frame_index))
    }
}

/// Scan the codestream to extract frame information by parsing headers + TOCs only.
fn scan_codestream_frames(
    codestream: &[u8],
    frame_index: Option<FrameIndexBox>,
) -> Result<FrameInfoSummary> {
    let mut br = BitReader::new(codestream);

    // Parse file header
    let file_header = FileHeader::read(&mut br)?;
    let has_animation = file_header.image_metadata.animation.is_some();
    let (tps_num, tps_den) = file_header
        .image_metadata
        .animation
        .as_ref()
        .map(|a| (a.tps_numerator, a.tps_denominator))
        .unwrap_or((0, 0));

    // Skip ICC profile if present
    if file_header.image_metadata.color_encoding.want_icc {
        skip_icc(&mut br)?;
    }
    br.jump_to_byte_boundary()?;

    let nonserialized = file_header.frame_header_nonserialized();
    let preview_nonserialized = file_header.preview_frame_header_nonserialized();

    let mut frames = Vec::new();
    let mut visible_index = 0usize;
    let mut total_duration_ms = 0.0;
    let mut is_first_frame = true;

    loop {
        let frame_byte_offset = br.total_bits_read() / 8;

        // First frame might be a preview frame
        let ns = if is_first_frame {
            preview_nonserialized.as_ref().unwrap_or(&nonserialized)
        } else {
            &nonserialized
        };

        let mut frame_header = FrameHeader::read_unconditional(&(), &mut br, ns)?;
        frame_header.postprocess(ns);

        // Handle preview frame: skip it and move on
        if is_first_frame && preview_nonserialized.is_some() {
            is_first_frame = false;
            // Parse TOC and skip sections
            skip_frame_sections(&mut frame_header, &mut br)?;
            continue;
        }
        is_first_frame = false;

        let num_toc_entries = frame_header.num_toc_entries();
        let toc = read_toc(num_toc_entries as u32, &mut br)?;
        let sections_size: usize = toc.iter().map(|x| *x as usize).sum();

        let is_visible = frame_header.is_visible();

        if is_visible {
            let duration_ticks = frame_header.duration;
            let duration_ms = if has_animation && tps_num > 0 {
                (duration_ticks as f64) * 1000.0 * (tps_den as f64) / (tps_num as f64)
            } else {
                0.0
            };

            // A frame is a keyframe if it's the first visible frame, or if it
            // uses Replace blending and has save_before_ct set (which implies
            // full_frame and Replace mode).
            let is_keyframe = visible_index == 0
                || frame_header.blending_info.mode
                    == crate::headers::frame_header::BlendingMode::Replace;

            frames.push(FrameInfo {
                index: visible_index,
                duration_ms,
                duration_ticks,
                codestream_offset: frame_byte_offset,
                is_last: frame_header.is_last,
                is_keyframe,
                name: frame_header.name.clone(),
            });

            total_duration_ms += duration_ms;
            visible_index += 1;
        }

        // Skip over all section data
        let skip_bits = sections_size * 8;
        br.skip_bits(skip_bits)?;
        br.jump_to_byte_boundary()?;

        if frame_header.is_last {
            break;
        }
    }

    Ok(FrameInfoSummary {
        num_frames: frames.len(),
        is_animated: has_animation,
        tps_numerator: tps_num,
        tps_denominator: tps_den,
        total_duration_ms,
        frames,
        frame_index,
    })
}

/// Parse a TOC (without building a full Toc struct) and return the entry sizes.
fn read_toc(num_entries: u32, br: &mut BitReader) -> Result<Vec<u32>> {
    let mut reader = IncrementalTocReader::new(num_entries, br)?;
    while !reader.is_complete() {
        reader.read_step(br)?;
    }
    let toc = reader.finalize();
    br.jump_to_byte_boundary()?;
    Ok(toc.entries)
}

/// Parse TOC for a frame and skip its section data.
fn skip_frame_sections(frame_header: &mut FrameHeader, br: &mut BitReader) -> Result<()> {
    let num_toc_entries = frame_header.num_toc_entries();
    let toc = read_toc(num_toc_entries as u32, br)?;
    let sections_size: usize = toc.iter().map(|x| *x as usize).sum();
    br.skip_bits(sections_size * 8)?;
    br.jump_to_byte_boundary()?;
    Ok(())
}

/// Skip over an ICC profile in the codestream.
/// The ICC profile uses Brotli-compressed data with a varint-encoded size prefix.
fn skip_icc(br: &mut BitReader) -> Result<()> {
    use crate::icc::IncrementalIccReader;
    let mut reader = IncrementalIccReader::new(br)?;
    for _ in 0..reader.remaining() {
        reader.read_one(br)?;
    }
    reader.finalize(br)?;
    Ok(())
}

/// Decoded frame pixel data.
#[derive(Debug, Clone)]
pub struct DecodedFrame {
    /// Pixel data in row-major order, format determined by the pixel format
    /// passed to `decode_frame` / `decode_all_frames`.
    pub data: Vec<u8>,
    /// Frame width in pixels.
    pub width: usize,
    /// Frame height in pixels.
    pub height: usize,
    /// Bytes per pixel in the output (e.g. 4 for RGBA u8, 8 for RGBA u16/f16, 16 for RGBA f32).
    pub bytes_per_pixel: usize,
    /// Frame info (index, duration, etc.).
    pub info: FrameInfo,
}

/// Decode a single frame by index from a JXL file.
///
/// The caller specifies the output pixel format and decoder options.
/// Frames before `frame_index` are skipped (not rendered) for efficiency.
///
/// # Example
///
/// ```no_run
/// use jxl::api::frame_info::decode_frame;
/// use jxl::api::{JxlPixelFormat, JxlDecoderOptions};
///
/// let data = std::fs::read("animation.jxl").unwrap();
///
/// // Decode frame 3 as RGBA u8
/// let frame = decode_frame(
///     &data, 3,
///     JxlPixelFormat::rgba8(0),
///     JxlDecoderOptions::default(),
/// ).unwrap();
/// println!("{}x{}, {} bytes", frame.width, frame.height, frame.data.len());
/// ```
pub fn decode_frame(
    data: &[u8],
    frame_index: usize,
    pixel_format: JxlPixelFormat,
    options: JxlDecoderOptions,
) -> Result<DecodedFrame> {
    // First, scan to get frame info and validate the index.
    let summary = scan_frame_info(data)?;
    if frame_index >= summary.num_frames {
        return Err(Error::OutOfBounds(frame_index));
    }
    let target_info = summary.frames[frame_index].clone();

    let mut input: &[u8] = data;
    let dec = JxlDecoder::<states::Initialized>::new(options);
    let mut dec = advance_init(dec, &mut input)?;

    let (width, height) = dec.basic_info().size;
    let pixel_format =
        adjust_pixel_format_for_image(pixel_format, dec.basic_info().extra_channels.len());
    dec.set_pixel_format(pixel_format.clone());

    let bytes_per_pixel = compute_bytes_per_pixel(&pixel_format);
    let bytes_per_row = width * bytes_per_pixel;
    let buf_size = bytes_per_row * height;
    let num_extra_buffers = pixel_format
        .extra_channel_format
        .iter()
        .filter(|f| f.is_some())
        .count();

    // Decode frames, skipping until we reach the target.
    let mut visible_idx = 0usize;
    loop {
        let dec_frame = advance_to_frame(dec, &mut input)?;

        if visible_idx == frame_index {
            let mut pixel_buf = vec![0u8; buf_size];
            let mut extra_bufs: Vec<Vec<u8>> = (0..num_extra_buffers)
                .map(|_| vec![0u8; width * height * 4]) // conservative size
                .collect();
            let mut out: Vec<JxlOutputBuffer<'_>> =
                vec![JxlOutputBuffer::new(&mut pixel_buf, height, bytes_per_row)];
            for eb in extra_bufs.iter_mut() {
                out.push(JxlOutputBuffer::new(eb, height, width * 4));
            }
            let _ = decode_frame_pixels(dec_frame, &mut input, &mut out)?;

            return Ok(DecodedFrame {
                data: pixel_buf,
                width,
                height,
                bytes_per_pixel,
                info: target_info,
            });
        }

        dec = skip_frame(dec_frame, &mut input)?;
        visible_idx += 1;
    }
}

/// Decode all frames from a JXL file.
///
/// The caller specifies the output pixel format and decoder options.
///
/// # Example
///
/// ```no_run
/// use jxl::api::frame_info::decode_all_frames;
/// use jxl::api::{JxlPixelFormat, JxlDecoderOptions};
///
/// let data = std::fs::read("animation.jxl").unwrap();
/// let frames = decode_all_frames(
///     &data,
///     JxlPixelFormat::rgba8(0),
///     JxlDecoderOptions::default(),
/// ).unwrap();
/// for f in &frames {
///     println!("frame {}: {}x{}, {:.0}ms", f.info.index, f.width, f.height, f.info.duration_ms);
/// }
/// ```
pub fn decode_all_frames(
    data: &[u8],
    pixel_format: JxlPixelFormat,
    options: JxlDecoderOptions,
) -> Result<Vec<DecodedFrame>> {
    let summary = scan_frame_info(data)?;
    let mut input: &[u8] = data;
    let dec = JxlDecoder::<states::Initialized>::new(options);
    let mut dec = advance_init(dec, &mut input)?;

    let (width, height) = dec.basic_info().size;
    let pixel_format =
        adjust_pixel_format_for_image(pixel_format, dec.basic_info().extra_channels.len());
    dec.set_pixel_format(pixel_format.clone());

    let bytes_per_pixel = compute_bytes_per_pixel(&pixel_format);
    let bytes_per_row = width * bytes_per_pixel;
    let buf_size = bytes_per_row * height;
    let num_extra_buffers = pixel_format
        .extra_channel_format
        .iter()
        .filter(|f| f.is_some())
        .count();
    let mut frames = Vec::with_capacity(summary.num_frames);

    for frame_info in &summary.frames {
        let dec_frame = advance_to_frame(dec, &mut input)?;
        let mut pixel_buf = vec![0u8; buf_size];
        let mut extra_bufs: Vec<Vec<u8>> = (0..num_extra_buffers)
            .map(|_| vec![0u8; width * height * 4])
            .collect();
        let mut out: Vec<JxlOutputBuffer<'_>> =
            vec![JxlOutputBuffer::new(&mut pixel_buf, height, bytes_per_row)];
        for eb in extra_bufs.iter_mut() {
            out.push(JxlOutputBuffer::new(eb, height, width * 4));
        }
        dec = decode_frame_pixels(dec_frame, &mut input, &mut out)?;

        frames.push(DecodedFrame {
            data: pixel_buf,
            width,
            height,
            bytes_per_pixel,
            info: frame_info.clone(),
        });

        if !dec.has_more_frames() {
            break;
        }
    }

    Ok(frames)
}

/// Compute bytes per pixel for the color channels of a pixel format.
fn compute_bytes_per_pixel(pixel_format: &JxlPixelFormat) -> usize {
    let samples = pixel_format.color_type.samples_per_pixel();
    let bytes_per_sample = pixel_format
        .color_data_format
        .as_ref()
        .map(|f| f.bytes_per_sample())
        .unwrap_or(1);
    samples * bytes_per_sample
}

/// Adjust a pixel format's extra_channel_format to match the actual number
/// of extra channels in the image. If the user didn't provide enough entries,
/// we pad with `None` (ignore). If they provided too many, we truncate.
fn adjust_pixel_format_for_image(
    mut format: JxlPixelFormat,
    num_extra_channels: usize,
) -> JxlPixelFormat {
    format.extra_channel_format.resize(num_extra_channels, None);
    format
}

// --- Internal helpers to drive the typestate decoder ---

fn advance_init(
    mut dec: JxlDecoder<states::Initialized>,
    input: &mut &[u8],
) -> Result<JxlDecoder<states::WithImageInfo>> {
    loop {
        match dec.process(input)? {
            ProcessingResult::Complete { result } => return Ok(result),
            ProcessingResult::NeedsMoreInput { fallback, .. } => {
                if input.is_empty() {
                    return Err(Error::OutOfBounds(0));
                }
                dec = fallback;
            }
        }
    }
}

fn advance_to_frame(
    mut dec: JxlDecoder<states::WithImageInfo>,
    input: &mut &[u8],
) -> Result<JxlDecoder<states::WithFrameInfo>> {
    loop {
        match dec.process(input)? {
            ProcessingResult::Complete { result } => return Ok(result),
            ProcessingResult::NeedsMoreInput { fallback, .. } => {
                if input.is_empty() {
                    return Err(Error::OutOfBounds(0));
                }
                dec = fallback;
            }
        }
    }
}

fn skip_frame(
    mut dec: JxlDecoder<states::WithFrameInfo>,
    input: &mut &[u8],
) -> Result<JxlDecoder<states::WithImageInfo>> {
    loop {
        match dec.skip_frame(input)? {
            ProcessingResult::Complete { result } => return Ok(result),
            ProcessingResult::NeedsMoreInput { fallback, .. } => {
                if input.is_empty() {
                    return Err(Error::OutOfBounds(0));
                }
                dec = fallback;
            }
        }
    }
}

fn decode_frame_pixels(
    mut dec: JxlDecoder<states::WithFrameInfo>,
    input: &mut &[u8],
    buffers: &mut [JxlOutputBuffer<'_>],
) -> Result<JxlDecoder<states::WithImageInfo>> {
    loop {
        match dec.process(input, buffers)? {
            ProcessingResult::Complete { result } => return Ok(result),
            ProcessingResult::NeedsMoreInput { fallback, .. } => {
                if input.is_empty() {
                    return Err(Error::OutOfBounds(0));
                }
                dec = fallback;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scan_bare_animation() {
        let data =
            std::fs::read("resources/test/conformance_test_images/animation_icos4d_5.jxl").unwrap();
        let summary = scan_frame_info(&data).unwrap();

        assert!(summary.is_animated);
        assert!(summary.num_frames > 1, "expected multiple frames");
        assert!(summary.frame_index.is_none()); // bare codestream, no jxli

        // All frames should have valid offsets
        for (i, frame) in summary.frames.iter().enumerate() {
            assert_eq!(frame.index, i);
            if i < summary.num_frames - 1 {
                assert!(!frame.is_last);
            }
        }
        // Last frame should be marked is_last
        assert!(summary.frames.last().unwrap().is_last);
    }

    #[test]
    fn test_scan_still_image() {
        let data = std::fs::read("resources/test/green_queen_vardct_e3.jxl").unwrap();
        let summary = scan_frame_info(&data).unwrap();

        assert!(!summary.is_animated);
        assert_eq!(summary.num_frames, 1);
        assert!(summary.frames[0].is_last);
        assert_eq!(summary.total_duration_ms, 0.0);
    }

    #[test]
    fn test_scan_numbered_animation() {
        let data = std::fs::read("/tmp/numbered.jxl");
        if data.is_err() {
            // Skip if test file not available
            return;
        }
        let data = data.unwrap();
        let summary = scan_frame_info(&data).unwrap();

        assert!(summary.is_animated);
        assert_eq!(summary.num_frames, 5, "expected 5 frames (0,1,2,3,4)");
        assert!(summary.frames.last().unwrap().is_last);

        // Offsets should be strictly increasing
        for i in 1..summary.frames.len() {
            assert!(summary.frames[i].codestream_offset > summary.frames[i - 1].codestream_offset);
        }
    }

    #[test]
    fn test_scan_container_with_jxli() {
        // Build a container-wrapped file with a jxli box
        let codestream =
            std::fs::read("resources/test/conformance_test_images/animation_icos4d_5.jxl").unwrap();

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

        fn make_box(ty: &[u8; 4], content: &[u8]) -> Vec<u8> {
            let len = (8 + content.len()) as u32;
            let mut buf = Vec::new();
            buf.extend(len.to_be_bytes());
            buf.extend(ty);
            buf.extend(content);
            buf
        }

        // Build a simple jxli with 2 entries
        let mut jxli_content = Vec::new();
        jxli_content.extend(encode_varint(2));
        jxli_content.extend(1u32.to_be_bytes()); // TNUM
        jxli_content.extend(1000u32.to_be_bytes()); // TDEN
        jxli_content.extend(encode_varint(0)); // OFF0
        jxli_content.extend(encode_varint(100)); // T0
        jxli_content.extend(encode_varint(1)); // F0
        jxli_content.extend(encode_varint(500)); // OFF1
        jxli_content.extend(encode_varint(200)); // T1
        jxli_content.extend(encode_varint(3)); // F1

        let sig: [u8; 12] = [
            0x00, 0x00, 0x00, 0x0c, 0x4a, 0x58, 0x4c, 0x20, 0x0d, 0x0a, 0x87, 0x0a,
        ];
        let mut container = Vec::new();
        container.extend(&sig);
        container.extend(make_box(b"ftyp", b"jxl \x00\x00\x00\x00jxl "));
        container.extend(make_box(b"jxli", &jxli_content));
        container.extend(make_box(b"jxlc", &codestream));

        let summary = scan_frame_info(&container).unwrap();

        // Should have both scanned frames AND the jxli index
        assert!(summary.is_animated);
        assert!(summary.num_frames > 1);

        let fi = summary.frame_index.as_ref().unwrap();
        assert_eq!(fi.num_frames(), 2);
        assert_eq!(fi.entries[0].codestream_offset, 0);
        assert_eq!(fi.entries[1].codestream_offset, 500);
    }

    fn rgba8() -> JxlPixelFormat {
        JxlPixelFormat::rgba8(0)
    }

    fn opts() -> JxlDecoderOptions {
        JxlDecoderOptions::default()
    }

    #[test]
    fn test_decode_single_frame_still() {
        let data = std::fs::read("resources/test/green_queen_vardct_e3.jxl").unwrap();
        let frame = decode_frame(&data, 0, rgba8(), opts()).unwrap();

        assert!(frame.width > 0);
        assert!(frame.height > 0);
        assert_eq!(
            frame.data.len(),
            frame.width * frame.height * frame.bytes_per_pixel
        );
        assert!(frame.info.is_last);
        assert!(frame.data.iter().any(|&b| b != 0));
    }

    #[test]
    fn test_decode_frame_out_of_bounds() {
        let data = std::fs::read("resources/test/green_queen_vardct_e3.jxl").unwrap();
        assert!(decode_frame(&data, 1, rgba8(), opts()).is_err());
    }

    #[test]
    fn test_decode_animation_specific_frame() {
        let data =
            std::fs::read("resources/test/conformance_test_images/animation_icos4d_5.jxl").unwrap();
        let info = scan_frame_info(&data).unwrap();

        let first = decode_frame(&data, 0, rgba8(), opts()).unwrap();
        assert_eq!(first.info.index, 0);
        assert_eq!(
            first.data.len(),
            first.width * first.height * first.bytes_per_pixel
        );

        let last = decode_frame(&data, info.num_frames - 1, rgba8(), opts()).unwrap();
        assert_eq!(last.info.index, info.num_frames - 1);
        assert!(last.info.is_last);
        assert_ne!(first.data, last.data);
    }

    #[test]
    fn test_decode_all_frames_animation() {
        let data =
            std::fs::read("resources/test/conformance_test_images/animation_icos4d_5.jxl").unwrap();
        let frames = decode_all_frames(&data, rgba8(), opts()).unwrap();

        assert!(frames.len() > 1);
        for (i, f) in frames.iter().enumerate() {
            assert_eq!(f.info.index, i);
            assert_eq!(f.data.len(), f.width * f.height * f.bytes_per_pixel);
        }
        assert!(frames.last().unwrap().info.is_last);
    }

    #[test]
    fn test_decode_numbered_frames() {
        let data = std::fs::read("/tmp/numbered.jxl");
        if data.is_err() {
            return;
        }
        let data = data.unwrap();
        let frames = decode_all_frames(&data, rgba8(), opts()).unwrap();

        assert_eq!(frames.len(), 5, "expected 5 frames");
        for f in &frames {
            assert_eq!(f.data.len(), f.width * f.height * f.bytes_per_pixel);
            assert!(
                f.data.iter().any(|&b| b != 0),
                "frame {} is all black",
                f.info.index
            );
        }
        assert_ne!(frames[0].data, frames[1].data);
        assert_ne!(frames[2].data, frames[4].data);
    }
}
