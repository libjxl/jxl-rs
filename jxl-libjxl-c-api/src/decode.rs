// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::ffi::{c_char, c_int, c_void};
use std::mem::MaybeUninit;

use jxl::api::{
    JxlColorEncoding as RsColorEncoding, JxlColorProfile as RsColorProfile,
    JxlParallelRunner as RsParallelRunner, JxlSignature as RsSignature, check_signature,
};

use crate::conversions::{
    convert_basic_info, convert_color_encoding, convert_extra_channel_info, to_rs_color_encoding,
    to_rs_data_format,
};
use crate::parallel_runner::RayonParallelRunner;
use crate::pixel_buffer::{BoxOutBuffer, ImageOutBuffer, InputBuffer, compute_buffer_size};
use crate::state_machine::{
    JxlDecoder, SUPPORTED_EVENTS_MASK, Stage, StreamState, TOP_LEVEL_EVENTS_MASK,
    make_main_output_buffers,
};
use crate::types::{
    JPEGXL_NUMERIC_VERSION, JXL_BIT_DEPTH_CUSTOM, JXL_BIT_DEPTH_FROM_CODESTREAM,
    JXL_BIT_DEPTH_FROM_PIXEL_FORMAT, JXL_BLEND_REPLACE, JXL_BOOL, JXL_DEC_ERROR,
    JXL_DEC_NEED_MORE_INPUT, JXL_DEC_SUCCESS, JXL_FALSE, JXL_SIG_CODESTREAM, JXL_SIG_CONTAINER,
    JXL_SIG_INVALID, JXL_SIG_NOT_ENOUGH_BYTES, JxlBasicInfo, JxlBitDepth, JxlBlendInfo, JxlBoxType,
    JxlCmsInterface, JxlColorEncoding, JxlColorProfileTarget, JxlDecoderStatus,
    JxlExtraChannelInfo, JxlFrameHeader, JxlGainMapBundle, JxlImageOutCallback,
    JxlImageOutDestroyCallback, JxlImageOutInitCallback, JxlImageOutRunCallback, JxlMemoryManager,
    JxlParallelRunner, JxlPixelFormat, JxlProgressiveDetail, JxlSignature,
};

/// Copies `src` as a NUL-terminated C string into `dst`.
///
/// # Safety
/// If `dst` is non-null and `size > src.len()`, the caller must ensure `dst` is valid for writes
/// of `src.len() + 1` bytes (within a single allocation), exclusively accessible, and
/// non-overlapping with `src` for the duration of this call.
unsafe fn copy_c_string(src: &str, dst: *mut c_char, size: usize) -> JxlDecoderStatus {
    let bytes = src.as_bytes();
    if dst.is_null() || size <= bytes.len() {
        return JXL_DEC_ERROR;
    }
    let dst = dst.cast::<u8>();
    // SAFETY: `dst` is non-null and `size > bytes.len()`, so by caller contract `dst` is valid for
    // writes of `bytes.len() + 1` non-overlapping bytes within one allocation.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len());
        dst.add(bytes.len()).write(0);
    }
    JXL_DEC_SUCCESS
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderVersion() -> u32 {
    JPEGXL_NUMERIC_VERSION
}

/// Checks if the input buffer starts with a valid JPEG XL codestream or container signature.
///
/// # Safety
/// If `buf` is non-null and `len > 0`, the caller must ensure `buf` is valid for reads of `len`
/// initialized bytes (within a single allocation) and is not modified concurrently for the
/// duration of this function call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn JxlSignatureCheck(buf: *const u8, len: usize) -> JxlSignature {
    if len == 0 {
        return JXL_SIG_NOT_ENOUGH_BYTES;
    }
    if buf.is_null() {
        return JXL_SIG_INVALID;
    }
    let check_len = len.min(12);
    // SAFETY: `buf` is non-null and `0 < check_len <= 12 <= len`, so by caller contract `buf` is
    // valid for reads of `check_len` initialized, immutable bytes within a single allocation for
    // the duration of this function call.
    let slice = unsafe { std::slice::from_raw_parts(buf, check_len) };
    match check_signature(slice) {
        RsSignature::NeedMoreInput { .. } => JXL_SIG_NOT_ENOUGH_BYTES,
        RsSignature::None => JXL_SIG_INVALID,
        RsSignature::Codestream => JXL_SIG_CODESTREAM,
        RsSignature::Container => JXL_SIG_CONTAINER,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderCreate(
    _memory_manager: Option<&JxlMemoryManager>,
) -> Option<Box<JxlDecoder>> {
    // The memory manager is ignored, and callers must not assume that `alloc` was used to allocate
    // the returned handle or any internal buffers; always free via `JxlDecoderDestroy`.
    Some(Box::new(JxlDecoder::new()))
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderReset(dec: Option<&mut JxlDecoder>) {
    if let Some(dec) = dec {
        *dec = JxlDecoder::new();
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderDestroy(_dec: Option<Box<JxlDecoder>>) {}

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderRewind(dec: Option<&mut JxlDecoder>) {
    if let Some(dec) = dec {
        dec.rs_decoder.reset();
        dec.stream = StreamState::default();
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderSkipFrames(dec: Option<&mut JxlDecoder>, amount: usize) {
    if let Some(dec) = dec {
        dec.rs_decoder.skip_frames(amount);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderSkipCurrentFrame(dec: Option<&mut JxlDecoder>) -> JxlDecoderStatus {
    let Some(dec) = dec else {
        return JXL_DEC_ERROR;
    };
    match dec.stream.stage {
        Stage::MainNeedOutBuffer | Stage::MainDecode => {
            dec.stream.image_out_target = None;
            dec.stream.extra_channel_targets.fill_with(|| None);
            dec.stream.stage = Stage::MainDecode;
            JXL_DEC_SUCCESS
        }
        _ => JXL_DEC_ERROR,
    }
}

// Unsupported feature: custom `JxlParallelRunner` callbacks are ignored; when a non-null
// `parallel_runner` is set, decoding uses the internal Rayon thread pool, and when `NULL` (or not
// set), decoding runs sequentially.
#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderSetParallelRunner(
    dec: Option<&mut JxlDecoder>,
    parallel_runner: JxlParallelRunner,
    _parallel_runner_opaque: *mut c_void,
) -> JxlDecoderStatus {
    dec.map_or(JXL_DEC_ERROR, |d| {
        d.set_before_process(|s| s.parallel = parallel_runner.is_some())
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderSizeHintBasicInfo(dec: Option<&JxlDecoder>) -> usize {
    let Some(dec) = dec else {
        return 0;
    };
    if dec.has_basic_info() {
        0
    } else {
        dec.stream.basic_info_size_hint.unwrap_or(64)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderSubscribeEvents(
    dec: Option<&mut JxlDecoder>,
    events_wanted: c_int,
) -> JxlDecoderStatus {
    let Some(dec) = dec else {
        return JXL_DEC_ERROR;
    };
    if dec.stream.process_started || (events_wanted & !SUPPORTED_EVENTS_MASK) != 0 {
        return JXL_DEC_ERROR;
    }
    let combined = dec.events_wanted | events_wanted;
    if combined != 0 && (combined & TOP_LEVEL_EVENTS_MASK) == 0 {
        return JXL_DEC_ERROR;
    }
    dec.events_wanted = combined;
    JXL_DEC_SUCCESS
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderSetKeepOrientation(
    dec: Option<&mut JxlDecoder>,
    skip_reorientation: JXL_BOOL,
) -> JxlDecoderStatus {
    dec.map_or(JXL_DEC_ERROR, |d| {
        d.set_before_process(|s| s.keep_orientation = skip_reorientation != JXL_FALSE)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderSetUnpremultiplyAlpha(
    dec: Option<&mut JxlDecoder>,
    unpremul_alpha: JXL_BOOL,
) -> JxlDecoderStatus {
    dec.map_or(JXL_DEC_ERROR, |d| {
        d.set_before_process(|s| s.unpremultiply_alpha = unpremul_alpha != JXL_FALSE)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderSetRenderSpotcolors(
    dec: Option<&mut JxlDecoder>,
    render_spotcolors: JXL_BOOL,
) -> JxlDecoderStatus {
    dec.map_or(JXL_DEC_ERROR, |d| {
        d.set_before_process(|s| s.render_spotcolors = render_spotcolors != JXL_FALSE)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderSetCoalescing(
    dec: Option<&mut JxlDecoder>,
    coalescing: JXL_BOOL,
) -> JxlDecoderStatus {
    if coalescing == JXL_FALSE {
        return JXL_DEC_ERROR;
    }
    dec.map_or(JXL_DEC_ERROR, |d| d.set_before_process(|_| {}))
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderProcessInput(dec: Option<&mut JxlDecoder>) -> JxlDecoderStatus {
    dec.map_or(JXL_DEC_ERROR, JxlDecoder::process_input)
}

/// Sets the input byte buffer for the decoder.
///
/// # Safety
/// If `data` is non-null and `size > 0`, the caller must ensure that during calls to functions in
/// this crate until the buffer is released by `JxlDecoderReleaseInput`, `JxlDecoderRewind`,
/// `JxlDecoderReset`, or `JxlDecoderDestroy`, `data` is valid for reads of `size` initialized
/// bytes (within a single allocation of at most `isize::MAX` bytes) and the pointed-to bytes are
/// not modified concurrently (including via overlapping output buffers registered on `dec`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn JxlDecoderSetInput(
    dec: Option<&mut JxlDecoder>,
    data: *const u8,
    size: usize,
) -> JxlDecoderStatus {
    let Some(dec) = dec else {
        return JXL_DEC_ERROR;
    };
    if dec.stream.input.is_some() || dec.stream.input_closed {
        return JXL_DEC_ERROR;
    }
    // SAFETY: Guaranteed by caller contract on `JxlDecoderSetInput`.
    let Some(mut buf) = (unsafe { InputBuffer::new(data, size) }) else {
        return JXL_DEC_ERROR;
    };
    let held_back = std::mem::replace(&mut dec.stream.held_back_byte, false);
    if held_back && size > 0 {
        buf.consume(1);
    }
    dec.stream.input_had_bytes = size > 0;
    if dec.stream.have_container.is_none() && buf.remaining() > 0 {
        let slice = buf.as_slice();
        let need = 16usize
            .saturating_sub(dec.stream.signature_probe.len())
            .min(slice.len());
        dec.stream.signature_probe.extend_from_slice(&slice[..need]);
        match check_signature(&dec.stream.signature_probe) {
            RsSignature::Container => dec.stream.have_container = Some(true),
            RsSignature::Codestream => dec.stream.have_container = Some(false),
            _ => {}
        }
    }
    dec.stream.input = Some(buf);
    JXL_DEC_SUCCESS
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderReleaseInput(dec: Option<&mut JxlDecoder>) -> usize {
    let Some(dec) = dec else {
        return 0;
    };
    let Some(inp) = dec.stream.input.take() else {
        return 0;
    };
    let rem = inp.remaining();
    if rem == 0 && dec.stream.input_had_bytes && dec.stream.can_continue {
        dec.stream.held_back_byte = true;
        1
    } else {
        rem
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderCloseInput(dec: Option<&mut JxlDecoder>) {
    if let Some(dec) = dec {
        dec.stream.input_closed = true;
        dec.rs_decoder.close_input();
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderGetBasicInfo(
    dec: Option<&JxlDecoder>,
    info: Option<&mut MaybeUninit<JxlBasicInfo>>,
) -> JxlDecoderStatus {
    let Some(dec) = dec else {
        return JXL_DEC_ERROR;
    };
    let Some(rs_info) = dec.basic_info_ref() else {
        return JXL_DEC_NEED_MORE_INPUT;
    };
    if let Some(info) = info {
        info.write(convert_basic_info(
            rs_info,
            dec.stream.have_container.unwrap_or(false),
            dec.is_gray(),
            dec.keep_orientation,
            dec.unpremultiply_alpha,
        ));
    }
    JXL_DEC_SUCCESS
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderGetExtraChannelInfo(
    dec: Option<&JxlDecoder>,
    index: usize,
    info: Option<&mut MaybeUninit<JxlExtraChannelInfo>>,
) -> JxlDecoderStatus {
    let (Some(dec), Some(info)) = (dec, info) else {
        return JXL_DEC_ERROR;
    };
    let Some(rs_info) = dec.basic_info_ref() else {
        return JXL_DEC_NEED_MORE_INPUT;
    };
    let Some(ec) = rs_info.extra_channels.get(index) else {
        return JXL_DEC_ERROR;
    };
    info.write(convert_extra_channel_info(ec));
    JXL_DEC_SUCCESS
}

/// Writes the null-terminated name of extra channel `index` into `name`.
///
/// # Safety
/// If `name` is non-null and `size > 0`, the caller must ensure `name` is valid for writes of
/// `size` bytes (within a single allocation), exclusively accessible, and non-overlapping with
/// `dec` for the duration of this function call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn JxlDecoderGetExtraChannelName(
    dec: Option<&JxlDecoder>,
    index: usize,
    name: *mut c_char,
    size: usize,
) -> JxlDecoderStatus {
    let Some(ec) = dec
        .and_then(JxlDecoder::basic_info_ref)
        .and_then(|b| b.extra_channels.get(index))
    else {
        return JXL_DEC_ERROR;
    };
    // SAFETY: Guaranteed by caller contract on `JxlDecoderGetExtraChannelName` (`ec.name` lives
    // inside `dec`, which the caller guarantees does not overlap `name`).
    unsafe { copy_c_string(&ec.name, name, size) }
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderGetColorAsEncodedProfile(
    dec: Option<&JxlDecoder>,
    target: JxlColorProfileTarget,
    color_encoding: Option<&mut MaybeUninit<JxlColorEncoding>>,
) -> JxlDecoderStatus {
    let Some(dec) = dec else {
        return JXL_DEC_ERROR;
    };
    if !dec.has_basic_info() {
        return JXL_DEC_NEED_MORE_INPUT;
    }
    let Some(RsColorProfile::Simple(enc)) = dec.select_color_profile(target) else {
        return JXL_DEC_ERROR;
    };
    if let Some(color_encoding) = color_encoding {
        color_encoding.write(convert_color_encoding(enc));
    }
    JXL_DEC_SUCCESS
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderGetICCProfileSize(
    dec: Option<&JxlDecoder>,
    target: JxlColorProfileTarget,
    size: Option<&mut MaybeUninit<usize>>,
) -> JxlDecoderStatus {
    let Some(dec) = dec else {
        return JXL_DEC_ERROR;
    };
    if !dec.has_basic_info() {
        return JXL_DEC_NEED_MORE_INPUT;
    }
    let Some(icc) = dec
        .select_color_profile(target)
        .and_then(RsColorProfile::try_as_icc)
    else {
        return JXL_DEC_ERROR;
    };
    if let Some(size) = size {
        size.write(icc.len());
    }
    JXL_DEC_SUCCESS
}

/// Copies the ICC profile for `target` into `icc_profile`.
///
/// # Safety
/// If `icc_profile` is non-null and `size > 0`, the caller must ensure `icc_profile` is valid for
/// writes of `size` bytes (within a single allocation), exclusively accessible, and
/// non-overlapping with `dec` for the duration of this function call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn JxlDecoderGetColorAsICCProfile(
    dec: Option<&JxlDecoder>,
    target: JxlColorProfileTarget,
    icc_profile: *mut u8,
    size: usize,
) -> JxlDecoderStatus {
    let Some(dec) = dec else {
        return JXL_DEC_ERROR;
    };
    if !dec.has_basic_info() {
        return JXL_DEC_NEED_MORE_INPUT;
    }
    let Some(icc) = dec
        .select_color_profile(target)
        .and_then(RsColorProfile::try_as_icc)
    else {
        return JXL_DEC_ERROR;
    };
    if icc_profile.is_null() || size < icc.len() {
        return JXL_DEC_ERROR;
    }
    // SAFETY: `icc_profile` is non-null and `size >= icc.len()`, so by caller contract
    // `icc_profile` is valid for writes of `icc.len()` bytes and does not overlap `icc`.
    unsafe {
        std::ptr::copy_nonoverlapping(icc.as_ptr(), icc_profile, icc.len());
    }
    JXL_DEC_SUCCESS
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderSetPreferredColorProfile(
    dec: Option<&mut JxlDecoder>,
    color_encoding: Option<&JxlColorEncoding>,
) -> JxlDecoderStatus {
    let (Some(dec), Some(color_encoding)) = (dec, color_encoding) else {
        return JXL_DEC_ERROR;
    };
    if !dec.can_set_color_profile() {
        return JXL_DEC_ERROR;
    }
    let Some(rs_enc) = to_rs_color_encoding(color_encoding) else {
        return JXL_DEC_ERROR;
    };
    let profile = RsColorProfile::Simple(rs_enc);
    if !profile.can_output_to() {
        return JXL_DEC_ERROR;
    }
    // In libjxl, JxlDecoderSetPreferredColorProfile is only a hint; the caller must
    // query JXL_COLOR_PROFILE_TARGET_DATA to obtain the actual output color profile.
    JXL_DEC_SUCCESS
}

// Unsupported feature: HDR tone mapping to a target display luminance is not implemented.
#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderSetDesiredIntensityTarget(
    _dec: Option<&mut JxlDecoder>,
    _desired_intensity_target: f32,
) -> JxlDecoderStatus {
    JXL_DEC_ERROR
}

// Unsupported feature: custom output color profiles require a CMS (`JxlDecoderSetCms`),
// which is not implemented.
#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderSetOutputColorProfile(
    _dec: Option<&mut JxlDecoder>,
    _color_encoding: Option<&JxlColorEncoding>,
    _icc_data: *const u8,
    _icc_size: usize,
) -> JxlDecoderStatus {
    JXL_DEC_ERROR
}

// Unsupported feature: custom CMS interfaces (`JxlCmsInterface`) are not implemented.
#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderSetCms(
    _dec: Option<&mut JxlDecoder>,
    _cms: JxlCmsInterface,
) -> JxlDecoderStatus {
    JXL_DEC_ERROR
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderPreviewOutBufferSize(
    dec: Option<&JxlDecoder>,
    format: Option<&JxlPixelFormat>,
    size: Option<&mut MaybeUninit<usize>>,
) -> JxlDecoderStatus {
    let (Some(dec), Some(format), Some(size)) = (dec, format, size) else {
        return JXL_DEC_ERROR;
    };
    let Some(preview_size) = dec.basic_info_ref().and_then(|b| b.preview_size) else {
        return JXL_DEC_ERROR;
    };
    let Some(bytes) = dec.color_buffer_size(preview_size, format) else {
        return JXL_DEC_ERROR;
    };
    size.write(bytes);
    JXL_DEC_SUCCESS
}

/// Sets the output buffer for the preview image.
///
/// # Safety
/// If `buffer` is non-null and `size > 0`, the caller must ensure that during calls to functions
/// in this crate until `JXL_DEC_PREVIEW_IMAGE` is returned or the decoder is rewound, reset, or
/// destroyed, `buffer` is valid for writes of `size` bytes (within a single allocation) and
/// remains exclusively accessible (non-overlapping with `dec` and any other buffers registered on
/// `dec`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn JxlDecoderSetPreviewOutBuffer(
    dec: Option<&mut JxlDecoder>,
    format: Option<&JxlPixelFormat>,
    buffer: *mut c_void,
    size: usize,
) -> JxlDecoderStatus {
    let (Some(dec), Some(format)) = (dec, format) else {
        return JXL_DEC_ERROR;
    };
    let Some(preview_size) = dec.basic_info_ref().and_then(|b| b.preview_size) else {
        return JXL_DEC_ERROR;
    };
    let Some(min_size) = dec.color_buffer_size(preview_size, format) else {
        return JXL_DEC_ERROR;
    };
    let codestream_bd = &dec.basic_info_ref().unwrap().bit_depth;
    if to_rs_data_format(format, &dec.output_bit_depth, codestream_bd).is_none() {
        return JXL_DEC_ERROR;
    }
    // SAFETY: Guaranteed by caller contract on `JxlDecoderSetPreviewOutBuffer`.
    let Some(target) =
        (unsafe { ImageOutBuffer::new(buffer.cast::<u8>(), size, min_size, *format) })
    else {
        return JXL_DEC_ERROR;
    };
    dec.stream.preview_out_target = Some(target);
    JXL_DEC_SUCCESS
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderGetFrameHeader(
    dec: Option<&JxlDecoder>,
    header: Option<&mut MaybeUninit<JxlFrameHeader>>,
) -> JxlDecoderStatus {
    let (Some(dec), Some(header)) = (dec, header) else {
        return JXL_DEC_ERROR;
    };
    let Some((val, _)) = dec.stream.current_frame_header.as_ref() else {
        return JXL_DEC_ERROR;
    };
    header.write(*val);
    JXL_DEC_SUCCESS
}

/// Writes the null-terminated name of the current frame into `name`.
///
/// # Safety
/// If `name` is non-null and `size > 0`, the caller must ensure `name` is valid for writes of
/// `size` bytes (within a single allocation), exclusively accessible, and non-overlapping with
/// `dec` for the duration of this function call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn JxlDecoderGetFrameName(
    dec: Option<&JxlDecoder>,
    name: *mut c_char,
    size: usize,
) -> JxlDecoderStatus {
    let Some((_, frame_name)) = dec.and_then(|d| d.stream.current_frame_header.as_ref()) else {
        return JXL_DEC_ERROR;
    };
    // SAFETY: Guaranteed by caller contract on `JxlDecoderGetFrameName` (`frame_name` lives inside
    // `dec`, which the caller guarantees does not overlap `name`).
    unsafe { copy_c_string(frame_name, name, size) }
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderGetExtraChannelBlendInfo(
    dec: Option<&JxlDecoder>,
    index: usize,
    blend_info: Option<&mut MaybeUninit<JxlBlendInfo>>,
) -> JxlDecoderStatus {
    let (Some(dec), Some(blend_info)) = (dec, blend_info) else {
        return JXL_DEC_ERROR;
    };
    let has_ec = dec
        .basic_info_ref()
        .is_some_and(|b| index < b.extra_channels.len());
    if !has_ec || dec.stream.current_frame_header.is_none() {
        return JXL_DEC_ERROR;
    }
    // Note: non-coalesced decoding is unsupported, so coalesced frames always report
    // `JXL_BLEND_REPLACE` for extra channels.
    blend_info.write(JxlBlendInfo {
        blendmode: JXL_BLEND_REPLACE,
        source: 0,
        alpha: 0,
        clamp: JXL_FALSE,
    });
    JXL_DEC_SUCCESS
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderImageOutBufferSize(
    dec: Option<&JxlDecoder>,
    format: Option<&JxlPixelFormat>,
    size: Option<&mut MaybeUninit<usize>>,
) -> JxlDecoderStatus {
    let (Some(dec), Some(format), Some(size)) = (dec, format, size) else {
        return JXL_DEC_ERROR;
    };
    let Some(img_size) = dec.basic_info_ref().map(|b| b.size) else {
        return JXL_DEC_ERROR;
    };
    let Some(bytes) = dec.color_buffer_size(img_size, format) else {
        return JXL_DEC_ERROR;
    };
    size.write(bytes);
    JXL_DEC_SUCCESS
}

/// Sets the output buffer for the current image frame.
///
/// # Safety
/// If `buffer` is non-null and `size > 0`, the caller must ensure that during calls to functions
/// in this crate until `JXL_DEC_FULL_IMAGE` is returned or the decoder is rewound, reset, or
/// destroyed, `buffer` is valid for writes of `size` bytes (within a single allocation) and
/// remains exclusively accessible (non-overlapping with `dec` and any other buffers registered on
/// `dec`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn JxlDecoderSetImageOutBuffer(
    dec: Option<&mut JxlDecoder>,
    format: Option<&JxlPixelFormat>,
    buffer: *mut c_void,
    size: usize,
) -> JxlDecoderStatus {
    let (Some(dec), Some(format)) = (dec, format) else {
        return JXL_DEC_ERROR;
    };
    let Some(img_size) = dec.basic_info_ref().map(|b| b.size) else {
        return JXL_DEC_ERROR;
    };
    let Some(min_size) = dec.color_buffer_size(img_size, format) else {
        return JXL_DEC_ERROR;
    };
    // SAFETY: Guaranteed by caller contract on `JxlDecoderSetImageOutBuffer`.
    let Some(target) =
        (unsafe { ImageOutBuffer::new(buffer.cast::<u8>(), size, min_size, *format) })
    else {
        return JXL_DEC_ERROR;
    };
    let prev = dec.stream.image_out_target.replace(target);
    if !dec.apply_main_pixel_format() {
        dec.stream.image_out_target = prev;
        return JXL_DEC_ERROR;
    }
    JXL_DEC_SUCCESS
}

// Unsupported feature: row-by-row output callbacks (`JxlImageOutCallback` and
// `JxlImageOutInitCallback` / `JxlImageOutRunCallback`) are not implemented; use
// `JxlDecoderSetImageOutBuffer` instead.
#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderSetImageOutCallback(
    _dec: Option<&mut JxlDecoder>,
    _format: Option<&JxlPixelFormat>,
    _callback: JxlImageOutCallback,
    _opaque: *mut c_void,
) -> JxlDecoderStatus {
    JXL_DEC_ERROR
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderSetMultithreadedImageOutCallback(
    _dec: Option<&mut JxlDecoder>,
    _format: Option<&JxlPixelFormat>,
    _init_callback: JxlImageOutInitCallback,
    _run_callback: JxlImageOutRunCallback,
    _destroy_callback: JxlImageOutDestroyCallback,
    _init_opaque: *mut c_void,
) -> JxlDecoderStatus {
    JXL_DEC_ERROR
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderExtraChannelBufferSize(
    dec: Option<&JxlDecoder>,
    format: Option<&JxlPixelFormat>,
    size: Option<&mut MaybeUninit<usize>>,
    index: u32,
) -> JxlDecoderStatus {
    let (Some(dec), Some(format), Some(size)) = (dec, format, size) else {
        return JXL_DEC_ERROR;
    };
    let Some(rs_info) = dec.basic_info_ref() else {
        return JXL_DEC_ERROR;
    };
    if (index as usize) >= rs_info.extra_channels.len() {
        return JXL_DEC_ERROR;
    };
    let mut ec_format = *format;
    ec_format.num_channels = 1;
    let Some(bytes) = compute_buffer_size(rs_info.size.0, rs_info.size.1, &ec_format) else {
        return JXL_DEC_ERROR;
    };
    size.write(bytes);
    JXL_DEC_SUCCESS
}

/// Sets the output buffer for extra channel `index` of the current frame.
///
/// # Safety
/// If `buffer` is non-null and `size > 0`, the caller must ensure that during calls to functions
/// in this crate until `JXL_DEC_FULL_IMAGE` is returned or the decoder is rewound, reset, or
/// destroyed, `buffer` is valid for writes of `size` bytes (within a single allocation) and
/// remains exclusively accessible (non-overlapping with `dec` and any other buffers registered on
/// `dec`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn JxlDecoderSetExtraChannelBuffer(
    dec: Option<&mut JxlDecoder>,
    format: Option<&JxlPixelFormat>,
    buffer: *mut c_void,
    size: usize,
    index: u32,
) -> JxlDecoderStatus {
    let (Some(dec), Some(format)) = (dec, format) else {
        return JXL_DEC_ERROR;
    };
    let idx = index as usize;
    let Some(rs_info) = dec.basic_info_ref() else {
        return JXL_DEC_ERROR;
    };
    if idx >= rs_info.extra_channels.len() || idx >= dec.stream.extra_channel_targets.len() {
        return JXL_DEC_ERROR;
    };
    let mut ec_format = *format;
    ec_format.num_channels = 1;
    let Some(min_size) = compute_buffer_size(rs_info.size.0, rs_info.size.1, &ec_format) else {
        return JXL_DEC_ERROR;
    };
    // SAFETY: Guaranteed by caller contract on `JxlDecoderSetExtraChannelBuffer`.
    let Some(target) =
        (unsafe { ImageOutBuffer::new(buffer.cast::<u8>(), size, min_size, ec_format) })
    else {
        return JXL_DEC_ERROR;
    };
    let prev = dec.stream.extra_channel_targets[idx].replace(target);
    if !dec.apply_main_pixel_format() {
        dec.stream.extra_channel_targets[idx] = prev;
        return JXL_DEC_ERROR;
    }
    JXL_DEC_SUCCESS
}

// Unsupported features: JPEG reconstruction (`JXL_DEC_JPEG_RECONSTRUCTION`) and progressive
// detail events (`JXL_DEC_FRAME_PROGRESSION`) are not implemented.

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderSetJPEGBuffer(
    _dec: Option<&mut JxlDecoder>,
    _data: *mut u8,
    _size: usize,
) -> JxlDecoderStatus {
    JXL_DEC_ERROR
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderReleaseJPEGBuffer(_dec: Option<&mut JxlDecoder>) -> usize {
    0
}

/// Sets the output buffer for the current container box payload.
///
/// # Safety
/// If `data` is non-null and `size > 0`, the caller must ensure that during calls to functions in
/// this crate until the buffer is released by `JxlDecoderReleaseBoxBuffer`, `JxlDecoderRewind`,
/// `JxlDecoderReset`, or `JxlDecoderDestroy`, `data` is valid for writes of `size` bytes (within a
/// single allocation of at most `isize::MAX` bytes) and remains exclusively accessible
/// (non-overlapping with `dec` and any other buffers registered on `dec`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn JxlDecoderSetBoxBuffer(
    dec: Option<&mut JxlDecoder>,
    data: *mut u8,
    size: usize,
) -> JxlDecoderStatus {
    let Some(dec) = dec else {
        return JXL_DEC_ERROR;
    };
    if dec.stream.box_out_buffer.is_some() {
        return JXL_DEC_ERROR;
    }
    // SAFETY: Guaranteed by caller contract on `JxlDecoderSetBoxBuffer`.
    let Some(buf) = (unsafe { BoxOutBuffer::new(data, size) }) else {
        return JXL_DEC_ERROR;
    };
    dec.stream.box_out_buffer = Some(buf);
    if let Some(active) = dec.stream.current_box.as_mut() {
        active.buffer_set = true;
    }
    JXL_DEC_SUCCESS
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderReleaseBoxBuffer(dec: Option<&mut JxlDecoder>) -> usize {
    let Some(dec) = dec else {
        return 0;
    };
    if let Some(active) = dec.stream.current_box.as_mut() {
        active.buffer_set = false;
    }
    dec.stream
        .box_out_buffer
        .take()
        .map_or(0, |buf| buf.remaining())
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderSetDecompressBoxes(
    dec: Option<&mut JxlDecoder>,
    decompress: JXL_BOOL,
) -> JxlDecoderStatus {
    let Some(dec) = dec else {
        return JXL_DEC_ERROR;
    };
    #[cfg(not(feature = "brotli"))]
    if decompress != JXL_FALSE {
        return JXL_DEC_ERROR;
    }
    dec.decompress_boxes = decompress != JXL_FALSE;
    JXL_DEC_SUCCESS
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderGetBoxType(
    dec: Option<&mut JxlDecoder>,
    box_type: Option<&mut MaybeUninit<JxlBoxType>>,
    decompressed: JXL_BOOL,
) -> JxlDecoderStatus {
    let (Some(dec), Some(box_type)) = (dec, box_type) else {
        return JXL_DEC_ERROR;
    };
    let Some(active) = dec.stream.current_box.as_ref() else {
        return JXL_DEC_ERROR;
    };
    let ty = if decompressed == JXL_FALSE && active.pending.is_compressed {
        *b"brob"
    } else {
        active.pending.box_type
    };
    box_type.write(ty.map(|b| b as c_char));
    JXL_DEC_SUCCESS
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderGetBoxSizeRaw(
    dec: Option<&JxlDecoder>,
    size: Option<&mut MaybeUninit<u64>>,
) -> JxlDecoderStatus {
    let Some(active) = dec.and_then(|d| d.stream.current_box.as_ref()) else {
        return JXL_DEC_ERROR;
    };
    if let Some(size) = size {
        size.write(active.pending.raw_size);
    }
    JXL_DEC_SUCCESS
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderGetBoxSizeContents(
    dec: Option<&JxlDecoder>,
    size: Option<&mut MaybeUninit<u64>>,
) -> JxlDecoderStatus {
    let Some(active) = dec.and_then(|d| d.stream.current_box.as_ref()) else {
        return JXL_DEC_ERROR;
    };
    if let Some(size) = size {
        size.write(active.pending.contents_size);
    }
    JXL_DEC_SUCCESS
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderSetProgressiveDetail(
    _dec: Option<&mut JxlDecoder>,
    _detail: JxlProgressiveDetail,
) -> JxlDecoderStatus {
    JXL_DEC_ERROR
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderGetIntendedDownsamplingRatio(dec: Option<&JxlDecoder>) -> usize {
    usize::from(dec.is_some())
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderFlushImage(dec: Option<&mut JxlDecoder>) -> JxlDecoderStatus {
    let Some(dec) = dec else {
        return JXL_DEC_ERROR;
    };
    if dec.stream.stage != Stage::MainDecode || !dec.has_any_main_output_target() {
        return JXL_DEC_ERROR;
    }
    let Some((width, height)) = dec.basic_info_ref().map(|b| b.size) else {
        return JXL_DEC_ERROR;
    };
    let Some(mut output_buffers) = make_main_output_buffers(
        width,
        height,
        &mut dec.stream.image_out_target,
        &mut dec.stream.extra_channel_targets,
    ) else {
        return JXL_DEC_ERROR;
    };
    let mut runner = RayonParallelRunner;
    match dec.rs_decoder.flush_pixels(
        &mut output_buffers,
        dec.parallel
            .then_some(&mut runner as &mut dyn RsParallelRunner),
    ) {
        Ok(flushed) => {
            dec.stream.frame_flushed |= flushed;
            if dec.stream.frame_flushed {
                JXL_DEC_SUCCESS
            } else {
                JXL_DEC_ERROR
            }
        }
        Err(_) => JXL_DEC_ERROR,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderSetImageOutBitDepth(
    dec: Option<&mut JxlDecoder>,
    bit_depth: Option<&JxlBitDepth>,
) -> JxlDecoderStatus {
    let (Some(dec), Some(bit_depth)) = (dec, bit_depth) else {
        return JXL_DEC_ERROR;
    };
    if dec.stream.stage == Stage::MainDecode {
        return JXL_DEC_ERROR;
    }
    match bit_depth.type_ {
        JXL_BIT_DEPTH_FROM_PIXEL_FORMAT | JXL_BIT_DEPTH_FROM_CODESTREAM => {}
        JXL_BIT_DEPTH_CUSTOM if (1..=32).contains(&bit_depth.bits_per_sample) => {}
        _ => return JXL_DEC_ERROR,
    }
    let prev = dec.output_bit_depth;
    dec.output_bit_depth = *bit_depth;
    if dec.has_basic_info() && !dec.apply_main_pixel_format() {
        dec.output_bit_depth = prev;
        return JXL_DEC_ERROR;
    }
    JXL_DEC_SUCCESS
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlColorEncodingSetToSRGB(
    color_encoding: Option<&mut MaybeUninit<JxlColorEncoding>>,
    is_gray: JXL_BOOL,
) {
    if let Some(color_encoding) = color_encoding {
        color_encoding.write(convert_color_encoding(&RsColorEncoding::srgb(
            is_gray != JXL_FALSE,
        )));
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlColorEncodingSetToLinearSRGB(
    color_encoding: Option<&mut MaybeUninit<JxlColorEncoding>>,
    is_gray: JXL_BOOL,
) {
    if let Some(color_encoding) = color_encoding {
        color_encoding.write(convert_color_encoding(&RsColorEncoding::linear_srgb(
            is_gray != JXL_FALSE,
        )));
    }
}

// Unsupported feature: reading `JxlGainMapBundle` (`JxlGainMapReadBundle`) is not implemented.
#[unsafe(no_mangle)]
pub extern "C" fn JxlGainMapReadBundle(
    _map_bundle: Option<&mut MaybeUninit<JxlGainMapBundle>>,
    _input_buffer: *const u8,
    _input_buffer_size: usize,
    _bytes_read: Option<&mut MaybeUninit<usize>>,
) -> JXL_BOOL {
    JXL_FALSE
}
