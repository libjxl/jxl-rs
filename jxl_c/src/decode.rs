// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::ffi::{c_char, c_int, c_void};

use jxl::api::{
    JxlColorEncoding as RsColorEncoding, JxlColorProfile as RsColorProfile,
    JxlSignature as RsSignature, check_signature,
};

use crate::parallel_runner::CParallelRunner;
use crate::state_machine::{JxlDecoder, convert_color_encoding};
use crate::types::{
    JPEGXL_NUMERIC_VERSION, JXL_BLEND_REPLACE, JXL_BOOL, JXL_DEC_ERROR, JXL_DEC_NEED_MORE_INPUT,
    JXL_DEC_SUCCESS, JXL_FALSE, JXL_SIG_CODESTREAM, JXL_SIG_CONTAINER, JXL_SIG_INVALID,
    JXL_SIG_NOT_ENOUGH_BYTES, JXL_TRUE, JxlBasicInfo, JxlBitDepth, JxlBlendInfo, JxlCmsInterface,
    JxlColorEncoding, JxlColorProfileTarget, JxlDecoderStatus, JxlExtraChannelInfo, JxlFrameHeader,
    JxlGainMapBundle, JxlImageOutCallback, JxlImageOutDestroyCallback, JxlImageOutInitCallback,
    JxlImageOutRunCallback, JxlMemoryManager, JxlParallelRunner, JxlPixelFormat,
    JxlProgressiveDetail, JxlSignature,
};

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderVersion() -> u32 {
    JPEGXL_NUMERIC_VERSION
}

/// Checks if the input buffer starts with a valid JPEG XL codestream or container signature.
///
/// # Safety
/// If `len > 0`, `buf` must be a valid non-null pointer to at least `len` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn JxlSignatureCheck(buf: *const u8, len: usize) -> JxlSignature {
    if len == 0 {
        return JXL_SIG_NOT_ENOUGH_BYTES;
    }
    if buf.is_null() {
        return JXL_SIG_INVALID;
    }
    // SAFETY: Caller guarantees `buf` is valid for `len` bytes.
    let slice = unsafe { std::slice::from_raw_parts(buf, len) };
    match check_signature(slice) {
        RsSignature::NeedMoreInput { .. } => JXL_SIG_NOT_ENOUGH_BYTES,
        RsSignature::None => JXL_SIG_INVALID,
        RsSignature::Codestream => JXL_SIG_CODESTREAM,
        RsSignature::Container => JXL_SIG_CONTAINER,
    }
}

/// Creates a new `JxlDecoder` instance.
///
/// # Safety
/// `memory_manager` must be null or point to a valid `JxlMemoryManager`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn JxlDecoderCreate(
    memory_manager: *const JxlMemoryManager,
) -> Option<Box<JxlDecoder>> {
    if !memory_manager.is_null() {
        // SAFETY: `memory_manager` is non-null and valid per caller contract.
        let mm = unsafe { &*memory_manager };
        if mm.alloc.is_some() != mm.free.is_some() {
            return None;
        }
    }
    Some(Box::new(JxlDecoder::new()))
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderReset(dec: Option<&mut JxlDecoder>) {
    if let Some(dec) = dec {
        dec.reset();
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderDestroy(_dec: Option<Box<JxlDecoder>>) {}

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderRewind(dec: Option<&mut JxlDecoder>) {
    if let Some(dec) = dec {
        dec.rewind();
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderSkipFrames(dec: Option<&mut JxlDecoder>, amount: usize) {
    if let Some(dec) = dec {
        dec.skip_frames(amount);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderSkipCurrentFrame(dec: Option<&mut JxlDecoder>) -> JxlDecoderStatus {
    let Some(dec) = dec else {
        return JXL_DEC_ERROR;
    };
    dec.skip_current_frame()
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderSetParallelRunner(
    dec: Option<&mut JxlDecoder>,
    parallel_runner: JxlParallelRunner,
    parallel_runner_opaque: *mut c_void,
) -> JxlDecoderStatus {
    let Some(dec) = dec else {
        return JXL_DEC_ERROR;
    };
    let runner = parallel_runner.map(|runner| CParallelRunner {
        runner,
        runner_opaque: parallel_runner_opaque,
    });
    dec.set_parallel_runner(runner)
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderSizeHintBasicInfo(dec: Option<&JxlDecoder>) -> usize {
    dec.map_or(0, |d| d.size_hint_basic_info())
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderSubscribeEvents(
    dec: Option<&mut JxlDecoder>,
    events_wanted: c_int,
) -> JxlDecoderStatus {
    let Some(dec) = dec else {
        return JXL_DEC_ERROR;
    };
    dec.subscribe_events(events_wanted)
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderSetKeepOrientation(
    dec: Option<&mut JxlDecoder>,
    skip_reorientation: JXL_BOOL,
) -> JxlDecoderStatus {
    let Some(dec) = dec else {
        return JXL_DEC_ERROR;
    };
    dec.set_keep_orientation(skip_reorientation != JXL_FALSE)
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderSetUnpremultiplyAlpha(
    dec: Option<&mut JxlDecoder>,
    unpremul_alpha: JXL_BOOL,
) -> JxlDecoderStatus {
    let Some(dec) = dec else {
        return JXL_DEC_ERROR;
    };
    dec.set_unpremultiply_alpha(unpremul_alpha != JXL_FALSE)
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderSetRenderSpotcolors(
    dec: Option<&mut JxlDecoder>,
    render_spotcolors: JXL_BOOL,
) -> JxlDecoderStatus {
    let Some(dec) = dec else {
        return JXL_DEC_ERROR;
    };
    dec.set_render_spotcolors(render_spotcolors != JXL_FALSE)
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderSetCoalescing(
    dec: Option<&mut JxlDecoder>,
    coalescing: JXL_BOOL,
) -> JxlDecoderStatus {
    let Some(dec) = dec else {
        return JXL_DEC_ERROR;
    };
    dec.set_coalescing(coalescing != JXL_FALSE)
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderProcessInput(dec: Option<&mut JxlDecoder>) -> JxlDecoderStatus {
    let Some(dec) = dec else {
        return JXL_DEC_ERROR;
    };
    dec.process_input()
}

/// Sets the input byte buffer for the decoder.
///
/// # Safety
/// If `size > 0`, `data` must point to at least `size` readable bytes that remain valid until
/// `JxlDecoderReleaseInput` is called.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn JxlDecoderSetInput(
    dec: Option<&mut JxlDecoder>,
    data: *const u8,
    size: usize,
) -> JxlDecoderStatus {
    let Some(dec) = dec else {
        return JXL_DEC_ERROR;
    };
    dec.set_input(data, size)
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderReleaseInput(dec: Option<&mut JxlDecoder>) -> usize {
    dec.map_or(0, |d| d.release_input())
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderCloseInput(dec: Option<&mut JxlDecoder>) {
    if let Some(dec) = dec {
        dec.close_input();
    }
}

/// Populates `info` with basic image metadata.
///
/// # Safety
/// `info` must be null or point to a valid writable `JxlBasicInfo`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn JxlDecoderGetBasicInfo(
    dec: Option<&JxlDecoder>,
    info: *mut JxlBasicInfo,
) -> JxlDecoderStatus {
    let Some(dec) = dec else {
        return JXL_DEC_ERROR;
    };
    let Some(val) = dec.get_basic_info() else {
        return JXL_DEC_NEED_MORE_INPUT;
    };
    if !info.is_null() {
        // SAFETY: `info` is non-null and points to writable memory for `JxlBasicInfo`.
        unsafe {
            info.write(val);
        }
    }
    JXL_DEC_SUCCESS
}

/// Populates `info` with metadata for extra channel `index`.
///
/// # Safety
/// `info` must be null or point to a valid writable `JxlExtraChannelInfo`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn JxlDecoderGetExtraChannelInfo(
    dec: Option<&JxlDecoder>,
    index: usize,
    info: *mut JxlExtraChannelInfo,
) -> JxlDecoderStatus {
    let Some(dec) = dec else {
        return JXL_DEC_ERROR;
    };
    if info.is_null() {
        return JXL_DEC_ERROR;
    }
    if !dec.has_basic_info() {
        return JXL_DEC_NEED_MORE_INPUT;
    }
    let Some(val) = dec.get_extra_channel_info(index) else {
        return JXL_DEC_ERROR;
    };
    // SAFETY: `info` is non-null and points to writable memory for `JxlExtraChannelInfo`.
    unsafe {
        info.write(val);
    }
    JXL_DEC_SUCCESS
}

/// Writes the null-terminated name of extra channel `index` into `name`.
///
/// # Safety
/// If `size > 0`, `name` must point to at least `size` writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn JxlDecoderGetExtraChannelName(
    dec: Option<&JxlDecoder>,
    index: usize,
    name: *mut c_char,
    size: usize,
) -> JxlDecoderStatus {
    let Some(dec) = dec else {
        return JXL_DEC_ERROR;
    };
    dec.get_extra_channel_name(index, name.cast::<u8>(), size)
}

/// Retrieves the color profile as a `JxlColorEncoding` if representable as a parameterized profile.
///
/// # Safety
/// `color_encoding` must be null or point to a valid writable `JxlColorEncoding`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn JxlDecoderGetColorAsEncodedProfile(
    dec: Option<&JxlDecoder>,
    target: JxlColorProfileTarget,
    color_encoding: *mut JxlColorEncoding,
) -> JxlDecoderStatus {
    let Some(dec) = dec else {
        return JXL_DEC_ERROR;
    };
    if !dec.has_basic_info() {
        return JXL_DEC_NEED_MORE_INPUT;
    }
    let Some(val) = dec.get_color_as_encoded_profile(target) else {
        return JXL_DEC_ERROR;
    };
    if !color_encoding.is_null() {
        // SAFETY: `color_encoding` is non-null and points to writable memory for `JxlColorEncoding`.
        unsafe {
            color_encoding.write(val);
        }
    }
    JXL_DEC_SUCCESS
}

/// Retrieves the byte size of the ICC profile for `target`.
///
/// # Safety
/// `size` must be null or point to a valid writable `usize`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn JxlDecoderGetICCProfileSize(
    dec: Option<&JxlDecoder>,
    target: JxlColorProfileTarget,
    size: *mut usize,
) -> JxlDecoderStatus {
    let Some(dec) = dec else {
        return JXL_DEC_ERROR;
    };
    if !dec.has_basic_info() {
        return JXL_DEC_NEED_MORE_INPUT;
    }
    let Some(val) = dec.get_icc_profile_size(target) else {
        return JXL_DEC_ERROR;
    };
    if !size.is_null() {
        // SAFETY: `size` is non-null and points to writable `usize`.
        unsafe {
            size.write(val);
        }
    }
    JXL_DEC_SUCCESS
}

/// Copies the ICC profile for `target` into `icc_profile`.
///
/// # Safety
/// `icc_profile` must point to at least `size` writable bytes.
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
    dec.get_color_as_icc_profile(target, icc_profile, size)
}

/// Sets the preferred output color profile.
///
/// # Safety
/// `color_encoding` must be null or point to a valid `JxlColorEncoding`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn JxlDecoderSetPreferredColorProfile(
    dec: Option<&mut JxlDecoder>,
    color_encoding: *const JxlColorEncoding,
) -> JxlDecoderStatus {
    let (Some(dec), Some(color_encoding)) = (
        dec,
        // SAFETY: `color_encoding` is null or a valid `JxlColorEncoding` pointer.
        unsafe { color_encoding.as_ref() },
    ) else {
        return JXL_DEC_ERROR;
    };
    dec.set_preferred_color_profile(color_encoding)
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderSetDesiredIntensityTarget(
    dec: Option<&mut JxlDecoder>,
    desired_intensity_target: f32,
) -> JxlDecoderStatus {
    let Some(dec) = dec else {
        return JXL_DEC_ERROR;
    };
    dec.set_desired_intensity_target(desired_intensity_target)
}

/// Sets an explicit output color profile (or ICC profile).
///
/// # Safety
/// `color_encoding` must be null or point to a valid `JxlColorEncoding`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn JxlDecoderSetOutputColorProfile(
    dec: Option<&mut JxlDecoder>,
    color_encoding: *const JxlColorEncoding,
    icc_data: *const u8,
    icc_size: usize,
) -> JxlDecoderStatus {
    if (!icc_data.is_null() && icc_size > 0) || color_encoding.is_null() {
        return JXL_DEC_ERROR;
    }
    // SAFETY: `color_encoding` was checked for null above.
    unsafe { JxlDecoderSetPreferredColorProfile(dec, color_encoding) }
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderSetCms(
    _dec: Option<&mut JxlDecoder>,
    _cms: JxlCmsInterface,
) -> JxlDecoderStatus {
    JXL_DEC_ERROR
}

/// Computes the minimum byte size required for the preview output buffer.
///
/// # Safety
/// `format` and `size` must be null or point to valid `JxlPixelFormat` and writable `usize`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn JxlDecoderPreviewOutBufferSize(
    dec: Option<&JxlDecoder>,
    format: *const JxlPixelFormat,
    size: *mut usize,
) -> JxlDecoderStatus {
    let (Some(dec), Some(format)) = (
        dec,
        // SAFETY: `format` is null or a valid `JxlPixelFormat` pointer.
        unsafe { format.as_ref() },
    ) else {
        return JXL_DEC_ERROR;
    };
    if size.is_null() {
        return JXL_DEC_ERROR;
    }
    let Some(bytes) = dec.preview_out_buffer_size(format) else {
        return JXL_DEC_ERROR;
    };
    // SAFETY: `size` is non-null and writable.
    unsafe {
        size.write(bytes);
    }
    JXL_DEC_SUCCESS
}

/// Sets the output buffer for decoding the preview frame.
///
/// # Safety
/// `format` must be null or point to a valid `JxlPixelFormat`, and `buffer` must point to at least
/// `size` writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn JxlDecoderSetPreviewOutBuffer(
    dec: Option<&mut JxlDecoder>,
    format: *const JxlPixelFormat,
    buffer: *mut c_void,
    size: usize,
) -> JxlDecoderStatus {
    let (Some(dec), Some(format)) = (
        dec,
        // SAFETY: `format` is null or a valid `JxlPixelFormat` pointer.
        unsafe { format.as_ref() },
    ) else {
        return JXL_DEC_ERROR;
    };
    dec.set_preview_out_buffer(format, buffer, size)
}

/// Populates `header` with information about the current frame.
///
/// # Safety
/// `header` must be null or point to a valid writable `JxlFrameHeader`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn JxlDecoderGetFrameHeader(
    dec: Option<&JxlDecoder>,
    header: *mut JxlFrameHeader,
) -> JxlDecoderStatus {
    let Some(dec) = dec else {
        return JXL_DEC_ERROR;
    };
    if header.is_null() {
        return JXL_DEC_ERROR;
    }
    let Some(val) = dec.get_frame_header() else {
        return JXL_DEC_ERROR;
    };
    // SAFETY: `header` is non-null and points to writable memory for `JxlFrameHeader`.
    unsafe {
        header.write(val);
    }
    JXL_DEC_SUCCESS
}

/// Writes the null-terminated name of the current frame into `name`.
///
/// # Safety
/// If `size > 0`, `name` must point to at least `size` writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn JxlDecoderGetFrameName(
    dec: Option<&JxlDecoder>,
    name: *mut c_char,
    size: usize,
) -> JxlDecoderStatus {
    let Some(dec) = dec else {
        return JXL_DEC_ERROR;
    };
    dec.get_frame_name(name.cast::<u8>(), size)
}

/// Populates `blend_info` with extra channel blend information for the current frame.
///
/// # Safety
/// `blend_info` must be null or point to a valid writable `JxlBlendInfo`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn JxlDecoderGetExtraChannelBlendInfo(
    dec: Option<&JxlDecoder>,
    index: usize,
    blend_info: *mut JxlBlendInfo,
) -> JxlDecoderStatus {
    let Some(dec) = dec else {
        return JXL_DEC_ERROR;
    };
    if blend_info.is_null()
        || dec.get_extra_channel_info(index).is_none()
        || dec.get_frame_header().is_none()
    {
        return JXL_DEC_ERROR;
    }
    // SAFETY: `blend_info` is non-null and points to writable memory for `JxlBlendInfo`.
    unsafe {
        blend_info.write(JxlBlendInfo {
            blendmode: JXL_BLEND_REPLACE,
            source: 0,
            alpha: 0,
            clamp: JXL_FALSE,
        });
    }
    JXL_DEC_SUCCESS
}

/// Computes the minimum byte size required for the main image output buffer.
///
/// # Safety
/// `format` and `size` must be null or point to valid `JxlPixelFormat` and writable `usize`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn JxlDecoderImageOutBufferSize(
    dec: Option<&JxlDecoder>,
    format: *const JxlPixelFormat,
    size: *mut usize,
) -> JxlDecoderStatus {
    let (Some(dec), Some(format)) = (
        dec,
        // SAFETY: `format` is null or a valid `JxlPixelFormat` pointer.
        unsafe { format.as_ref() },
    ) else {
        return JXL_DEC_ERROR;
    };
    if size.is_null() {
        return JXL_DEC_ERROR;
    }
    let Some(bytes) = dec.image_out_buffer_size(format) else {
        return JXL_DEC_ERROR;
    };
    // SAFETY: `size` is non-null and writable.
    unsafe {
        size.write(bytes);
    }
    JXL_DEC_SUCCESS
}

/// Sets the output buffer for the main image.
///
/// # Safety
/// `format` must be null or point to a valid `JxlPixelFormat`, and `buffer` must point to at least
/// `size` writable bytes that remain valid until `JXL_DEC_FULL_IMAGE` is emitted.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn JxlDecoderSetImageOutBuffer(
    dec: Option<&mut JxlDecoder>,
    format: *const JxlPixelFormat,
    buffer: *mut c_void,
    size: usize,
) -> JxlDecoderStatus {
    let (Some(dec), Some(format)) = (
        dec,
        // SAFETY: `format` is null or a valid `JxlPixelFormat` pointer.
        unsafe { format.as_ref() },
    ) else {
        return JXL_DEC_ERROR;
    };
    dec.set_image_out_buffer(format, buffer, size)
}

/// Sets a row callback for receiving decoded image pixels.
///
/// # Safety
/// `format` must be null or point to a valid `JxlPixelFormat`, and `callback` / `opaque` must
/// satisfy the `JxlImageOutCallback` contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn JxlDecoderSetImageOutCallback(
    dec: Option<&mut JxlDecoder>,
    format: *const JxlPixelFormat,
    callback: JxlImageOutCallback,
    opaque: *mut c_void,
) -> JxlDecoderStatus {
    let (Some(dec), Some(format)) = (
        dec,
        // SAFETY: `format` is null or a valid `JxlPixelFormat` pointer.
        unsafe { format.as_ref() },
    ) else {
        return JXL_DEC_ERROR;
    };
    dec.set_image_out_callback(format, callback, opaque)
}

/// Sets a multithreaded callback for receiving decoded image pixels.
///
/// # Safety
/// `format` must be null or point to a valid `JxlPixelFormat`, and the callbacks must satisfy the
/// `libjxl` multithreaded callback contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn JxlDecoderSetMultithreadedImageOutCallback(
    dec: Option<&mut JxlDecoder>,
    format: *const JxlPixelFormat,
    init_callback: JxlImageOutInitCallback,
    run_callback: JxlImageOutRunCallback,
    destroy_callback: JxlImageOutDestroyCallback,
    init_opaque: *mut c_void,
) -> JxlDecoderStatus {
    let (Some(dec), Some(format)) = (
        dec,
        // SAFETY: `format` is null or a valid `JxlPixelFormat` pointer.
        unsafe { format.as_ref() },
    ) else {
        return JXL_DEC_ERROR;
    };
    dec.set_multithreaded_image_out_callback(
        format,
        init_callback,
        run_callback,
        destroy_callback,
        init_opaque,
    )
}

/// Computes the minimum byte size required for extra channel `index`.
///
/// # Safety
/// `format` and `size` must be null or point to valid `JxlPixelFormat` and writable `usize`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn JxlDecoderExtraChannelBufferSize(
    dec: Option<&JxlDecoder>,
    format: *const JxlPixelFormat,
    size: *mut usize,
    index: u32,
) -> JxlDecoderStatus {
    let (Some(dec), Some(format)) = (
        dec,
        // SAFETY: `format` is null or a valid `JxlPixelFormat` pointer.
        unsafe { format.as_ref() },
    ) else {
        return JXL_DEC_ERROR;
    };
    if size.is_null() {
        return JXL_DEC_ERROR;
    }
    let Some(bytes) = dec.extra_channel_buffer_size(format, index) else {
        return JXL_DEC_ERROR;
    };
    // SAFETY: `size` is non-null and writable.
    unsafe {
        size.write(bytes);
    }
    JXL_DEC_SUCCESS
}

/// Sets the output buffer for extra channel `index`.
///
/// # Safety
/// `format` must be null or point to a valid `JxlPixelFormat`, and `buffer` must point to at least
/// `size` writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn JxlDecoderSetExtraChannelBuffer(
    dec: Option<&mut JxlDecoder>,
    format: *const JxlPixelFormat,
    buffer: *mut c_void,
    size: usize,
    index: u32,
) -> JxlDecoderStatus {
    let (Some(dec), Some(format)) = (
        dec,
        // SAFETY: `format` is null or a valid `JxlPixelFormat` pointer.
        unsafe { format.as_ref() },
    ) else {
        return JXL_DEC_ERROR;
    };
    dec.set_extra_channel_buffer(format, buffer, size, index)
}

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

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderSetBoxBuffer(
    _dec: Option<&mut JxlDecoder>,
    _data: *mut u8,
    _size: usize,
) -> JxlDecoderStatus {
    JXL_DEC_ERROR
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderReleaseBoxBuffer(_dec: Option<&mut JxlDecoder>) -> usize {
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderSetDecompressBoxes(
    dec: Option<&mut JxlDecoder>,
    _decompress: JXL_BOOL,
) -> JxlDecoderStatus {
    let Some(dec) = dec else {
        return JXL_DEC_ERROR;
    };
    dec.set_decompress_boxes()
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderGetBoxType(
    _dec: Option<&mut JxlDecoder>,
    _box_type: *mut c_char,
    _decompressed: JXL_BOOL,
) -> JxlDecoderStatus {
    JXL_DEC_ERROR
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderGetBoxSizeRaw(
    _dec: Option<&JxlDecoder>,
    _size: *mut u64,
) -> JxlDecoderStatus {
    JXL_DEC_ERROR
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderGetBoxSizeContents(
    _dec: Option<&JxlDecoder>,
    _size: *mut u64,
) -> JxlDecoderStatus {
    JXL_DEC_ERROR
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
    dec.flush_image()
}

/// Sets the bit depth interpretation of integer output buffers.
///
/// # Safety
/// `bit_depth` must be null or point to a valid `JxlBitDepth`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn JxlDecoderSetImageOutBitDepth(
    dec: Option<&mut JxlDecoder>,
    bit_depth: *const JxlBitDepth,
) -> JxlDecoderStatus {
    let (Some(dec), Some(bit_depth)) = (
        dec,
        // SAFETY: `bit_depth` is null or a valid `JxlBitDepth` pointer.
        unsafe { bit_depth.as_ref() },
    ) else {
        return JXL_DEC_ERROR;
    };
    dec.set_output_bit_depth(bit_depth)
}

/// Populates `color_encoding` with standard sRGB (or grayscale sRGB).
///
/// # Safety
/// `color_encoding` must be null or point to a valid writable `JxlColorEncoding`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn JxlColorEncodingSetToSRGB(
    color_encoding: *mut JxlColorEncoding,
    is_gray: JXL_BOOL,
) {
    if !color_encoding.is_null() {
        let val = convert_color_encoding(&RsColorEncoding::srgb(is_gray != JXL_FALSE));
        // SAFETY: `color_encoding` is non-null and points to writable memory for `JxlColorEncoding`.
        unsafe {
            color_encoding.write(val);
        }
    }
}

/// Populates `color_encoding` with linear sRGB (or linear grayscale).
///
/// # Safety
/// `color_encoding` must be null or point to a valid writable `JxlColorEncoding`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn JxlColorEncodingSetToLinearSRGB(
    color_encoding: *mut JxlColorEncoding,
    is_gray: JXL_BOOL,
) {
    if !color_encoding.is_null() {
        let val = convert_color_encoding(&RsColorEncoding::linear_srgb(is_gray != JXL_FALSE));
        // SAFETY: `color_encoding` is non-null and points to writable memory for `JxlColorEncoding`.
        unsafe {
            color_encoding.write(val);
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlGainMapGetBundleSize(
    _map_bundle: *const JxlGainMapBundle,
    _bundle_size: *mut usize,
) -> JXL_BOOL {
    JXL_FALSE
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlGainMapWriteBundle(
    _map_bundle: *const JxlGainMapBundle,
    _output_buffer: *mut u8,
    _output_buffer_size: usize,
    _bytes_written: *mut usize,
) -> JXL_BOOL {
    JXL_FALSE
}

/// Deserializes a `JxlGainMapBundle` from an uncompressed `jhgm` box payload.
///
/// # Safety
/// `map_bundle` must be null or point to a writable `JxlGainMapBundle`, `input_buffer` must point
/// to at least `input_buffer_size` readable bytes, and `bytes_read` must be null or point to a
/// writable `usize`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn JxlGainMapReadBundle(
    map_bundle: *mut JxlGainMapBundle,
    input_buffer: *const u8,
    input_buffer_size: usize,
    bytes_read: *mut usize,
) -> JXL_BOOL {
    if map_bundle.is_null() || input_buffer.is_null() || input_buffer_size == 0 {
        return JXL_FALSE;
    }
    // SAFETY: Caller guarantees `input_buffer` is valid for `input_buffer_size` readable bytes.
    let mut data = unsafe { std::slice::from_raw_parts(input_buffer, input_buffer_size) };
    let Some((&jhgm_version, rem)) = data.split_first() else {
        return JXL_FALSE;
    };
    if jhgm_version != 0 {
        return JXL_FALSE;
    }
    data = rem;

    let Some((meta_len, rem)) = data.split_first_chunk::<2>() else {
        return JXL_FALSE;
    };
    let gain_map_metadata_size = u16::from_be_bytes(*meta_len);
    data = rem;
    let Some(gain_map_metadata) = data.split_off(..gain_map_metadata_size as usize) else {
        return JXL_FALSE;
    };

    let Some((&color_len, rem)) = data.split_first() else {
        return JXL_FALSE;
    };
    data = rem;
    let Some(color_bytes) = data.split_off(..color_len as usize) else {
        return JXL_FALSE;
    };
    let (has_color_encoding, color_encoding) = if color_len > 0 {
        let Ok(RsColorProfile::Simple(enc)) = RsColorProfile::from_raw_bytes(color_bytes, &[])
        else {
            return JXL_FALSE;
        };
        (JXL_TRUE, convert_color_encoding(&enc))
    } else {
        (JXL_FALSE, JxlColorEncoding::default())
    };

    let Some((icc_len, rem)) = data.split_first_chunk::<4>() else {
        return JXL_FALSE;
    };
    let alt_icc_size = u32::from_be_bytes(*icc_len);
    data = rem;
    let Some(alt_icc) = data.split_off(..alt_icc_size as usize) else {
        return JXL_FALSE;
    };

    let Ok(gain_map_size) = u32::try_from(data.len()) else {
        return JXL_FALSE;
    };

    // SAFETY: `map_bundle` is non-null and writable; `bytes_read` is checked for null.
    unsafe {
        map_bundle.write(JxlGainMapBundle {
            jhgm_version,
            gain_map_metadata_size,
            gain_map_metadata: gain_map_metadata.as_ptr(),
            has_color_encoding,
            color_encoding,
            alt_icc_size,
            alt_icc: alt_icc.as_ptr(),
            gain_map_size,
            gain_map: data.as_ptr(),
        });
        if !bytes_read.is_null() {
            bytes_read.write(input_buffer_size);
        }
    }
    JXL_TRUE
}
