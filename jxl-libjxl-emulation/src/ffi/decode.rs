// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::ffi::{c_char, c_int, c_void};

use jxl::api::{
    JxlColorEncoding as RsColorEncoding, JxlColorProfile as RsColorProfile,
    JxlSignature as RsSignature, check_signature,
};

use crate::convert::{
    convert_basic_info, convert_color_encoding, convert_extra_channel_info, to_rs_color_encoding,
};
use crate::decoder::output::{ImageOutTarget, compute_buffer_size};
use crate::decoder::{ALL_INFORMATIVE_EVENTS_MASK, DecodeSession, DecoderSettings, JxlDecoder};
use crate::parallel_runner::CParallelRunner;
use crate::types::{
    JXL_BLEND_REPLACE, JXL_BOOL, JXL_DEC_ERROR, JXL_DEC_NEED_MORE_INPUT, JXL_DEC_SUCCESS,
    JXL_FALSE, JXL_SIG_CODESTREAM, JXL_SIG_CONTAINER, JXL_SIG_INVALID, JXL_SIG_NOT_ENOUGH_BYTES,
    JXL_TRUE, JxlBasicInfo, JxlBitDepth, JxlBlendInfo, JxlBoxType, JxlCmsInterface,
    JxlColorEncoding, JxlColorProfileTarget, JxlDecoderStatus, JxlExtraChannelInfo, JxlFrameHeader,
    JxlImageOutCallback, JxlLayerInfo, JxlMemoryManager, JxlParallelRunner, JxlPixelFormat,
    JxlSignature,
};

fn set_before_start(
    dec: &mut JxlDecoder,
    f: impl FnOnce(&mut DecoderSettings),
) -> JxlDecoderStatus {
    if dec.session.process_started {
        return JXL_DEC_ERROR;
    }
    f(&mut dec.settings);
    JXL_DEC_SUCCESS
}

/// Returns the decoder library version in `MAJOR * 1_000_000 + MINOR * 1_000 + PATCH` format.
#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderVersion() -> u32 {
    11_001
}

/// Checks if the input buffer starts with a valid JPEG XL codestream or container signature.
///
/// # Safety
/// If `len > 0`, `buf` must point to at least `len` readable bytes.
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
#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderCreate(memory_manager: Option<&JxlMemoryManager>) -> *mut JxlDecoder {
    if let Some(mm) = memory_manager
        && (mm.alloc.is_none() || mm.free.is_none())
    {
        return std::ptr::null_mut();
    }
    Box::into_raw(Box::new(JxlDecoder::new()))
}

/// Resets a `JxlDecoder` to its initial newly-created state.
#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderReset(dec: &mut JxlDecoder) {
    *dec = JxlDecoder::new();
}

/// Destroys a `JxlDecoder` instance.
///
/// # Safety
/// `dec` must be null or a valid pointer returned by `JxlDecoderCreate`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn JxlDecoderDestroy(dec: *mut JxlDecoder) {
    if !dec.is_null() {
        // SAFETY: `dec` was allocated by `Box::into_raw` in `JxlDecoderCreate`.
        unsafe {
            drop(Box::from_raw(dec));
        }
    }
}

/// Rewinds a `JxlDecoder` to the beginning of the stream while keeping settings and subscriptions.
#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderRewind(dec: &mut JxlDecoder) {
    dec.session = DecodeSession::default();
}

/// Skips `amount` visible frames.
#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderSkipFrames(dec: &mut JxlDecoder, amount: usize) {
    dec.session.frames_to_skip = dec.session.frames_to_skip.saturating_add(amount);
}

/// Skips decoding the current frame.
#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderSkipCurrentFrame(dec: &mut JxlDecoder) -> JxlDecoderStatus {
    if dec.session.can_skip_current_frame {
        dec.session.can_skip_current_frame = false;
        dec.session.skip_current_frame = true;
        JXL_DEC_SUCCESS
    } else {
        JXL_DEC_ERROR
    }
}

/// Sets the parallel runner used by the decoder.
///
/// # Safety
/// `parallel_runner` and `parallel_runner_opaque` must obey the `JxlParallelRunner` C-ABI contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn JxlDecoderSetParallelRunner(
    dec: &mut JxlDecoder,
    parallel_runner: JxlParallelRunner,
    parallel_runner_opaque: *mut c_void,
) -> JxlDecoderStatus {
    set_before_start(dec, |s| {
        s.parallel_runner = parallel_runner.map(|runner| CParallelRunner {
            runner,
            runner_opaque: parallel_runner_opaque,
        });
    })
}

/// Returns a size hint for how many more bytes are needed to reach basic info.
#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderSizeHintBasicInfo(dec: &JxlDecoder) -> usize {
    if dec.has_basic_info() { 0 } else { 64 }
}

/// Subscribes the decoder to status events.
#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderSubscribeEvents(
    dec: &mut JxlDecoder,
    events_wanted: c_int,
) -> JxlDecoderStatus {
    if (events_wanted & 63) != 0 || (events_wanted & !ALL_INFORMATIVE_EVENTS_MASK) != 0 {
        return JXL_DEC_ERROR;
    }
    set_before_start(dec, |s| s.events_wanted = events_wanted)
}

/// Configures whether the decoder skips applying the codestream's Exif orientation.
#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderSetKeepOrientation(
    dec: &mut JxlDecoder,
    skip_reorientation: JXL_BOOL,
) -> JxlDecoderStatus {
    set_before_start(dec, |s| {
        s.keep_orientation = skip_reorientation != JXL_FALSE
    })
}

/// Configures whether associated alpha channels are unpremultiplied before output.
#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderSetUnpremultiplyAlpha(
    dec: &mut JxlDecoder,
    unpremul_alpha: JXL_BOOL,
) -> JxlDecoderStatus {
    set_before_start(dec, |s| s.unpremultiply_alpha = unpremul_alpha != JXL_FALSE)
}

/// Configures whether spot color channels are rendered onto color channels.
#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderSetRenderSpotcolors(
    dec: &mut JxlDecoder,
    render_spotcolors: JXL_BOOL,
) -> JxlDecoderStatus {
    set_before_start(dec, |s| {
        s.render_spotcolors = render_spotcolors != JXL_FALSE
    })
}

/// Configures whether frames are coalesced.
#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderSetCoalescing(
    dec: &mut JxlDecoder,
    coalescing: JXL_BOOL,
) -> JxlDecoderStatus {
    set_before_start(dec, |s| s.coalescing = coalescing != JXL_FALSE)
}

/// Advances the decoding state machine.
#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderProcessInput(dec: &mut JxlDecoder) -> JxlDecoderStatus {
    dec.process_input()
}

/// Sets the input byte buffer for the decoder.
///
/// # Safety
/// If `size > 0`, `data` must point to at least `size` readable bytes that remain valid until
/// `JxlDecoderReleaseInput` is called.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn JxlDecoderSetInput(
    dec: &mut JxlDecoder,
    data: *const u8,
    size: usize,
) -> JxlDecoderStatus {
    // SAFETY: Caller guarantees `data` is valid for `size` readable bytes until `JxlDecoderReleaseInput`.
    if unsafe { dec.session.input.set_input(data, size) } {
        JXL_DEC_SUCCESS
    } else {
        JXL_DEC_ERROR
    }
}

/// Releases the input buffer set by `JxlDecoderSetInput` and returns the number of unconsumed bytes.
#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderReleaseInput(dec: &mut JxlDecoder) -> usize {
    dec.session.input.release_input()
}

/// Marks the input stream as complete (no further bytes will be provided).
#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderCloseInput(dec: &mut JxlDecoder) {
    dec.session.input.close();
    if let Some(rs_dec) = dec.session.rs_decoder.as_mut() {
        rs_dec.close_input();
    }
}

/// Populates `info` with basic image metadata.
#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderGetBasicInfo(
    dec: &JxlDecoder,
    info: Option<&mut JxlBasicInfo>,
) -> JxlDecoderStatus {
    let Some(rs_info) = dec.rs_basic_info() else {
        return JXL_DEC_NEED_MORE_INPUT;
    };
    if let Some(info) = info {
        *info = convert_basic_info(
            rs_info,
            dec.session.input.have_container().unwrap_or(false),
            dec.session.is_gray,
            dec.settings.keep_orientation,
            dec.settings.desired_intensity_target,
        );
    }
    JXL_DEC_SUCCESS
}

/// Populates `info` with metadata for extra channel `index`.
#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderGetExtraChannelInfo(
    dec: &JxlDecoder,
    index: usize,
    info: &mut JxlExtraChannelInfo,
) -> JxlDecoderStatus {
    let Some(rs_info) = dec.rs_basic_info() else {
        return JXL_DEC_NEED_MORE_INPUT;
    };
    let Some(ec) = rs_info.extra_channels.get(index) else {
        return JXL_DEC_ERROR;
    };
    *info = convert_extra_channel_info(ec);
    JXL_DEC_SUCCESS
}

/// Writes the null-terminated name of extra channel `index` into `name`.
///
/// # Safety
/// `name` must point to at least `size` writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn JxlDecoderGetExtraChannelName(
    dec: &JxlDecoder,
    index: usize,
    name: *mut c_char,
    size: usize,
) -> JxlDecoderStatus {
    let Some(ec) = dec
        .rs_basic_info()
        .and_then(|info| info.extra_channels.get(index))
    else {
        return JXL_DEC_ERROR;
    };
    let bytes: Vec<u8> = ec.name.chars().map(|c| c as u8).collect();
    // SAFETY: `name` must be valid for `size` writable bytes per C-ABI contract.
    unsafe { write_c_string(&bytes, name, size) }
}

/// Retrieves the color profile as a `JxlColorEncoding` if representable as a parameterized profile.
#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderGetColorAsEncodedProfile(
    dec: &JxlDecoder,
    target: JxlColorProfileTarget,
    color_encoding: Option<&mut JxlColorEncoding>,
) -> JxlDecoderStatus {
    if !dec.has_basic_info() {
        return JXL_DEC_NEED_MORE_INPUT;
    }
    let Some(RsColorProfile::Simple(enc)) = dec.select_color_profile(target) else {
        return JXL_DEC_ERROR;
    };
    if let Some(out) = color_encoding {
        *out = convert_color_encoding(enc);
    }
    JXL_DEC_SUCCESS
}

/// Retrieves the byte size of the ICC profile for `target`.
#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderGetICCProfileSize(
    dec: &JxlDecoder,
    target: JxlColorProfileTarget,
    size: Option<&mut usize>,
) -> JxlDecoderStatus {
    if !dec.has_basic_info() {
        return JXL_DEC_NEED_MORE_INPUT;
    }
    let Some(icc) = dec
        .select_color_profile(target)
        .and_then(|p| p.try_as_icc())
    else {
        return JXL_DEC_ERROR;
    };
    if let Some(out) = size {
        *out = icc.len();
    }
    JXL_DEC_SUCCESS
}

/// Copies the ICC profile for `target` into `icc_profile`.
///
/// # Safety
/// `icc_profile` must point to at least `size` writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn JxlDecoderGetColorAsICCProfile(
    dec: &JxlDecoder,
    target: JxlColorProfileTarget,
    icc_profile: *mut u8,
    size: usize,
) -> JxlDecoderStatus {
    if !dec.has_basic_info() {
        return JXL_DEC_NEED_MORE_INPUT;
    }
    let Some(icc) = dec
        .select_color_profile(target)
        .and_then(|p| p.try_as_icc())
    else {
        return JXL_DEC_ERROR;
    };
    if icc_profile.is_null() || size < icc.len() {
        return JXL_DEC_ERROR;
    }
    // SAFETY: `icc_profile` is non-null and valid for at least `size >= icc.len()` bytes.
    unsafe {
        std::ptr::copy_nonoverlapping(icc.as_ptr(), icc_profile, icc.len());
    }
    JXL_DEC_SUCCESS
}

/// Sets the preferred output color profile.
#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderSetPreferredColorProfile(
    dec: &mut JxlDecoder,
    color_encoding: &JxlColorEncoding,
) -> JxlDecoderStatus {
    if dec.session.mid_frame_decode {
        return JXL_DEC_ERROR;
    }
    let Some(rs_enc) = to_rs_color_encoding(color_encoding) else {
        return JXL_DEC_ERROR;
    };
    let Some(rs_dec) = dec.session.rs_decoder.as_mut() else {
        return JXL_DEC_ERROR;
    };
    if rs_dec
        .set_output_color_profile(RsColorProfile::Simple(rs_enc))
        .is_ok()
    {
        JXL_DEC_SUCCESS
    } else {
        JXL_DEC_ERROR
    }
}

/// Sets the desired intensity target in nits for tone mapping.
#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderSetDesiredIntensityTarget(
    dec: &mut JxlDecoder,
    desired_intensity_target: f32,
) -> JxlDecoderStatus {
    if desired_intensity_target < 0.0 || !desired_intensity_target.is_finite() {
        return JXL_DEC_ERROR;
    }
    set_before_start(dec, |s| {
        s.desired_intensity_target = desired_intensity_target
    })
}

/// Sets an explicit output color profile (or ICC profile).
#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderSetOutputColorProfile(
    dec: &mut JxlDecoder,
    color_encoding: Option<&JxlColorEncoding>,
    icc_data: *const u8,
    icc_size: usize,
) -> JxlDecoderStatus {
    if (!icc_data.is_null() && icc_size > 0) || color_encoding.is_none() {
        return JXL_DEC_ERROR;
    }
    JxlDecoderSetPreferredColorProfile(dec, color_encoding.unwrap())
}

/// Sets the color management system (CMS) interface.
#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderSetCms(dec: &mut JxlDecoder, cms: JxlCmsInterface) -> JxlDecoderStatus {
    dec.settings.cms = Some(cms);
    JXL_DEC_SUCCESS
}

/// Populates `header` with information about the current frame.
#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderGetFrameHeader(
    dec: &JxlDecoder,
    header: &mut JxlFrameHeader,
) -> JxlDecoderStatus {
    let Some(rs_dec) = dec.session.rs_decoder.as_ref() else {
        return JXL_DEC_ERROR;
    };
    let Some(fh) = rs_dec.frame_header() else {
        return JXL_DEC_ERROR;
    };
    let scanned = rs_dec.scanned_frames().last();
    let duration_ticks = scanned.map(|f| f.duration_ticks).unwrap_or(0);
    let is_last = scanned
        .map(|f| f.is_last)
        .unwrap_or(!dec.session.has_more_frames);
    *header = JxlFrameHeader {
        duration: duration_ticks,
        timecode: 0,
        name_length: fh.name.chars().count() as u32,
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
    JXL_DEC_SUCCESS
}

/// Writes the null-terminated name of the current frame into `name`.
///
/// # Safety
/// `name` must point to at least `size` writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn JxlDecoderGetFrameName(
    dec: &JxlDecoder,
    name: *mut c_char,
    size: usize,
) -> JxlDecoderStatus {
    let Some(fh) = dec
        .session
        .rs_decoder
        .as_ref()
        .and_then(|d| d.frame_header())
    else {
        return JXL_DEC_ERROR;
    };
    let bytes: Vec<u8> = fh.name.chars().map(|c| c as u8).collect();
    // SAFETY: `name` must be valid for `size` writable bytes per C-ABI contract.
    unsafe { write_c_string(&bytes, name, size) }
}

/// Populates `blend_info` with extra channel blend information for the current frame.
#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderGetExtraChannelBlendInfo(
    dec: &JxlDecoder,
    index: usize,
    blend_info: &mut JxlBlendInfo,
) -> JxlDecoderStatus {
    let Some(rs_dec) = dec.session.rs_decoder.as_ref() else {
        return JXL_DEC_ERROR;
    };
    let Some(rs_info) = rs_dec.basic_info() else {
        return JXL_DEC_ERROR;
    };
    if index >= rs_info.extra_channels.len() || rs_dec.frame_header().is_none() {
        return JXL_DEC_ERROR;
    }
    *blend_info = JxlBlendInfo {
        blendmode: JXL_BLEND_REPLACE,
        source: 0,
        alpha: 0,
        clamp: JXL_FALSE,
    };
    JXL_DEC_SUCCESS
}

/// Computes the minimum byte size required for the main image output buffer.
#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderImageOutBufferSize(
    dec: &JxlDecoder,
    format: &JxlPixelFormat,
    size: &mut usize,
) -> JxlDecoderStatus {
    let Some(rs_info) = dec.rs_basic_info() else {
        return JXL_DEC_ERROR;
    };
    if !dec.session.is_gray && format.num_channels < 3 {
        return JXL_DEC_ERROR;
    }
    let Some(bytes) = compute_buffer_size(rs_info.size.0, rs_info.size.1, format) else {
        return JXL_DEC_ERROR;
    };
    *size = bytes;
    JXL_DEC_SUCCESS
}

/// Sets the output buffer for the main image.
///
/// # Safety
/// `buffer` must point to at least `size` writable bytes that remain valid until
/// `JXL_DEC_FULL_IMAGE` is emitted.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn JxlDecoderSetImageOutBuffer(
    dec: &mut JxlDecoder,
    format: &JxlPixelFormat,
    buffer: *mut c_void,
    size: usize,
) -> JxlDecoderStatus {
    if buffer.is_null() || dec.session.mid_frame_decode {
        return JXL_DEC_ERROR;
    }
    let mut min_size = 0;
    if JxlDecoderImageOutBufferSize(dec, format, &mut min_size) != JXL_DEC_SUCCESS
        || size < min_size
    {
        return JXL_DEC_ERROR;
    }
    dec.session.output.image_out_target = Some(ImageOutTarget::Buffer {
        ptr: buffer.cast::<u8>(),
        size,
        format: *format,
    });
    let _ = dec.refresh_pixel_format();
    JXL_DEC_SUCCESS
}

/// Sets a row callback for receiving decoded image pixels.
///
/// # Safety
/// `callback` and `opaque` must satisfy the `JxlImageOutCallback` C-ABI contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn JxlDecoderSetImageOutCallback(
    dec: &mut JxlDecoder,
    format: &JxlPixelFormat,
    callback: JxlImageOutCallback,
    opaque: *mut c_void,
) -> JxlDecoderStatus {
    if dec.session.mid_frame_decode
        || (dec.has_basic_info() && !dec.session.is_gray && format.num_channels < 3)
        || compute_buffer_size(1, 1, format).is_none()
    {
        return JXL_DEC_ERROR;
    }
    let Some(target) = ImageOutTarget::new_callback(*format, callback, opaque) else {
        return JXL_DEC_ERROR;
    };
    dec.session.output.image_out_target = Some(target);
    let _ = dec.refresh_pixel_format();
    JXL_DEC_SUCCESS
}

/// Computes the minimum byte size required for extra channel `index`.
#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderExtraChannelBufferSize(
    dec: &JxlDecoder,
    format: &JxlPixelFormat,
    size: &mut usize,
    index: u32,
) -> JxlDecoderStatus {
    let Some(rs_info) = dec.rs_basic_info() else {
        return JXL_DEC_ERROR;
    };
    if (index as usize) >= rs_info.extra_channels.len() {
        return JXL_DEC_ERROR;
    }
    let mut ec_format = *format;
    ec_format.num_channels = 1;
    let Some(bytes) = compute_buffer_size(rs_info.size.0, rs_info.size.1, &ec_format) else {
        return JXL_DEC_ERROR;
    };
    *size = bytes;
    JXL_DEC_SUCCESS
}

/// Sets the output buffer for extra channel `index`.
///
/// # Safety
/// `buffer` must point to at least `size` writable bytes that remain valid until
/// `JXL_DEC_FULL_IMAGE` is emitted.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn JxlDecoderSetExtraChannelBuffer(
    dec: &mut JxlDecoder,
    format: &JxlPixelFormat,
    buffer: *mut c_void,
    size: usize,
    index: u32,
) -> JxlDecoderStatus {
    if buffer.is_null() || dec.session.mid_frame_decode {
        return JXL_DEC_ERROR;
    }
    let mut min_size = 0;
    if JxlDecoderExtraChannelBufferSize(dec, format, &mut min_size, index) != JXL_DEC_SUCCESS
        || size < min_size
    {
        return JXL_DEC_ERROR;
    }
    let idx = index as usize;
    if idx >= dec.session.output.extra_channel_targets.len() {
        return JXL_DEC_ERROR;
    }
    let mut ec_format = *format;
    ec_format.num_channels = 1;
    dec.session.output.extra_channel_targets[idx] = Some(ImageOutTarget::Buffer {
        ptr: buffer.cast::<u8>(),
        size,
        format: ec_format,
    });
    let _ = dec.refresh_pixel_format();
    JXL_DEC_SUCCESS
}

/// Sets the output buffer for the current container box.
///
/// # Safety
/// `data` must point to at least `size` writable bytes that remain valid until
/// `JxlDecoderReleaseBoxBuffer` is called.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn JxlDecoderSetBoxBuffer(
    dec: &mut JxlDecoder,
    data: *mut u8,
    size: usize,
) -> JxlDecoderStatus {
    // SAFETY: Caller guarantees `data` has at least `size` writable bytes until `JxlDecoderReleaseBoxBuffer`.
    if unsafe { dec.session.boxes.set_buffer(data, size) } {
        JXL_DEC_SUCCESS
    } else {
        JXL_DEC_ERROR
    }
}

/// Releases the current box buffer and returns the number of unused bytes in it.
#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderReleaseBoxBuffer(dec: &mut JxlDecoder) -> usize {
    dec.session.boxes.release_buffer()
}

/// Configures whether `brob` (Brotli-compressed) boxes are decompressed automatically.
#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderSetDecompressBoxes(
    dec: &mut JxlDecoder,
    decompress: JXL_BOOL,
) -> JxlDecoderStatus {
    set_before_start(dec, |s| s.decompress_boxes = decompress != JXL_FALSE)
}

/// Retrieves the 4-character box type of the current box.
#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderGetBoxType(
    dec: &mut JxlDecoder,
    box_type: &mut JxlBoxType,
    decompressed: JXL_BOOL,
) -> JxlDecoderStatus {
    let Some(ty) = dec
        .session
        .boxes
        .box_type(dec.settings.decompress_boxes, decompressed != JXL_FALSE)
    else {
        return JXL_DEC_ERROR;
    };
    *box_type = ty;
    JXL_DEC_SUCCESS
}

/// Retrieves the raw byte size of the current box (including header), or `0` if until EOF.
#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderGetBoxSizeRaw(dec: &JxlDecoder, size: &mut u64) -> JxlDecoderStatus {
    let Some(s) = dec.session.boxes.raw_size() else {
        return JXL_DEC_ERROR;
    };
    *size = s;
    JXL_DEC_SUCCESS
}

/// Retrieves the raw content size of the current box (excluding header).
#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderGetBoxSizeContents(
    dec: &JxlDecoder,
    size: &mut u64,
) -> JxlDecoderStatus {
    let Some(s) = dec.session.boxes.contents_size() else {
        return JXL_DEC_ERROR;
    };
    *size = s;
    JXL_DEC_SUCCESS
}

/// Sets the bit depth interpretation of integer output buffers.
#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderSetImageOutBitDepth(
    dec: &mut JxlDecoder,
    bit_depth: &JxlBitDepth,
) -> JxlDecoderStatus {
    if dec.session.mid_frame_decode {
        return JXL_DEC_ERROR;
    }
    dec.session.output.output_bit_depth = *bit_depth;
    let _ = dec.refresh_pixel_format();
    JXL_DEC_SUCCESS
}

/// Populates `color_encoding` with standard sRGB (or grayscale sRGB).
#[unsafe(no_mangle)]
pub extern "C" fn JxlColorEncodingSetToSRGB(
    color_encoding: &mut JxlColorEncoding,
    is_gray: JXL_BOOL,
) {
    *color_encoding = convert_color_encoding(&RsColorEncoding::srgb(is_gray != JXL_FALSE));
}

/// Populates `color_encoding` with linear sRGB (or linear grayscale).
#[unsafe(no_mangle)]
pub extern "C" fn JxlColorEncodingSetToLinearSRGB(
    color_encoding: &mut JxlColorEncoding,
    is_gray: JXL_BOOL,
) {
    *color_encoding = convert_color_encoding(&RsColorEncoding::linear_srgb(is_gray != JXL_FALSE));
}

unsafe fn write_c_string(bytes: &[u8], out: *mut c_char, size: usize) -> JxlDecoderStatus {
    if out.is_null() || size <= bytes.len() {
        return JXL_DEC_ERROR;
    }
    // SAFETY: `out` is non-null and valid for at least `size > bytes.len()` writable bytes.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.cast::<u8>(), bytes.len());
        *out.add(bytes.len()) = 0;
    }
    JXL_DEC_SUCCESS
}
