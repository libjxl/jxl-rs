// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::ffi::c_void;

use crate::decoder::JxlDecoder;
use crate::types::{
    JXL_BOOL, JXL_DEC_ERROR, JXL_FALSE, JxlDecoderStatus, JxlGainMapBundle,
    JxlImageOutDestroyCallback, JxlImageOutInitCallback, JxlImageOutRunCallback, JxlPixelFormat,
    JxlProgressiveDetail,
};

/// Preview frame decoding is unimplemented; returns `JXL_DEC_ERROR`.
#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderPreviewOutBufferSize(
    _dec: &JxlDecoder,
    _format: &JxlPixelFormat,
    _size: &mut usize,
) -> JxlDecoderStatus {
    JXL_DEC_ERROR
}

/// Preview frame decoding is unimplemented; returns `JXL_DEC_ERROR`.
#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderSetPreviewOutBuffer(
    _dec: &mut JxlDecoder,
    _format: &JxlPixelFormat,
    _buffer: *mut c_void,
    _size: usize,
) -> JxlDecoderStatus {
    JXL_DEC_ERROR
}

/// Multithreaded image out callback is unimplemented; returns `JXL_DEC_ERROR`.
#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderSetMultithreadedImageOutCallback(
    _dec: &mut JxlDecoder,
    _format: &JxlPixelFormat,
    _init_callback: JxlImageOutInitCallback,
    _run_callback: JxlImageOutRunCallback,
    _destroy_callback: JxlImageOutDestroyCallback,
    _init_opaque: *mut c_void,
) -> JxlDecoderStatus {
    JXL_DEC_ERROR
}

/// JPEG reconstruction is unimplemented; returns `JXL_DEC_ERROR`.
#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderSetJPEGBuffer(
    _dec: &mut JxlDecoder,
    _data: *mut u8,
    _size: usize,
) -> JxlDecoderStatus {
    JXL_DEC_ERROR
}

/// JPEG reconstruction is unimplemented; returns `0`.
#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderReleaseJPEGBuffer(_dec: &mut JxlDecoder) -> usize {
    0
}

/// Progressive decoding is unimplemented; returns `JXL_DEC_ERROR`.
#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderSetProgressiveDetail(
    _dec: &mut JxlDecoder,
    _detail: JxlProgressiveDetail,
) -> JxlDecoderStatus {
    JXL_DEC_ERROR
}

/// Returns `1` (progressive downsampling is unimplemented).
#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderGetIntendedDownsamplingRatio(_dec: &JxlDecoder) -> usize {
    1
}

/// Progressive flushing is unimplemented; returns `JXL_DEC_ERROR`.
#[unsafe(no_mangle)]
pub extern "C" fn JxlDecoderFlushImage(_dec: &mut JxlDecoder) -> JxlDecoderStatus {
    JXL_DEC_ERROR
}

/// Gain map bundle serialization is unimplemented; returns `JXL_FALSE`.
#[unsafe(no_mangle)]
pub extern "C" fn JxlGainMapGetBundleSize(
    _map_bundle: *const JxlGainMapBundle,
    _bundle_size: *mut usize,
) -> JXL_BOOL {
    JXL_FALSE
}

/// Gain map bundle serialization is unimplemented; returns `JXL_FALSE`.
#[unsafe(no_mangle)]
pub extern "C" fn JxlGainMapWriteBundle(
    _map_bundle: *const JxlGainMapBundle,
    _output_buffer: *mut u8,
    _output_buffer_size: usize,
    _bytes_written: *mut usize,
) -> JXL_BOOL {
    JXL_FALSE
}

/// Gain map bundle deserialization is unimplemented; returns `JXL_FALSE`.
#[unsafe(no_mangle)]
pub extern "C" fn JxlGainMapReadBundle(
    _map_bundle: *mut JxlGainMapBundle,
    _input_buffer: *const u8,
    _input_buffer_size: usize,
    _bytes_read: *mut usize,
) -> JXL_BOOL {
    JXL_FALSE
}
