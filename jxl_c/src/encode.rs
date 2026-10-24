// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::ffi::{c_char, c_int, c_void};

use crate::types::{
    JPEGXL_NUMERIC_VERSION, JXL_BLEND_REPLACE, JXL_BOOL, JXL_ENC_ERR_NOT_SUPPORTED, JXL_ENC_ERROR,
    JXL_FALSE, JXL_ORIENT_IDENTITY, JXL_TRUE, JxlAnimationHeader, JxlBasicInfo, JxlBitDepth,
    JxlBlendInfo, JxlBoxType, JxlChunkedFrameInputSource, JxlCmsInterface, JxlColorEncoding,
    JxlDebugImageCallback, JxlEncoder, JxlEncoderError, JxlEncoderFrameSettingId,
    JxlEncoderFrameSettings, JxlEncoderOutputProcessor, JxlEncoderStats, JxlEncoderStatsKey,
    JxlEncoderStatus, JxlExtraChannelInfo, JxlExtraChannelType, JxlFrameHeader, JxlLayerInfo,
    JxlMemoryManager, JxlParallelRunner, JxlPixelFormat,
};

#[unsafe(no_mangle)]
pub extern "C" fn JxlEncoderVersion() -> u32 {
    JPEGXL_NUMERIC_VERSION
}

/// Initializes a `JxlBasicInfo` struct with default values matching `libjxl`.
///
/// # Safety
/// `info` must be null or point to a valid writable `JxlBasicInfo`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn JxlEncoderInitBasicInfo(info: *mut JxlBasicInfo) {
    if !info.is_null() {
        // SAFETY: `info` is non-null and points to writable memory for `JxlBasicInfo`.
        unsafe {
            info.write(JxlBasicInfo {
                bits_per_sample: 8,
                orientation: JXL_ORIENT_IDENTITY,
                num_color_channels: 3,
                animation: JxlAnimationHeader {
                    tps_numerator: 10,
                    tps_denominator: 1,
                    ..Default::default()
                },
                ..Default::default()
            });
        }
    }
}

/// Initializes a `JxlFrameHeader` struct with default values matching `libjxl`.
///
/// # Safety
/// `frame_header` must be null or point to a valid writable `JxlFrameHeader`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn JxlEncoderInitFrameHeader(frame_header: *mut JxlFrameHeader) {
    if !frame_header.is_null() {
        // SAFETY: `frame_header` is non-null and points to writable memory for `JxlFrameHeader`.
        unsafe {
            frame_header.write(JxlFrameHeader {
                is_last: JXL_TRUE,
                layer_info: JxlLayerInfo {
                    blend_info: JxlBlendInfo {
                        blendmode: JXL_BLEND_REPLACE,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                ..Default::default()
            });
        }
    }
}

/// Initializes a `JxlBlendInfo` struct with default values matching `libjxl`.
///
/// # Safety
/// `blend_info` must be null or point to a valid writable `JxlBlendInfo`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn JxlEncoderInitBlendInfo(blend_info: *mut JxlBlendInfo) {
    if !blend_info.is_null() {
        // SAFETY: `blend_info` is non-null and points to writable memory for `JxlBlendInfo`.
        unsafe {
            blend_info.write(JxlBlendInfo {
                blendmode: JXL_BLEND_REPLACE,
                ..Default::default()
            });
        }
    }
}

/// Initializes a `JxlExtraChannelInfo` struct with default values matching `libjxl`.
///
/// # Safety
/// `info` must be null or point to a valid writable `JxlExtraChannelInfo`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn JxlEncoderInitExtraChannelInfo(
    type_: JxlExtraChannelType,
    info: *mut JxlExtraChannelInfo,
) {
    if !info.is_null() {
        // SAFETY: `info` is non-null and points to writable memory for `JxlExtraChannelInfo`.
        unsafe {
            info.write(JxlExtraChannelInfo {
                type_,
                bits_per_sample: 8,
                alpha_premultiplied: JXL_FALSE,
                ..Default::default()
            });
        }
    }
}

macro_rules! enc_stubs {
    ($(fn $name:ident($($arg:ident: $ty:ty),* $(,)?) $(-> $ret:ty)? = $val:expr;)*) => {
        $(
            #[unsafe(no_mangle)]
            pub extern "C" fn $name($($arg: $ty),*) $(-> $ret)? {
                $val
            }
        )*
    };
}

enc_stubs! {
    fn JxlEncoderCreate(_mm: *const JxlMemoryManager) -> *mut JxlEncoder = std::ptr::null_mut();
    fn JxlEncoderReset(_enc: Option<&mut JxlEncoder>) = ();
    fn JxlEncoderDestroy(_enc: Option<&mut JxlEncoder>) = ();
    fn JxlEncoderGetError(_enc: Option<&mut JxlEncoder>) -> JxlEncoderError = JXL_ENC_ERR_NOT_SUPPORTED;
    fn JxlEncoderSetCms(_enc: Option<&mut JxlEncoder>, _cms: JxlCmsInterface) = ();
    fn JxlEncoderSetParallelRunner(_enc: Option<&mut JxlEncoder>, _runner: JxlParallelRunner, _opaque: *mut c_void) -> JxlEncoderStatus = JXL_ENC_ERROR;
    fn JxlEncoderProcessOutput(_enc: Option<&mut JxlEncoder>, _next_out: *mut *mut u8, _avail_out: *mut usize) -> JxlEncoderStatus = JXL_ENC_ERROR;
    fn JxlEncoderSetFrameHeader(_fs: Option<&mut JxlEncoderFrameSettings>, _fh: *const JxlFrameHeader) -> JxlEncoderStatus = JXL_ENC_ERROR;
    fn JxlEncoderSetExtraChannelBlendInfo(_fs: Option<&mut JxlEncoderFrameSettings>, _index: usize, _bi: *const JxlBlendInfo) -> JxlEncoderStatus = JXL_ENC_ERROR;
    fn JxlEncoderSetFrameName(_fs: Option<&mut JxlEncoderFrameSettings>, _name: *const c_char) -> JxlEncoderStatus = JXL_ENC_ERROR;
    fn JxlEncoderSetFrameBitDepth(_fs: Option<&mut JxlEncoderFrameSettings>, _bd: *const JxlBitDepth) -> JxlEncoderStatus = JXL_ENC_ERROR;
    fn JxlEncoderAddJPEGFrame(_fs: Option<&JxlEncoderFrameSettings>, _buf: *const u8, _size: usize) -> JxlEncoderStatus = JXL_ENC_ERROR;
    fn JxlEncoderAddImageFrame(_fs: Option<&JxlEncoderFrameSettings>, _pf: *const JxlPixelFormat, _buf: *const c_void, _size: usize) -> JxlEncoderStatus = JXL_ENC_ERROR;
    fn JxlEncoderSetOutputProcessor(_enc: Option<&mut JxlEncoder>, _op: JxlEncoderOutputProcessor) -> JxlEncoderStatus = JXL_ENC_ERROR;
    fn JxlEncoderFlushInput(_enc: Option<&mut JxlEncoder>) -> JxlEncoderStatus = JXL_ENC_ERROR;
    fn JxlEncoderAddChunkedFrame(_fs: Option<&JxlEncoderFrameSettings>, _last: JXL_BOOL, _src: JxlChunkedFrameInputSource) -> JxlEncoderStatus = JXL_ENC_ERROR;
    fn JxlEncoderSetExtraChannelBuffer(_fs: Option<&JxlEncoderFrameSettings>, _pf: *const JxlPixelFormat, _buf: *const c_void, _size: usize, _idx: u32) -> JxlEncoderStatus = JXL_ENC_ERROR;
    fn JxlEncoderAddBox(_enc: Option<&mut JxlEncoder>, _ty: *const JxlBoxType, _contents: *const u8, _size: usize, _compress: JXL_BOOL) -> JxlEncoderStatus = JXL_ENC_ERROR;
    fn JxlEncoderUseBoxes(_enc: Option<&mut JxlEncoder>) -> JxlEncoderStatus = JXL_ENC_ERROR;
    fn JxlEncoderCloseBoxes(_enc: Option<&mut JxlEncoder>) = ();
    fn JxlEncoderCloseFrames(_enc: Option<&mut JxlEncoder>) = ();
    fn JxlEncoderCloseInput(_enc: Option<&mut JxlEncoder>) = ();
    fn JxlEncoderSetColorEncoding(_enc: Option<&mut JxlEncoder>, _color: *const JxlColorEncoding) -> JxlEncoderStatus = JXL_ENC_ERROR;
    fn JxlEncoderSetICCProfile(_enc: Option<&mut JxlEncoder>, _icc: *const u8, _size: usize) -> JxlEncoderStatus = JXL_ENC_ERROR;
    fn JxlEncoderSetBasicInfo(_enc: Option<&mut JxlEncoder>, _info: *const JxlBasicInfo) -> JxlEncoderStatus = JXL_ENC_ERROR;
    fn JxlEncoderSetUpsamplingMode(_enc: Option<&mut JxlEncoder>, _factor: i64, _mode: i64) -> JxlEncoderStatus = JXL_ENC_ERROR;
    fn JxlEncoderSetExtraChannelInfo(_enc: Option<&mut JxlEncoder>, _index: usize, _info: *const JxlExtraChannelInfo) -> JxlEncoderStatus = JXL_ENC_ERROR;
    fn JxlEncoderSetExtraChannelName(_enc: Option<&mut JxlEncoder>, _index: usize, _name: *const c_char, _size: usize) -> JxlEncoderStatus = JXL_ENC_ERROR;
    fn JxlEncoderFrameSettingsSetOption(_fs: Option<&mut JxlEncoderFrameSettings>, _opt: JxlEncoderFrameSettingId, _val: i64) -> JxlEncoderStatus = JXL_ENC_ERROR;
    fn JxlEncoderFrameSettingsSetFloatOption(_fs: Option<&mut JxlEncoderFrameSettings>, _opt: JxlEncoderFrameSettingId, _val: f32) -> JxlEncoderStatus = JXL_ENC_ERROR;
    fn JxlEncoderUseContainer(_enc: Option<&mut JxlEncoder>, _use_container: JXL_BOOL) -> JxlEncoderStatus = JXL_ENC_ERROR;
    fn JxlEncoderStoreJPEGMetadata(_enc: Option<&mut JxlEncoder>, _store: JXL_BOOL) -> JxlEncoderStatus = JXL_ENC_ERROR;
    fn JxlEncoderSetCodestreamLevel(_enc: Option<&mut JxlEncoder>, _level: c_int) -> JxlEncoderStatus = JXL_ENC_ERROR;
    fn JxlEncoderGetRequiredCodestreamLevel(_enc: Option<&JxlEncoder>) -> c_int = -1;
    fn JxlEncoderSetFrameLossless(_fs: Option<&mut JxlEncoderFrameSettings>, _lossless: JXL_BOOL) -> JxlEncoderStatus = JXL_ENC_ERROR;
    fn JxlEncoderSetFrameDistance(_fs: Option<&mut JxlEncoderFrameSettings>, _distance: f32) -> JxlEncoderStatus = JXL_ENC_ERROR;
    fn JxlEncoderSetExtraChannelDistance(_fs: Option<&mut JxlEncoderFrameSettings>, _idx: usize, _distance: f32) -> JxlEncoderStatus = JXL_ENC_ERROR;
    fn JxlEncoderDistanceFromQuality(_quality: f32) -> f32 = -1.0;
    fn JxlEncoderFrameSettingsCreate(_enc: Option<&mut JxlEncoder>, _src: Option<&JxlEncoderFrameSettings>) -> *mut JxlEncoderFrameSettings = std::ptr::null_mut();
    fn JxlEncoderAllowExpertOptions(_enc: Option<&mut JxlEncoder>) = ();
    fn JxlEncoderSetDebugImageCallback(_fs: Option<&mut JxlEncoderFrameSettings>, _cb: JxlDebugImageCallback, _opaque: *mut c_void) = ();
    fn JxlEncoderCollectStats(_fs: Option<&mut JxlEncoderFrameSettings>, _stats: Option<&mut JxlEncoderStats>) = ();
    fn JxlEncoderStatsCreate() -> *mut JxlEncoderStats = std::ptr::null_mut();
    fn JxlEncoderStatsDestroy(_stats: Option<&mut JxlEncoderStats>) = ();
    fn JxlEncoderStatsGet(_stats: Option<&JxlEncoderStats>, _key: JxlEncoderStatsKey) -> usize = 0;
    fn JxlEncoderStatsMerge(_stats: Option<&mut JxlEncoderStats>, _other: Option<&JxlEncoderStats>) = ();
}
