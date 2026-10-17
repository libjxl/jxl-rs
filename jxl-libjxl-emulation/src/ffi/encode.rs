// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::ffi::{c_char, c_int, c_void};

use crate::types::{
    JXL_BLEND_REPLACE, JXL_BOOL, JXL_ENC_ERR_NOT_SUPPORTED, JXL_ENC_ERROR, JXL_FALSE,
    JXL_ORIENT_IDENTITY, JXL_TRUE, JxlAnimationHeader, JxlBasicInfo, JxlBitDepth, JxlBlendInfo,
    JxlBoxType, JxlChunkedFrameInputSource, JxlCmsInterface, JxlColorEncoding,
    JxlDebugImageCallback, JxlEncoder, JxlEncoderError, JxlEncoderFrameSettingId,
    JxlEncoderFrameSettings, JxlEncoderOutputProcessor, JxlEncoderStats, JxlEncoderStatsKey,
    JxlEncoderStatus, JxlExtraChannelInfo, JxlExtraChannelType, JxlFrameHeader, JxlLayerInfo,
    JxlMemoryManager, JxlParallelRunner, JxlPixelFormat, JxlPreviewHeader,
};

#[unsafe(no_mangle)]
pub extern "C" fn JxlEncoderVersion() -> u32 {
    11_001
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlEncoderInitBasicInfo(info: Option<&mut JxlBasicInfo>) {
    if let Some(info) = info {
        *info = JxlBasicInfo {
            have_container: JXL_FALSE,
            xsize: 0,
            ysize: 0,
            bits_per_sample: 8,
            exponent_bits_per_sample: 0,
            intensity_target: 0.0,
            min_nits: 0.0,
            relative_to_max_display: JXL_FALSE,
            linear_below: 0.0,
            uses_original_profile: JXL_FALSE,
            have_preview: JXL_FALSE,
            have_animation: JXL_FALSE,
            orientation: JXL_ORIENT_IDENTITY,
            num_color_channels: 3,
            num_extra_channels: 0,
            alpha_bits: 0,
            alpha_exponent_bits: 0,
            alpha_premultiplied: JXL_FALSE,
            preview: JxlPreviewHeader { xsize: 0, ysize: 0 },
            animation: JxlAnimationHeader {
                tps_numerator: 10,
                tps_denominator: 1,
                num_loops: 0,
                have_timecodes: JXL_FALSE,
            },
            intrinsic_xsize: 0,
            intrinsic_ysize: 0,
            padding: [0u8; 100],
        };
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlEncoderInitFrameHeader(frame_header: Option<&mut JxlFrameHeader>) {
    if let Some(frame_header) = frame_header {
        *frame_header = JxlFrameHeader {
            duration: 0,
            timecode: 0,
            name_length: 0,
            is_last: JXL_TRUE,
            layer_info: JxlLayerInfo {
                have_crop: JXL_FALSE,
                crop_x0: 0,
                crop_y0: 0,
                xsize: 0,
                ysize: 0,
                blend_info: JxlBlendInfo {
                    blendmode: JXL_BLEND_REPLACE,
                    source: 0,
                    alpha: 0,
                    clamp: JXL_FALSE,
                },
                save_as_reference: 0,
            },
        };
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlEncoderInitBlendInfo(blend_info: Option<&mut JxlBlendInfo>) {
    if let Some(blend_info) = blend_info {
        *blend_info = JxlBlendInfo {
            blendmode: JXL_BLEND_REPLACE,
            source: 0,
            alpha: 0,
            clamp: JXL_FALSE,
        };
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlEncoderInitExtraChannelInfo(
    type_: JxlExtraChannelType,
    info: Option<&mut JxlExtraChannelInfo>,
) {
    if let Some(info) = info {
        *info = JxlExtraChannelInfo {
            type_,
            bits_per_sample: 8,
            exponent_bits_per_sample: 0,
            dim_shift: 0,
            name_length: 0,
            alpha_premultiplied: JXL_FALSE,
            spot_color: [0.0; 4],
            cfa_channel: 0,
        };
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlEncoderCreate(_memory_manager: Option<&JxlMemoryManager>) -> *mut JxlEncoder {
    std::ptr::null_mut()
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlEncoderReset(_enc: *mut JxlEncoder) {}

#[unsafe(no_mangle)]
pub extern "C" fn JxlEncoderDestroy(_enc: *mut JxlEncoder) {}

#[unsafe(no_mangle)]
pub extern "C" fn JxlEncoderGetError(_enc: *mut JxlEncoder) -> JxlEncoderError {
    JXL_ENC_ERR_NOT_SUPPORTED
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlEncoderSetCms(_enc: *mut JxlEncoder, _cms: JxlCmsInterface) {}

#[unsafe(no_mangle)]
pub extern "C" fn JxlEncoderSetParallelRunner(
    _enc: *mut JxlEncoder,
    _parallel_runner: JxlParallelRunner,
    _parallel_runner_opaque: *mut c_void,
) -> JxlEncoderStatus {
    JXL_ENC_ERROR
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlEncoderProcessOutput(
    _enc: *mut JxlEncoder,
    _next_out: *mut *mut u8,
    _avail_out: *mut usize,
) -> JxlEncoderStatus {
    JXL_ENC_ERROR
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlEncoderSetFrameHeader(
    _frame_settings: *mut JxlEncoderFrameSettings,
    _frame_header: *const JxlFrameHeader,
) -> JxlEncoderStatus {
    JXL_ENC_ERROR
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlEncoderSetExtraChannelBlendInfo(
    _frame_settings: *mut JxlEncoderFrameSettings,
    _index: usize,
    _blend_info: *const JxlBlendInfo,
) -> JxlEncoderStatus {
    JXL_ENC_ERROR
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlEncoderSetFrameName(
    _frame_settings: *mut JxlEncoderFrameSettings,
    _frame_name: *const c_char,
) -> JxlEncoderStatus {
    JXL_ENC_ERROR
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlEncoderSetFrameBitDepth(
    _frame_settings: *mut JxlEncoderFrameSettings,
    _bit_depth: *const JxlBitDepth,
) -> JxlEncoderStatus {
    JXL_ENC_ERROR
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlEncoderAddJPEGFrame(
    _frame_settings: *const JxlEncoderFrameSettings,
    _buffer: *const u8,
    _size: usize,
) -> JxlEncoderStatus {
    JXL_ENC_ERROR
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlEncoderAddImageFrame(
    _frame_settings: *const JxlEncoderFrameSettings,
    _pixel_format: *const JxlPixelFormat,
    _buffer: *const c_void,
    _size: usize,
) -> JxlEncoderStatus {
    JXL_ENC_ERROR
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlEncoderSetOutputProcessor(
    _enc: *mut JxlEncoder,
    _output_processor: JxlEncoderOutputProcessor,
) -> JxlEncoderStatus {
    JXL_ENC_ERROR
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlEncoderFlushInput(_enc: *mut JxlEncoder) -> JxlEncoderStatus {
    JXL_ENC_ERROR
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlEncoderAddChunkedFrame(
    _frame_settings: *const JxlEncoderFrameSettings,
    _is_last_frame: JXL_BOOL,
    _chunked_frame_input: JxlChunkedFrameInputSource,
) -> JxlEncoderStatus {
    JXL_ENC_ERROR
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlEncoderSetExtraChannelBuffer(
    _frame_settings: *const JxlEncoderFrameSettings,
    _pixel_format: *const JxlPixelFormat,
    _buffer: *const c_void,
    _size: usize,
    _index: u32,
) -> JxlEncoderStatus {
    JXL_ENC_ERROR
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlEncoderAddBox(
    _enc: *mut JxlEncoder,
    _type_: *const JxlBoxType,
    _contents: *const u8,
    _size: usize,
    _compress_box: JXL_BOOL,
) -> JxlEncoderStatus {
    JXL_ENC_ERROR
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlEncoderUseBoxes(_enc: *mut JxlEncoder) -> JxlEncoderStatus {
    JXL_ENC_ERROR
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlEncoderCloseBoxes(_enc: *mut JxlEncoder) {}

#[unsafe(no_mangle)]
pub extern "C" fn JxlEncoderCloseFrames(_enc: *mut JxlEncoder) {}

#[unsafe(no_mangle)]
pub extern "C" fn JxlEncoderCloseInput(_enc: *mut JxlEncoder) {}

#[unsafe(no_mangle)]
pub extern "C" fn JxlEncoderSetColorEncoding(
    _enc: *mut JxlEncoder,
    _color: *const JxlColorEncoding,
) -> JxlEncoderStatus {
    JXL_ENC_ERROR
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlEncoderSetICCProfile(
    _enc: *mut JxlEncoder,
    _icc_profile: *const u8,
    _size: usize,
) -> JxlEncoderStatus {
    JXL_ENC_ERROR
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlEncoderSetBasicInfo(
    _enc: *mut JxlEncoder,
    _info: *const JxlBasicInfo,
) -> JxlEncoderStatus {
    JXL_ENC_ERROR
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlEncoderSetUpsamplingMode(
    _enc: *mut JxlEncoder,
    _factor: i64,
    _mode: i64,
) -> JxlEncoderStatus {
    JXL_ENC_ERROR
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlEncoderSetExtraChannelInfo(
    _enc: *mut JxlEncoder,
    _index: usize,
    _info: *const JxlExtraChannelInfo,
) -> JxlEncoderStatus {
    JXL_ENC_ERROR
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlEncoderSetExtraChannelName(
    _enc: *mut JxlEncoder,
    _index: usize,
    _name: *const c_char,
    _size: usize,
) -> JxlEncoderStatus {
    JXL_ENC_ERROR
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlEncoderFrameSettingsSetOption(
    _frame_settings: *mut JxlEncoderFrameSettings,
    _option: JxlEncoderFrameSettingId,
    _value: i64,
) -> JxlEncoderStatus {
    JXL_ENC_ERROR
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlEncoderFrameSettingsSetFloatOption(
    _frame_settings: *mut JxlEncoderFrameSettings,
    _option: JxlEncoderFrameSettingId,
    _value: f32,
) -> JxlEncoderStatus {
    JXL_ENC_ERROR
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlEncoderUseContainer(
    _enc: *mut JxlEncoder,
    _use_container: JXL_BOOL,
) -> JxlEncoderStatus {
    JXL_ENC_ERROR
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlEncoderStoreJPEGMetadata(
    _enc: *mut JxlEncoder,
    _store_jpeg_metadata: JXL_BOOL,
) -> JxlEncoderStatus {
    JXL_ENC_ERROR
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlEncoderSetCodestreamLevel(
    _enc: *mut JxlEncoder,
    _level: c_int,
) -> JxlEncoderStatus {
    JXL_ENC_ERROR
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlEncoderGetRequiredCodestreamLevel(_enc: *const JxlEncoder) -> c_int {
    -1
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlEncoderSetFrameLossless(
    _frame_settings: *mut JxlEncoderFrameSettings,
    _lossless: JXL_BOOL,
) -> JxlEncoderStatus {
    JXL_ENC_ERROR
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlEncoderSetFrameDistance(
    _frame_settings: *mut JxlEncoderFrameSettings,
    _distance: f32,
) -> JxlEncoderStatus {
    JXL_ENC_ERROR
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlEncoderSetExtraChannelDistance(
    _frame_settings: *mut JxlEncoderFrameSettings,
    _index: usize,
    _distance: f32,
) -> JxlEncoderStatus {
    JXL_ENC_ERROR
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlEncoderDistanceFromQuality(_quality: f32) -> f32 {
    -1.0
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlEncoderFrameSettingsCreate(
    _enc: *mut JxlEncoder,
    _source: *const JxlEncoderFrameSettings,
) -> *mut JxlEncoderFrameSettings {
    std::ptr::null_mut()
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlEncoderAllowExpertOptions(_enc: *mut JxlEncoder) {}

#[unsafe(no_mangle)]
pub extern "C" fn JxlEncoderSetDebugImageCallback(
    _frame_settings: *mut JxlEncoderFrameSettings,
    _callback: JxlDebugImageCallback,
    _opaque: *mut c_void,
) {
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlEncoderCollectStats(
    _frame_settings: *mut JxlEncoderFrameSettings,
    _stats: *mut JxlEncoderStats,
) {
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlEncoderStatsCreate() -> *mut JxlEncoderStats {
    std::ptr::null_mut()
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlEncoderStatsDestroy(_stats: *mut JxlEncoderStats) {}

#[unsafe(no_mangle)]
pub extern "C" fn JxlEncoderStatsGet(
    _stats: *const JxlEncoderStats,
    _key: JxlEncoderStatsKey,
) -> usize {
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlEncoderStatsMerge(
    _stats: *mut JxlEncoderStats,
    _other: *const JxlEncoderStats,
) {
}
