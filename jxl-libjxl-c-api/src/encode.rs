// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::ffi::{c_char, c_int, c_void};
use std::mem::MaybeUninit;

use crate::types::{
    JPEGXL_NUMERIC_VERSION, JXL_BLEND_REPLACE, JXL_BOOL, JXL_ENC_ERR_NOT_SUPPORTED, JXL_ENC_ERROR,
    JXL_FALSE, JXL_ORIENT_IDENTITY, JXL_TRUE, JxlAnimationHeader, JxlBasicInfo, JxlBitDepth,
    JxlBlendInfo, JxlBoxType, JxlChunkedFrameInputSource, JxlCmsInterface, JxlColorEncoding,
    JxlDebugImageCallback, JxlEncoder, JxlEncoderError, JxlEncoderFrameSettingId,
    JxlEncoderFrameSettings, JxlEncoderOutputProcessor, JxlEncoderStats, JxlEncoderStatsKey,
    JxlEncoderStatus, JxlExtraChannelInfo, JxlExtraChannelType, JxlFrameHeader, JxlGainMapBundle,
    JxlLayerInfo, JxlMemoryManager, JxlParallelRunner, JxlPixelFormat, JxlPreviewHeader,
};

#[unsafe(no_mangle)]
pub extern "C" fn JxlEncoderVersion() -> u32 {
    JPEGXL_NUMERIC_VERSION
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlEncoderInitBasicInfo(info: Option<&mut MaybeUninit<JxlBasicInfo>>) {
    if let Some(info) = info {
        info.write(JxlBasicInfo {
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
            padding: [0; 100],
        });
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlEncoderInitFrameHeader(
    frame_header: Option<&mut MaybeUninit<JxlFrameHeader>>,
) {
    if let Some(frame_header) = frame_header {
        frame_header.write(JxlFrameHeader {
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
        });
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlEncoderInitBlendInfo(blend_info: Option<&mut MaybeUninit<JxlBlendInfo>>) {
    if let Some(blend_info) = blend_info {
        blend_info.write(JxlBlendInfo {
            blendmode: JXL_BLEND_REPLACE,
            source: 0,
            alpha: 0,
            clamp: JXL_FALSE,
        });
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn JxlEncoderInitExtraChannelInfo(
    type_: JxlExtraChannelType,
    info: Option<&mut MaybeUninit<JxlExtraChannelInfo>>,
) {
    if let Some(info) = info {
        info.write(JxlExtraChannelInfo {
            type_,
            bits_per_sample: 8,
            exponent_bits_per_sample: 0,
            dim_shift: 0,
            name_length: 0,
            alpha_premultiplied: JXL_FALSE,
            spot_color: [0.0; 4],
            cfa_channel: 0,
        });
    }
}

// Unsupported feature: JPEG XL encoding is not implemented in `jxl-rs`.
// Struct initialization helpers (`JxlEncoderInit*`) and `JxlEncoderVersion` are provided above,
// while all stateful encoder operations below are stubs returning `NULL` / `JXL_ENC_ERROR` /
// `JXL_ENC_ERR_NOT_SUPPORTED` for symbol compatibility.
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
    fn JxlEncoderCreate(_mm: Option<&JxlMemoryManager>) -> *mut JxlEncoder = std::ptr::null_mut();
    fn JxlEncoderReset(_enc: Option<&mut JxlEncoder>) = ();
    fn JxlEncoderDestroy(_enc: Option<&mut JxlEncoder>) = ();
    fn JxlEncoderGetError(
        _enc: Option<&mut JxlEncoder>,
    ) -> JxlEncoderError = JXL_ENC_ERR_NOT_SUPPORTED;
    fn JxlEncoderSetCms(_enc: Option<&mut JxlEncoder>, _cms: JxlCmsInterface) = ();
    fn JxlEncoderSetParallelRunner(
        _enc: Option<&mut JxlEncoder>,
        _runner: JxlParallelRunner,
        _opaque: *mut c_void,
    ) -> JxlEncoderStatus = JXL_ENC_ERROR;
    fn JxlEncoderProcessOutput(
        _enc: Option<&mut JxlEncoder>,
        _next_out: *mut *mut u8,
        _avail_out: *mut usize,
    ) -> JxlEncoderStatus = JXL_ENC_ERROR;
    fn JxlEncoderSetFrameHeader(
        _fs: Option<&mut JxlEncoderFrameSettings>,
        _fh: Option<&JxlFrameHeader>,
    ) -> JxlEncoderStatus = JXL_ENC_ERROR;
    fn JxlEncoderSetExtraChannelBlendInfo(
        _fs: Option<&mut JxlEncoderFrameSettings>,
        _index: usize,
        _bi: Option<&JxlBlendInfo>,
    ) -> JxlEncoderStatus = JXL_ENC_ERROR;
    fn JxlEncoderSetFrameName(
        _fs: Option<&mut JxlEncoderFrameSettings>,
        _name: *const c_char,
    ) -> JxlEncoderStatus = JXL_ENC_ERROR;
    fn JxlEncoderSetFrameBitDepth(
        _fs: Option<&mut JxlEncoderFrameSettings>,
        _bd: Option<&JxlBitDepth>,
    ) -> JxlEncoderStatus = JXL_ENC_ERROR;
    fn JxlEncoderAddJPEGFrame(
        _fs: Option<&mut JxlEncoderFrameSettings>,
        _buf: *const u8,
        _size: usize,
    ) -> JxlEncoderStatus = JXL_ENC_ERROR;
    fn JxlEncoderAddImageFrame(
        _fs: Option<&mut JxlEncoderFrameSettings>,
        _pf: Option<&JxlPixelFormat>,
        _buf: *const c_void,
        _size: usize,
    ) -> JxlEncoderStatus = JXL_ENC_ERROR;
    fn JxlEncoderSetOutputProcessor(
        _enc: Option<&mut JxlEncoder>,
        _op: JxlEncoderOutputProcessor,
    ) -> JxlEncoderStatus = JXL_ENC_ERROR;
    fn JxlEncoderFlushInput(_enc: Option<&mut JxlEncoder>) -> JxlEncoderStatus = JXL_ENC_ERROR;
    fn JxlEncoderAddChunkedFrame(
        _fs: Option<&mut JxlEncoderFrameSettings>,
        _last: JXL_BOOL,
        _src: JxlChunkedFrameInputSource,
    ) -> JxlEncoderStatus = JXL_ENC_ERROR;
    fn JxlEncoderSetExtraChannelBuffer(
        _fs: Option<&mut JxlEncoderFrameSettings>,
        _pf: Option<&JxlPixelFormat>,
        _buf: *const c_void,
        _size: usize,
        _idx: u32,
    ) -> JxlEncoderStatus = JXL_ENC_ERROR;
    fn JxlEncoderAddBox(
        _enc: Option<&mut JxlEncoder>,
        _ty: Option<&JxlBoxType>,
        _contents: *const u8,
        _size: usize,
        _compress: JXL_BOOL,
    ) -> JxlEncoderStatus = JXL_ENC_ERROR;
    fn JxlEncoderUseBoxes(_enc: Option<&mut JxlEncoder>) -> JxlEncoderStatus = JXL_ENC_ERROR;
    fn JxlEncoderCloseBoxes(_enc: Option<&mut JxlEncoder>) = ();
    fn JxlEncoderCloseFrames(_enc: Option<&mut JxlEncoder>) = ();
    fn JxlEncoderCloseInput(_enc: Option<&mut JxlEncoder>) = ();
    fn JxlEncoderSetColorEncoding(
        _enc: Option<&mut JxlEncoder>,
        _color: Option<&JxlColorEncoding>,
    ) -> JxlEncoderStatus = JXL_ENC_ERROR;
    fn JxlEncoderSetICCProfile(
        _enc: Option<&mut JxlEncoder>,
        _icc: *const u8,
        _size: usize,
    ) -> JxlEncoderStatus = JXL_ENC_ERROR;
    fn JxlEncoderSetBasicInfo(
        _enc: Option<&mut JxlEncoder>,
        _info: Option<&JxlBasicInfo>,
    ) -> JxlEncoderStatus = JXL_ENC_ERROR;
    fn JxlEncoderSetUpsamplingMode(
        _enc: Option<&mut JxlEncoder>,
        _factor: i64,
        _mode: i64,
    ) -> JxlEncoderStatus = JXL_ENC_ERROR;
    fn JxlEncoderSetExtraChannelInfo(
        _enc: Option<&mut JxlEncoder>,
        _index: usize,
        _info: Option<&JxlExtraChannelInfo>,
    ) -> JxlEncoderStatus = JXL_ENC_ERROR;
    fn JxlEncoderSetExtraChannelName(
        _enc: Option<&mut JxlEncoder>,
        _index: usize,
        _name: *const c_char,
        _size: usize,
    ) -> JxlEncoderStatus = JXL_ENC_ERROR;
    fn JxlEncoderFrameSettingsSetOption(
        _fs: Option<&mut JxlEncoderFrameSettings>,
        _opt: JxlEncoderFrameSettingId,
        _val: i64,
    ) -> JxlEncoderStatus = JXL_ENC_ERROR;
    fn JxlEncoderFrameSettingsSetFloatOption(
        _fs: Option<&mut JxlEncoderFrameSettings>,
        _opt: JxlEncoderFrameSettingId,
        _val: f32,
    ) -> JxlEncoderStatus = JXL_ENC_ERROR;
    fn JxlEncoderUseContainer(
        _enc: Option<&mut JxlEncoder>,
        _use_container: JXL_BOOL,
    ) -> JxlEncoderStatus = JXL_ENC_ERROR;
    fn JxlEncoderStoreJPEGMetadata(
        _enc: Option<&mut JxlEncoder>,
        _store: JXL_BOOL,
    ) -> JxlEncoderStatus = JXL_ENC_ERROR;
    fn JxlEncoderSetCodestreamLevel(
        _enc: Option<&mut JxlEncoder>,
        _level: c_int,
    ) -> JxlEncoderStatus = JXL_ENC_ERROR;
    fn JxlEncoderGetRequiredCodestreamLevel(_enc: Option<&JxlEncoder>) -> c_int = -1;
    fn JxlEncoderSetFrameLossless(
        _fs: Option<&mut JxlEncoderFrameSettings>,
        _lossless: JXL_BOOL,
    ) -> JxlEncoderStatus = JXL_ENC_ERROR;
    fn JxlEncoderSetFrameDistance(
        _fs: Option<&mut JxlEncoderFrameSettings>,
        _distance: f32,
    ) -> JxlEncoderStatus = JXL_ENC_ERROR;
    fn JxlEncoderSetExtraChannelDistance(
        _fs: Option<&mut JxlEncoderFrameSettings>,
        _idx: usize,
        _distance: f32,
    ) -> JxlEncoderStatus = JXL_ENC_ERROR;
    fn JxlEncoderDistanceFromQuality(_quality: f32) -> f32 = -1.0;
    fn JxlEncoderFrameSettingsCreate(
        _enc: Option<&mut JxlEncoder>,
        _src: Option<&JxlEncoderFrameSettings>,
    ) -> *mut JxlEncoderFrameSettings = std::ptr::null_mut();
    fn JxlEncoderAllowExpertOptions(_enc: Option<&mut JxlEncoder>) = ();
    fn JxlEncoderSetDebugImageCallback(
        _fs: Option<&mut JxlEncoderFrameSettings>,
        _cb: JxlDebugImageCallback,
        _opaque: *mut c_void,
    ) = ();
    fn JxlEncoderCollectStats(
        _fs: Option<&mut JxlEncoderFrameSettings>,
        _stats: Option<&mut JxlEncoderStats>,
    ) = ();
    fn JxlEncoderStatsCreate() -> *mut JxlEncoderStats = std::ptr::null_mut();
    fn JxlEncoderStatsDestroy(_stats: Option<&mut JxlEncoderStats>) = ();
    fn JxlEncoderStatsGet(_stats: Option<&JxlEncoderStats>, _key: JxlEncoderStatsKey) -> usize = 0;
    fn JxlEncoderStatsMerge(
        _stats: Option<&mut JxlEncoderStats>,
        _other: Option<&JxlEncoderStats>,
    ) = ();
    fn JxlGainMapGetBundleSize(
        _map_bundle: Option<&JxlGainMapBundle>,
        _bundle_size: Option<&mut MaybeUninit<usize>>,
    ) -> JXL_BOOL = JXL_FALSE;
    fn JxlGainMapWriteBundle(
        _map_bundle: Option<&JxlGainMapBundle>,
        _output_buffer: *mut u8,
        _output_buffer_size: usize,
        _bytes_written: Option<&mut MaybeUninit<usize>>,
    ) -> JXL_BOOL = JXL_FALSE;
}
