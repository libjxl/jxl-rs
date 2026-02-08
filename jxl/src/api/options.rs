// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::api::JxlCms;

/// Default maximum aggregate size for EXIF metadata (1MB).
/// Typical EXIF data is 10-64KB.
pub const DEFAULT_EXIF_SIZE_LIMIT: u64 = 1024 * 1024;

/// Default maximum aggregate size for XMP metadata (1MB).
/// Typical XMP data is 1-100KB.
pub const DEFAULT_XMP_SIZE_LIMIT: u64 = 1024 * 1024;

/// Default maximum aggregate size for JUMBF metadata (16MB).
/// JUMBF can be larger due to C2PA embedded images.
pub const DEFAULT_JUMBF_SIZE_LIMIT: u64 = 16 * 1024 * 1024;

pub enum JxlProgressiveMode {
    /// Renders all pixels in every call to Process.
    Eager,
    /// Renders pixels once passes are completed.
    Pass,
    /// Renders pixels only once the final frame is ready.
    FullFrame,
}

/// Options for capturing metadata boxes during container parsing.
/// All capture flags default to false (opt-in) to avoid memory overhead.
#[derive(Debug, Clone, Default)]
pub struct JxlMetadataCaptureOptions {
    /// Whether to capture EXIF metadata boxes during container parsing.
    /// When enabled, EXIF boxes can be retrieved via `exif_boxes()` after parsing.
    pub capture_exif: bool,
    /// Whether to capture XMP metadata boxes during container parsing.
    /// When enabled, XMP boxes can be retrieved via `xmp_boxes()` after parsing.
    pub capture_xmp: bool,
    /// Whether to capture JUMBF metadata boxes during container parsing.
    /// JUMBF (JPEG Universal Metadata Box Format) is used for C2PA content credentials.
    /// When enabled, JUMBF boxes can be retrieved via `jumbf_boxes()` after parsing.
    pub capture_jumbf: bool,
    /// Maximum aggregate size in bytes for all EXIF boxes combined.
    /// Once this limit is reached, additional EXIF boxes are skipped.
    /// Set to `None` to disable the limit.
    /// Default: 1MB ([`DEFAULT_EXIF_SIZE_LIMIT`])
    pub exif_size_limit: Option<u64>,
    /// Maximum aggregate size in bytes for all XMP boxes combined.
    /// Once this limit is reached, additional XMP boxes are skipped.
    /// Set to `None` to disable the limit.
    /// Default: 1MB ([`DEFAULT_XMP_SIZE_LIMIT`])
    pub xmp_size_limit: Option<u64>,
    /// Maximum aggregate size in bytes for all JUMBF boxes combined.
    /// Once this limit is reached, additional JUMBF boxes are skipped.
    /// Set to `None` to disable the limit.
    /// Default: 16MB ([`DEFAULT_JUMBF_SIZE_LIMIT`])
    pub jumbf_size_limit: Option<u64>,
}

impl JxlMetadataCaptureOptions {
    /// Create options with all metadata capture enabled and default size limits.
    pub fn capture_all_with_limits() -> Self {
        Self {
            capture_exif: true,
            capture_xmp: true,
            capture_jumbf: true,
            exif_size_limit: Some(DEFAULT_EXIF_SIZE_LIMIT),
            xmp_size_limit: Some(DEFAULT_XMP_SIZE_LIMIT),
            jumbf_size_limit: Some(DEFAULT_JUMBF_SIZE_LIMIT),
        }
    }

    /// Create options with all metadata capture enabled and no size limits.
    pub fn capture_all() -> Self {
        Self {
            capture_exif: true,
            capture_xmp: true,
            capture_jumbf: true,
            exif_size_limit: None,
            xmp_size_limit: None,
            jumbf_size_limit: None,
        }
    }

    /// Create options with all metadata capture disabled
    pub fn no_capture() -> Self {
        Self {
            capture_exif: false,
            capture_xmp: false,
            capture_jumbf: false,
            exif_size_limit: Some(0),
            xmp_size_limit: Some(0),
            jumbf_size_limit: Some(0),
        }
    }
}

#[non_exhaustive]
pub struct JxlDecoderOptions {
    pub adjust_orientation: bool,
    pub render_spot_colors: bool,
    pub coalescing: bool,
    pub desired_intensity_target: Option<f32>,
    pub skip_preview: bool,
    pub progressive_mode: JxlProgressiveMode,
    pub cms: Option<Box<dyn JxlCms>>,
    /// Fail decoding images with more than this number of pixels, or with frames with
    /// more than this number of pixels. The limit counts the product of pixels and
    /// channels, so for example an image with 1 extra channel of size 1024x1024 has 4
    /// million pixels.
    pub pixel_limit: Option<usize>,
    /// Use high precision mode for decoding.
    /// When false (default), uses lower precision settings that match libjxl's default.
    /// When true, uses higher precision at the cost of performance.
    ///
    /// This affects multiple decoder decisions including spline rendering precision
    /// and potentially intermediate buffer storage (e.g., using f32 vs f16).
    pub high_precision: bool,
    /// If true, multiply RGB by alpha before writing to output buffer.
    /// This produces premultiplied alpha output, which is useful for compositing.
    /// Default: false (output straight alpha)
    pub premultiply_output: bool,
    /// Options for capturing metadata boxes (EXIF, XMP, JUMBF) during container parsing.
    pub metadata_capture: JxlMetadataCaptureOptions,
}

impl Default for JxlDecoderOptions {
    fn default() -> Self {
        Self {
            adjust_orientation: true,
            render_spot_colors: true,
            coalescing: true,
            skip_preview: true,
            desired_intensity_target: None,
            progressive_mode: JxlProgressiveMode::Pass,
            cms: None,
            pixel_limit: None,
            high_precision: false,
            premultiply_output: false,
            metadata_capture: JxlMetadataCaptureOptions::no_capture(),
        }
    }
}
