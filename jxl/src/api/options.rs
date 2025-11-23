// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::api::JxlCms;

pub enum JxlProgressiveMode {
    /// Renders all pixels in every call to Process.
    Eager,
    /// Renders pixels once passes are completed.
    Pass,
    /// Renders pixels only once the final frame is ready.
    FullFrame,
}

#[non_exhaustive]
pub struct JxlDecoderOptions {
    pub adjust_orientation: bool,
    pub unpremultiply_alpha: bool,
    pub render_spot_colors: bool,
    pub coalescing: bool,
    pub desired_intensity_target: Option<f32>,
    pub skip_preview: bool,
    pub progressive_mode: JxlProgressiveMode,
    pub xyb_output_linear: bool,
    pub enable_output: bool,
    pub cms: Option<Box<dyn JxlCms>>,
    /// Use high precision mode for spline rendering.
    /// When false (default), uses lower precision settings that match libjxl's default.
    /// When true, uses higher precision at the cost of performance.
    pub high_precision: bool,
}

impl Default for JxlDecoderOptions {
    fn default() -> Self {
        Self {
            adjust_orientation: true,
            unpremultiply_alpha: false,
            render_spot_colors: true,
            coalescing: true,
            skip_preview: true,
            desired_intensity_target: None,
            progressive_mode: JxlProgressiveMode::Pass,
            xyb_output_linear: true,
            enable_output: true,
            cms: None,
            high_precision: false,
        }
    }
}
