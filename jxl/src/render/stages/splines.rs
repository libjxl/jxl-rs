// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO(firsching): remove once we use this!
#![allow(dead_code)]

use crate::{
    features::spline::Splines,
    render::{RenderPipelineInPlaceStage, RenderPipelineStage},
};

pub struct SplinesStage {
    splines: Splines,
}

impl std::fmt::Display for SplinesStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "splines")
    }
}

impl RenderPipelineStage for SplinesStage {
    type Type = RenderPipelineInPlaceStage<f32>;

    fn uses_channel(&self, c: usize) -> bool {
        c < 3
    }

    fn process_row_chunk(
        &mut self,
        position: (usize, usize),
        xsize: usize,
        row: &mut [&mut [f32]],
    ) {
        self.splines.draw_segments(row, position, xsize);
    }
}
