// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::render::{Channels, ChannelsMut, ErasedLocalState, RenderPipelineInOutStage};

/// Scalar box reduction. Each call contributes one input row to an output row.
/// Edge blocks average only the samples that actually belong to the image.
pub struct Downsample<const X: u8, const Y: u8> {
    channel: usize,
    input_size: (usize, usize),
}

impl<const X: u8, const Y: u8> Downsample<X, Y> {
    pub fn new(channel: usize, input_size: (usize, usize)) -> Self {
        assert!(X <= 3 && Y <= 3);
        Self {
            channel,
            input_size,
        }
    }
}

impl<const X: u8, const Y: u8> std::fmt::Display for Downsample<X, Y> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "box downsample {}x{} channel {}",
            1 << X,
            1 << Y,
            self.channel
        )
    }
}

impl<const X: u8, const Y: u8> RenderPipelineInOutStage for Downsample<X, Y> {
    type InputT = f32;
    type OutputT = f32;
    const BORDER: (u8, u8) = (0, 0);
    const SHIFT: (i8, i8) = (-(X as i8), -(Y as i8));

    fn uses_channel(&self, c: usize) -> bool {
        c == self.channel
    }

    fn process_row_chunk(
        &self,
        (x, y): (usize, usize),
        xsize: usize,
        input_rows: &Channels<f32>,
        output_rows: &mut ChannelsMut<f32>,
        _state: Option<&mut ErasedLocalState>,
        _previous_call_was_previous_row: bool,
    ) {
        let fx = 1 << X;
        let fy = 1 << Y;
        assert_eq!(x % fx, 0);
        assert!(xsize % fx == 0 || x + xsize == self.input_size.0);
        let input = input_rows.row(0);
        let output = output_rows.row_mut(0, 0);
        let first = y % fy == 0;
        let last = y % fy == fy - 1 || y + 1 == self.input_size.1;
        let height = fy.min(self.input_size.1 - (y / fy) * fy);
        for (ox, ix) in (0..xsize).step_by(fx).enumerate() {
            let width = fx.min(xsize - ix);
            let sum: f32 = input[ix..ix + width].iter().sum();
            if first {
                output[ox] = sum;
            } else {
                output[ox] += sum;
            }
            if last {
                output[ox] /= (width * height) as f32;
            }
        }
    }
}
