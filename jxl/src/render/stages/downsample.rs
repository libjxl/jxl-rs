// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::error::Result;
use crate::render::{Channels, ChannelsMut, ErasedLocalState, RenderPipelineInOutStage};

/// Reduce all frame channels together, retaining the original alpha samples
/// until every channel's weighted sum has been computed.
pub struct DownsampleFrame<const SHIFT: u8> {
    input_size: (usize, usize),
    /// Alpha weights for straight channels; associated samples use an ordinary average.
    alpha: Vec<Option<usize>>,
}

impl<const SHIFT: u8> DownsampleFrame<SHIFT> {
    pub fn new(input_size: (usize, usize), alpha: Vec<Option<usize>>) -> Self {
        assert!((1..=3).contains(&SHIFT));
        Self { input_size, alpha }
    }
}

#[derive(Default)]
struct Accumulators {
    sums: Vec<f32>,
    weights: Vec<f32>,
}

impl<const SHIFT: u8> std::fmt::Display for DownsampleFrame<SHIFT> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "frame downsample {}x", 1 << SHIFT)
    }
}

impl<const SHIFT: u8> RenderPipelineInOutStage for DownsampleFrame<SHIFT> {
    type InputT = f32;
    type OutputT = f32;
    const BORDER: (u8, u8) = (0, 0);
    const SHIFT: (i8, i8) = (-(SHIFT as i8), -(SHIFT as i8));

    fn uses_channel(&self, c: usize) -> bool {
        c < self.alpha.len()
    }

    fn init_local_state(&self) -> Result<Option<Box<ErasedLocalState>>> {
        Ok(Some(Box::new(Accumulators::default())))
    }

    fn process_row_chunk(
        &self,
        (x, y): (usize, usize),
        xsize: usize,
        input: &Channels<f32>,
        output: &mut ChannelsMut<f32>,
        state: Option<&mut ErasedLocalState>,
        _previous_call_was_previous_row: bool,
    ) {
        let factor = 1 << SHIFT;
        debug_assert!(x.is_multiple_of(factor));
        debug_assert!(xsize.is_multiple_of(factor) || x + xsize == self.input_size.0);
        let width = self.input_size.0.div_ceil(factor);
        let scratch = state.unwrap().downcast_mut::<Accumulators>().unwrap();
        // Index by absolute x, so different chunks can accumulate independently.
        scratch.sums.resize(width * self.alpha.len(), 0.0);
        scratch.weights.resize(width * self.alpha.len(), 0.0);
        let first = y % factor == 0;
        let last = y % factor == factor - 1 || y + 1 == self.input_size.1;
        for (c, alpha) in self.alpha.iter().enumerate() {
            let samples = input.row(c);
            let dest = output.row_mut(c, 0);
            for (ox, ix) in (0..xsize).step_by(factor).enumerate() {
                let idx = c * width + x / factor + ox;
                if first {
                    scratch.sums[idx] = 0.0;
                    scratch.weights[idx] = 0.0;
                }

                let ix_end = (ix + factor).min(xsize);
                let src = &samples[ix..ix_end];
                let alpha = alpha.map(|ac| &input.row(ac)[ix..ix_end]);
                for (dx, &sample) in src.iter().enumerate() {
                    let (multiplier, weight) = if let Some(src_a) = alpha {
                        let a = src_a[dx];
                        (a, a)
                    } else {
                        (1.0, 1.0)
                    };
                    scratch.sums[idx] += sample * multiplier;
                    scratch.weights[idx] += weight;
                }

                if last {
                    dest[ox] = if scratch.weights[idx] > 0.0 {
                        scratch.sums[idx] / scratch.weights[idx]
                    } else {
                        0.0
                    };
                }
            }
        }
    }
}
