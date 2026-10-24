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
    /// For each output channel: optional alpha channel and associated-alpha flag.
    alpha: Vec<Option<(usize, bool)>>,
}

impl<const SHIFT: u8> DownsampleFrame<SHIFT> {
    pub fn new(input_size: (usize, usize), alpha: Vec<Option<(usize, bool)>>) -> Self {
        assert!((1..=3).contains(&SHIFT));
        Self { input_size, alpha }
    }
}

#[derive(Default)]
struct Accumulators {
    sums: Vec<f64>,
    weights: Vec<f64>,
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
        debug_assert_eq!(x % factor, 0);
        debug_assert!(xsize % factor == 0 || x + xsize == self.input_size.0);
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
                for sx in ix..(ix + factor).min(xsize) {
                    let (multiplier, weight) = if let Some((ac, associated)) = alpha {
                        let a = input.row(*ac)[sx] as f64;
                        (if *associated { 1.0 } else { a }, a)
                    } else {
                        (1.0, 1.0)
                    };
                    scratch.sums[idx] += samples[sx] as f64 * multiplier;
                    scratch.weights[idx] += weight;
                }
                if last {
                    dest[ox] = if scratch.weights[idx] > 0.0 {
                        (scratch.sums[idx] / scratch.weights[idx]) as f32
                    } else {
                        0.0
                    };
                }
            }
        }
    }
}
