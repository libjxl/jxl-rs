// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::error::Result;
use crate::render::{Channels, ChannelsMut, ErasedLocalState, RenderPipelineInOutStage};
use crate::util::ShiftRightCeil;

pub struct DownsampleStage<const SHIFT: u8> {
    from_channel: usize,
    // (alpha_channel, clamp)
    alpha: Vec<Option<(usize, bool)>>,
}

impl<const SHIFT: u8> DownsampleStage<SHIFT> {
    pub fn new(alpha: Vec<Option<(usize, bool)>>) -> Self {
        const {
            assert!(SHIFT >= 1 && SHIFT <= 3);
        }

        Self {
            from_channel: 0,
            alpha,
        }
    }

    pub fn single(from_channel: usize) -> Self {
        const {
            assert!(SHIFT >= 1 && SHIFT <= 3);
        }

        Self {
            from_channel,
            alpha: vec![None],
        }
    }
}

struct DownsampleState {
    need_reset: bool,
    weights: Vec<f32>,
}

impl<const SHIFT: u8> std::fmt::Display for DownsampleStage<SHIFT> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let from = self.from_channel;
        let num_channels = self.alpha.len();
        let to = from + num_channels;
        write!(f, "downsample channel {from}..{to} by {}x", 1 << SHIFT)
    }
}

impl<const SHIFT: u8> RenderPipelineInOutStage for DownsampleStage<SHIFT> {
    type InputT = f32;
    type OutputT = f32;
    const BORDER: (u8, u8) = (0, 0);
    const SHIFT: (i8, i8) = (-(SHIFT as i8), -(SHIFT as i8));

    fn uses_channel(&self, c: usize) -> bool {
        let from = self.from_channel;
        let num_channels = self.alpha.len();
        (from..from + num_channels).contains(&c)
    }

    fn init_local_state(&self) -> Result<Option<Box<ErasedLocalState>>> {
        Ok(Some(Box::new(DownsampleState {
            need_reset: true,
            weights: Vec::new(),
        })))
    }

    fn process_row_chunk(
        &self,
        _position: (usize, usize),
        xsize: usize,
        input: &Channels<f32>,
        output: &mut ChannelsMut<f32>,
        state: Option<&mut ErasedLocalState>,
        _previous_call_was_previous_row: bool,
        flush_output_row: bool,
    ) {
        let num_channels = self.alpha.len();
        let out_width = xsize.shrc(SHIFT);
        let scratch = state.unwrap().downcast_mut::<DownsampleState>().unwrap();
        if xsize == 0 {
            scratch.need_reset = flush_output_row;
            return;
        }
        if scratch.need_reset {
            scratch.weights.clear();
            scratch.weights.resize(out_width * num_channels, 0.0);
        }

        let weights = scratch.weights.chunks_exact_mut(out_width);
        for (c, (weights, alpha)) in weights.zip(&self.alpha).enumerate() {
            let samples = input.row(c);
            let dest = output.row_mut(c, 0);
            if scratch.need_reset {
                dest.fill(0.0);
            }

            for ox in 0..out_width {
                let ix = ox << SHIFT;
                let ix_end = (ix + (1 << SHIFT)).min(xsize);

                let src = &samples[ix..ix_end];
                if let &Some((ac, clamp)) = alpha {
                    let src_a = &input.row(ac)[ix..ix_end];
                    for (&sample, &alpha) in src.iter().zip(src_a) {
                        let weight = if clamp {
                            alpha.clamp(0.0, 1.0)
                        } else {
                            alpha.max(0.0)
                        };
                        dest[ox] += sample * weight;
                        weights[ox] += weight;
                    }
                } else {
                    for &sample in src.iter() {
                        dest[ox] += sample;
                        weights[ox] += 1.0;
                    }
                }
            }

            if flush_output_row {
                for ox in 0..out_width {
                    dest[ox] = if weights[ox] > 0.0 {
                        dest[ox] / weights[ox]
                    } else {
                        0.0
                    };
                }
            }
        }

        scratch.need_reset = flush_output_row;
    }
}
