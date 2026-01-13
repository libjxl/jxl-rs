// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::any::Any;
use std::sync::Mutex;

use crate::api::JxlCmsTransformer;
use crate::error::Result;
use crate::render::simd_utils::{
    deinterleave_2_dispatch, deinterleave_3_dispatch, deinterleave_4_dispatch,
    interleave_2_dispatch, interleave_3_dispatch, interleave_4_dispatch,
};
use crate::render::RenderPipelineInPlaceStage;

/// Thread-local state for CMS transform.
struct CmsLocalState {
    transformer_idx: usize,
    buffer: Vec<f32>,
}

/// Applies CMS color transform between color profiles.
pub struct CmsStage {
    transformers: Vec<Mutex<Box<dyn JxlCmsTransformer + Send>>>,
    first_channel: usize,
    num_channels: usize,
    buffer_size: usize,
}

impl CmsStage {
    pub fn new(
        transformers: Vec<Box<dyn JxlCmsTransformer + Send>>,
        first_channel: usize,
        num_channels: usize,
        max_pixels: usize,
    ) -> Self {
        Self {
            transformers: transformers.into_iter().map(Mutex::new).collect(),
            first_channel,
            num_channels,
            buffer_size: max_pixels
                .checked_mul(num_channels)
                .expect("CMS buffer size overflow"),
        }
    }
}

impl std::fmt::Display for CmsStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CMS transform for channels [{}, {})",
            self.first_channel,
            self.first_channel + self.num_channels
        )
    }
}

impl RenderPipelineInPlaceStage for CmsStage {
    type Type = f32;

    fn uses_channel(&self, c: usize) -> bool {
        (self.first_channel..self.first_channel + self.num_channels).contains(&c)
    }

    fn init_local_state(&self, thread_index: usize) -> Result<Option<Box<dyn Any>>> {
        if self.transformers.is_empty() {
            return Ok(None);
        }
        // Use thread index modulo transformer count to assign transformer to this thread
        let idx = thread_index % self.transformers.len();
        Ok(Some(Box::new(CmsLocalState {
            transformer_idx: idx,
            buffer: vec![0.0f32; self.buffer_size],
        })))
    }

    fn process_row_chunk(
        &self,
        _position: (usize, usize),
        xsize: usize,
        row: &mut [&mut [f32]],
        state: Option<&mut dyn Any>,
    ) {
        let Some(state) = state else { return };
        let state: &mut CmsLocalState = state.downcast_mut().unwrap();
        let nc = self.num_channels;

        // Interleave planar -> packed using SIMD where available
        match nc {
            1 => {
                // No interleaving needed for single channel
                state.buffer[..xsize].copy_from_slice(&row[0][..xsize]);
            }
            2 if row.len() >= 2 => {
                interleave_2_dispatch(
                    &row[0][..xsize],
                    &row[1][..xsize],
                    &mut state.buffer[..xsize * 2],
                );
            }
            3 if row.len() >= 3 => {
                interleave_3_dispatch(
                    &row[0][..xsize],
                    &row[1][..xsize],
                    &row[2][..xsize],
                    &mut state.buffer[..xsize * 3],
                );
            }
            4 if row.len() >= 4 => {
                interleave_4_dispatch(
                    &row[0][..xsize],
                    &row[1][..xsize],
                    &row[2][..xsize],
                    &row[3][..xsize],
                    &mut state.buffer[..xsize * 4],
                );
            }
            _ => {
                // Scalar fallback for other channel counts
                for x in 0..xsize {
                    for (c, row_c) in row.iter().enumerate().take(nc) {
                        state.buffer[x * nc + c] = row_c[x];
                    }
                }
            }
        }

        // Apply transform via mutex-guarded transformer
        let mut transformer = self.transformers[state.transformer_idx]
            .lock()
            .expect("CMS transformer mutex poisoned");
        transformer
            .do_transform_inplace(&mut state.buffer[..xsize * nc])
            .expect("CMS transform failed");

        // De-interleave packed -> planar using SIMD where available
        match nc {
            1 => {
                // No deinterleaving needed for single channel
                row[0][..xsize].copy_from_slice(&state.buffer[..xsize]);
            }
            2 if row.len() >= 2 => {
                let (r0, r1) = row.split_at_mut(1);
                deinterleave_2_dispatch(
                    &state.buffer[..xsize * 2],
                    &mut r0[0][..xsize],
                    &mut r1[0][..xsize],
                );
            }
            3 if row.len() >= 3 => {
                let (r0, rest) = row.split_at_mut(1);
                let (r1, r2) = rest.split_at_mut(1);
                deinterleave_3_dispatch(
                    &state.buffer[..xsize * 3],
                    &mut r0[0][..xsize],
                    &mut r1[0][..xsize],
                    &mut r2[0][..xsize],
                );
            }
            4 if row.len() >= 4 => {
                let (r0, rest) = row.split_at_mut(1);
                let (r1, rest) = rest.split_at_mut(1);
                let (r2, r3) = rest.split_at_mut(1);
                deinterleave_4_dispatch(
                    &state.buffer[..xsize * 4],
                    &mut r0[0][..xsize],
                    &mut r1[0][..xsize],
                    &mut r2[0][..xsize],
                    &mut r3[0][..xsize],
                );
            }
            _ => {
                // Scalar fallback for other channel counts
                for x in 0..xsize {
                    for (c, row_c) in row.iter_mut().enumerate().take(nc) {
                        row_c[x] = state.buffer[x * nc + c];
                    }
                }
            }
        }
    }
}
