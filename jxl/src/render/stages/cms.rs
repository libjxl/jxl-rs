// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::any::Any;

use crate::api::JxlCmsTransformer;
use crate::error::Result;
use crate::render::RenderPipelineInPlaceStage;

use crate::render::simd_utils::{
    deinterleave_2_dispatch, deinterleave_3_dispatch, deinterleave_4_dispatch,
    interleave_2_dispatch, interleave_3_dispatch, interleave_4_dispatch,
};
use crate::util::AtomicRefCell;

/// Thread-local state for CMS transform.
struct CmsLocalState {
    transformer_idx: usize,
    buffer: Vec<f32>,
}

/// Applies CMS color transform between color profiles.
pub struct CmsStage {
    transformers: Vec<AtomicRefCell<Box<dyn JxlCmsTransformer + Send>>>,
    first_channel: usize,
    num_channels: usize,
    buffer_size: usize,
}

/// Wrapper to prevent LLVM from pessimizing unrelated code paths.
///
/// Without this wrapper, the mere presence of the dyn trait call causes a ~20% performance
/// regression in the modular decoder's `process_output` function, even when CMS is never
/// invoked (no CMS provider configured). The regression appears to be caused by LLVM's
/// interprocedural analysis of the dyn trait vtable affecting code generation elsewhere.
///
/// Things we tried that did NOT fix the regression:
/// - `#[cold]` on various functions
/// - `#[inline(never)]` on `process_output` and functions it calls
/// - `std::hint::black_box` around the transformer
/// - Moving SIMD code to separate modules/crates
/// - Returning `()` or `Result<()>` from this wrapper
///
/// Things that DO fix the regression:
/// - `#[inline(never)]` wrapper returning `bool` or `Option<()>` (this solution)
/// - `extern "C"` ABI boundary
/// - Global `lto = "thin"` or `codegen-units = 1`
///
/// Theory: LLVM's analysis of the dyn trait vtable (even when never executed) influences
/// register allocation or instruction selection in distant hot paths. Isolating the dyn
/// call in a non-inlined function with a non-trivial return type creates enough of an
/// optimization barrier to prevent this cross-function interference.
#[inline(never)]
fn call_cms_transform(transformer: &mut dyn JxlCmsTransformer, buf: &mut [f32]) -> Option<()> {
    transformer.do_transform_inplace(buf).ok()
}

impl CmsStage {
    pub fn new(
        transformers: Vec<Box<dyn JxlCmsTransformer + Send>>,
        first_channel: usize,
        num_channels: usize,
        max_pixels: usize,
    ) -> Self {
        // Pad buffer to SIMD alignment (max vector length is 16)
        let padded_pixels = max_pixels.next_multiple_of(16);
        Self {
            transformers: transformers.into_iter().map(AtomicRefCell::new).collect(),
            first_channel,
            num_channels,
            buffer_size: padded_pixels
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

        // Single channel: transform directly in place without copying
        if nc == 1 {
            let mut transformer = self.transformers[state.transformer_idx].borrow_mut();
            call_cms_transform(&mut **transformer, &mut row[0][..xsize])
                .expect("CMS transform failed");
            return;
        }

        // Pad to SIMD alignment (pipeline rows are already padded)
        let xsize_padded = xsize.next_multiple_of(16);

        // Interleave planar -> packed using SIMD
        match nc {
            2 if row.len() >= 2 => {
                interleave_2_dispatch(
                    &row[0][..xsize_padded],
                    &row[1][..xsize_padded],
                    &mut state.buffer[..xsize_padded * 2],
                );
            }
            3 if row.len() >= 3 => {
                interleave_3_dispatch(
                    &row[0][..xsize_padded],
                    &row[1][..xsize_padded],
                    &row[2][..xsize_padded],
                    &mut state.buffer[..xsize_padded * 3],
                );
            }
            4 if row.len() >= 4 => {
                interleave_4_dispatch(
                    &row[0][..xsize_padded],
                    &row[1][..xsize_padded],
                    &row[2][..xsize_padded],
                    &row[3][..xsize_padded],
                    &mut state.buffer[..xsize_padded * 4],
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

        // Apply transform (only on actual pixels, not padding)
        let mut transformer = self.transformers[state.transformer_idx].borrow_mut();
        call_cms_transform(&mut **transformer, &mut state.buffer[..xsize * nc])
            .expect("CMS transform failed");

        // De-interleave packed -> planar using SIMD
        match nc {
            2 if row.len() >= 2 => {
                let (r0, r1) = row.split_at_mut(1);
                deinterleave_2_dispatch(
                    &state.buffer[..xsize_padded * 2],
                    &mut r0[0][..xsize_padded],
                    &mut r1[0][..xsize_padded],
                );
            }
            3 if row.len() >= 3 => {
                let (r0, rest) = row.split_at_mut(1);
                let (r1, r2) = rest.split_at_mut(1);
                deinterleave_3_dispatch(
                    &state.buffer[..xsize_padded * 3],
                    &mut r0[0][..xsize_padded],
                    &mut r1[0][..xsize_padded],
                    &mut r2[0][..xsize_padded],
                );
            }
            4 if row.len() >= 4 => {
                let (r0, rest) = row.split_at_mut(1);
                let (r1, rest) = rest.split_at_mut(1);
                let (r2, r3) = rest.split_at_mut(1);
                deinterleave_4_dispatch(
                    &state.buffer[..xsize_padded * 4],
                    &mut r0[0][..xsize_padded],
                    &mut r1[0][..xsize_padded],
                    &mut r2[0][..xsize_padded],
                    &mut r3[0][..xsize_padded],
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
