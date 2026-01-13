// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(unsafe_code)]

use std::any::Any;

use crate::api::JxlCmsTransformer;
use crate::error::Result;
use crate::render::RenderPipelineInPlaceStage;
use crate::render::simd_utils::{
    deinterleave_2_dispatch, deinterleave_3_dispatch, deinterleave_4_dispatch,
    interleave_2_dispatch, interleave_3_dispatch, interleave_4_dispatch,
};

/// Trampoline function pointer type.
type TransformFn = fn(*const (), &mut [f32]) -> std::result::Result<(), String>;

/// Trampoline that calls through the double-boxed trait object.
/// Using a function pointer prevents LLVM from analyzing the vtable
/// and affecting optimization of unrelated code paths.
fn cms_trampoline(ptr: *const (), buf: &mut [f32]) -> std::result::Result<(), String> {
    // SAFETY: ptr was created from a valid Box<Box<dyn JxlCmsTransformer + Send>>
    // and the Box is kept alive by CmsStage._owned
    let boxed = unsafe { &mut *(ptr as *mut Box<dyn JxlCmsTransformer + Send>) };
    boxed.do_transform_inplace(buf).map_err(|e| e.to_string())
}

/// Thread-local state for CMS transform.
struct CmsLocalState {
    transformer_idx: usize,
    buffer: Vec<f32>,
}

/// Opaque handle to a transformer, stored as thin pointer + function pointer.
struct OpaqueTransformer {
    ptr: *const (),
    call: TransformFn,
}

// SAFETY: The pointer points to a Box<dyn JxlCmsTransformer + Send> which is Send.
// The data is owned by CmsStage._owned and outlives the OpaqueTransformer.
unsafe impl Send for OpaqueTransformer {}
// SAFETY: Access is synchronized via the pipeline's thread-local state assignment.
// Each thread gets a unique transformer_idx, so no concurrent mutable access occurs.
unsafe impl Sync for OpaqueTransformer {}

/// Applies CMS color transform between color profiles.
pub struct CmsStage {
    transformers: Vec<OpaqueTransformer>,
    // Double-boxed to get thin pointers. The outer Box is what we point to.
    // This indirection is intentional - it gives us a thin pointer to the inner fat pointer.
    #[allow(clippy::vec_box)]
    _owned: Vec<Box<Box<dyn JxlCmsTransformer + Send>>>,
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
        // Pad buffer to SIMD alignment (max vector length is 16)
        let padded_pixels = max_pixels.next_multiple_of(16);

        // Double-box each transformer to get thin pointers.
        // The inner Box is a fat pointer (dyn Trait), the outer Box gives us a thin pointer.
        let mut owned: Vec<Box<Box<dyn JxlCmsTransformer + Send>>> =
            transformers.into_iter().map(Box::new).collect();

        // Create opaque handles pointing to the inner Box
        let opaque: Vec<OpaqueTransformer> = owned
            .iter_mut()
            .map(|b| OpaqueTransformer {
                ptr: (&mut **b) as *mut Box<dyn JxlCmsTransformer + Send> as *const (),
                call: cms_trampoline,
            })
            .collect();

        Self {
            transformers: opaque,
            _owned: owned,
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
            let t = &self.transformers[state.transformer_idx];
            (t.call)(t.ptr, &mut row[0][..xsize]).expect("CMS transform failed");
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
        let t = &self.transformers[state.transformer_idx];
        (t.call)(t.ptr, &mut state.buffer[..xsize * nc]).expect("CMS transform failed");

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
