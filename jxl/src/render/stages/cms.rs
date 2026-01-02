// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::any::Any;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::api::JxlCmsTransformer;
use crate::error::Result;
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
    next_transformer_idx: AtomicUsize,
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
            next_transformer_idx: AtomicUsize::new(0),
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

    fn init_local_state(&self) -> Result<Option<Box<dyn Any>>> {
        if self.transformers.is_empty() {
            return Ok(None);
        }
        // Assign unique transformer index to each thread's local state.
        // Relaxed ordering is sufficient: we only need atomicity for the increment,
        // not happens-before relationships. Wraparound is handled by modulo.
        let idx =
            self.next_transformer_idx.fetch_add(1, Ordering::Relaxed) % self.transformers.len();
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

        // Interleave planar -> packed
        for x in 0..xsize {
            for (c, row_c) in row.iter().enumerate().take(nc) {
                state.buffer[x * nc + c] = row_c[x];
            }
        }

        // Apply transform via mutex-guarded transformer
        let mut transformer = self.transformers[state.transformer_idx]
            .lock()
            .expect("CMS transformer mutex poisoned");
        transformer
            .do_transform_inplace(&mut state.buffer[..xsize * nc])
            .expect("CMS transform failed");

        // De-interleave packed -> planar
        for x in 0..xsize {
            for (c, row_c) in row.iter_mut().enumerate().take(nc) {
                row_c[x] = state.buffer[x * nc + c];
            }
        }
    }
}
