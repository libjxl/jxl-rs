// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::GROUP_DIM;
use jxl_transforms::transform_map::MAX_COEFF_AREA;

/// Size of the LF scratch buffer (32x32 block).
const LF_BUFFER_SIZE: usize = 32 * 32;

/// Per-thread cache for parallel VarDCT group decoding.
///
/// This cache stores reusable buffers to avoid allocations during parallel
/// decoding. Each thread has its own GroupDecodeCache to eliminate contention.
///
/// The buffers are lazily allocated on first use and reused for subsequent
/// group decodes, significantly reducing allocation overhead.
pub struct GroupDecodeCache {
    /// Scratch buffer for LF operations (LF_BUFFER_SIZE f32s)
    scratch: Vec<f32>,
    /// Coefficient storage for X, Y, B channels (3 * GROUP_DIM * GROUP_DIM i32s)
    coeffs_storage: Vec<i32>,
    /// Transform buffers for X, Y, B channels (3 * MAX_COEFF_AREA f32s each)
    transform_buffer: [Vec<f32>; 3],
}

impl GroupDecodeCache {
    /// Create a new empty cache. Buffers are allocated lazily on first use.
    pub fn new() -> Self {
        Self {
            scratch: Vec::new(),
            coeffs_storage: Vec::new(),
            transform_buffer: [Vec::new(), Vec::new(), Vec::new()],
        }
    }

    /// Get a mutable reference to the scratch buffer, ensuring it has the required capacity.
    /// The buffer is zeroed before returning.
    #[inline]
    pub fn get_scratch(&mut self) -> &mut [f32] {
        if self.scratch.len() < LF_BUFFER_SIZE {
            self.scratch.resize(LF_BUFFER_SIZE, 0.0);
        } else {
            // Zero out existing buffer
            self.scratch[..LF_BUFFER_SIZE].fill(0.0);
        }
        &mut self.scratch[..LF_BUFFER_SIZE]
    }

    /// Get coefficient storage split into X, Y, B channels.
    /// The buffer is zeroed before returning.
    #[inline]
    pub fn get_coeffs(&mut self) -> [&mut [i32]; 3] {
        const COEFFS_SIZE: usize = 3 * GROUP_DIM * GROUP_DIM;
        if self.coeffs_storage.len() < COEFFS_SIZE {
            self.coeffs_storage.resize(COEFFS_SIZE, 0);
        } else {
            // Zero out existing buffer
            self.coeffs_storage[..COEFFS_SIZE].fill(0);
        }
        let (coeffs_x, coeffs_y_b) = self.coeffs_storage.split_at_mut(GROUP_DIM * GROUP_DIM);
        let (coeffs_y, coeffs_b) = coeffs_y_b.split_at_mut(GROUP_DIM * GROUP_DIM);
        [coeffs_x, coeffs_y, coeffs_b]
    }

    /// Get transform buffers for X, Y, B channels.
    /// The buffers are zeroed before returning.
    #[inline]
    pub fn get_transform_buffers(&mut self) -> &mut [Vec<f32>; 3] {
        for buf in &mut self.transform_buffer {
            if buf.len() < MAX_COEFF_AREA {
                buf.resize(MAX_COEFF_AREA, 0.0);
            } else {
                // Zero out existing buffer
                buf[..MAX_COEFF_AREA].fill(0.0);
            }
        }
        &mut self.transform_buffer
    }

    /// Get all buffers at once to avoid borrow checker issues.
    /// Returns (scratch, coeffs, transform_buffers).
    /// All buffers are zeroed before returning.
    #[inline]
    pub fn get_all_buffers(&mut self) -> (&mut [f32], [&mut [i32]; 3], &mut [Vec<f32>; 3]) {
        // Ensure scratch has capacity
        if self.scratch.len() < LF_BUFFER_SIZE {
            self.scratch.resize(LF_BUFFER_SIZE, 0.0);
        } else {
            self.scratch[..LF_BUFFER_SIZE].fill(0.0);
        }

        // Ensure coeffs has capacity
        const COEFFS_SIZE: usize = 3 * GROUP_DIM * GROUP_DIM;
        if self.coeffs_storage.len() < COEFFS_SIZE {
            self.coeffs_storage.resize(COEFFS_SIZE, 0);
        } else {
            self.coeffs_storage[..COEFFS_SIZE].fill(0);
        }

        // Ensure transform buffers have capacity
        for buf in &mut self.transform_buffer {
            if buf.len() < MAX_COEFF_AREA {
                buf.resize(MAX_COEFF_AREA, 0.0);
            } else {
                buf[..MAX_COEFF_AREA].fill(0.0);
            }
        }

        // Split coeffs into channels
        let (coeffs_x, coeffs_y_b) = self.coeffs_storage.split_at_mut(GROUP_DIM * GROUP_DIM);
        let (coeffs_y, coeffs_b) = coeffs_y_b.split_at_mut(GROUP_DIM * GROUP_DIM);

        (
            &mut self.scratch[..LF_BUFFER_SIZE],
            [coeffs_x, coeffs_y, coeffs_b],
            &mut self.transform_buffer,
        )
    }
}

impl Default for GroupDecodeCache {
    fn default() -> Self {
        Self::new()
    }
}
