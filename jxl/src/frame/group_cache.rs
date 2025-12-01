// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/// Per-thread cache for parallel VarDCT group decoding.
///
/// This cache stores reusable coefficient buffers to avoid allocations
/// during parallel decoding. Each thread has its own GroupDecodeCache
/// to eliminate contention.
pub struct GroupDecodeCache {
    /// Coefficient storage for X, Y, B channels
    pub coeffs: [Vec<i32>; 3],
}

impl GroupDecodeCache {
    /// Create a new empty cache
    pub fn new() -> Self {
        Self {
            coeffs: [Vec::new(), Vec::new(), Vec::new()],
        }
    }
}

impl Default for GroupDecodeCache {
    fn default() -> Self {
        Self::new()
    }
}
