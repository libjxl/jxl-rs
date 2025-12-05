// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::GROUP_DIM;
use crate::image::Image;
use jxl_transforms::transform_map::MAX_COEFF_AREA;

/// Size of the LF scratch buffer (32x32 block).
const LF_BUFFER_SIZE: usize = 32 * 32;

/// Maximum dimension for num_nzeros images (64 blocks = GROUP_DIM / 8).
const MAX_NUM_NZEROS_DIM: usize = GROUP_DIM / 8;

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
    /// Coefficient storage for X, Y, B channels (separate Vecs for borrow splitting)
    coeffs_x: Vec<i32>,
    coeffs_y: Vec<i32>,
    coeffs_b: Vec<i32>,
    /// Transform buffers for X, Y, B channels (3 * MAX_COEFF_AREA f32s each)
    transform_buffer: [Vec<f32>; 3],
    /// Cached num_nzeros images for VarDCT decoding (per channel).
    num_nzeros: Option<[Image<u32>; 3]>,
}

impl GroupDecodeCache {
    /// Create a new empty cache. Buffers are allocated lazily on first use.
    pub fn new() -> Self {
        Self {
            scratch: Vec::new(),
            coeffs_x: Vec::new(),
            coeffs_y: Vec::new(),
            coeffs_b: Vec::new(),
            transform_buffer: [Vec::new(), Vec::new(), Vec::new()],
            num_nzeros: None,
        }
    }

    /// Get or create num_nzeros images for VarDCT decoding.
    /// The returned images are zeroed and sized to cover the requested dimensions.
    #[inline]
    pub fn get_num_nzeros(
        &mut self,
        sizes: [(usize, usize); 3],
    ) -> crate::error::Result<&mut [Image<u32>; 3]> {
        let need_alloc = match &self.num_nzeros {
            None => true,
            Some(imgs) => {
                sizes[0].0 > imgs[0].size().0
                    || sizes[0].1 > imgs[0].size().1
                    || sizes[1].0 > imgs[1].size().0
                    || sizes[1].1 > imgs[1].size().1
                    || sizes[2].0 > imgs[2].size().0
                    || sizes[2].1 > imgs[2].size().1
            }
        };

        if need_alloc {
            let max_sizes = [
                (
                    sizes[0].0.max(MAX_NUM_NZEROS_DIM),
                    sizes[0].1.max(MAX_NUM_NZEROS_DIM),
                ),
                (
                    sizes[1].0.max(MAX_NUM_NZEROS_DIM),
                    sizes[1].1.max(MAX_NUM_NZEROS_DIM),
                ),
                (
                    sizes[2].0.max(MAX_NUM_NZEROS_DIM),
                    sizes[2].1.max(MAX_NUM_NZEROS_DIM),
                ),
            ];
            self.num_nzeros = Some([
                Image::new(max_sizes[0])?,
                Image::new(max_sizes[1])?,
                Image::new(max_sizes[2])?,
            ]);
        }

        // Zero out the images
        let imgs = self.num_nzeros.as_mut().unwrap();
        for img in imgs.iter_mut() {
            let (w, h) = img.size();
            for y in 0..h {
                let row = img.row_mut(y);
                row[..w].fill(0);
            }
        }

        Ok(imgs)
    }

    /// Get a mutable reference to the scratch buffer.
    #[inline]
    pub fn get_scratch(&mut self) -> &mut [f32] {
        if self.scratch.len() < LF_BUFFER_SIZE {
            self.scratch.resize(LF_BUFFER_SIZE, 0.0);
        } else {
            self.scratch[..LF_BUFFER_SIZE].fill(0.0);
        }
        &mut self.scratch[..LF_BUFFER_SIZE]
    }

    /// Get coefficient storage split into X, Y, B channels.
    #[inline]
    pub fn get_coeffs(&mut self) -> [&mut [i32]; 3] {
        const COEFFS_PER_CHANNEL: usize = GROUP_DIM * GROUP_DIM;
        for coeffs in [&mut self.coeffs_x, &mut self.coeffs_y, &mut self.coeffs_b] {
            if coeffs.len() < COEFFS_PER_CHANNEL {
                coeffs.resize(COEFFS_PER_CHANNEL, 0);
            } else {
                coeffs[..COEFFS_PER_CHANNEL].fill(0);
            }
        }
        [
            &mut self.coeffs_x[..COEFFS_PER_CHANNEL],
            &mut self.coeffs_y[..COEFFS_PER_CHANNEL],
            &mut self.coeffs_b[..COEFFS_PER_CHANNEL],
        ]
    }

    /// Get transform buffers for X, Y, B channels.
    #[inline]
    pub fn get_transform_buffers(&mut self) -> &mut [Vec<f32>; 3] {
        for buf in &mut self.transform_buffer {
            if buf.len() < MAX_COEFF_AREA {
                buf.resize(MAX_COEFF_AREA, 0.0);
            } else {
                buf[..MAX_COEFF_AREA].fill(0.0);
            }
        }
        &mut self.transform_buffer
    }
}

/// Bundled buffer references for VarDCT decoding.
/// This struct borrows from GroupDecodeCache and provides all buffers needed
/// for parallel-safe VarDCT group decoding.
pub struct VarDctBufferRefs<'a> {
    pub scratch: &'a mut [f32],
    pub coeffs_x: &'a mut [i32],
    pub coeffs_y: &'a mut [i32],
    pub coeffs_b: &'a mut [i32],
    pub transform_buffer: &'a mut [Vec<f32>; 3],
    pub num_nzeros: &'a mut [Image<u32>; 3],
}

impl GroupDecodeCache {
    /// Get all VarDCT decode buffers at once.
    /// Returns bundled buffer references for use in parallel decoding.
    #[inline]
    pub fn get_vardct_buffers(
        &mut self,
        nzeros_sizes: [(usize, usize); 3],
    ) -> crate::error::Result<VarDctBufferRefs<'_>> {
        const COEFFS_PER_CHANNEL: usize = GROUP_DIM * GROUP_DIM;

        // Ensure scratch buffer is allocated
        if self.scratch.len() < LF_BUFFER_SIZE {
            self.scratch.resize(LF_BUFFER_SIZE, 0.0);
        }
        self.scratch[..LF_BUFFER_SIZE].fill(0.0);

        // Ensure coefficients are allocated
        for coeffs in [&mut self.coeffs_x, &mut self.coeffs_y, &mut self.coeffs_b] {
            if coeffs.len() < COEFFS_PER_CHANNEL {
                coeffs.resize(COEFFS_PER_CHANNEL, 0);
            } else {
                coeffs[..COEFFS_PER_CHANNEL].fill(0);
            }
        }

        // Ensure transform buffers are allocated
        for buf in &mut self.transform_buffer {
            if buf.len() < MAX_COEFF_AREA {
                buf.resize(MAX_COEFF_AREA, 0.0);
            } else {
                buf[..MAX_COEFF_AREA].fill(0.0);
            }
        }

        // Ensure num_nzeros images are allocated
        let need_alloc = match &self.num_nzeros {
            None => true,
            Some(imgs) => {
                nzeros_sizes[0].0 > imgs[0].size().0
                    || nzeros_sizes[0].1 > imgs[0].size().1
                    || nzeros_sizes[1].0 > imgs[1].size().0
                    || nzeros_sizes[1].1 > imgs[1].size().1
                    || nzeros_sizes[2].0 > imgs[2].size().0
                    || nzeros_sizes[2].1 > imgs[2].size().1
            }
        };

        if need_alloc {
            let max_sizes = [
                (
                    nzeros_sizes[0].0.max(MAX_NUM_NZEROS_DIM),
                    nzeros_sizes[0].1.max(MAX_NUM_NZEROS_DIM),
                ),
                (
                    nzeros_sizes[1].0.max(MAX_NUM_NZEROS_DIM),
                    nzeros_sizes[1].1.max(MAX_NUM_NZEROS_DIM),
                ),
                (
                    nzeros_sizes[2].0.max(MAX_NUM_NZEROS_DIM),
                    nzeros_sizes[2].1.max(MAX_NUM_NZEROS_DIM),
                ),
            ];
            self.num_nzeros = Some([
                Image::new(max_sizes[0])?,
                Image::new(max_sizes[1])?,
                Image::new(max_sizes[2])?,
            ]);
        }

        // Zero out num_nzeros images
        let imgs = self.num_nzeros.as_mut().unwrap();
        for img in imgs.iter_mut() {
            let (w, h) = img.size();
            for y in 0..h {
                let row = img.row_mut(y);
                row[..w].fill(0);
            }
        }

        Ok(VarDctBufferRefs {
            scratch: &mut self.scratch[..LF_BUFFER_SIZE],
            coeffs_x: &mut self.coeffs_x[..COEFFS_PER_CHANNEL],
            coeffs_y: &mut self.coeffs_y[..COEFFS_PER_CHANNEL],
            coeffs_b: &mut self.coeffs_b[..COEFFS_PER_CHANNEL],
            transform_buffer: &mut self.transform_buffer,
            num_nzeros: self.num_nzeros.as_mut().unwrap(),
        })
    }
}

impl Default for GroupDecodeCache {
    fn default() -> Self {
        Self::new()
    }
}
