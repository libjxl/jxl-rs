// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::image::Image;
use crate::GROUP_DIM;
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
    /// Stored with maximum size to allow reuse across different group sizes.
    num_nzeros: Option<[Image<u32>; 3]>,
    /// Cached pixel output buffers for group rendering.
    /// Size is (GROUP_DIM + padding) per dimension.
    pixel_buffers: Option<[Image<f32>; 3]>,
    /// Size of the cached pixel buffers.
    pixel_buffer_size: (usize, usize),
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
            pixel_buffers: None,
            pixel_buffer_size: (0, 0),
        }
    }

    /// Get or create num_nzeros images for VarDCT decoding.
    /// The returned images are zeroed and sized to cover the requested dimensions.
    /// Sizes are provided as (width, height) for each of the 3 channels.
    #[inline]
    pub fn get_num_nzeros(
        &mut self,
        sizes: [(usize, usize); 3],
    ) -> crate::error::Result<&mut [Image<u32>; 3]> {
        // Check if we need to allocate or if existing images are large enough
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
            // Allocate with maximum size to handle future requests
            let max_sizes = [
                (sizes[0].0.max(MAX_NUM_NZEROS_DIM), sizes[0].1.max(MAX_NUM_NZEROS_DIM)),
                (sizes[1].0.max(MAX_NUM_NZEROS_DIM), sizes[1].1.max(MAX_NUM_NZEROS_DIM)),
                (sizes[2].0.max(MAX_NUM_NZEROS_DIM), sizes[2].1.max(MAX_NUM_NZEROS_DIM)),
            ];
            self.num_nzeros = Some([
                Image::new(max_sizes[0])?,
                Image::new(max_sizes[1])?,
                Image::new(max_sizes[2])?,
            ]);
        }

        // Zero out the entire image to avoid stale data from previous frames
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

    /// Get or create pixel output buffers for group rendering.
    /// The returned images are sized to match the requested size.
    #[inline]
    pub fn get_pixel_buffers(
        &mut self,
        size: (usize, usize),
    ) -> crate::error::Result<&mut [Image<f32>; 3]> {
        // Check if we need to allocate
        let need_alloc = match &self.pixel_buffers {
            None => true,
            Some(_) => size.0 > self.pixel_buffer_size.0 || size.1 > self.pixel_buffer_size.1,
        };

        if need_alloc {
            self.pixel_buffers = Some([
                Image::new(size)?,
                Image::new(size)?,
                Image::new(size)?,
            ]);
            self.pixel_buffer_size = size;
        }

        Ok(self.pixel_buffers.as_mut().unwrap())
    }

    /// Take ownership of the pixel buffers (for passing to pipeline).
    /// Returns None if not allocated.
    #[inline]
    pub fn take_pixel_buffers(&mut self) -> Option<[Image<f32>; 3]> {
        self.pixel_buffer_size = (0, 0);
        self.pixel_buffers.take()
    }

    /// Initialize num_nzeros buffers for VarDCT decode, allocating if needed.
    /// Must be called before get_all_buffers() since they can't share the borrow.
    /// Returns the required sizes for use later.
    #[inline]
    pub fn init_num_nzeros(&mut self, sizes: [(usize, usize); 3]) -> crate::error::Result<()> {
        // Check if we need to allocate or if existing images are large enough
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
            // Allocate with maximum size to handle future requests
            let max_sizes = [
                (sizes[0].0.max(MAX_NUM_NZEROS_DIM), sizes[0].1.max(MAX_NUM_NZEROS_DIM)),
                (sizes[1].0.max(MAX_NUM_NZEROS_DIM), sizes[1].1.max(MAX_NUM_NZEROS_DIM)),
                (sizes[2].0.max(MAX_NUM_NZEROS_DIM), sizes[2].1.max(MAX_NUM_NZEROS_DIM)),
            ];
            self.num_nzeros = Some([
                Image::new(max_sizes[0])?,
                Image::new(max_sizes[1])?,
                Image::new(max_sizes[2])?,
            ]);
        }

        // Zero out the entire image to avoid stale data from previous frames
        let imgs = self.num_nzeros.as_mut().unwrap();
        for img in imgs.iter_mut() {
            let (w, h) = img.size();
            for y in 0..h {
                let row = img.row_mut(y);
                row[..w].fill(0);
            }
        }

        Ok(())
    }

    /// Get a reference to the num_nzeros images (must call init_num_nzeros first).
    #[inline]
    pub fn num_nzeros(&mut self) -> &mut [Image<u32>; 3] {
        self.num_nzeros.as_mut().expect("init_num_nzeros must be called first")
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
        const COEFFS_PER_CHANNEL: usize = GROUP_DIM * GROUP_DIM;

        // Ensure scratch has capacity
        if self.scratch.len() < LF_BUFFER_SIZE {
            self.scratch.resize(LF_BUFFER_SIZE, 0.0);
        } else {
            self.scratch[..LF_BUFFER_SIZE].fill(0.0);
        }

        // Ensure coeffs have capacity (separate Vecs)
        for coeffs in [&mut self.coeffs_x, &mut self.coeffs_y, &mut self.coeffs_b] {
            if coeffs.len() < COEFFS_PER_CHANNEL {
                coeffs.resize(COEFFS_PER_CHANNEL, 0);
            } else {
                coeffs[..COEFFS_PER_CHANNEL].fill(0);
            }
        }

        // Ensure transform buffers have capacity
        for buf in &mut self.transform_buffer {
            if buf.len() < MAX_COEFF_AREA {
                buf.resize(MAX_COEFF_AREA, 0.0);
            } else {
                buf[..MAX_COEFF_AREA].fill(0.0);
            }
        }

        (
            &mut self.scratch[..LF_BUFFER_SIZE],
            [
                &mut self.coeffs_x[..COEFFS_PER_CHANNEL],
                &mut self.coeffs_y[..COEFFS_PER_CHANNEL],
                &mut self.coeffs_b[..COEFFS_PER_CHANNEL],
            ],
            &mut self.transform_buffer,
        )
    }

    /// Get all VarDCT buffers at once including num_nzeros.
    /// This is the primary method for VarDCT decoding - returns all needed buffers.
    /// All buffers are allocated if needed and zeroed before returning.
    #[inline]
    pub fn get_vardct_buffers(
        &mut self,
        nzeros_sizes: [(usize, usize); 3],
    ) -> crate::error::Result<VardctBuffers<'_>> {
        const COEFFS_PER_CHANNEL: usize = GROUP_DIM * GROUP_DIM;

        // Initialize num_nzeros (allocates if needed)
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

        // Zero out the entire num_nzeros images to avoid stale data from previous frames
        let nzeros = self.num_nzeros.as_mut().unwrap();
        for img in nzeros.iter_mut() {
            let (w, h) = img.size();
            for y in 0..h {
                let row = img.row_mut(y);
                row[..w].fill(0);
            }
        }

        // Ensure scratch has capacity
        if self.scratch.len() < LF_BUFFER_SIZE {
            self.scratch.resize(LF_BUFFER_SIZE, 0.0);
        } else {
            self.scratch[..LF_BUFFER_SIZE].fill(0.0);
        }

        // Ensure coeffs have capacity (separate Vecs for each channel)
        for coeffs in [&mut self.coeffs_x, &mut self.coeffs_y, &mut self.coeffs_b] {
            if coeffs.len() < COEFFS_PER_CHANNEL {
                coeffs.resize(COEFFS_PER_CHANNEL, 0);
            } else {
                coeffs[..COEFFS_PER_CHANNEL].fill(0);
            }
        }

        // Ensure transform buffers have capacity
        for buf in &mut self.transform_buffer {
            if buf.len() < MAX_COEFF_AREA {
                buf.resize(MAX_COEFF_AREA, 0.0);
            } else {
                buf[..MAX_COEFF_AREA].fill(0.0);
            }
        }

        Ok(VardctBuffers {
            scratch: &mut self.scratch[..LF_BUFFER_SIZE],
            coeffs_x: &mut self.coeffs_x[..COEFFS_PER_CHANNEL],
            coeffs_y: &mut self.coeffs_y[..COEFFS_PER_CHANNEL],
            coeffs_b: &mut self.coeffs_b[..COEFFS_PER_CHANNEL],
            transform_buffer: &mut self.transform_buffer,
            num_nzeros: self.num_nzeros.as_mut().unwrap(),
        })
    }
}

/// Struct holding all VarDCT decode buffers with proper lifetime.
/// This allows returning multiple mutable references from the cache.
pub struct VardctBuffers<'a> {
    pub scratch: &'a mut [f32],
    pub coeffs_x: &'a mut [i32],
    pub coeffs_y: &'a mut [i32],
    pub coeffs_b: &'a mut [i32],
    pub transform_buffer: &'a mut [Vec<f32>; 3],
    pub num_nzeros: &'a mut [Image<u32>; 3],
}

impl Default for GroupDecodeCache {
    fn default() -> Self {
        Self::new()
    }
}
