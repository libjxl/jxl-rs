// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::alloc::Layout;
use std::sync::Mutex;

use super::ImageDataType;
use super::internal::{BufferInitialization, RawImageBuffer};
use super::raw::OwnedRawImage;
use super::typed::Image;
use crate::error::Result;
use crate::util::CACHE_LINE_BYTE_SIZE;

pub const MIN_BUCKET_SIZE: usize = 1024;
pub const MAX_BUCKET_SIZE: usize = 4 * 1024 * 1024;
pub const NUM_BUCKETS: usize = 25;

/// Maps a requested byte count to `Some((bucket_index, bucket_capacity))` if
/// within the recyclable range [1KB, 4MB], or `None` if unbucketed.
#[inline(always)]
pub fn bucket_index_and_size(bytes: usize) -> Option<(usize, usize)> {
    if !(MIN_BUCKET_SIZE..=MAX_BUCKET_SIZE).contains(&bytes) {
        return None;
    }
    let x = (bytes - 1).ilog2() as usize; // 9 <= x <= 21
    if x == 9 {
        return Some((0, 1024));
    }
    let p = 1 << x;
    let mid = p + (p >> 1);
    if bytes <= mid {
        Some(((x - 10) * 2 + 1, mid))
    } else {
        Some(((x - 10) * 2 + 2, p << 1))
    }
}

pub(crate) struct RawBuffer {
    pub(crate) buf: *mut u8,
    pub(crate) capacity: usize,
}

// SAFETY: RawBuffer owns its raw memory buffer uniquely.
unsafe impl Send for RawBuffer {}
// SAFETY: RawBuffer contains no interior mutability and is guarded by locks when shared in DecoderBufferPool.
unsafe impl Sync for RawBuffer {}

/// Per-decoder shared buffer reservoir.
pub struct DecoderBufferPool {
    buckets: [Mutex<Vec<RawBuffer>>; NUM_BUCKETS],
    limit_per_bucket: Mutex<Option<usize>>,
}

impl std::fmt::Debug for DecoderBufferPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecoderBufferPool")
            .field("limit_per_bucket", &self.limit_per_bucket)
            .finish()
    }
}

impl DecoderBufferPool {
    pub fn new(limit_per_bucket: Option<usize>) -> Self {
        Self {
            buckets: std::array::from_fn(|_| Mutex::new(vec![])),
            limit_per_bucket: Mutex::new(limit_per_bucket),
        }
    }

    pub fn set_limit_per_bucket(&self, limit: Option<usize>) {
        *self.limit_per_bucket.lock().unwrap() = limit;
    }

    pub(crate) fn pop_raw_parts(&self, bucket_idx: usize) -> Option<RawBuffer> {
        self.buckets[bucket_idx].lock().unwrap().pop()
    }

    pub(crate) fn push_raw_parts(&self, buf: *mut u8, capacity: usize) {
        if let Some((bucket_idx, _)) = bucket_index_and_size(capacity) {
            let limit = *self.limit_per_bucket.lock().unwrap();
            let mut bucket = self.buckets[bucket_idx].lock().unwrap();
            if limit.is_none_or(|lim| bucket.len() < lim) {
                bucket.push(RawBuffer { buf, capacity });
                return;
            }
        }
        // SAFETY: `buf` was allocated with the layout corresponding to `capacity` and `CACHE_LINE_BYTE_SIZE`.
        unsafe {
            if let Ok(layout) = Layout::from_size_align(capacity, CACHE_LINE_BYTE_SIZE) {
                std::alloc::dealloc(buf, layout);
            }
        }
    }

    pub fn set_limit(&self, limit: Option<usize>) {
        *self.limit_per_bucket.lock().unwrap() = limit;
    }

    pub fn clear(&self) {
        for bucket in &self.buckets {
            let mut b = bucket.lock().unwrap();
            for raw in b.drain(..) {
                // SAFETY: `raw.buf` was allocated with the layout corresponding to `raw.capacity` and `CACHE_LINE_BYTE_SIZE`.
                unsafe {
                    if let Ok(layout) = Layout::from_size_align(raw.capacity, CACHE_LINE_BYTE_SIZE)
                    {
                        std::alloc::dealloc(raw.buf, layout);
                    }
                }
            }
        }
    }
}

impl Drop for DecoderBufferPool {
    fn drop(&mut self) {
        self.clear();
    }
}

/// Task-scoped thread recycler that manages lock-free allocation and recycling,
/// flushing all retained buffers to `DecoderBufferPool` on drop.
pub struct LocalBufferRecycler<'a> {
    pool: &'a DecoderBufferPool,
    local_buckets: [Vec<RawBuffer>; NUM_BUCKETS],
}

impl<'a> LocalBufferRecycler<'a> {
    pub fn new(pool: &'a DecoderBufferPool) -> Self {
        Self {
            pool,
            local_buckets: std::array::from_fn(|_| vec![]),
        }
    }

    pub fn alloc_raw(
        &mut self,
        byte_size: (usize, usize),
        require_zero: bool,
    ) -> Result<OwnedRawImage> {
        let (bytes_per_row, num_rows) = byte_size;
        let bytes_between_rows =
            bytes_per_row.div_ceil(CACHE_LINE_BYTE_SIZE) * CACHE_LINE_BYTE_SIZE;
        let req_len = (num_rows.saturating_sub(1)) * bytes_between_rows + bytes_per_row;

        if let Some((bucket_idx, bucket_cap)) = bucket_index_and_size(req_len) {
            // 1. Try local cache
            if let Some(raw) = self.local_buckets[bucket_idx].pop() {
                let init = if require_zero {
                    BufferInitialization::Zeroed
                } else {
                    BufferInitialization::Undefined
                };
                // SAFETY: `raw.buf` is valid for `raw.capacity >= req_len` bytes aligned to cache line.
                return unsafe {
                    RawImageBuffer::from_recycled(raw.buf, byte_size, raw.capacity, init)
                        .map(OwnedRawImage::from_raw_buffer)
                };
            }
            // 2. Try pool
            if let Some(raw) = self.pool.pop_raw_parts(bucket_idx) {
                let init = if require_zero {
                    BufferInitialization::Zeroed
                } else {
                    BufferInitialization::Undefined
                };
                // SAFETY: `raw.buf` is valid for `raw.capacity >= req_len` bytes aligned to cache line.
                return unsafe {
                    RawImageBuffer::from_recycled(raw.buf, byte_size, raw.capacity, init)
                        .map(OwnedRawImage::from_raw_buffer)
                };
            }

            // 3. Fall back to fresh bucket-sized allocation
            let layout = Layout::from_size_align(bucket_cap, CACHE_LINE_BYTE_SIZE).unwrap();
            let memory = if require_zero {
                // SAFETY: `layout` has non-zero size aligned to CACHE_LINE_BYTE_SIZE.
                unsafe { std::alloc::alloc_zeroed(layout) }
            } else {
                // SAFETY: `layout` has non-zero size aligned to CACHE_LINE_BYTE_SIZE.
                unsafe { std::alloc::alloc(layout) }
            };
            if memory.is_null() {
                return Err(crate::error::Error::ImageOutOfMemory(
                    bytes_per_row,
                    num_rows,
                ));
            }
            // SAFETY: `memory` was allocated with `layout` and satisfies validity requirements.
            let raw_img = unsafe {
                RawImageBuffer::new_from_ptr(
                    memory,
                    num_rows,
                    bytes_per_row,
                    bytes_between_rows,
                    bucket_cap,
                )
            };
            return Ok(OwnedRawImage::from_raw_buffer(raw_img));
        }

        // 4. Fall back to unbucketed allocation
        OwnedRawImage::new(byte_size, require_zero)
    }

    pub fn alloc_image<T: ImageDataType>(&mut self, size: (usize, usize)) -> Result<Image<T>> {
        let s = T::DATA_TYPE_ID.size();
        let raw = self.alloc_raw((size.0 * s, size.1), false)?;
        Ok(Image::from_raw(raw))
    }

    pub fn alloc_image_zeroed<T: ImageDataType>(
        &mut self,
        size: (usize, usize),
    ) -> Result<Image<T>> {
        let s = T::DATA_TYPE_ID.size();
        let raw = self.alloc_raw((size.0 * s, size.1), true)?;
        Ok(Image::from_raw(raw))
    }

    /// Moves an OwnedRawImage into the local recycler. Subsumes drop.
    pub fn recycle_raw(&mut self, image: OwnedRawImage) {
        let (buf, capacity) = image.into_raw_parts();
        if buf.is_null() || capacity == 0 {
            return;
        }
        if let Some((bucket_idx, bucket_cap)) = bucket_index_and_size(capacity)
            && capacity == bucket_cap
        {
            self.local_buckets[bucket_idx].push(RawBuffer { buf, capacity });
            return;
        }
        // SAFETY: `buf` was allocated with layout corresponding to `capacity` and `CACHE_LINE_BYTE_SIZE`.
        unsafe {
            if let Ok(layout) = Layout::from_size_align(capacity, CACHE_LINE_BYTE_SIZE) {
                std::alloc::dealloc(buf, layout);
            }
        }
    }

    pub fn recycle_image<T: ImageDataType>(&mut self, image: Image<T>) {
        self.recycle_raw(image.into_raw());
    }
}

impl Drop for LocalBufferRecycler<'_> {
    fn drop(&mut self) {
        for bucket in &mut self.local_buckets {
            for raw in bucket.drain(..) {
                self.pool.push_raw_parts(raw.buf, raw.capacity);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bucket_mapping() {
        assert_eq!(bucket_index_and_size(1023), None);
        assert_eq!(bucket_index_and_size(1024), Some((0, 1024)));
        assert_eq!(bucket_index_and_size(1025), Some((1, 1536)));
        assert_eq!(bucket_index_and_size(1536), Some((1, 1536)));
        assert_eq!(bucket_index_and_size(1537), Some((2, 2048)));
        assert_eq!(bucket_index_and_size(2048), Some((2, 2048)));
        assert_eq!(bucket_index_and_size(2049), Some((3, 3072)));
        assert_eq!(bucket_index_and_size(3072), Some((3, 3072)));
        assert_eq!(bucket_index_and_size(3073), Some((4, 4096)));
        assert_eq!(
            bucket_index_and_size(4 * 1024 * 1024),
            Some((24, 4 * 1024 * 1024))
        );
        assert_eq!(bucket_index_and_size(4 * 1024 * 1024 + 1), None);
    }

    #[test]
    fn test_local_recycler_and_pool_flush() {
        let pool = DecoderBufferPool::new(Some(2));

        // Task 1: allocate, write, recycle, drop local recycler
        {
            let mut recycler = LocalBufferRecycler::new(&pool);
            let mut img: Image<f32> = recycler.alloc_image((16, 16)).unwrap();
            img.fill(42.0);
            recycler.recycle_image(img);
        }

        // Buffer was flushed to pool
        assert_eq!(pool.buckets[0].lock().unwrap().len(), 1);

        // Task 2: allocate from pool
        {
            let mut recycler = LocalBufferRecycler::new(&pool);
            let img: Image<f32> = recycler.alloc_image((16, 16)).unwrap();
            // Local pop drained the pool
            assert_eq!(pool.buckets[0].lock().unwrap().len(), 0);
            recycler.recycle_image(img);
        }

        assert_eq!(pool.buckets[0].lock().unwrap().len(), 1);
    }

    #[test]
    fn test_limit_discard() {
        let pool = DecoderBufferPool::new(Some(1));

        {
            let mut recycler = LocalBufferRecycler::new(&pool);
            let img1: Image<f32> = recycler.alloc_image((16, 16)).unwrap();
            let img2: Image<f32> = recycler.alloc_image((16, 16)).unwrap();
            recycler.recycle_image(img1);
            recycler.recycle_image(img2);
        }

        // Limit was 1, so only 1 buffer retained in pool, 1 was discarded
        assert_eq!(pool.buckets[0].lock().unwrap().len(), 1);
    }

    #[test]
    fn test_alloc_image_zeroed() {
        let pool = DecoderBufferPool::new(Some(2));

        {
            let mut recycler = LocalBufferRecycler::new(&pool);
            let mut img: Image<u32> = recycler.alloc_image((16, 16)).unwrap();
            img.fill(0xDEADBEEF);
            recycler.recycle_image(img);
        }

        {
            let mut recycler = LocalBufferRecycler::new(&pool);
            let zeroed: Image<u32> = recycler.alloc_image_zeroed((16, 16)).unwrap();
            for y in 0..16 {
                for x in 0..16 {
                    assert_eq!(zeroed.row(y)[x], 0);
                }
            }
        }
    }
}
