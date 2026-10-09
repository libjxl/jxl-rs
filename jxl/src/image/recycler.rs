// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::error::Result;
use crate::image::{Image, ImageDataType, OwnedRawImage};
use crate::util::sync::Mutex;
use crate::util::sync::atomic::{AtomicUsize, Ordering};

const NUM_SHARDS: usize = 16;

type BucketList = Vec<((usize, usize), Vec<OwnedRawImage>)>;

#[repr(align(64))]
struct RecyclerShard {
    count: AtomicUsize,
    buckets: Mutex<BucketList>,
}

impl RecyclerShard {
    const fn new() -> Self {
        Self {
            count: AtomicUsize::new(0),
            buckets: Mutex::new(Vec::new()),
        }
    }
}

pub struct BufferRecycler {
    max_total_bytes: usize,
    shards: [RecyclerShard; NUM_SHARDS],
}

impl BufferRecycler {
    pub fn new(group_dim: usize) -> Self {
        Self {
            max_total_bytes: group_dim * group_dim * 4,
            shards: [const { RecyclerShard::new() }; NUM_SHARDS],
        }
    }

    #[inline(always)]
    fn can_recycle(&self, byte_size: (usize, usize)) -> bool {
        byte_size.0 != 0
            && byte_size.1 != 0
            && byte_size.0.saturating_mul(byte_size.1) <= self.max_total_bytes
    }

    #[inline(always)]
    fn shard_hint() -> usize {
        let stack_var = 0usize;
        let addr = std::ptr::addr_of!(stack_var) as usize;
        (addr >> 13) ^ (addr >> 17) ^ (addr >> 21)
    }

    #[inline(always)]
    fn pop_from_buckets(
        buckets: &mut [((usize, usize), Vec<OwnedRawImage>)],
        byte_size: (usize, usize),
    ) -> Option<OwnedRawImage> {
        for (sz, vec) in buckets.iter_mut() {
            if *sz == byte_size {
                return vec.pop();
            }
        }
        None
    }

    #[inline(always)]
    fn push_to_buckets(
        buckets: &mut Vec<((usize, usize), Vec<OwnedRawImage>)>,
        byte_size: (usize, usize),
        buffer: OwnedRawImage,
    ) {
        for (sz, vec) in buckets.iter_mut() {
            if *sz == byte_size {
                vec.push(buffer);
                return;
            }
        }
        buckets.push((byte_size, vec![buffer]));
    }

    pub fn get_buffer<T: ImageDataType>(&self, size: (usize, usize)) -> Result<Image<T>> {
        self.get_raw_buffer((std::mem::size_of::<T>() * size.0, size.1), false)
            .map(Image::from_raw)
    }

    pub fn get_raw_buffer(
        &self,
        byte_size: (usize, usize),
        zero_if_recycled: bool,
    ) -> Result<OwnedRawImage> {
        if !self.can_recycle(byte_size) {
            return OwnedRawImage::new(byte_size);
        }
        let start = Self::shard_hint() & (NUM_SHARDS - 1);
        for i in 0..NUM_SHARDS {
            let shard = &self.shards[(start + i) & (NUM_SHARDS - 1)];
            if shard.count.load(Ordering::Relaxed) == 0 {
                continue;
            }
            let popped = {
                let mut buckets = shard.buckets.lock().unwrap();
                Self::pop_from_buckets(&mut buckets, byte_size)
            };
            if let Some(mut img) = popped {
                shard.count.fetch_sub(1, Ordering::Relaxed);
                if zero_if_recycled {
                    img.fill_zero();
                }
                return Ok(img);
            }
        }
        OwnedRawImage::new(byte_size)
    }

    pub fn recycle_buffer<T: ImageDataType>(&self, buffer: Image<T>) {
        let buffer = buffer.into_raw();
        self.recycle_raw_buffer(buffer);
    }

    pub fn recycle_raw_buffer(&self, buffer: OwnedRawImage) {
        let byte_size = buffer.byte_size();
        if !self.can_recycle(byte_size) {
            return;
        }
        let start = Self::shard_hint() & (NUM_SHARDS - 1);
        for i in 0..NUM_SHARDS {
            let shard = &self.shards[(start + i) & (NUM_SHARDS - 1)];
            if let Ok(mut buckets) = shard.buckets.try_lock() {
                Self::push_to_buckets(&mut buckets, byte_size, buffer);
                shard.count.fetch_add(1, Ordering::Relaxed);
                return;
            }
        }
        let shard = &self.shards[start];
        let mut buckets = shard.buckets.lock().unwrap();
        Self::push_to_buckets(&mut buckets, byte_size, buffer);
        shard.count.fetch_add(1, Ordering::Relaxed);
    }
}

impl std::fmt::Debug for BufferRecycler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BufferRecycler").finish()
    }
}
