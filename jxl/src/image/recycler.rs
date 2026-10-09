// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::collections::HashMap;

use crate::error::Result;
use crate::frame::group::VarDctBuffers;
use crate::frame::modular::ScratchSpace;
use crate::image::{Image, ImageDataType, OwnedRawImage};
use crate::util::sync::Mutex;
use crate::util::{PerThreadStorage, PerThreadStorageRef};

const MAX_RECYCLED_BUFFER_BYTES: usize = 1024 * 1024 * 4;

pub(crate) struct BufferRecycler {
    buckets: Mutex<HashMap<usize, Vec<OwnedRawImage>>>,
    modular_scratch: PerThreadStorage<ScratchSpace>,
    vardct_buffers: PerThreadStorage<VarDctBuffers>,
    lz77_windows: PerThreadStorage<Vec<u32>>,
}

impl Default for BufferRecycler {
    fn default() -> Self {
        Self::new()
    }
}

impl BufferRecycler {
    pub fn new() -> Self {
        Self {
            buckets: Mutex::new(HashMap::new()),
            modular_scratch: PerThreadStorage::new(ScratchSpace::new),
            vardct_buffers: PerThreadStorage::new(VarDctBuffers::new),
            lz77_windows: PerThreadStorage::new(Vec::new),
        }
    }

    pub(crate) fn get_modular_scratch(&self) -> PerThreadStorageRef<'_, ScratchSpace> {
        self.modular_scratch.get()
    }

    pub(crate) fn get_vardct_buffers(&self) -> PerThreadStorageRef<'_, VarDctBuffers> {
        self.vardct_buffers.get()
    }

    pub(crate) fn get_lz77_window(&self) -> PerThreadStorageRef<'_, Vec<u32>> {
        self.lz77_windows.get()
    }

    fn can_recycle(&self, alloc_size: usize) -> bool {
        alloc_size != 0 && alloc_size <= MAX_RECYCLED_BUFFER_BYTES
    }

    pub fn get_buffer<T: ImageDataType>(&self, size: (usize, usize)) -> Result<Image<T>> {
        self.get_raw_buffer((std::mem::size_of::<T>() * size.0, size.1), false)
            .map(Image::from_raw)
    }

    pub fn get_zeroed_buffer<T: ImageDataType>(&self, size: (usize, usize)) -> Result<Image<T>> {
        self.get_raw_buffer((std::mem::size_of::<T>() * size.0, size.1), true)
            .map(Image::from_raw)
    }

    pub fn get_raw_buffer(
        &self,
        byte_size: (usize, usize),
        zero_if_recycled: bool,
    ) -> Result<OwnedRawImage> {
        let Some(alloc_size) = OwnedRawImage::allocation_size(byte_size) else {
            return OwnedRawImage::new(byte_size);
        };
        if !self.can_recycle(alloc_size) {
            return OwnedRawImage::new(byte_size);
        }
        let popped = self
            .buckets
            .lock()
            .unwrap()
            .get_mut(&alloc_size)
            .and_then(Vec::pop);
        if let Some(mut img) = popped {
            img.reshape(byte_size);
            if zero_if_recycled {
                img.fill_zero();
            }
            Ok(img)
        } else {
            OwnedRawImage::new(byte_size)
        }
    }

    pub fn recycle_buffer<T: ImageDataType>(&self, buffer: Image<T>) {
        let buffer = buffer.into_raw();
        self.recycle_raw_buffer(buffer);
    }

    pub fn recycle_raw_buffer(&self, buffer: OwnedRawImage) {
        let alloc_size = buffer.owned_allocation_size();
        if !self.can_recycle(alloc_size) {
            return;
        }
        self.buckets
            .lock()
            .unwrap()
            .entry(alloc_size)
            .or_default()
            .push(buffer);
    }
}

impl std::fmt::Debug for BufferRecycler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BufferRecycler").finish()
    }
}
