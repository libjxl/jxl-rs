// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::image::OwnedRawImage;
use crate::util::sync::Mutex;

pub const KIND_GROUP: usize = 0;
pub const KIND_TOPBOTTOM: usize = 1;
pub const KIND_LEFTRIGHT: usize = 2;

pub const BUFFER_KIND_GROUP: usize = KIND_GROUP;
pub const BUFFER_KIND_TOPBOTTOM: usize = KIND_TOPBOTTOM;
pub const BUFFER_KIND_LEFTRIGHT: usize = KIND_LEFTRIGHT;

pub const NUM_KINDS: usize = 3;
pub const NUM_GROUP_SHIFTS: usize = 9;
pub const NUM_TOPBOTTOM_SHIFTS: usize = 3;
pub const NUM_LEFTRIGHT_SHIFTS: usize = 3;
pub const NUM_BUCKETS_PER_TYPE_SIZE: usize =
    NUM_GROUP_SHIFTS + NUM_TOPBOTTOM_SHIFTS + NUM_LEFTRIGHT_SHIFTS;
pub const NUM_SUPPORTED_TYPE_SIZES: usize = 2;
pub const NUM_BUCKETS: usize = NUM_BUCKETS_PER_TYPE_SIZE * NUM_SUPPORTED_TYPE_SIZES;

pub fn group_shift_to_bucket(shift: (usize, usize)) -> Option<usize> {
    if shift.0 <= 2 && shift.1 <= 2 {
        Some(shift.0 * 3 + shift.1)
    } else {
        None
    }
}

pub fn topbottom_shift_to_bucket(shift: usize) -> Option<usize> {
    if shift <= 2 {
        Some(shift)
    } else {
        None
    }
}

pub fn leftright_shift_to_bucket(shift: usize) -> Option<usize> {
    if shift <= 2 {
        Some(shift)
    } else {
        None
    }
}

pub fn data_type_size_to_bucket(type_size: usize) -> Option<usize> {
    match type_size {
        2 => Some(0),
        4 => Some(1),
        _ => None,
    }
}

pub fn bucket_id(kind: usize, shift_bucket: usize, type_size: usize) -> Option<usize> {
    let type_bucket = data_type_size_to_bucket(type_size)?;
    let (kind_offset, max_shift) = match kind {
        KIND_GROUP => (0, NUM_GROUP_SHIFTS),
        KIND_TOPBOTTOM => (NUM_GROUP_SHIFTS, NUM_TOPBOTTOM_SHIFTS),
        KIND_LEFTRIGHT => (NUM_GROUP_SHIFTS + NUM_TOPBOTTOM_SHIFTS, NUM_LEFTRIGHT_SHIFTS),
        _ => return None,
    };
    if shift_bucket < max_shift {
        Some(type_bucket * NUM_BUCKETS_PER_TYPE_SIZE + kind_offset + shift_bucket)
    } else {
        None
    }
}

pub struct BufferRecycler {
    buckets: [Mutex<Vec<OwnedRawImage>>; NUM_BUCKETS],
}

impl BufferRecycler {
    pub fn new() -> Self {
        Self {
            buckets: std::array::from_fn(|_| Mutex::new(Vec::new())),
        }
    }

    pub fn get_buffer_by_bucket(&self, bucket_id: usize) -> Option<OwnedRawImage> {
        if bucket_id < NUM_BUCKETS {
            self.buckets[bucket_id].try_lock().ok()?.pop()
        } else {
            None
        }
    }

    pub fn recycle_buffer_by_bucket(&self, bucket_id: usize, buffer: OwnedRawImage) {
        if bucket_id < NUM_BUCKETS {
            if let Ok(mut bucket) = self.buckets[bucket_id].try_lock() {
                bucket.push(buffer);
            }
        }
    }

    pub fn maybe_get_buffer(
        &self,
        kind: usize,
        shift_bucket: usize,
        type_size: usize,
    ) -> Option<OwnedRawImage> {
        let id = bucket_id(kind, shift_bucket, type_size)?;
        self.get_buffer_by_bucket(id)
    }

    pub fn store_buffer(
        &self,
        kind: usize,
        shift_bucket: usize,
        type_size: usize,
        buffer: OwnedRawImage,
    ) {
        if let Some(id) = bucket_id(kind, shift_bucket, type_size) {
            self.recycle_buffer_by_bucket(id, buffer);
        }
    }

    pub fn get_group_buffer(
        &self,
        shift: (usize, usize),
        type_size: usize,
    ) -> Option<OwnedRawImage> {
        let shift_bucket = group_shift_to_bucket(shift)?;
        self.maybe_get_buffer(KIND_GROUP, shift_bucket, type_size)
    }

    pub fn recycle_group_buffer(
        &self,
        shift: (usize, usize),
        type_size: usize,
        buffer: OwnedRawImage,
    ) {
        if let Some(shift_bucket) = group_shift_to_bucket(shift) {
            self.store_buffer(KIND_GROUP, shift_bucket, type_size, buffer);
        }
    }

    pub fn get_topbottom_buffer(&self, shift: usize, type_size: usize) -> Option<OwnedRawImage> {
        let shift_bucket = topbottom_shift_to_bucket(shift)?;
        self.maybe_get_buffer(KIND_TOPBOTTOM, shift_bucket, type_size)
    }

    pub fn recycle_topbottom_buffer(&self, shift: usize, type_size: usize, buffer: OwnedRawImage) {
        if let Some(shift_bucket) = topbottom_shift_to_bucket(shift) {
            self.store_buffer(KIND_TOPBOTTOM, shift_bucket, type_size, buffer);
        }
    }

    pub fn get_leftright_buffer(&self, shift: usize, type_size: usize) -> Option<OwnedRawImage> {
        let shift_bucket = leftright_shift_to_bucket(shift)?;
        self.maybe_get_buffer(KIND_LEFTRIGHT, shift_bucket, type_size)
    }

    pub fn recycle_leftright_buffer(&self, shift: usize, type_size: usize, buffer: OwnedRawImage) {
        if let Some(shift_bucket) = leftright_shift_to_bucket(shift) {
            self.store_buffer(KIND_LEFTRIGHT, shift_bucket, type_size, buffer);
        }
    }
}

impl Default for BufferRecycler {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for BufferRecycler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BufferRecycler").finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_group_shift_to_bucket() {
        for sx in 0..=2 {
            for sy in 0..=2 {
                assert_eq!(group_shift_to_bucket((sx, sy)), Some(sx * 3 + sy));
            }
        }
        assert_eq!(group_shift_to_bucket((3, 0)), None);
        assert_eq!(group_shift_to_bucket((0, 3)), None);
        assert_eq!(group_shift_to_bucket((3, 3)), None);
    }

    #[test]
    fn test_side_shift_to_bucket() {
        for s in 0..=2 {
            assert_eq!(topbottom_shift_to_bucket(s), Some(s));
            assert_eq!(leftright_shift_to_bucket(s), Some(s));
        }
        assert_eq!(topbottom_shift_to_bucket(3), None);
        assert_eq!(leftright_shift_to_bucket(3), None);
    }

    #[test]
    fn test_data_type_size_to_bucket() {
        assert_eq!(data_type_size_to_bucket(1), None);
        assert_eq!(data_type_size_to_bucket(2), Some(0));
        assert_eq!(data_type_size_to_bucket(3), None);
        assert_eq!(data_type_size_to_bucket(4), Some(1));
        assert_eq!(data_type_size_to_bucket(8), None);
    }

    #[test]
    fn test_bucket_id_uniqueness() {
        let mut seen = std::collections::HashSet::new();
        for &type_size in &[2, 4] {
            for kind in [KIND_GROUP, KIND_TOPBOTTOM, KIND_LEFTRIGHT] {
                let max_shift = match kind {
                    KIND_GROUP => 9,
                    KIND_TOPBOTTOM => 3,
                    KIND_LEFTRIGHT => 3,
                    _ => unreachable!(),
                };
                for shift in 0..max_shift {
                    let id = bucket_id(kind, shift, type_size).unwrap();
                    assert!(id < NUM_BUCKETS);
                    assert!(seen.insert(id), "Duplicate bucket ID: {id}");
                }
            }
        }
        assert_eq!(seen.len(), NUM_BUCKETS);
    }

    #[test]
    fn test_buffer_recycling() {
        let recycler = BufferRecycler::new();
        assert!(recycler.get_group_buffer((0, 0), 4).is_none());

        let img = OwnedRawImage::new((256 * 4, 256)).unwrap();
        recycler.recycle_group_buffer((0, 0), 4, img);

        // Should not be retrievable with size-2 or different shift
        assert!(recycler.get_group_buffer((0, 0), 2).is_none());
        assert!(recycler.get_group_buffer((1, 0), 4).is_none());

        let retrieved = recycler.get_group_buffer((0, 0), 4);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().byte_size(), (256 * 4, 256));

        assert!(recycler.get_group_buffer((0, 0), 4).is_none());
    }

    #[test]
    fn test_side_buffer_recycling() {
        let recycler = BufferRecycler::new();

        // TopBottom
        let tb_img_s4 = OwnedRawImage::new((256 * 4, 4)).unwrap();
        recycler.recycle_topbottom_buffer(1, 4, tb_img_s4);
        assert!(recycler.get_topbottom_buffer(1, 2).is_none());
        assert!(recycler.get_topbottom_buffer(0, 4).is_none());
        assert_eq!(
            recycler.get_topbottom_buffer(1, 4).unwrap().byte_size(),
            (256 * 4, 4)
        );

        // LeftRight
        let lr_img_s2 = OwnedRawImage::new((4 * 2, 256)).unwrap();
        recycler.recycle_leftright_buffer(2, 2, lr_img_s2);
        assert!(recycler.get_leftright_buffer(2, 4).is_none());
        assert!(recycler.get_leftright_buffer(1, 2).is_none());
        assert_eq!(
            recycler.get_leftright_buffer(2, 2).unwrap().byte_size(),
            (4 * 2, 256)
        );
    }

    #[test]
    fn test_all_shifts_and_kinds() {
        let recycler = BufferRecycler::new();

        for &type_size in &[2, 4] {
            // Test all 9 group shifts
            for sx in 0..=2 {
                for sy in 0..=2 {
                    let img = OwnedRawImage::new((128 * type_size, 128)).unwrap();
                    recycler.recycle_group_buffer((sx, sy), type_size, img);
                    let got = recycler.get_group_buffer((sx, sy), type_size);
                    assert!(got.is_some());
                    assert_eq!(got.unwrap().byte_size(), (128 * type_size, 128));
                }
            }

            // Test all 3 topbottom shifts
            for s in 0..=2 {
                let img = OwnedRawImage::new((128 * type_size, 4)).unwrap();
                recycler.recycle_topbottom_buffer(s, type_size, img);
                let got = recycler.get_topbottom_buffer(s, type_size);
                assert!(got.is_some());
                assert_eq!(got.unwrap().byte_size(), (128 * type_size, 4));
            }

            // Test all 3 leftright shifts
            for s in 0..=2 {
                let img = OwnedRawImage::new((4 * type_size, 128)).unwrap();
                recycler.recycle_leftright_buffer(s, type_size, img);
                let got = recycler.get_leftright_buffer(s, type_size);
                assert!(got.is_some());
                assert_eq!(got.unwrap().byte_size(), (4 * type_size, 128));
            }
        }
    }
}
