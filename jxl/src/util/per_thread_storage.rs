// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::ops::{Deref, DerefMut};

use crate::util::AtomicRefCell;

// Note: this is meant to be used only in low-contention
// scenarios.
pub struct PerThreadStorage<T> {
    storage: AtomicRefCell<Vec<T>>,
    init: fn() -> T,
}

pub struct PerThreadStorageRef<'a, T> {
    r: &'a PerThreadStorage<T>,
    val: Option<T>,
}

impl<T> PerThreadStorage<T> {
    pub fn new(init: fn() -> T) -> Self {
        Self {
            storage: AtomicRefCell::new(vec![]),
            init,
        }
    }

    pub fn get(&self) -> PerThreadStorageRef<'_, T> {
        let t = loop {
            if let Some(mut a) = self.storage.try_borrow_mut() {
                if let Some(x) = a.pop() {
                    break x;
                }
                // go to the call to `init()` if storage is empty
            } else {
                continue;
            }
            break (self.init)();
        };
        PerThreadStorageRef {
            r: self,
            val: Some(t),
        }
    }
}

impl<'a, T> Drop for PerThreadStorageRef<'a, T> {
    fn drop(&mut self) {
        let v = self.val.take().unwrap();
        loop {
            if let Some(mut a) = self.r.storage.try_borrow_mut() {
                a.push(v);
                break;
            }
        }
    }
}

impl<'a, T> Deref for PerThreadStorageRef<'a, T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        self.val.as_ref().unwrap()
    }
}

impl<'a, T> DerefMut for PerThreadStorageRef<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.val.as_mut().unwrap()
    }
}
