// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::{
    fmt::Debug,
    ops::{Deref, DerefMut},
};

use crate::{error::Result, util::AtomicRefCell};

// Note: this is meant to be used only in low-contention
// scenarios.
pub struct PerThreadStorage<T> {
    storage: Vec<AtomicRefCell<Option<T>>>,
    available: AtomicRefCell<Vec<usize>>,
}

impl<T> Debug for PerThreadStorage<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<thread storage>")
    }
}

pub struct PerThreadStorageRef<'a, T> {
    r: &'a PerThreadStorage<T>,
    val: Option<T>,
    index: usize,
}

impl<T> PerThreadStorage<T> {
    pub fn new() -> Self {
        Self {
            storage: vec![],
            available: AtomicRefCell::new(vec![]),
        }
    }

    pub fn prepare_for_threads(
        &mut self,
        num: usize,
        make_new: impl Fn() -> Result<T>,
    ) -> Result<()> {
        while self.storage.len() < num {
            self.available.borrow_mut().push(self.storage.len());
            self.storage.push(AtomicRefCell::new(Some(make_new()?)));
        }
        Ok(())
    }

    pub fn get(&self) -> PerThreadStorageRef<'_, T> {
        let idx = loop {
            if let Some(mut a) = self.available.try_borrow_mut() {
                break a.pop().unwrap();
            }
        };
        let t = self.storage[idx].borrow_mut().take().unwrap();
        PerThreadStorageRef {
            r: self,
            val: Some(t),
            index: idx,
        }
    }
}

impl<'a, T> Drop for PerThreadStorageRef<'a, T> {
    fn drop(&mut self) {
        *self.r.storage[self.index].borrow_mut() = self.val.take();
        loop {
            if let Some(mut a) = self.r.available.try_borrow_mut() {
                a.push(self.index);
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
