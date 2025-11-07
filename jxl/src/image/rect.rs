// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use super::DataTypeTag;

#[derive(Clone, Copy, Debug)]
pub struct Rect {
    pub origin: (usize, usize),
    // width, height
    pub size: (usize, usize),
}

impl Rect {
    pub fn check_within(&self, size: (usize, usize)) {
        if self.origin.0.checked_add(self.size.0).unwrap() > size.0
            || self.origin.1.checked_add(self.size.1).unwrap() > size.1
        {
            panic!(
                "Rect out of bounds: {}x{}+{}+{} rect in {}x{} view",
                self.size.0, self.size.1, self.origin.0, self.origin.1, size.0, size.1
            );
        }
    }

    pub const fn to_byte_rect(&self, data_type: DataTypeTag) -> Rect {
        Rect {
            origin: (self.origin.0 * data_type.size(), self.origin.1),
            size: (self.size.0 * data_type.size(), self.size.1),
        }
    }
}
