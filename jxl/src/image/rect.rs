// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::error::{Error, Result};

use super::DataTypeTag;

#[derive(Clone, Copy, Debug)]
pub struct Rect {
    pub origin: (usize, usize),
    // width, height
    pub size: (usize, usize),
}

impl Rect {
    pub fn is_within(&self, size: (usize, usize)) -> Result<()> {
        if self
            .origin
            .0
            .checked_add(self.size.0)
            .ok_or(Error::ArithmeticOverflow)?
            > size.0
            || self
                .origin
                .1
                .checked_add(self.size.1)
                .ok_or(Error::ArithmeticOverflow)?
                > size.1
        {
            Err(Error::RectOutOfBounds(
                self.size.0,
                self.size.1,
                self.origin.0,
                self.origin.1,
                size.0,
                size.1,
            ))
        } else {
            Ok(())
        }
    }

    pub const fn to_byte_rect(&self, data_type: DataTypeTag) -> Rect {
        Rect {
            origin: (self.origin.0 * data_type.size(), self.origin.1),
            size: (self.size.0 * data_type.size(), self.size.1),
        }
    }
}
