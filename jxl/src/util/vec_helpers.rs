// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

pub trait TryWithCapacity {
    type Output;
    type Error;
    fn try_with_capacity(capacity: usize) -> Result<Self::Output, Self::Error>;
}

impl<T> TryWithCapacity for Vec<T> {
    type Output = Vec<T>;
    type Error = std::collections::TryReserveError;

    fn try_with_capacity(capacity: usize) -> Result<Self::Output, Self::Error> {
        let mut vec = Vec::new();
        vec.try_reserve(capacity)?;
        Ok(vec)
    }
}

impl TryWithCapacity for String {
    type Output = String;
    type Error = std::collections::TryReserveError;
    fn try_with_capacity(capacity: usize) -> Result<Self::Output, Self::Error> {
        let mut s = String::new();
        s.try_reserve(capacity)?;
        Ok(s)
    }
}
