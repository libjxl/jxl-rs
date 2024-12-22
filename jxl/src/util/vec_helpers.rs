// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

pub fn try_with_capacity<T>(capacity: usize) -> Result<Vec<T>, std::collections::TryReserveError> {
    let mut vec = Vec::new();
    vec.try_reserve(capacity)?;
    Ok(vec)
}
