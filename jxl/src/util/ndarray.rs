// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

macro_rules! slice_mut {
    ($data:expr, $range:expr) => {
        $data[$range]
    };
    ($data:expr, $range:expr $(, $rest_ranges:expr)+) => {
        $data[$range].iter_mut().map(|inner_data| {
            &mut slice_mut!(inner_data, $($rest_ranges),+)
        }).collect::<Vec<_>>()
    };
}
pub(crate) use slice_mut;
