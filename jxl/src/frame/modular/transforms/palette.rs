// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
use crate::{error::Result, frame::modular::ModularBufferInfo};

use super::TransformStep;

pub fn do_palette_step(
    _step: TransformStep,
    _buffers: &mut [ModularBufferInfo],
    (_gx, _gy): (usize, usize),
) -> Result<Vec<(usize, usize)>> {
    todo!()
}
