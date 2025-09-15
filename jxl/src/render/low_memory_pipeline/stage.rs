// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::any::Any;

use crate::render::save::SaveStage;

pub enum Stage {
    Save(SaveStage),
    Process(Box<dyn Any>), // TODO
}
