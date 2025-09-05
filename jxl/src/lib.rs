// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![deny(unsafe_code)]
pub mod api;
pub mod bit_reader;
pub mod color;
pub mod container;
pub mod entropy_coding;
pub mod error;
pub mod features;
pub mod frame;
pub mod headers;
pub mod icc;
pub mod image;
pub mod render;
mod simd;
pub mod util;
pub mod var_dct;

// TODO: Move these to a more appropriate location.
const GROUP_DIM: usize = 256;
const BLOCK_DIM: usize = 8;
const BLOCK_SIZE: usize = BLOCK_DIM * BLOCK_DIM;
const SIGMA_PADDING: usize = 2;
#[allow(clippy::excessive_precision)]
const MIN_SIGMA: f32 = -3.90524291751269967465540850526868;
