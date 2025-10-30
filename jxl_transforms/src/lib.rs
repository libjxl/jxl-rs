// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

pub mod dct;
pub mod idct2d;
pub mod scales;
pub mod transform;
pub mod transform_map;

mod idct128;
mod idct16;
mod idct2;
mod idct256;
mod idct32;
mod idct4;
mod idct64;
mod idct8;

use idct128::*;
use idct16::*;
use idct2::*;
use idct256::*;
use idct32::*;
use idct4::*;
use idct64::*;
use idct8::*;

pub use idct2d::*;

#[cfg(test)]
mod tests;
