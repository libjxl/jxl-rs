// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Helpers for inspecting the payload of a JPEG XL `jhgm` gain-map box.
//!
//! The parser borrows the envelope fields and decodes the optional alternate
//! color encoding and ICC profile. It does not validate ISO gain-map metadata,
//! capture the box from a JPEG XL stream, decode the embedded image, or apply
//! the gain map.

mod gain_map;

pub use gain_map::JxlGainMapBundle;
