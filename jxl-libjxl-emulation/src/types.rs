// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(
    non_upper_case_globals,
    non_camel_case_types,
    non_snake_case,
    dead_code,
    unpredictable_function_pointer_comparisons,
    clippy::undocumented_unsafe_blocks
)]

pub type JXL_BOOL = std::ffi::c_int;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
