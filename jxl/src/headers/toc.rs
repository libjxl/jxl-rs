// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use jxl_macros::UnconditionalCoder;

use crate::{
    bit_reader::BitReader,
    error::{Error, Result},
    headers::encodings::*,
};

use super::permutation::Permutation;

pub struct TocNonserialized {
    pub num_entries: u32,
}

#[derive(UnconditionalCoder, Debug, PartialEq)]
#[nonserialized(TocNonserialized)]
pub struct Toc {
    #[default(false)]
    pub permuted: bool,

    // Here we don't use `condition(permuted)`, because `jump_to_byte_boundary` needs to be executed in both cases
    #[default(Permutation::default())]
    #[nonserialized(num_entries: nonserialized.num_entries, permuted: permuted)]
    pub permutation: Permutation,

    #[coder(u2S(Bits(10), Bits(14) + 1024, Bits(22) + 17408, Bits(30) + 4211712))]
    #[size_coder(explicit(nonserialized.num_entries))]
    pub entries: Vec<u32>,
}
