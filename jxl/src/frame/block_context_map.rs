// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    bit_reader::BitReader,
    entropy_coding::context_map::*,
    entropy_coding::decode::unpack_signed,
    error::{Error, Result},
    frame::coeff_order::NUM_ORDERS,
};

#[derive(Debug)]
#[allow(dead_code)]
pub struct BlockContextMap {
    pub lf_thresholds: [Vec<i32>; 3],
    pub qf_thresholds: Vec<u32>,
    pub context_map: Vec<u8>,
    pub num_lf_contexts: usize,
    pub num_contexts: usize,
}

impl BlockContextMap {
    pub fn read(br: &mut BitReader) -> Result<BlockContextMap, Error> {
        if br.read(1)? == 1 {
            Ok(BlockContextMap {
                lf_thresholds: [vec![], vec![], vec![]],
                qf_thresholds: vec![],
                context_map: vec![
                    0, 1, 2, 2, 3, 3, 4, 5, 6, 6, 6, 6, 6, //
                    7, 8, 9, 9, 10, 11, 12, 13, 14, 14, 14, 14, 14, //
                    7, 8, 9, 9, 10, 11, 12, 13, 14, 14, 14, 14, 14, //
                ],
                num_lf_contexts: 1,
                num_contexts: 15,
            })
        } else {
            let mut num_lf_contexts: usize = 1;
            let mut lf_thresholds: [Vec<i32>; 3] = [vec![], vec![], vec![]];
            for thr in lf_thresholds.iter_mut() {
                let num_lf_thresholds = br.read(4)? as usize;
                let mut v: Vec<i32> = vec![0; num_lf_thresholds];
                for val in v.iter_mut() {
                    let uval = match br.read(2)? {
                        0 => 4,
                        1 => br.read(8)? + 16,
                        2 => br.read(16)? + 272,
                        _ => br.read(32)? + 65808,
                    };
                    *val = unpack_signed(uval as u32)
                }
                *thr = v;
                num_lf_contexts *= num_lf_thresholds + 1;
            }
            let num_qf_thresholds = br.read(4)? as usize;
            let mut qf_thresholds: Vec<u32> = vec![0; num_qf_thresholds];
            for val in qf_thresholds.iter_mut() {
                *val = match br.read(2)? {
                    0 => 2,
                    1 => br.read(3)? + 4,
                    2 => br.read(5)? + 12,
                    _ => br.read(8)? + 44,
                } as u32;
            }
            if num_lf_contexts * (num_qf_thresholds + 1) > 64 {
                return Err(Error::BlockContextMapSizeTooBig);
            }
            let context_map_size = 3 * NUM_ORDERS * num_lf_contexts * (num_qf_thresholds + 1);
            let context_map = decode_context_map(context_map_size, br)?;
            assert_eq!(context_map.len(), context_map_size);
            let num_contexts = *context_map.iter().max().unwrap() as usize + 1;
            if num_contexts > 16 {
                Err(Error::TooManyBlockContexts)
            } else {
                Ok(BlockContextMap {
                    lf_thresholds,
                    qf_thresholds,
                    context_map,
                    num_lf_contexts,
                    num_contexts,
                })
            }
        }
    }
}
