// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::bit_reader::BitReader;
use crate::entropy_coding::decode::Reader;
use crate::error::{Error, Result};
use crate::util::CeilLog2;

pub struct Permutation(Vec<usize>);

impl std::ops::Deref for Permutation {
    type Target = [usize];

    fn deref(&self) -> &[usize] {
        &self.0
    }
}

impl Permutation {
    /// Decode a permutation from entropy-coded stream.
    pub fn decode(
        size: u32,
        skip: u32,
        br: &mut BitReader,
        entropy_reader: &mut Reader,
    ) -> Result<Self> {
        let end = entropy_reader.read(br, get_context(size))?;
        Self::decode_inner(size, skip, end, |ctx| entropy_reader.read(br, ctx))
    }

    fn decode_inner(
        size: u32,
        skip: u32,
        end: u32,
        mut read: impl FnMut(usize) -> Result<u32>,
    ) -> Result<Self> {
        if end > size - skip {
            return Err(Error::InvalidPermutationSize { size, skip, end });
        }

        let mut lehmer = Vec::new();
        lehmer.try_reserve(end as usize)?;

        let mut prev_val = 0u32;
        for idx in skip..(skip + end) {
            let val = read(get_context(prev_val))?;
            if val >= size - idx {
                return Err(Error::InvalidPermutationLehmerCode {
                    size,
                    idx,
                    lehmer: val,
                });
            }
            lehmer.push(val);
            prev_val = val;
        }

        let mut temp = Vec::new();
        temp.try_reserve((size - skip) as usize)?;
        temp.extend((skip as usize)..(size as usize));

        let mut permutation = Vec::new();
        permutation.try_reserve(size as usize)?;

        permutation.extend(0..skip as usize);
        for idx in lehmer {
            permutation.push(temp.remove(idx as usize));
        }
        permutation.extend(temp);

        Ok(Self(permutation))
    }
}

fn get_context(x: u32) -> usize {
    (x + 1).ceil_log2().min(7) as usize
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn simple() {
        // 1, 1, 2, 3, 3, 6, 0, 1 => 5, 6, 8, 10, 11, 15, 4, 9
        let mut syms_for_ctx = [
            vec![1, 1].into_iter(),
            vec![1, 2].into_iter(),
            vec![3, 3, 6].into_iter(),
            vec![0].into_iter(),
        ];

        let permutation =
            Permutation::decode_inner(16, 4, 8, |ctx| Ok(syms_for_ctx[ctx].next().unwrap()))
                .unwrap();

        assert_eq!(
            &*permutation,
            &[0, 1, 2, 3, 5, 6, 8, 10, 11, 15, 4, 9, 7, 12, 13, 14],
        );
    }

    #[test]
    fn lehmer_out_of_bounds() {
        let result = Permutation::decode_inner(8, 4, 1, |_| Ok(4));
        assert!(result.is_err());
    }
}
