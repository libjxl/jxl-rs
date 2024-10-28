// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::bit_reader::BitReader;
use crate::entropy_coding::decode::Reader;
use crate::error::{Error, Result};
use crate::util::CeilLog2;
use crate::util::value_of_lowest_1_bit;

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

        // Create a temporary permutation vector for the elements to permute
        let mut perm_temp: Vec<u32> = (skip..size).collect();

        // Decode the Lehmer code
        decode_lehmer_code(&lehmer, &mut perm_temp)?;

        // Construct the full permutation vector
        let mut permutation = Vec::with_capacity(size as usize);
        permutation.extend(0..(skip as usize)); // Add skipped elements

        // Append the permuted elements
        permutation.extend(perm_temp.iter().map(|&x| x as usize));

        // Ensure the permutation has the correct size
        assert_eq!(permutation.len(), size as usize);

        Ok(Self(permutation))
    }
}


// Decodes the Lehmer code in code[0..n) into permutation[0..n).
fn decode_lehmer_code(code: &[u32], permutation: &mut [u32]) -> Result<()> {
    println!("code: {:?}", code);
    println!("permutation: {:?}", permutation);
    let permutation_copy: Vec<u32> = permutation.to_vec();
    let n = permutation.len();
    if n == 0 {
        return Err(Error::InvalidPermutationLehmerCode {
            size: 0,
            idx: 0,
            lehmer: 0,
        });
    }

    let log2n = (n as u32).ceil_log2();
    let padded_n = 1 << log2n;

    // Allocate temp array inside the function
    let mut temp = vec![0u32; padded_n];

    // Initialize temp array
    for i in 0..padded_n {
        let i1 = (i + 1) as u32;
        temp[i] = value_of_lowest_1_bit(i1);
    }

    for i in 0..n {
        let code_i = *code.get(i).unwrap_or(&0);

        // Adjust the maximum allowed value for code_i
        if code_i as usize > n - i - 1 {
            return Err(Error::InvalidPermutationLehmerCode {
                size: n as u32,
                idx: i as u32,
                lehmer: code_i,
            });
        }

        let mut rank = code_i + 1;

        // Extract i-th unused element via implicit order-statistics tree.
        let mut bit = padded_n;
        let mut next = 0usize;
        while bit != 0 {
            let cand = next + bit;
            if cand == 0 || cand > padded_n {
                return Err(Error::InvalidPermutationLehmerCode {
                    size: n as u32,
                    idx: i as u32,
                    lehmer: code_i,
                });
            }
            bit >>= 1;
            if temp[cand - 1] < rank {
                next = cand;
                rank -= temp[cand - 1];
            }
        }

        permutation[i] = permutation_copy[next];

        next += 1;
        while next <= padded_n {
            temp[next - 1] -= 1;
            next += value_of_lowest_1_bit(next as u32) as usize;
        }
    }

    Ok(())
}

// Used in testing to check that `decode_lehmer_code` implements the same function.
#[allow(dead_code)]
fn decode_lehmer_code_naive(code: &[u32], permutation: &mut [u32]) -> Result<()> {
    let n = code.len();
    if n == 0 {
        return Err(Error::InvalidPermutationLehmerCode {
            size: 0,
            idx: 0,
            lehmer: 0,
        });
    }

    // Ensure permutation has sufficient length
    if permutation.len() < n {
        return Err(Error::InvalidPermutationLehmerCode {
            size: n as u32,
            idx: 0,
            lehmer: 0,
        });
    }

    // Create temp array inside the function with values from 0 to n - 1
    let mut temp = permutation.to_vec();

    let mut perm_index = 0;

    // Iterate over the Lehmer code
    for (i, &idx) in code.iter().enumerate() {
        if idx as usize >= temp.len() {
            return Err(Error::InvalidPermutationLehmerCode {
                size: n as u32,
                idx: i as u32,
                lehmer: idx,
            });
        }

        // Assign temp[idx] to permutation[perm_index]
        permutation[perm_index] = temp.remove(idx as usize);
        perm_index += 1;
    }

    // Copy any remaining elements from temp to permutation
    for val in temp {
        permutation[perm_index] = val;
        perm_index += 1;
    }

    Ok(())
}

fn get_context(x: u32) -> usize {
    (x + 1).ceil_log2().min(7) as usize
}

#[cfg(test)]
mod test {
    use super::*;

    use super::{decode_lehmer_code, decode_lehmer_code_naive};
    use crate::error::Result;
    use arbtest::arbitrary::{self, Arbitrary, Unstructured};

    #[test]
    fn generate_permutation_arbtest() {
        arbtest::arbtest(|u| {
            let mut input = PermutationInput::arbitrary(u)?;

            let perm1 = decode_lehmer_code(&input.code, &mut input.permutation);
            let perm2 = decode_lehmer_code_naive(&input.code, &mut input.permutation);

            match (perm1, perm2) {
                // Both Ok, check if permutations are equal
                (Ok(p1), Ok(p2)) => assert_eq!(p1, p2),
                // Both Err, compare error strings
                (Err(e1), Err(e2)) => assert_eq!(e1.to_string(), e2.to_string()),
                // One is Ok, the other is Err
                (res1, res2) => panic!("Mismatched results: {:?} != {:?}", res1, res2),
            }

            Ok(())
        });
    }

    #[derive(Debug)]
    struct PermutationInput {
        code: Vec<u32>,
        permutation: Vec<u32>,
    }

    impl<'a> Arbitrary<'a> for PermutationInput {
        fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self, arbitrary::Error> {
            // Generate a reasonable size to prevent tests from taking too long
            let size_lehmer = u.int_in_range(1..=1000)?;

            let mut lehmer: Vec<u32> = Vec::with_capacity(size_lehmer as usize);
            for i in 0..size_lehmer {
                let max_val = size_lehmer - i - 1;
                let val = if max_val > 0 {
                    u.int_in_range(0..=max_val)?
                } else {
                    0
                };
                lehmer.push(val);
            }

            let mut permutation = Vec::new();
            let size_permutation = u.int_in_range(size_lehmer..=1000)?;
            permutation.extend(0..size_permutation as u32);
            let mut num_of_swaps = u.int_in_range(0..=100)?;
            while 0 < num_of_swaps {
                // Randomly swap two positions
                let pos1 = u.int_in_range(0..=size_permutation - 1)?;
                let pos2 = u.int_in_range(0..=size_permutation - 1)?;
                num_of_swaps -= 1;
                permutation.swap(pos1.try_into().unwrap(), pos2.try_into().unwrap());
            }
            Ok(PermutationInput {
                code: lehmer,
                permutation,
            })
        }
    }

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
    fn decode_lehmer_compare_different_length() -> Result<(), Box<dyn std::error::Error>> {
        // Lehmer code: [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        let code = vec![1u32, 1, 2, 3, 3, 6, 0, 1];

        // Prepare temp and permutation arrays for the optimized function
        let mut permutation_optimized: Vec<u32> = (4..16 as u32).collect();

        // Decode using the optimized function
        decode_lehmer_code(&code, &mut permutation_optimized)?;
        // Prepare temp and permutation arrays for the naive function
        let mut permutation_naive: Vec<u32> = (4..16 as u32).collect();
        // Decode using the naive function
        decode_lehmer_code_naive(&code, &mut permutation_naive)?;

        // Expected permutation: [2, 4, 0, 1, 3]
        let expected_permutation = vec![5u32, 6, 8, 10, 11, 15, 4, 9, 7, 12, 13, 14];

        // Assert that both permutations match the expected permutation
        assert_eq!(permutation_optimized, expected_permutation);
        assert_eq!(permutation_naive, expected_permutation);

        // Assert that both functions produce the same permutation
        assert_eq!(permutation_optimized, permutation_naive);

        Ok(())
    }

    #[test]
    fn decode_lehmer_compare_same_length() -> Result<(), Box<dyn std::error::Error>> {
        // Lehmer code: [2, 3, 0, 0, 0]
        let code = vec![2u32, 3, 0, 0, 0];
        let n = code.len();

        // Prepare temp and permutation arrays for the optimized function
        let mut permutation_optimized: Vec<u32> = (0..n as u32).collect();

        // Decode using the optimized function
        decode_lehmer_code(&code, &mut permutation_optimized)?;

        // Prepare temp and permutation arrays for the naive function
        let mut permutation_naive: Vec<u32> = (0..n as u32).collect();

        // Decode using the naive function
        decode_lehmer_code_naive(&code, &mut permutation_naive)?;

        // Expected permutation: [2, 4, 0, 1, 3]
        let expected_permutation = vec![2u32, 4, 0, 1, 3];

        // Assert that both permutations match the expected permutation
        assert_eq!(permutation_optimized, expected_permutation);
        assert_eq!(permutation_naive, expected_permutation);

        // Assert that both functions produce the same permutation
        assert_eq!(permutation_optimized, permutation_naive);

        Ok(())
    }

    #[test]
    fn lehmer_out_of_bounds() {
        let result = Permutation::decode_inner(8, 4, 1, |_| Ok(4));
        assert!(result.is_err());
    }
}
