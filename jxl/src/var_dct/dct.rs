// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(dead_code)]

use std::f64::consts::SQRT_2;

use super::dct_scales::WcMultipliers;

struct CoeffBundle<const N: usize, const SZ: usize>;

// impl<const SZ: usize> CoeffBundle<4, SZ> {
//     fn add_reverse(a_in1: &[f32], a_in2: &[f32], a_out: &mut [f32]) {
//         for i in 0..4 {
//             for j in 0..SZ {
//                 a_out[i * SZ + j] = a_in1[i * SZ + j] + a_in2[(4 - i - 1) * SZ + j];
//             }
//         }
//     }

//     fn sub_reverse(a_in1: &[f32], a_in2: &[f32], a_out: &mut [f32]) {
//         for i in 0..4 {
//             for j in 0..SZ {
//                 a_out[i * SZ + j] = a_in1[i * SZ + j] - a_in2[(4 - i - 1) * SZ + j];
//             }
//         }
//     }

//     fn b(coeff: &mut [f32]) {
//         for j in 0..SZ {
//             coeff[j] = (SQRT_2 as f32) * coeff[j] + coeff[SZ + j];
//         }

//         for i in 1..(4 - 1) {
//             for j in 0..SZ {
//                 coeff[i * SZ + j] += coeff[(i + 1) * SZ + j];
//             }
//         }
//     }

//     fn inverse_even_odd(a_in: &[f32], a_out: &mut [f32]) {
//         for i in 0..4 / 2 {
//             for j in 0..SZ {
//                 a_out[2 * i * SZ + j] = a_in[i * SZ + j];
//             }
//         }
//         for i in 4 / 2..4 {
//             for j in 0..SZ {
//                 a_out[(2 * (i - 4 / 2) + 1) * SZ + j] = a_in[i * SZ + j];
//             }
//         }
//     }

//     fn multiply(coeff: &mut [f32]) {
//         for i in 0..4 / 2 {
//             for j in 0..SZ {
//                 coeff[(4 / 2 + i) * SZ + j] *= WcMultipliers::<4>::K_MULTIPLIERS[i];
//             }
//         }
//     }

//     fn load_from_block(block: &[f32], block_stride: usize, off: usize, coeff: &mut [f32]) {
//         for i in 0..4 {
//             for j in 0..SZ {
//                 coeff[i * SZ + j] = block[i * block_stride + off + j];
//             }
//         }
//     }

//     fn store_to_block_and_scale(coeff: &[f32], block: &mut [f32], block_stride: usize, off: usize) {
//         for i in 0..4 {
//             for j in 0..SZ {
//                 block[i * block_stride + off + j] = (1.0 / (4 as f32)) * coeff[i * SZ + j];
//             }
//         }
//     }
// }

struct IDCT1DImpl<const SIZE: usize>;

trait IDCT1D {
    fn do_idct<const COLUMNS: usize>(data: &mut [f32]);
}

impl IDCT1D for IDCT1DImpl<1> {
    fn do_idct<const COLUMNS: usize>(_data: &mut [f32]) {
        // Do nothing
    }
}

impl IDCT1D for IDCT1DImpl<2> {
    fn do_idct<const COLUMNS: usize>(data: &mut [f32]) {
        let temp0 = data[0];
        let temp1 = data[1];
        data[0] = temp0 + temp1;
        data[1] = temp0 - temp1;
    }
}

macro_rules! define_idct_1d {
    ($n:literal, $nhalf: literal) => {
        impl<const SZ: usize> CoeffBundle<$nhalf, SZ> {
            fn b_transpose(coeff: &mut [f32]) {
                for i in (1..$nhalf).rev() {
                    for j in 0..SZ {
                        coeff[i * SZ + j] += coeff[(i - 1) * SZ + j];
                    }
                }
                for j in 0..SZ {
                    coeff[j] *= SQRT_2 as f32;
                }
            }
        }

        impl<const SZ: usize> CoeffBundle<$n, SZ> {
            fn forward_even_odd(a_in: &[f32], a_in_stride: usize, a_out: &mut [f32]) {
                for i in 0..($nhalf) {
                    for j in 0..SZ {
                        a_out[i * SZ + j] = a_in[2 * i * a_in_stride + j];
                    }
                }
                for i in ($nhalf)..$n {
                    for j in 0..SZ {
                        a_out[i * SZ + j] = a_in[(2 * (i - $nhalf) + 1) * a_in_stride + j];
                    }
                }
            }
            fn multiply_and_add(coeff: &[f32], out: &mut [f32], out_stride: usize) {
                for i in 0..($nhalf) {
                    for j in 0..SZ {
                        let mul = WcMultipliers::<$n>::K_MULTIPLIERS[i];
                        let in1 = coeff[i * SZ + j];
                        let in2 = coeff[($nhalf + i) * SZ + j];
                        out[i * out_stride + j] = mul * in2 + in1;
                        out[($n - i - 1) * out_stride + j] = -mul * in2 + in1;
                    }
                }
            }
        }

        impl IDCT1D for IDCT1DImpl<$n> {
            fn do_idct<const COLUMNS: usize>(data: &mut [f32]) {
                const { assert!($nhalf * 2 == $n) }

                // We assume `data` is arranged as a 4xCOLUMNS matrix in a flat array.

                // TODO(firsching): make tmp buffer here of size $n*COLUMNS or something, right now
                // it will only work for COLUMNS=1
                let mut tmp = [0.0f32; $n]; // Temporary buffer

                // 1. ForwardEvenOdd
                CoeffBundle::<$n, COLUMNS>::forward_even_odd(data, COLUMNS, &mut tmp);
                // 2. First Recursive Call (IDCT1DImpl::do_idct)
                // first half
                IDCT1DImpl::<$nhalf>::do_idct::<COLUMNS>(&mut tmp[0..$nhalf]);
                // 3. BTranspose.
                // only the second half
                CoeffBundle::<$nhalf, COLUMNS>::b_transpose(&mut tmp[$nhalf..$n]);
                // 4. Second Recursive Call (IDCT1DImpl::do_idct)
                // second half
                IDCT1DImpl::<$nhalf>::do_idct::<COLUMNS>(&mut tmp[$nhalf..$n]);
                // 5. MultiplyAndAdd.
                CoeffBundle::<$n, COLUMNS>::multiply_and_add(&tmp, data, COLUMNS);
            }
        }
    };
}
define_idct_1d!(4, 2);
define_idct_1d!(8, 4);
//define_idct_1d!(16, 8);

fn transpose<const N: usize, const M: usize>(_data: &mut [f32]) {
    todo!();
}

fn idct2d<const ROWS: usize, const COLS: usize>(_data: &mut [f32])
where
    IDCT1DImpl<ROWS>: IDCT1D,
    IDCT1DImpl<COLS>: IDCT1D,
{
    todo!();
    // TODO(firsching): Add something like
    // IDCT1DImpl::<ROWS>::do_idct::<COLS>(data);
    // transpose::<ROWS, COLS>(data);
    // IDCT1DImpl::<COLS>::do_idct::<ROWS>(data);
}

#[cfg(test)]
mod tests {
    use crate::{
        util::test::assert_all_almost_eq,
        var_dct::{
            dct::{IDCT1DImpl, IDCT1D},
            dct_slow::idct1d,
        },
    };

    #[test]
    fn test_dct1d_4x1_eq_slow() {
        const N: usize = 4;
        const M: usize = 1;
        const NM: usize = N * M;
        let input_f64: [f64; NM] = [1.0, 2.0, 3.0, 4.0];
        let mut output_slow = [0.0; NM];

        idct1d::<N, M, NM>(&input_f64, &mut output_slow);
        let mut input: [f32; N] = [0.0; N];
        for i in 0..N {
            input[i] = input_f64[i] as f32;
        }
        let mut output = input;
        IDCT1DImpl::<N>::do_idct::<M>(&mut output);
        assert_all_almost_eq!(output, output_slow, 1e-6);
        assert!(!output.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_dct1d_8x1_eq_slow() {
        const N: usize = 8;
        const M: usize = 1;
        const NM: usize = N * M;
        let input_f64: [f64; NM] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut output_slow = [0.0; NM];

        idct1d::<N, M, NM>(&input_f64, &mut output_slow);
        let mut input: [f32; N] = [0.0; N];
        for i in 0..N {
            input[i] = input_f64[i] as f32;
        }
        let mut output = input;
        IDCT1DImpl::<N>::do_idct::<M>(&mut output);
        assert_all_almost_eq!(output, output_slow, 1e-5);
        assert!(!output.iter().all(|&x| x == 0.0));
    }
}
