// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(dead_code)]

use std::f64::consts::SQRT_2;

use super::dct_scales::WcMultipliers;

struct CoeffBundle<const N: usize, const SZ: usize>;

struct IDCT1DImpl<const SIZE: usize>;

trait IDCT1D {
    fn do_idct<const COLUMNS: usize>(data: &mut [[f32; COLUMNS]]);
}

impl IDCT1D for IDCT1DImpl<1> {
    fn do_idct<const COLUMNS: usize>(_data: &mut [[f32; COLUMNS]]) {
        // Do nothing
    }
}

impl IDCT1D for IDCT1DImpl<2> {
    fn do_idct<const COLUMNS: usize>(data: &mut [[f32; COLUMNS]]) {
        for i in 0..COLUMNS {
            let temp0 = data[0][i];
            let temp1 = data[1][i];
            data[0][i] = temp0 + temp1;
            data[1][i] = temp0 - temp1;
        }
    }
}

macro_rules! define_idct_1d {
    ($n:literal, $nhalf: literal) => {
        impl<const SZ: usize> CoeffBundle<$nhalf, SZ> {
            fn b_transpose(coeff: &mut [[f32; SZ]]) {
                for i in (1..$nhalf).rev() {
                    for j in 0..SZ {
                        coeff[i][j] += coeff[i - 1][j];
                    }
                }
                for j in 0..SZ {
                    coeff[0][j] *= SQRT_2 as f32;
                }
            }
        }

        impl<const SZ: usize> CoeffBundle<$n, SZ> {
            fn forward_even_odd(a_in: &[[f32; SZ]], a_out: &mut [[f32; SZ]]) {
                for i in 0..($nhalf) {
                    for j in 0..SZ {
                        a_out[i][j] = a_in[2 * i][j];
                    }
                }
                for i in ($nhalf)..$n {
                    for j in 0..SZ {
                        a_out[i][j] = a_in[2 * (i - $nhalf) + 1][j];
                    }
                }
            }
            fn multiply_and_add(coeff: &[[f32; SZ]], out: &mut [[f32; SZ]]) {
                for i in 0..($nhalf) {
                    for j in 0..SZ {
                        let mul = WcMultipliers::<$n>::K_MULTIPLIERS[i];
                        let in1 = coeff[i][j];
                        let in2 = coeff[$nhalf + i][j];
                        out[i][j] = mul * in2 + in1;
                        out[($n - i - 1)][j] = -mul * in2 + in1;
                    }
                }
            }
        }

        impl IDCT1D for IDCT1DImpl<$n> {
            fn do_idct<const COLUMNS: usize>(data: &mut [[f32; COLUMNS]]) {
                const { assert!($nhalf * 2 == $n) }

                // We assume `data` is arranged as a nxCOLUMNS matrix in a flat array.

                let mut tmp = [[0.0f32; COLUMNS]; $n]; // Temporary buffer

                // 1. ForwardEvenOdd
                CoeffBundle::<$n, COLUMNS>::forward_even_odd(data, &mut tmp);
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
                CoeffBundle::<$n, COLUMNS>::multiply_and_add(&tmp, data);
            }
        }
    };
}
define_idct_1d!(4, 2);
define_idct_1d!(8, 4);
define_idct_1d!(16, 8);
define_idct_1d!(32, 16);
define_idct_1d!(64, 32);
define_idct_1d!(128, 64);
define_idct_1d!(256, 128);

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
    use std::array;
    use test_log::test;

    use crate::{
        util::test::assert_almost_eq,
        var_dct::{
            dct::{IDCT1DImpl, IDCT1D},
            dct_slow::idct1d,
        },
    };
    macro_rules! test_idct1d_eq_slow_n {
        ($test_name:ident, $n_val:expr, $tolerance:expr) => {
            #[test]
            fn $test_name() {
                const N: usize = $n_val;
                const M: usize = 1;
                const NM: usize = N * M;

                // Generate input data
                let input_f64_vec: Vec<f64> = (1..=N).map(|i| i as f64).collect();
                let input_f64: [f64; NM] = input_f64_vec
                    .try_into()
                    .expect("Vec to array conversion failed");

                // Run reference implementation
                let mut output_slow = [0.0; NM];
                idct1d::<N, M, NM>(&input_f64, &mut output_slow);

                // Prepare input for tested implementation
                let mut input_arr_2d = [[0.0f32; M]; N];
                for i in 0..N {
                    input_arr_2d[i][0] = input_f64[i] as f32;
                }

                // Run tested implementation (in-place)
                let mut output = input_arr_2d;
                IDCT1DImpl::<N>::do_idct::<M>(&mut output);

                // Compare results
                for i in 0..N {
                    assert_almost_eq!(output[i][0], output_slow[i] as f32, $tolerance, 1e-3);
                }
            }
        };
    }

    test_idct1d_eq_slow_n!(test_dct1d_2x1_eq_slow, 2, 1e-6);
    test_idct1d_eq_slow_n!(test_dct1d_4x1_eq_slow, 4, 1e-6);
    test_idct1d_eq_slow_n!(test_dct1d_8x1_eq_slow, 8, 1e-5);
    test_idct1d_eq_slow_n!(test_dct1d_16x1_eq_slow, 16, 1e-4);
    test_idct1d_eq_slow_n!(test_dct1d_32x1_eq_slow, 32, 1e-3);
    test_idct1d_eq_slow_n!(test_dct1d_64x1_eq_slow, 64, 1e-2);
    test_idct1d_eq_slow_n!(test_dct1d_128x1_eq_slow, 128, 1e-2);
    test_idct1d_eq_slow_n!(test_dct1d_256x1_eq_slow, 256, 1e-1);

    #[test]
    fn test_dct1d_8x3_eq_slow() {
        const N: usize = 8;
        const M: usize = 3;
        const NM: usize = N * M; // 24

        // Initialize input_f64 with values 1.0 to 24.0
        let input_f64: [f64; NM] = array::from_fn(|i| (i + 1) as f64);

        let mut output_slow = [0.0; NM];

        // Call slow implementation (operates on flat data)
        idct1d::<N, M, NM>(&input_f64, &mut output_slow);

        // Prepare input for the implementation under test (2D array: [N][M])
        let mut input = [[0.0; M]; N];
        for j in 0..M {
            for i in 0..N {
                input[i][j] = input_f64[i * M + j] as f32;
            }
        }
        let mut output = input;

        // Call the implementation under test (operates on 2D data)
        IDCT1DImpl::<N>::do_idct::<M>(&mut output);

        // Compare results element-wise
        for j in 0..M {
            for i in 0..N {
                assert_almost_eq!(output[i][j], output_slow[i * M + j] as f32, 1e-5);
            }
        }
    }
}
