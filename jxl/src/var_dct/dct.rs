// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(dead_code)]

use std::f64::consts::SQRT_2;

use super::dct_scales::WcMultipliers;

struct CoeffBundle<const N: usize, const SZ: usize>;

struct DCT1DImpl<const SIZE: usize>;
struct IDCT1DImpl<const SIZE: usize>;

trait DCT1D {
    fn do_dct<const COLUMNS: usize>(data: &mut [[f32; COLUMNS]]);
}
trait IDCT1D {
    fn do_idct<const COLUMNS: usize>(data: &mut [[f32; COLUMNS]]);
}

impl DCT1D for DCT1DImpl<1> {
    fn do_dct<const COLUMNS: usize>(_data: &mut [[f32; COLUMNS]]) {
        // Do nothing
    }
}
impl IDCT1D for IDCT1DImpl<1> {
    fn do_idct<const COLUMNS: usize>(_data: &mut [[f32; COLUMNS]]) {
        // Do nothing
    }
}

impl DCT1D for DCT1DImpl<2> {
    fn do_dct<const COLUMNS: usize>(data: &mut [[f32; COLUMNS]]) {
        for i in 0..COLUMNS {
            let temp0 = data[0][i];
            let temp1 = data[1][i];
            data[0][i] = temp0 + temp1;
            data[1][i] = temp0 - temp1;
        }
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

macro_rules! define_dct_1d {
    ($n:literal, $nhalf: literal) => {
        // Helper functions for CoeffBundle operating on $nhalf rows
        impl<const SZ: usize> CoeffBundle<$nhalf, SZ> {
            /// Adds a_in1[i] and a_in2[$nhalf - 1 - i], storing in a_out[i].
            fn add_reverse(a_in1: &[[f32; SZ]], a_in2: &[[f32; SZ]], a_out: &mut [[f32; SZ]]) {
                const N_HALF_CONST: usize = $nhalf;
                for i in 0..N_HALF_CONST {
                    for j in 0..SZ {
                        a_out[i][j] = a_in1[i][j] + a_in2[N_HALF_CONST - 1 - i][j];
                    }
                }
            }

            /// Subtracts a_in2[$nhalf - 1 - i] from a_in1[i], storing in a_out[i].
            fn sub_reverse(a_in1: &[[f32; SZ]], a_in2: &[[f32; SZ]], a_out: &mut [[f32; SZ]]) {
                const N_HALF_CONST: usize = $nhalf;
                for i in 0..N_HALF_CONST {
                    for j in 0..SZ {
                        a_out[i][j] = a_in1[i][j] - a_in2[N_HALF_CONST - 1 - i][j];
                    }
                }
            }

            /// Applies the B transform (forward DCT step).
            /// Operates on a slice of $nhalf rows.
            fn b(coeff: &mut [[f32; SZ]]) {
                const N_HALF_CONST: usize = $nhalf;
                for j in 0..SZ {
                    coeff[0][j] = coeff[0][j] * (SQRT_2 as f32) + coeff[1][j];
                }
                // empty in the case N_HALF_CONST == 2
                #[allow(clippy::reversed_empty_ranges)]
                for i in 1..(N_HALF_CONST - 1) {
                    for j in 0..SZ {
                        coeff[i][j] += coeff[i + 1][j];
                    }
                }
            }
        }

        // Helper functions for CoeffBundle operating on $n rows
        impl<const SZ: usize> CoeffBundle<$n, SZ> {
            /// Multiplies the second half of `coeff` by WcMultipliers.
            fn multiply(coeff: &mut [[f32; SZ]]) {
                const N_CONST: usize = $n;
                const N_HALF_CONST: usize = $nhalf;
                for i in 0..N_HALF_CONST {
                    let mul_val = WcMultipliers::<N_CONST>::K_MULTIPLIERS[i];
                    for j in 0..SZ {
                        coeff[N_HALF_CONST + i][j] *= mul_val;
                    }
                }
            }

            /// De-interleaves `a_in` into `a_out`.
            /// Even indexed rows of `a_out` get first half of `a_in`.
            /// Odd indexed rows of `a_out` get second half of `a_in`.
            fn inverse_even_odd(a_in: &[[f32; SZ]], a_out: &mut [[f32; SZ]]) {
                const N_CONST: usize = $n;
                const N_HALF_CONST: usize = $nhalf;
                for i in 0..N_HALF_CONST {
                    for j in 0..SZ {
                        a_out[2 * i][j] = a_in[i][j];
                    }
                }
                for i in 0..N_HALF_CONST {
                    for j in 0..SZ {
                        a_out[2 * i + 1][j] = a_in[N_HALF_CONST + i][j];
                    }
                }
            }
        }

        impl DCT1D for DCT1DImpl<$n> {
            fn do_dct<const COLUMNS: usize>(data: &mut [[f32; COLUMNS]]) {
                const { assert!($nhalf * 2 == $n, "N/2 * 2 must be N") }
                assert!(
                    data.len() == $n,
                    "Input data must have $n rows for DCT1DImpl<$n>"
                );

                let mut tmp_buffer = [[0.0f32; COLUMNS]; $n];

                // 1. AddReverse
                //
                //    Inputs: first N/2 rows of data, second N/2 rows of data
                //    Output: first N/2 rows of tmp_buffer
                CoeffBundle::<$nhalf, COLUMNS>::add_reverse(
                    &data[0..$nhalf],
                    &data[$nhalf..$n],
                    &mut tmp_buffer[0..$nhalf],
                );

                // 2. First Recursive Call (do_dct)
                //    first half
                DCT1DImpl::<$nhalf>::do_dct::<COLUMNS>(&mut tmp_buffer[0..$nhalf]);

                // 3. SubReverse
                //    Inputs: first N/2 rows of data, second N/2 rows of data
                //    Output: second N/2 rows of tmp_buffer
                CoeffBundle::<$nhalf, COLUMNS>::sub_reverse(
                    &data[0..$nhalf],
                    &data[$nhalf..$n],
                    &mut tmp_buffer[$nhalf..$n],
                );

                // 4. Multiply(tmp);
                //    Operates on the entire tmp_buffer.
                CoeffBundle::<$n, COLUMNS>::multiply(&mut tmp_buffer);

                // 5. Second Recursive Call (do_dct)
                //    second half.
                DCT1DImpl::<$nhalf>::do_dct::<COLUMNS>(&mut tmp_buffer[$nhalf..$n]);

                // 6. B
                //    Operates on the second N/2 rows of tmp_buffer.
                CoeffBundle::<$nhalf, COLUMNS>::b(&mut tmp_buffer[$nhalf..$n]);

                // 7. InverseEvenOdd
                CoeffBundle::<$n, COLUMNS>::inverse_even_odd(&tmp_buffer, data);
            }
        }
    };
}
define_dct_1d!(4, 2);
define_dct_1d!(8, 4);
define_dct_1d!(16, 8);
define_dct_1d!(32, 16);
define_dct_1d!(64, 32);
define_dct_1d!(128, 64);
define_dct_1d!(256, 128);

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
                const { assert!($nhalf * 2 == $n, "N/2 * 2 must be N") }

                // We assume `data` is arranged as a nxCOLUMNS matrix.

                let mut tmp = [[0.0f32; COLUMNS]; $n];

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

fn transpose<const ROWS: usize, const COLS: usize>(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), ROWS * COLS);
    assert_eq!(output.len(), ROWS * COLS);

    for r in 0..ROWS {
        for c in 0..COLS {
            let input_idx = r * COLS + c;
            let output_idx = c * ROWS + r;
            output[output_idx] = input[input_idx];
        }
    }
}

fn dct2d<const ROWS: usize, const COLS: usize>(data: &mut [f32])
where
    DCT1DImpl<ROWS>: DCT1D,
    DCT1DImpl<COLS>: DCT1D,
{
    assert_eq!(data.len(), ROWS * COLS, "Data length mismatch");

    // Copy data from flat slice `data` into a temporary Vec of arrays (rows).
    let mut temp_rows: Vec<[f32; COLS]> = vec![[0.0f32; COLS]; ROWS];
    for (r, column) in temp_rows.iter_mut().enumerate().take(ROWS) {
        let start = r * COLS;
        let end = start + COLS;
        column.copy_from_slice(&data[start..end]);
    }

    DCT1DImpl::<ROWS>::do_dct::<COLS>(&mut temp_rows);

    for (r, column) in temp_rows.iter().enumerate().take(ROWS) {
        let start = r * COLS;
        let end = start + COLS;
        data[start..end].copy_from_slice(column);
    }

    // Create a temporary flat buffer for the transposed data.
    let mut transposed_data = vec![0.0f32; ROWS * COLS];
    transpose::<ROWS, COLS>(data, &mut transposed_data);

    // Copy data from flat `transposed_data` into a temporary Vec of arrays.
    let mut temp_cols: Vec<[f32; ROWS]> = vec![[0.0f32; ROWS]; COLS];
    for (c, row) in temp_cols.iter_mut().enumerate().take(COLS) {
        let start = c * ROWS;
        let end = start + ROWS;
        row.copy_from_slice(&transposed_data[start..end]);
    }

    // Perform DCT on the temporary structure (treating original columns as rows).
    DCT1DImpl::<COLS>::do_dct::<ROWS>(&mut temp_cols);

    // Copy results back from the temporary structure into the flat `transposed_data`.
    for (c, row) in temp_cols.iter().enumerate().take(COLS) {
        let start = c * ROWS;
        let end = start + ROWS;
        transposed_data[start..end].copy_from_slice(row);
    }
    transpose::<COLS, ROWS>(&transposed_data, data);
}

fn idct2d<const ROWS: usize, const COLS: usize>(data: &mut [f32])
where
    IDCT1DImpl<ROWS>: IDCT1D,
    IDCT1DImpl<COLS>: IDCT1D,
{
    assert_eq!(data.len(), ROWS * COLS, "Data length mismatch");

    // Copy data from flat slice `data` into a temporary Vec of arrays (rows).
    let mut temp_rows: Vec<[f32; COLS]> = vec![[0.0f32; COLS]; ROWS];
    for (r, column) in temp_rows.iter_mut().enumerate().take(ROWS) {
        let start = r * COLS;
        let end = start + COLS;
        column.copy_from_slice(&data[start..end]);
    }

    IDCT1DImpl::<ROWS>::do_idct::<COLS>(&mut temp_rows);

    for (r, column) in temp_rows.iter().enumerate().take(ROWS) {
        let start = r * COLS;
        let end = start + COLS;
        data[start..end].copy_from_slice(column);
    }

    // Create a temporary flat buffer for the transposed data.
    let mut transposed_data = vec![0.0f32; ROWS * COLS];
    transpose::<ROWS, COLS>(data, &mut transposed_data);

    // Copy data from flat `transposed_data` into a temporary Vec of arrays.
    let mut temp_cols: Vec<[f32; ROWS]> = vec![[0.0f32; ROWS]; COLS];
    for (c, row) in temp_cols.iter_mut().enumerate().take(COLS) {
        let start = c * ROWS;
        let end = start + ROWS;
        row.copy_from_slice(&transposed_data[start..end]);
    }

    // Perform IDCT on the temporary structure (treating original columns as rows).
    IDCT1DImpl::<COLS>::do_idct::<ROWS>(&mut temp_cols);

    // Copy results back from the temporary structure into the flat `transposed_data`.
    for (c, row) in temp_cols.iter().enumerate().take(COLS) {
        let start = c * ROWS;
        let end = start + ROWS;
        transposed_data[start..end].copy_from_slice(row);
    }
    transpose::<COLS, ROWS>(&transposed_data, data);
}

#[cfg(test)]
mod tests {
    use crate::{
        util::test::assert_almost_eq,
        var_dct::{
            dct::{dct2d, idct2d, DCT1DImpl, IDCT1DImpl, DCT1D, IDCT1D},
            dct_slow::{dct1d, idct1d},
        },
    };
    use std::array;
    use test_log::test;
    macro_rules! test_dct1d_eq_slow_n {
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
                dct1d::<N, M, NM>(&input_f64, &mut output_slow);

                // Prepare input for tested implementation
                let mut input_arr_2d = [[0.0f32; M]; N];
                for i in 0..N {
                    input_arr_2d[i][0] = input_f64[i] as f32;
                }

                // Run tested implementation (in-place)
                let mut output = input_arr_2d;
                DCT1DImpl::<N>::do_dct::<M>(&mut output);

                // Compare results

                for i in 0..N {
                    println!(
                        "fast: {:}, slow: {:}, ratio: {}",
                        output[i][0],
                        output_slow[i] as f32,
                        output[i][0] / (output_slow[i] as f32)
                    );
                }
                for i in 0..N {
                    assert_almost_eq!(output[i][0], output_slow[i] as f32, $tolerance);
                }
            }
        };
    }
    macro_rules! test_idct1d_eq_slow_n {
        ($test_name:ident, $n_val:expr, $tolerance:expr) => {
            #[test]
            fn $test_name() {
                const N: usize = $n_val;
                const M: usize = 1;
                const NM: usize = N * M;

                // Generate input data
                let input_f64_vec: Vec<f64> = (1..=N).map(|i| (i + 1) as f64).collect();
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

    test_dct1d_eq_slow_n!(test_dct1d_1x1_eq_slow, 1, 1e-6);
    test_idct1d_eq_slow_n!(test_idct1d_1x1_eq_slow, 1, 1e-6);
    test_dct1d_eq_slow_n!(test_dct1d_2x1_eq_slow, 2, 1e-6);
    test_idct1d_eq_slow_n!(test_idct1d_2x1_eq_slow, 2, 1e-6);
    test_dct1d_eq_slow_n!(test_dct1d_4x1_eq_slow, 4, 1e-6);
    test_idct1d_eq_slow_n!(test_idct1d_4x1_eq_slow, 4, 1e-6);
    test_dct1d_eq_slow_n!(test_dct1d_8x1_eq_slow, 8, 1e-5);
    test_idct1d_eq_slow_n!(test_idct1d_8x1_eq_slow, 8, 1e-5);
    test_dct1d_eq_slow_n!(test_dct1d_16x1_eq_slow, 16, 1e-4);
    test_idct1d_eq_slow_n!(test_idct1d_16x1_eq_slow, 16, 1e-4);
    test_dct1d_eq_slow_n!(test_dct1d_32x1_eq_slow, 32, 1e-3);
    test_idct1d_eq_slow_n!(test_idct1d_32x1_eq_slow, 32, 1e-3);
    test_dct1d_eq_slow_n!(test_dct1d_64x1_eq_slow, 64, 1e-2);
    test_idct1d_eq_slow_n!(test_idct1d_64x1_eq_slow, 64, 1e-2);
    test_dct1d_eq_slow_n!(test_dct1d_128x1_eq_slow, 128, 1e-2);
    test_idct1d_eq_slow_n!(test_idct1d_128x1_eq_slow, 128, 1e-2);
    test_dct1d_eq_slow_n!(test_dct1d_256x1_eq_slow, 256, 1e-1);
    test_idct1d_eq_slow_n!(test_idct1d_256x1_eq_slow, 256, 1e-1);

    #[test]
    fn test_idct1d_8x3_eq_slow() {
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

    #[test]
    fn test_dct1d_8x3_eq_slow() {
        const N: usize = 8;
        const M: usize = 3;
        const NM: usize = N * M; // 24

        // Initialize input_f64 with values 1.0 to 24.0
        let input_f64: [f64; NM] = array::from_fn(|i| (i + 1) as f64);

        let mut output_slow = [0.0; NM];

        // Call slow implementation (operates on flat data)
        dct1d::<N, M, NM>(&input_f64, &mut output_slow);

        // Prepare input for the implementation under test (2D array: [N][M])
        let mut input = [[0.0; M]; N];
        for j in 0..M {
            for i in 0..N {
                input[i][j] = input_f64[i * M + j] as f32;
            }
        }
        let mut output = input;

        // Call the implementation under test (operates on 2D data)
        DCT1DImpl::<N>::do_dct::<M>(&mut output);

        // Compare results element-wise
        for j in 0..M {
            for i in 0..N {
                assert_almost_eq!(output[i][j], output_slow[i * M + j] as f32, 1e-5);
            }
        }
    }

    // TODO(firsching): possibly change these tests to test against slow
    // (i)dct method (after adding 2d-variant there)
    macro_rules! test_idct2d_exists_n_m {
        ($test_name:ident, $n_val:expr, $m_val:expr) => {
            #[test]
            fn $test_name() {
                const N: usize = $n_val;
                const M: usize = $m_val;
                let mut data = [0.0f32; M * N];
                idct2d::<N, M>(&mut data);
            }
        };
    }
    macro_rules! test_dct2d_exists_n_m {
        ($test_name:ident, $n_val:expr, $m_val:expr) => {
            #[test]
            fn $test_name() {
                const N: usize = $n_val;
                const M: usize = $m_val;
                let mut data = [0.0f32; M * N];
                dct2d::<N, M>(&mut data);
            }
        };
    }
    test_dct2d_exists_n_m!(test_dct2d_exists_1_1, 1, 1);
    test_idct2d_exists_n_m!(test_idct2d_exists_1_1, 1, 1);
    test_dct2d_exists_n_m!(test_dct2d_exists_1_2, 1, 2);
    test_idct2d_exists_n_m!(test_idct2d_exists_1_2, 1, 2);
    test_dct2d_exists_n_m!(test_dct2d_exists_1_4, 1, 4);
    test_idct2d_exists_n_m!(test_idct2d_exists_1_4, 1, 4);
    test_dct2d_exists_n_m!(test_dct2d_exists_1_8, 1, 8);
    test_idct2d_exists_n_m!(test_idct2d_exists_1_8, 1, 8);
    test_dct2d_exists_n_m!(test_dct2d_exists_1_16, 1, 16);
    test_idct2d_exists_n_m!(test_idct2d_exists_1_16, 1, 16);
    test_dct2d_exists_n_m!(test_dct2d_exists_1_32, 1, 32);
    test_idct2d_exists_n_m!(test_idct2d_exists_1_32, 1, 32);
    test_dct2d_exists_n_m!(test_dct2d_exists_1_64, 1, 64);
    test_idct2d_exists_n_m!(test_idct2d_exists_1_64, 1, 64);
    test_dct2d_exists_n_m!(test_dct2d_exists_1_128, 1, 128);
    test_idct2d_exists_n_m!(test_idct2d_exists_1_128, 1, 128);
    test_dct2d_exists_n_m!(test_dct2d_exists_1_256, 1, 256);
    test_idct2d_exists_n_m!(test_idct2d_exists_1_256, 1, 256);
    test_dct2d_exists_n_m!(test_dct2d_exists_2_1, 2, 1);
    test_idct2d_exists_n_m!(test_idct2d_exists_2_1, 2, 1);
    test_dct2d_exists_n_m!(test_dct2d_exists_2_2, 2, 2);
    test_idct2d_exists_n_m!(test_idct2d_exists_2_2, 2, 2);
    test_dct2d_exists_n_m!(test_dct2d_exists_2_4, 2, 4);
    test_idct2d_exists_n_m!(test_idct2d_exists_2_4, 2, 4);
    test_dct2d_exists_n_m!(test_dct2d_exists_2_8, 2, 8);
    test_idct2d_exists_n_m!(test_idct2d_exists_2_8, 2, 8);
    test_dct2d_exists_n_m!(test_dct2d_exists_2_16, 2, 16);
    test_idct2d_exists_n_m!(test_idct2d_exists_2_16, 2, 16);
    test_dct2d_exists_n_m!(test_dct2d_exists_2_32, 2, 32);
    test_idct2d_exists_n_m!(test_idct2d_exists_2_32, 2, 32);
    test_dct2d_exists_n_m!(test_dct2d_exists_2_64, 2, 64);
    test_idct2d_exists_n_m!(test_idct2d_exists_2_64, 2, 64);
    test_dct2d_exists_n_m!(test_dct2d_exists_2_128, 2, 128);
    test_idct2d_exists_n_m!(test_idct2d_exists_2_128, 2, 128);
    test_dct2d_exists_n_m!(test_dct2d_exists_2_256, 2, 256);
    test_idct2d_exists_n_m!(test_idct2d_exists_2_256, 2, 256);
    test_dct2d_exists_n_m!(test_dct2d_exists_4_1, 4, 1);
    test_idct2d_exists_n_m!(test_idct2d_exists_4_1, 4, 1);
    test_dct2d_exists_n_m!(test_dct2d_exists_4_2, 4, 2);
    test_idct2d_exists_n_m!(test_idct2d_exists_4_2, 4, 2);
    test_dct2d_exists_n_m!(test_dct2d_exists_4_4, 4, 4);
    test_idct2d_exists_n_m!(test_idct2d_exists_4_4, 4, 4);
    test_dct2d_exists_n_m!(test_dct2d_exists_4_8, 4, 8);
    test_idct2d_exists_n_m!(test_idct2d_exists_4_8, 4, 8);
    test_dct2d_exists_n_m!(test_dct2d_exists_4_16, 4, 16);
    test_idct2d_exists_n_m!(test_idct2d_exists_4_16, 4, 16);
    test_dct2d_exists_n_m!(test_dct2d_exists_4_32, 4, 32);
    test_idct2d_exists_n_m!(test_idct2d_exists_4_32, 4, 32);
    test_dct2d_exists_n_m!(test_dct2d_exists_4_64, 4, 64);
    test_idct2d_exists_n_m!(test_idct2d_exists_4_64, 4, 64);
    test_dct2d_exists_n_m!(test_dct2d_exists_4_128, 4, 128);
    test_idct2d_exists_n_m!(test_idct2d_exists_4_128, 4, 128);
    test_dct2d_exists_n_m!(test_dct2d_exists_4_256, 4, 256);
    test_idct2d_exists_n_m!(test_idct2d_exists_4_256, 4, 256);
    test_dct2d_exists_n_m!(test_dct2d_exists_8_1, 8, 1);
    test_idct2d_exists_n_m!(test_idct2d_exists_8_1, 8, 1);
    test_dct2d_exists_n_m!(test_dct2d_exists_8_2, 8, 2);
    test_idct2d_exists_n_m!(test_idct2d_exists_8_2, 8, 2);
    test_dct2d_exists_n_m!(test_dct2d_exists_8_4, 8, 4);
    test_idct2d_exists_n_m!(test_idct2d_exists_8_4, 8, 4);
    test_dct2d_exists_n_m!(test_dct2d_exists_8_8, 8, 8);
    test_idct2d_exists_n_m!(test_idct2d_exists_8_8, 8, 8);
    test_dct2d_exists_n_m!(test_dct2d_exists_8_16, 8, 16);
    test_idct2d_exists_n_m!(test_idct2d_exists_8_16, 8, 16);
    test_dct2d_exists_n_m!(test_dct2d_exists_8_32, 8, 32);
    test_idct2d_exists_n_m!(test_idct2d_exists_8_32, 8, 32);
    test_dct2d_exists_n_m!(test_dct2d_exists_8_64, 8, 64);
    test_idct2d_exists_n_m!(test_idct2d_exists_8_64, 8, 64);
    test_dct2d_exists_n_m!(test_dct2d_exists_8_128, 8, 128);
    test_idct2d_exists_n_m!(test_idct2d_exists_8_128, 8, 128);
    test_dct2d_exists_n_m!(test_dct2d_exists_8_256, 8, 256);
    test_idct2d_exists_n_m!(test_idct2d_exists_8_256, 8, 256);
    test_dct2d_exists_n_m!(test_dct2d_exists_16_1, 16, 1);
    test_idct2d_exists_n_m!(test_idct2d_exists_16_1, 16, 1);
    test_dct2d_exists_n_m!(test_dct2d_exists_16_2, 16, 2);
    test_idct2d_exists_n_m!(test_idct2d_exists_16_2, 16, 2);
    test_dct2d_exists_n_m!(test_dct2d_exists_16_4, 16, 4);
    test_idct2d_exists_n_m!(test_idct2d_exists_16_4, 16, 4);
    test_dct2d_exists_n_m!(test_dct2d_exists_16_8, 16, 8);
    test_idct2d_exists_n_m!(test_idct2d_exists_16_8, 16, 8);
    test_dct2d_exists_n_m!(test_dct2d_exists_16_16, 16, 16);
    test_idct2d_exists_n_m!(test_idct2d_exists_16_16, 16, 16);
    test_dct2d_exists_n_m!(test_dct2d_exists_16_32, 16, 32);
    test_idct2d_exists_n_m!(test_idct2d_exists_16_32, 16, 32);
    test_dct2d_exists_n_m!(test_dct2d_exists_16_64, 16, 64);
    test_idct2d_exists_n_m!(test_idct2d_exists_16_64, 16, 64);
    test_dct2d_exists_n_m!(test_dct2d_exists_16_128, 16, 128);
    test_idct2d_exists_n_m!(test_idct2d_exists_16_128, 16, 128);
    test_dct2d_exists_n_m!(test_dct2d_exists_16_256, 16, 256);
    test_idct2d_exists_n_m!(test_idct2d_exists_16_256, 16, 256);
    test_dct2d_exists_n_m!(test_dct2d_exists_32_1, 32, 1);
    test_idct2d_exists_n_m!(test_idct2d_exists_32_1, 32, 1);
    test_dct2d_exists_n_m!(test_dct2d_exists_32_2, 32, 2);
    test_idct2d_exists_n_m!(test_idct2d_exists_32_2, 32, 2);
    test_dct2d_exists_n_m!(test_dct2d_exists_32_4, 32, 4);
    test_idct2d_exists_n_m!(test_idct2d_exists_32_4, 32, 4);
    test_dct2d_exists_n_m!(test_dct2d_exists_32_8, 32, 8);
    test_idct2d_exists_n_m!(test_idct2d_exists_32_8, 32, 8);
    test_dct2d_exists_n_m!(test_dct2d_exists_32_16, 32, 16);
    test_idct2d_exists_n_m!(test_idct2d_exists_32_16, 32, 16);
    test_dct2d_exists_n_m!(test_dct2d_exists_32_32, 32, 32);
    test_idct2d_exists_n_m!(test_idct2d_exists_32_32, 32, 32);
    test_dct2d_exists_n_m!(test_dct2d_exists_32_64, 32, 64);
    test_idct2d_exists_n_m!(test_idct2d_exists_32_64, 32, 64);
    test_dct2d_exists_n_m!(test_dct2d_exists_32_128, 32, 128);
    test_idct2d_exists_n_m!(test_idct2d_exists_32_128, 32, 128);
    test_dct2d_exists_n_m!(test_dct2d_exists_32_256, 32, 256);
    test_idct2d_exists_n_m!(test_idct2d_exists_32_256, 32, 256);
    test_dct2d_exists_n_m!(test_dct2d_exists_64_1, 64, 1);
    test_idct2d_exists_n_m!(test_idct2d_exists_64_1, 64, 1);
    test_dct2d_exists_n_m!(test_dct2d_exists_64_2, 64, 2);
    test_idct2d_exists_n_m!(test_idct2d_exists_64_2, 64, 2);
    test_dct2d_exists_n_m!(test_dct2d_exists_64_4, 64, 4);
    test_idct2d_exists_n_m!(test_idct2d_exists_64_4, 64, 4);
    test_dct2d_exists_n_m!(test_dct2d_exists_64_8, 64, 8);
    test_idct2d_exists_n_m!(test_idct2d_exists_64_8, 64, 8);
    test_dct2d_exists_n_m!(test_dct2d_exists_64_16, 64, 16);
    test_idct2d_exists_n_m!(test_idct2d_exists_64_16, 64, 16);
    test_dct2d_exists_n_m!(test_dct2d_exists_64_32, 64, 32);
    test_idct2d_exists_n_m!(test_idct2d_exists_64_32, 64, 32);
    test_dct2d_exists_n_m!(test_dct2d_exists_64_64, 64, 64);
    test_idct2d_exists_n_m!(test_idct2d_exists_64_64, 64, 64);
    test_dct2d_exists_n_m!(test_dct2d_exists_64_128, 64, 128);
    test_idct2d_exists_n_m!(test_idct2d_exists_64_128, 64, 128);
    test_dct2d_exists_n_m!(test_dct2d_exists_64_256, 64, 256);
    test_idct2d_exists_n_m!(test_idct2d_exists_64_256, 64, 256);
    test_dct2d_exists_n_m!(test_dct2d_exists_128_1, 128, 1);
    test_idct2d_exists_n_m!(test_idct2d_exists_128_1, 128, 1);
    test_dct2d_exists_n_m!(test_dct2d_exists_128_2, 128, 2);
    test_idct2d_exists_n_m!(test_idct2d_exists_128_2, 128, 2);
    test_dct2d_exists_n_m!(test_dct2d_exists_128_4, 128, 4);
    test_idct2d_exists_n_m!(test_idct2d_exists_128_4, 128, 4);
    test_dct2d_exists_n_m!(test_dct2d_exists_128_8, 128, 8);
    test_idct2d_exists_n_m!(test_idct2d_exists_128_8, 128, 8);
    test_dct2d_exists_n_m!(test_dct2d_exists_128_16, 128, 16);
    test_idct2d_exists_n_m!(test_idct2d_exists_128_16, 128, 16);
    test_dct2d_exists_n_m!(test_dct2d_exists_128_32, 128, 32);
    test_idct2d_exists_n_m!(test_idct2d_exists_128_32, 128, 32);
    test_dct2d_exists_n_m!(test_dct2d_exists_128_64, 128, 64);
    test_idct2d_exists_n_m!(test_idct2d_exists_128_64, 128, 64);
    test_dct2d_exists_n_m!(test_dct2d_exists_128_128, 128, 128);
    test_idct2d_exists_n_m!(test_idct2d_exists_128_128, 128, 128);
    test_dct2d_exists_n_m!(test_dct2d_exists_128_256, 128, 256);
    test_idct2d_exists_n_m!(test_idct2d_exists_128_256, 128, 256);
    test_dct2d_exists_n_m!(test_dct2d_exists_256_1, 256, 1);
    test_idct2d_exists_n_m!(test_idct2d_exists_256_1, 256, 1);
    test_dct2d_exists_n_m!(test_dct2d_exists_256_2, 256, 2);
    test_idct2d_exists_n_m!(test_idct2d_exists_256_2, 256, 2);
    test_dct2d_exists_n_m!(test_dct2d_exists_256_4, 256, 4);
    test_idct2d_exists_n_m!(test_idct2d_exists_256_4, 256, 4);
    test_dct2d_exists_n_m!(test_dct2d_exists_256_8, 256, 8);
    test_idct2d_exists_n_m!(test_idct2d_exists_256_8, 256, 8);
    test_dct2d_exists_n_m!(test_dct2d_exists_256_16, 256, 16);
    test_idct2d_exists_n_m!(test_idct2d_exists_256_16, 256, 16);
    test_dct2d_exists_n_m!(test_dct2d_exists_256_32, 256, 32);
    test_idct2d_exists_n_m!(test_idct2d_exists_256_32, 256, 32);
    test_dct2d_exists_n_m!(test_dct2d_exists_256_64, 256, 64);
    test_idct2d_exists_n_m!(test_idct2d_exists_256_64, 256, 64);
    test_dct2d_exists_n_m!(test_dct2d_exists_256_128, 256, 128);
    test_idct2d_exists_n_m!(test_idct2d_exists_256_128, 256, 128);
    test_dct2d_exists_n_m!(test_dct2d_exists_256_256, 256, 256);
    test_idct2d_exists_n_m!(test_idct2d_exists_256_256, 256, 256);
}
