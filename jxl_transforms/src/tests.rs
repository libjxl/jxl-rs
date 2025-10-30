// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
use super::*;
use crate::dct::{dct2d, DCT1DImpl, DCT1D};
use jxl_simd::{test_all_instruction_sets, ScalarDescriptor, SimdDescriptor};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use test_log::test;

use std::f64::consts::FRAC_1_SQRT_2;
use std::f64::consts::PI;
use std::f64::consts::SQRT_2;

#[inline(always)]
fn alpha(u: usize) -> f64 {
    if u == 0 {
        FRAC_1_SQRT_2
    } else {
        1.0
    }
}

pub fn dct1d(input_matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let num_rows = input_matrix.len();

    if num_rows == 0 {
        return Vec::new();
    }

    let num_cols = input_matrix[0].len();

    let mut output_matrix = vec![vec![0.0f64; num_cols]; num_rows];

    let scale: f64 = SQRT_2;

    // Precompute the DCT matrix (size: n_rows x n_rows)
    let mut dct_coeff_matrix = vec![vec![0.0f64; num_rows]; num_rows];
    for (u_freq, row) in dct_coeff_matrix.iter_mut().enumerate() {
        let alpha_u_val = alpha(u_freq);
        for (y_spatial, coeff) in row.iter_mut().enumerate() {
            *coeff = alpha_u_val
                * ((y_spatial as f64 + 0.5) * u_freq as f64 * PI / num_rows as f64).cos()
                * scale;
        }
    }

    // Perform the DCT calculation column by column
    for x_col_idx in 0..num_cols {
        for u_freq_idx in 0..num_rows {
            let mut sum = 0.0;
            for (y_spatial_idx, col) in input_matrix.iter().enumerate() {
                // This access `input_matrix[y_spatial_idx][x_col_idx]` assumes the input_matrix
                // is rectangular. If not, it might panic here.
                sum += dct_coeff_matrix[u_freq_idx][y_spatial_idx] * col[x_col_idx];
            }
            output_matrix[u_freq_idx][x_col_idx] = sum;
        }
    }

    output_matrix
}

pub fn idct1d(input_matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let num_rows = input_matrix.len();

    if num_rows == 0 {
        return Vec::new();
    }

    let num_cols = input_matrix[0].len();

    let mut output_matrix = vec![vec![0.0f64; num_cols]; num_rows];

    let scale: f64 = SQRT_2;

    // Precompute the DCT matrix (size: num_rows x num_rows)
    let mut dct_coeff_matrix = vec![vec![0.0f64; num_rows]; num_rows];
    for (u_freq, row) in dct_coeff_matrix.iter_mut().enumerate() {
        let alpha_u_val = alpha(u_freq);
        for (y_def_idx, coeff) in row.iter_mut().enumerate() {
            *coeff = alpha_u_val
                * ((y_def_idx as f64 + 0.5) * u_freq as f64 * PI / num_rows as f64).cos()
                * scale;
        }
    }

    // Perform the IDCT calculation column by column
    for x_col_idx in 0..num_cols {
        for (y_row_idx, row) in output_matrix.iter_mut().enumerate() {
            let mut sum = 0.0;
            for (u_freq_idx, col) in input_matrix.iter().enumerate() {
                // This access input_coeffs_matrix[u_freq_idx][x_col_idx] assumes input_coeffs_matrix
                // is rectangular. If not, it might panic here.
                sum += dct_coeff_matrix[u_freq_idx][y_row_idx] * col[x_col_idx];
            }
            row[x_col_idx] = sum;
        }
    }

    output_matrix
}

fn transpose_f64(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if matrix.is_empty() {
        return Vec::new();
    }
    let num_rows = matrix.len();
    let num_cols = matrix[0].len();
    let mut transposed = vec![vec![0.0; num_rows]; num_cols];
    for i in 0..num_rows {
        for j in 0..num_cols {
            transposed[j][i] = matrix[i][j];
        }
    }
    transposed
}

pub fn slow_idct2d(input: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows = input.len();
    let cols = input[0].len();
    let idct1 = if rows < cols {
        let transposed = transpose_f64(input);
        idct1d(&transposed)
    } else {
        let input: Vec<_> = input.iter().flat_map(|x| x.iter()).copied().collect();
        let input: Vec<_> = input.chunks_exact(rows).map(|x| x.to_vec()).collect();
        idct1d(&input)
    };
    let transposed1 = transpose_f64(&idct1);
    idct1d(&transposed1)
}

#[track_caller]
fn check_close(a: f64, b: f64, max_err: f64) {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs());
    assert!(
        abs < max_err || rel < max_err,
        "a: {a} b: {b} abs diff: {abs:?} rel diff: {rel:?}"
    );
}

#[track_caller]
fn check_all_close(a: &[f64], b: &[f64], max_abs: f64) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b.iter()) {
        check_close(*a, *b, max_abs);
    }
}

#[test]
fn test_dct_idct_scaling() {
    const N_ROWS: usize = 7;
    const M_COLS: usize = 13;
    let input_matrix: Vec<Vec<f64>> = (0..N_ROWS)
        .map(|r_idx| {
            (0..M_COLS)
                // some arbitrary pattern
                .map(|c_idx| (r_idx + c_idx) as f64 * 7.7)
                .collect::<Vec<f64>>()
        })
        .collect::<Vec<Vec<f64>>>();

    let dct_output = dct1d(&input_matrix);
    let idct_output = idct1d(&dct_output);

    // Verify that idct1d(dct1d(input)) == N_ROWS * input
    for r_idx in 0..N_ROWS {
        let expected_current_row_scaled: Vec<f64> = input_matrix[r_idx]
            .iter()
            .map(|&val| val * (N_ROWS as f64))
            .collect();

        check_all_close(&idct_output[r_idx], &expected_current_row_scaled, 1e-7);
    }
}

#[test]
fn test_idct_dct_scaling() {
    const N_ROWS: usize = 17;
    const M_COLS: usize = 11;
    let input_matrix: Vec<Vec<f64>> = (0..N_ROWS)
        .map(|r_idx| {
            (0..M_COLS)
                // some arbitrary pattern
                .map(|c_idx| (r_idx + c_idx) as f64 * 12.34)
                .collect::<Vec<f64>>()
        })
        .collect::<Vec<Vec<f64>>>();

    let idct_output = idct1d(&input_matrix);
    let dct_output = dct1d(&idct_output);

    // Verify that dct1d(idct1d(input)) == N_ROWS * input
    for r_idx in 0..N_ROWS {
        let expected_current_row_scaled: Vec<f64> = input_matrix[r_idx]
            .iter()
            .map(|&val| val * (N_ROWS as f64))
            .collect();

        check_all_close(&dct_output[r_idx], &expected_current_row_scaled, 1e-7);
    }
}

macro_rules! test_dct1d_eq_slow_n {
    ($test_name:ident, $n_val:expr, $tolerance:expr) => {
        #[test]
        fn $test_name() {
            const N: usize = $n_val;
            const M: usize = 1;
            const NM: usize = N * M;

            // Generate input data for the reference dct1d.
            // Results in vec![vec![1.0], vec![2.0], ..., vec![N.0]]
            let input_matrix_for_ref: Vec<Vec<f64>> =
                std::array::from_fn::<f64, NM, _>(|i| (i + 1) as f64)
                    .chunks(M)
                    .map(|row_slice| row_slice.to_vec())
                    .collect();

            let output_matrix_slow: Vec<Vec<f64>> = dct1d(&input_matrix_for_ref);

            // DCT1DImpl expects data in [[f32; M]; N] format.
            let mut input_arr_2d = [[0.0f32; M]; N];
            for r_idx in 0..N {
                for c_idx in 0..M {
                    input_arr_2d[r_idx][c_idx] = input_matrix_for_ref[r_idx][c_idx] as f32;
                }
            }

            let mut output = input_arr_2d;
            let d = ScalarDescriptor {};
            DCT1DImpl::<N>::do_dct::<_, M>(d, &mut output);

            for i in 0..N {
                check_close(output[i][0] as f64, output_matrix_slow[i][0], $tolerance);
            }
        }
    };
}

test_dct1d_eq_slow_n!(test_dct1d_1x1_eq_slow, 1, 1e-6);
test_dct1d_eq_slow_n!(test_dct1d_2x1_eq_slow, 2, 1e-6);
test_dct1d_eq_slow_n!(test_dct1d_4x1_eq_slow, 4, 1e-6);
test_dct1d_eq_slow_n!(test_dct1d_8x1_eq_slow, 8, 1e-5);
test_dct1d_eq_slow_n!(test_dct1d_16x1_eq_slow, 16, 1e-4);
test_dct1d_eq_slow_n!(test_dct1d_32x1_eq_slow, 32, 1e-3);
test_dct1d_eq_slow_n!(test_dct1d_64x1_eq_slow, 64, 1e-2);
test_dct1d_eq_slow_n!(test_dct1d_128x1_eq_slow, 128, 1e-2);
test_dct1d_eq_slow_n!(test_dct1d_256x1_eq_slow, 256, 1e-1);

fn random_matrix(n: usize, m: usize) -> Vec<Vec<f64>> {
    let mut rng = ChaCha12Rng::seed_from_u64(0);
    let mut data = vec![vec![0.0; m]; n];

    data.iter_mut()
        .flat_map(|x| x.iter_mut())
        .for_each(|x| *x = rng.random_range(-1.0..1.0));

    data
}

macro_rules! test_idct1d_eq_slow_n {
    ($test_name:ident, $n_val:expr, $do_idct_fun:path, $tolerance:expr) => {
        #[test]
        fn $test_name() {
            const N: usize = $n_val;

            let input_matrix_for_ref = random_matrix(N, 1);

            let output_matrix_slow: Vec<Vec<f64>> = idct1d(&input_matrix_for_ref);

            let mut output: Vec<_> = input_matrix_for_ref.iter().map(|x| x[0] as f32).collect();
            let d = ScalarDescriptor {};

            let (output_chunks, remainder) = output.as_chunks_mut::<1>();
            assert!(remainder.is_empty());
            $do_idct_fun(d, output_chunks, 1);

            for i in 0..N {
                check_close(output[i] as f64, output_matrix_slow[i][0], $tolerance);
            }
        }
    };
}

test_idct1d_eq_slow_n!(test_idct1d_2_eq_slow, 2, do_idct_2, 1e-6);
test_idct1d_eq_slow_n!(test_idct1d_4_eq_slow, 4, do_idct_4, 1e-6);
test_idct1d_eq_slow_n!(test_idct1d_8_eq_slow, 8, do_idct_8, 1e-6);
test_idct1d_eq_slow_n!(test_idct1d_16_eq_slow, 16, do_idct_16, 1e-6);
test_idct1d_eq_slow_n!(test_idct1d_32_eq_slow, 32, do_idct_32, 5e-6);
test_idct1d_eq_slow_n!(test_idct1d_64_eq_slow, 64, do_idct_64, 5e-6);
test_idct1d_eq_slow_n!(test_idct1d_128_eq_slow, 128, do_idct_128, 5e-5);
test_idct1d_eq_slow_n!(test_idct1d_256_eq_slow, 256, do_idct_256, 5e-5);

macro_rules! test_idct2d_eq_slow {
    ($test_name:ident, $rows:expr, $cols:expr, $fast_idct:ident, $tol:expr) => {
        fn $test_name<D: SimdDescriptor>(d: D) {
            const N: usize = $rows;
            const M: usize = $cols;

            let slow_input = random_matrix(N, M);

            let slow_output = slow_idct2d(&slow_input);

            let mut fast_input: Vec<_> = slow_input
                .iter()
                .flat_map(|x| x.iter())
                .map(|x| *x as f32)
                .collect();

            $fast_idct(d, &mut fast_input);

            for r in 0..N {
                for c in 0..M {
                    check_close(fast_input[r * M + c] as f64, slow_output[r][c], $tol);
                }
            }
        }
        test_all_instruction_sets!($test_name);
    };
}

test_idct2d_eq_slow!(test_idct2d_2_2_eq_slow, 2, 2, idct2d_2_2, 1e-6);
test_idct2d_eq_slow!(test_idct2d_4_4_eq_slow, 4, 4, idct2d_4_4, 1e-6);
test_idct2d_eq_slow!(test_idct2d_4_8_eq_slow, 4, 8, idct2d_4_8, 1e-6);
test_idct2d_eq_slow!(test_idct2d_8_4_eq_slow, 8, 4, idct2d_8_4, 1e-6);
test_idct2d_eq_slow!(test_idct2d_8_8_eq_slow, 8, 8, idct2d_8_8, 5e-6);
test_idct2d_eq_slow!(test_idct2d_16_8_eq_slow, 16, 8, idct2d_16_8, 5e-6);
test_idct2d_eq_slow!(test_idct2d_8_16_eq_slow, 8, 16, idct2d_8_16, 5e-6);
test_idct2d_eq_slow!(test_idct2d_16_16_eq_slow, 16, 16, idct2d_16_16, 1e-5);
test_idct2d_eq_slow!(test_idct2d_32_8_eq_slow, 32, 8, idct2d_32_8, 5e-6);
test_idct2d_eq_slow!(test_idct2d_8_32_eq_slow, 8, 32, idct2d_8_32, 5e-6);
test_idct2d_eq_slow!(test_idct2d_32_16_eq_slow, 32, 16, idct2d_32_16, 1e-5);
test_idct2d_eq_slow!(test_idct2d_16_32_eq_slow, 16, 32, idct2d_16_32, 1e-5);
test_idct2d_eq_slow!(test_idct2d_32_32_eq_slow, 32, 32, idct2d_32_32, 5e-5);
test_idct2d_eq_slow!(test_idct2d_64_32_eq_slow, 64, 32, idct2d_64_32, 1e-4);
test_idct2d_eq_slow!(test_idct2d_32_64_eq_slow, 32, 64, idct2d_32_64, 1e-4);
test_idct2d_eq_slow!(test_idct2d_64_64_eq_slow, 64, 64, idct2d_64_64, 1e-4);
test_idct2d_eq_slow!(test_idct2d_128_64_eq_slow, 128, 64, idct2d_128_64, 5e-4);
test_idct2d_eq_slow!(test_idct2d_64_128_eq_slow, 64, 128, idct2d_64_128, 5e-4);
test_idct2d_eq_slow!(test_idct2d_128_128_eq_slow, 128, 128, idct2d_128_128, 5e-4);
test_idct2d_eq_slow!(test_idct2d_256_128_eq_slow, 256, 128, idct2d_256_128, 1e-3);
test_idct2d_eq_slow!(test_idct2d_128_256_eq_slow, 128, 256, idct2d_128_256, 1e-3);
test_idct2d_eq_slow!(test_idct2d_256_256_eq_slow, 256, 256, idct2d_256_256, 5e-3);

// TODO(firsching): possibly change these tests to test against slow
// (i)dct method (after adding 2d-variant there)
macro_rules! test_dct2d_exists_n_m {
    ($test_name:ident, $n_val:expr, $m_val:expr) => {
        #[test]
        fn $test_name() {
            const N: usize = $n_val;
            const M: usize = $m_val;
            let mut data = [0.0f32; M * N];
            let mut scratch = vec![0.0; M * N];
            let d = ScalarDescriptor {};
            dct2d::<_, N, M>(d, &mut data, &mut scratch);
        }
    };
}
test_dct2d_exists_n_m!(test_dct2d_exists_1_1, 1, 1);
test_dct2d_exists_n_m!(test_dct2d_exists_1_2, 1, 2);
test_dct2d_exists_n_m!(test_dct2d_exists_1_4, 1, 4);
test_dct2d_exists_n_m!(test_dct2d_exists_1_8, 1, 8);
test_dct2d_exists_n_m!(test_dct2d_exists_1_16, 1, 16);
test_dct2d_exists_n_m!(test_dct2d_exists_1_32, 1, 32);
test_dct2d_exists_n_m!(test_dct2d_exists_1_64, 1, 64);
test_dct2d_exists_n_m!(test_dct2d_exists_1_128, 1, 128);
test_dct2d_exists_n_m!(test_dct2d_exists_1_256, 1, 256);
test_dct2d_exists_n_m!(test_dct2d_exists_2_1, 2, 1);
test_dct2d_exists_n_m!(test_dct2d_exists_2_2, 2, 2);
test_dct2d_exists_n_m!(test_dct2d_exists_2_4, 2, 4);
test_dct2d_exists_n_m!(test_dct2d_exists_2_8, 2, 8);
test_dct2d_exists_n_m!(test_dct2d_exists_2_16, 2, 16);
test_dct2d_exists_n_m!(test_dct2d_exists_2_32, 2, 32);
test_dct2d_exists_n_m!(test_dct2d_exists_2_64, 2, 64);
test_dct2d_exists_n_m!(test_dct2d_exists_2_128, 2, 128);
test_dct2d_exists_n_m!(test_dct2d_exists_2_256, 2, 256);
test_dct2d_exists_n_m!(test_dct2d_exists_4_1, 4, 1);
test_dct2d_exists_n_m!(test_dct2d_exists_4_2, 4, 2);
test_dct2d_exists_n_m!(test_dct2d_exists_4_4, 4, 4);
test_dct2d_exists_n_m!(test_dct2d_exists_4_8, 4, 8);
test_dct2d_exists_n_m!(test_dct2d_exists_4_16, 4, 16);
test_dct2d_exists_n_m!(test_dct2d_exists_4_32, 4, 32);
test_dct2d_exists_n_m!(test_dct2d_exists_4_64, 4, 64);
test_dct2d_exists_n_m!(test_dct2d_exists_4_128, 4, 128);
test_dct2d_exists_n_m!(test_dct2d_exists_4_256, 4, 256);
test_dct2d_exists_n_m!(test_dct2d_exists_8_1, 8, 1);
test_dct2d_exists_n_m!(test_dct2d_exists_8_2, 8, 2);
test_dct2d_exists_n_m!(test_dct2d_exists_8_4, 8, 4);
test_dct2d_exists_n_m!(test_dct2d_exists_8_8, 8, 8);
test_dct2d_exists_n_m!(test_dct2d_exists_8_16, 8, 16);
test_dct2d_exists_n_m!(test_dct2d_exists_8_32, 8, 32);
test_dct2d_exists_n_m!(test_dct2d_exists_8_64, 8, 64);
test_dct2d_exists_n_m!(test_dct2d_exists_8_128, 8, 128);
test_dct2d_exists_n_m!(test_dct2d_exists_8_256, 8, 256);
test_dct2d_exists_n_m!(test_dct2d_exists_16_1, 16, 1);
test_dct2d_exists_n_m!(test_dct2d_exists_16_2, 16, 2);
test_dct2d_exists_n_m!(test_dct2d_exists_16_4, 16, 4);
test_dct2d_exists_n_m!(test_dct2d_exists_16_8, 16, 8);
test_dct2d_exists_n_m!(test_dct2d_exists_16_16, 16, 16);
test_dct2d_exists_n_m!(test_dct2d_exists_16_32, 16, 32);
test_dct2d_exists_n_m!(test_dct2d_exists_16_64, 16, 64);
test_dct2d_exists_n_m!(test_dct2d_exists_16_128, 16, 128);
test_dct2d_exists_n_m!(test_dct2d_exists_16_256, 16, 256);
test_dct2d_exists_n_m!(test_dct2d_exists_32_1, 32, 1);
test_dct2d_exists_n_m!(test_dct2d_exists_32_2, 32, 2);
test_dct2d_exists_n_m!(test_dct2d_exists_32_4, 32, 4);
test_dct2d_exists_n_m!(test_dct2d_exists_32_8, 32, 8);
test_dct2d_exists_n_m!(test_dct2d_exists_32_16, 32, 16);
test_dct2d_exists_n_m!(test_dct2d_exists_32_32, 32, 32);
test_dct2d_exists_n_m!(test_dct2d_exists_32_64, 32, 64);
test_dct2d_exists_n_m!(test_dct2d_exists_32_128, 32, 128);
test_dct2d_exists_n_m!(test_dct2d_exists_32_256, 32, 256);
test_dct2d_exists_n_m!(test_dct2d_exists_64_1, 64, 1);
test_dct2d_exists_n_m!(test_dct2d_exists_64_2, 64, 2);
test_dct2d_exists_n_m!(test_dct2d_exists_64_4, 64, 4);
test_dct2d_exists_n_m!(test_dct2d_exists_64_8, 64, 8);
test_dct2d_exists_n_m!(test_dct2d_exists_64_16, 64, 16);
test_dct2d_exists_n_m!(test_dct2d_exists_64_32, 64, 32);
test_dct2d_exists_n_m!(test_dct2d_exists_64_64, 64, 64);
test_dct2d_exists_n_m!(test_dct2d_exists_64_128, 64, 128);
test_dct2d_exists_n_m!(test_dct2d_exists_64_256, 64, 256);
test_dct2d_exists_n_m!(test_dct2d_exists_128_1, 128, 1);
test_dct2d_exists_n_m!(test_dct2d_exists_128_2, 128, 2);
test_dct2d_exists_n_m!(test_dct2d_exists_128_4, 128, 4);
test_dct2d_exists_n_m!(test_dct2d_exists_128_8, 128, 8);
test_dct2d_exists_n_m!(test_dct2d_exists_128_16, 128, 16);
test_dct2d_exists_n_m!(test_dct2d_exists_128_32, 128, 32);
test_dct2d_exists_n_m!(test_dct2d_exists_128_64, 128, 64);
test_dct2d_exists_n_m!(test_dct2d_exists_128_128, 128, 128);
test_dct2d_exists_n_m!(test_dct2d_exists_128_256, 128, 256);
test_dct2d_exists_n_m!(test_dct2d_exists_256_1, 256, 1);
test_dct2d_exists_n_m!(test_dct2d_exists_256_2, 256, 2);
test_dct2d_exists_n_m!(test_dct2d_exists_256_4, 256, 4);
test_dct2d_exists_n_m!(test_dct2d_exists_256_8, 256, 8);
test_dct2d_exists_n_m!(test_dct2d_exists_256_16, 256, 16);
test_dct2d_exists_n_m!(test_dct2d_exists_256_32, 256, 32);
test_dct2d_exists_n_m!(test_dct2d_exists_256_64, 256, 64);
test_dct2d_exists_n_m!(test_dct2d_exists_256_128, 256, 128);
test_dct2d_exists_n_m!(test_dct2d_exists_256_256, 256, 256);
