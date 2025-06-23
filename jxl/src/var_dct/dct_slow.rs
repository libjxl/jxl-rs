// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![cfg(test)]

use std::f64::consts::FRAC_1_SQRT_2;
use std::f64::consts::PI;
use std::f64::consts::SQRT_2;

#[inline(always)]
fn alpha(u: usize) -> f64 {
    if u == 0 { FRAC_1_SQRT_2 } else { 1.0 }
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

#[cfg(test)]
mod tests {
    use test_log::test;

    use crate::util::test::assert_all_almost_eq;

    use super::*;

    #[test]
    fn test_slow_dct1d() {
        const N_ROWS: usize = 8;
        const M_COLS: usize = 1;

        let flat_input_data: [f64; N_ROWS] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

        // Prepare input_matrix for dct1d
        // It expects Vec<Vec<f64>> structured as input_matrix[row_idx][col_idx]
        // For N_ROWS=8, M_COLS=1, this means 8 rows, each containing a Vec with 1 element.
        let input_matrix: Vec<Vec<f64>> =
            flat_input_data.iter().map(|&value| vec![value]).collect();

        // Call the refactored dct1d function which returns a new matrix
        let output_matrix: Vec<Vec<f64>> = dct1d(&input_matrix);

        // Extract the first (and only) column from output_matrix for comparison
        let mut result_column: Vec<f64> = Vec::with_capacity(N_ROWS);
        if M_COLS > 0 {
            for row in output_matrix.iter() {
                result_column.push(row[0]);
            }
        }

        let expected = [
            2.80000000e+01,
            -1.82216412e+01,
            -1.38622135e-15,
            -1.90481783e+00,
            0.00000000e+00,
            -5.68239222e-01,
            -1.29520973e-15,
            -1.43407825e-01,
        ];
        // Ensure assert_all_almost_eq can compare Vec<f64> (or slice) with [f64; N]
        assert_all_almost_eq!(result_column.as_slice(), expected.as_slice(), 1e-7);
    }

    #[test]
    fn test_slow_dct1d_same_on_columns() {
        const N_ROWS: usize = 8;
        const M_COLS: usize = 5;

        // Prepare input_matrix for dct1d
        // It expects Vec<Vec<f64>> structured as input_matrix[row_idx][col_idx].
        // Each column of the input should be [0.0, 1.0, ..., N_ROWS-1.0].
        let input_matrix: Vec<Vec<f64>> = (0..N_ROWS).map(|r| vec![r as f64; M_COLS]).collect();

        // Call the refactored dct1d function which returns a new matrix
        let output_matrix: Vec<Vec<f64>> = dct1d(&input_matrix);

        // Expected output for a single column [0.0 .. N_ROWS-1.0]
        let single_column_dct_expected = [
            2.80000000e+01,
            -1.82216412e+01,
            -1.38622135e-15,
            -1.90481783e+00,
            0.00000000e+00,
            -5.68239222e-01,
            -1.29520973e-15,
            -1.43407825e-01,
        ];

        for r_freq_idx in 0..N_ROWS {
            let actual_row_slice: &[f64] = output_matrix[r_freq_idx].as_slice();
            let expected_row_values: Vec<f64> =
                vec![single_column_dct_expected[r_freq_idx]; M_COLS];
            assert_all_almost_eq!(actual_row_slice, expected_row_values.as_slice(), 1e-7);
        }
    }

    #[test]
    fn test_slow_idct1d() {
        const N: usize = 8;
        const M: usize = 1;

        let flat_input_data: [f64; N] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

        let input_coeffs_matrix_p1: Vec<Vec<f64>> =
            flat_input_data.iter().map(|&value| vec![value]).collect();
        // Prepare input_matrix for dct1d
        // It expects Vec<Vec<f64>> structured as input_matrix[row_idx][col_idx].
        // Each column of the input should be [0.0, 1.0, ..., N_ROWS-1.0].k
        let output_matrix: Vec<Vec<f64>> = idct1d(&input_coeffs_matrix_p1);

        let mut result_column: Vec<f64> = Vec::with_capacity(N);
        if M > 0 {
            for row_vec in output_matrix.iter() {
                result_column.push(row_vec[0]);
            }
        }

        let expected = [
            20.63473963,
            -22.84387206,
            8.99218712,
            -7.77138893,
            4.05078387,
            -3.47821595,
            1.32990088,
            -0.91413457,
        ];
        assert_all_almost_eq!(result_column.as_slice(), expected.as_slice(), 1e-7);
    }

    #[test]
    fn test_slow_idct1d_same_on_columns() {
        const N_ROWS: usize = 8;
        const M_COLS: usize = 5;

        // Prepare input_matrix for idct1d
        // It expects Vec<Vec<f64>> structured as input_matrix[row_idx][col_idx].
        // Each column of the input should be [0.0, 1.0, ..., N_ROWS-1.0].
        let input_matrix: Vec<Vec<f64>> = (0..N_ROWS).map(|r| vec![r as f64; M_COLS]).collect();

        // Call the refactored idct1d function which returns a new matrix
        let output_matrix: Vec<Vec<f64>> = idct1d(&input_matrix);

        // Expected spatial output for a single input coefficient column [0.0 .. N_FREQUENCIES-1.0]
        // This is taken from the single-column test `test_slow_idct1d`
        let single_column_idct_expected = [
            20.63473963,
            -22.84387206,
            8.99218712,
            -7.77138893,
            4.05078387,
            -3.47821595,
            1.32990088,
            -0.91413457,
        ];

        // Verify each row of output_spatial_matrix.
        // The row output_spatial_matrix[r_spatial_idx] should consist of M_COLS elements,
        // all equal to single_column_idct_expected[r_spatial_idx].
        for r_spatial_idx in 0..N_ROWS {
            // Iterate over spatial output rows
            let actual_row_slice: &[f64] = output_matrix[r_spatial_idx].as_slice();
            let expected_row_values: Vec<f64> =
                vec![single_column_idct_expected[r_spatial_idx]; M_COLS];
            assert_all_almost_eq!(actual_row_slice, expected_row_values.as_slice(), 1e-7);
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

            assert_all_almost_eq!(
                idct_output[r_idx].as_slice(),
                expected_current_row_scaled.as_slice(),
                1e-7
            );
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

            assert_all_almost_eq!(
                dct_output[r_idx].as_slice(),
                expected_current_row_scaled.as_slice(),
                1e-7
            );
        }
    }
}
