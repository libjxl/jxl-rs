// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(dead_code)]

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

// TODO: write commment/test showing that "i" in "idct" is not exactly inverse, but scaled (by sqrt N)
pub fn idct1d<const N: usize, const M: usize, const NM: usize>(
    input: &[f64; NM],
    out: &mut [f64; NM],
) {
    const { assert!(NM == N * M, "NM must be equal to N * M") };
    let scale = SQRT_2;

    let mut matrix = [[0.0f64; N]; N];
    for (u, row) in matrix.iter_mut().enumerate() {
        let alpha_u = alpha(u);
        for (y, element) in row.iter_mut().enumerate() {
            // Transpose of DCT matrix.
            *element = alpha_u * ((y as f64 + 0.5) * u as f64 * PI / N as f64).cos() * scale;
        }
    }

    for x in 0..M {
        for y in 0..N {
            let mut sum = 0.0;
            for u in 0..N {
                sum += matrix[u][y] * input[M * u + x];
            }
            out[M * y + x] = sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{array, iter};
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
        let input_matrix: Vec<Vec<f64>> = flat_input_data
            .iter()
            .map(|&value| vec![value])
            .collect();

        // Call the refactored dct1d function which returns a new matrix
        let output_matrix: Vec<Vec<f64>> = dct1d(&input_matrix);

        // Extract the first (and only) column from output_matrix for comparison
        let mut result_column: Vec<f64> = Vec::with_capacity(N_ROWS);
        if M_COLS > 0 {
            for i in 0..N_ROWS {
                result_column.push(output_matrix[i][0]);
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
        let input_matrix: Vec<Vec<f64>> = (0..N_ROWS)
            .map(|r| vec![r as f64; M_COLS])
            .collect();

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
        const NM: usize = N * M;
        let input: [f64; N] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let mut output = [0.0; NM];

        idct1d::<8, 1, 8>(&input, &mut output);
        // obtained with the following python code:
        //
        // import math
        // import scipy.fft
        // scipy.fft.idct(list(map(lambda x: x * 2 * math.sqrt(2), map(float, range(8)))), norm='ortho')
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
        assert_all_almost_eq!(output, expected, 1e-7);
        let input: [f64; 2] = [1.0, 3.0];
        let mut output = [0.0; 2];
        idct1d::<2, 1, 2>(&input, &mut output);
    }
    #[test]
    fn test_slow_idct1d_same_on_columns() {
        const N: usize = 8;
        const M: usize = 5;
        const NM: usize = N * M;

        let input: [f64; NM] = array::from_fn(|i| ((i / M) as f64));
        let mut output = [0.0; NM];

        idct1d::<N, M, NM>(&input, &mut output);

        // Expected output should be M copies of the result of applying idct1d to a single column [0.0 .. N-1.0]
        // We take the expected result from the single-column test `test_slow_idct1d`
        let initial = [
            20.63473963,
            -22.84387206,
            8.99218712,
            -7.77138893,
            4.05078387,
            -3.47821595,
            1.32990088,
            -0.91413457,
        ];

        // Create an iterator that repeats each element M times (column-wise)
        let generated_iter = initial
            .iter()
            .flat_map(|&element| iter::repeat_n(element, M));

        let expected: Vec<f64> = generated_iter.collect();
        assert_all_almost_eq!(output, expected, 1e-7);
    }
}
