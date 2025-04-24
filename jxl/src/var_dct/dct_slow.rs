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
// TODO(firsching): Consider to just use a Vec and skip the templates,
// since this is slow anyway
pub fn dct1d<const N: usize, const M: usize, const NM: usize>(
    input: &[f64; NM],
    out: &mut [f64; NM],
) {
    const { assert!(NM == N * M, "NM must be equal to N * M") };
    let scale: f64 = SQRT_2;

    let mut matrix = [[0.0f64; N]; N];
    for (u, row) in matrix.iter_mut().enumerate() {
        let alpha_u = alpha(u);
        for (y, element) in row.iter_mut().enumerate() {
            *element = alpha_u * ((y as f64 + 0.5) * u as f64 * PI / N as f64).cos() * scale;
        }
    }

    for x in 0..M {
        for u in 0..N {
            let mut sum = 0.0;
            for y in 0..N {
                sum += matrix[u][y] * input[M * y + x];
            }
            out[M * u + x] = sum;
        }
    }
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
        const N: usize = 8;
        const M: usize = 1;
        const NM: usize = N * M;
        let input: [f64; NM] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let mut output = [0.0; NM];

        dct1d::<8, 1, 8>(&input, &mut output);
        // obtained with the following python code:
        // import math
        // import scipy.fft
        // scipy.fft.dct(list(map(lambda x: 2*math.sqrt(x)*x , map(float, range(8)))), norm='ortho')
        // import scipy.fft; scipy.fft.dct(list(map(float, range(8))), norm='ortho')
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
        assert_all_almost_eq!(output, expected, 1e-7);
    }

    #[test]
    fn test_slow_dct1d_same_on_columns() {
        const N: usize = 8;
        const M: usize = 5;
        const NM: usize = N * M;

        let input: [f64; NM] = array::from_fn(|i| ((i as i64) / 5) as f64);
        let mut output = [0.0; NM];

        dct1d::<N, M, NM>(&input, &mut output);

        // Expected output should be M copies of the result of applying idct1d to a single column [0.0 .. N-1.0]
        // We take the expected result from the single-column test `test_slow_dct1d`
        let initial = [
            2.80000000e+01,
            -1.82216412e+01,
            -1.38622135e-15,
            -1.90481783e+00,
            0.00000000e+00,
            -5.68239222e-01,
            -1.29520973e-15,
            -1.43407825e-01,
        ];

        // Create an iterator that repeats each element 5 times and flattens the result
        let generated_iter = initial
            .iter()
            .flat_map(|&element| iter::repeat_n(element, M));

        let expected: Vec<f64> = generated_iter.collect();
        assert_all_almost_eq!(output, expected, 1e-7);
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
