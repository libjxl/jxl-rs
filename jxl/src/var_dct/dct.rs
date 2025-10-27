// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use super::dct_scales::WcMultipliers;
use crate::simd::{F32SimdVec, SimdDescriptor};
use std::f64::consts::SQRT_2;

struct CoeffBundle<const N: usize, const SZ: usize>;

pub struct DCT1DImpl<const SIZE: usize>;
pub struct IDCT1DImpl<const SIZE: usize>;

pub trait DCT1D {
    fn do_dct<D: SimdDescriptor, const COLUMNS: usize>(
        d: D,
        data: &mut [[f32; COLUMNS]],
        starting_column: usize,
        num_columns: usize,
    );
}
pub trait IDCT1D {
    fn do_idct<D: SimdDescriptor, const COLUMNS: usize>(
        d: D,
        data: &mut [[f32; COLUMNS]],
        starting_column: usize,
        num_columns: usize,
    );
}

impl DCT1D for DCT1DImpl<1> {
    #[inline(always)]
    fn do_dct<D: SimdDescriptor, const COLUMNS: usize>(
        _d: D,
        _data: &mut [[f32; COLUMNS]],
        _starting_column: usize,
        _num_columns: usize,
    ) {
        // Do nothing
    }
}
impl IDCT1D for IDCT1DImpl<1> {
    #[inline(always)]
    fn do_idct<D: SimdDescriptor, const COLUMNS: usize>(
        _d: D,
        _data: &mut [[f32; COLUMNS]],
        _starting_column: usize,
        _num_columns: usize,
    ) {
        // Do nothing
    }
}

impl DCT1D for DCT1DImpl<2> {
    #[inline(always)]
    fn do_dct<D: SimdDescriptor, const COLUMNS: usize>(
        d: D,
        data: &mut [[f32; COLUMNS]],
        starting_column: usize,
        num_columns: usize,
    ) {
        let temp0 = D::F32Vec::load_partial(d, num_columns, &data[0][starting_column..]);
        let temp1 = D::F32Vec::load_partial(d, num_columns, &data[1][starting_column..]);
        (temp0 + temp1).store_partial(num_columns, &mut data[0][starting_column..]);
        (temp0 - temp1).store_partial(num_columns, &mut data[1][starting_column..]);
    }
}

impl IDCT1D for IDCT1DImpl<2> {
    #[inline(always)]
    fn do_idct<D: SimdDescriptor, const COLUMNS: usize>(
        d: D,
        data: &mut [[f32; COLUMNS]],
        starting_column: usize,
        num_columns: usize,
    ) {
        DCT1DImpl::<2>::do_dct::<D, COLUMNS>(d, data, starting_column, num_columns)
    }
}

/// Helper macro to conditionally wrap recursive DCT calls in d.call() based on size.
/// For small sizes (≤4), call directly to reduce compilation time.
/// For larger sizes (>4), use d.call() to enable aggressive inlining within the boundary.
macro_rules! maybe_call_dct {
    // Small sizes: direct call
    ($d:expr, 2, $($call:tt)*) => {
        $($call)*
    };
    ($d:expr, 4, $($call:tt)*) => {
        $($call)*
    };
    // Larger sizes: wrap in d.call()
    ($d:expr, $size:literal, $($call:tt)*) => {
        $d.call(|_d| $($call)*)
    };
}

/// Helper macro for IDCT - same logic as maybe_call_dct
macro_rules! maybe_call_idct {
    // Small sizes: direct call
    ($d:expr, 2, $($call:tt)*) => {
        $($call)*
    };
    ($d:expr, 4, $($call:tt)*) => {
        $($call)*
    };
    // Larger sizes: wrap in d.call()
    ($d:expr, $size:literal, $($call:tt)*) => {
        $d.call(|_d| $($call)*)
    };
}

macro_rules! define_dct_1d {
    ($n:literal, $nhalf: literal) => {
        // Helper functions for CoeffBundle operating on $nhalf rows
        impl<const SZ: usize> CoeffBundle<$nhalf, SZ> {
            /// Adds a_in1[i] and a_in2[$nhalf - 1 - i], storing in a_out[i].
            #[inline(always)]
            fn add_reverse<D: SimdDescriptor>(
                d: D,
                a_in1: &[[f32; SZ]],
                a_in2: &[[f32; SZ]],
                a_out: &mut [[f32; SZ]],
                starting_column: usize,
                num_columns: usize,
            ) {
                const N_HALF_CONST: usize = $nhalf;
                for i in 0..N_HALF_CONST {
                    let j = starting_column;
                    let in1 = D::F32Vec::load_partial(d, num_columns, &a_in1[i][j..]);
                    let in2 =
                        D::F32Vec::load_partial(d, num_columns, &a_in2[N_HALF_CONST - 1 - i][j..]);
                    (in1 + in2).store_partial(num_columns, &mut a_out[i][j..]);
                }
            }

            /// Subtracts a_in2[$nhalf - 1 - i] from a_in1[i], storing in a_out[i].
            #[inline(always)]
            fn sub_reverse<D: SimdDescriptor>(
                d: D,
                a_in1: &[[f32; SZ]],
                a_in2: &[[f32; SZ]],
                a_out: &mut [[f32; SZ]],
                starting_column: usize,
                num_columns: usize,
            ) {
                const N_HALF_CONST: usize = $nhalf;
                for i in 0..N_HALF_CONST {
                    let j = starting_column;
                    let in1 = D::F32Vec::load_partial(d, num_columns, &a_in1[i][j..]);
                    let in2 =
                        D::F32Vec::load_partial(d, num_columns, &a_in2[N_HALF_CONST - 1 - i][j..]);
                    (in1 - in2).store_partial(num_columns, &mut a_out[i][j..]);
                }
            }

            /// Applies the B transform (forward DCT step).
            /// Operates on a slice of $nhalf rows.
            #[inline(always)]
            fn b<D: SimdDescriptor>(
                d: D,
                coeff: &mut [[f32; SZ]],
                starting_column: usize,
                num_columns: usize,
            ) {
                const N_HALF_CONST: usize = $nhalf;
                let sqrt2 = D::F32Vec::splat(d, SQRT_2 as f32);
                let j = starting_column;
                let coeff0 = D::F32Vec::load_partial(d, num_columns, &coeff[0][j..]);
                let coeff1 = D::F32Vec::load_partial(d, num_columns, &coeff[1][j..]);
                coeff0
                    .mul_add(sqrt2, coeff1)
                    .store_partial(num_columns, &mut coeff[0][j..]);
                // empty in the case N_HALF_CONST == 2
                #[allow(clippy::reversed_empty_ranges)]
                for i in 1..(N_HALF_CONST - 1) {
                    let coeffs_curr = D::F32Vec::load_partial(d, num_columns, &coeff[i][j..]);
                    let coeffs_next = D::F32Vec::load_partial(d, num_columns, &coeff[i + 1][j..]);
                    (coeffs_curr + coeffs_next).store_partial(num_columns, &mut coeff[i][j..]);
                }
            }
        }

        // Helper functions for CoeffBundle operating on $n rows
        impl<const SZ: usize> CoeffBundle<$n, SZ> {
            /// Multiplies the second half of `coeff` by WcMultipliers.
            #[inline(always)]
            fn multiply<D: SimdDescriptor>(
                d: D,
                coeff: &mut [[f32; SZ]],
                starting_column: usize,
                num_columns: usize,
            ) {
                const N_CONST: usize = $n;
                const N_HALF_CONST: usize = $nhalf;
                for i in 0..N_HALF_CONST {
                    let j = starting_column;
                    let mul_val = D::F32Vec::splat(d, WcMultipliers::<N_CONST>::K_MULTIPLIERS[i]);
                    let coeffs =
                        D::F32Vec::load_partial(d, num_columns, &coeff[N_HALF_CONST + i][j..]);
                    (coeffs * mul_val)
                        .store_partial(num_columns, &mut coeff[N_HALF_CONST + i][j..]);
                }
            }

            /// De-interleaves `a_in` into `a_out`.
            /// Even indexed rows of `a_out` get first half of `a_in`.
            /// Odd indexed rows of `a_out` get second half of `a_in`.
            #[inline(always)]
            fn inverse_even_odd<D: SimdDescriptor>(
                d: D,
                a_in: &[[f32; SZ]],
                a_out: &mut [[f32; SZ]],
                starting_column: usize,
                num_columns: usize,
            ) {
                const N_HALF_CONST: usize = $nhalf;
                for i in 0..N_HALF_CONST {
                    let j = starting_column;
                    D::F32Vec::load_partial(d, num_columns, &a_in[i][j..])
                        .store_partial(num_columns, &mut a_out[2 * i][j..]);
                }
                for i in 0..N_HALF_CONST {
                    let j = starting_column;
                    D::F32Vec::load_partial(d, num_columns, &a_in[N_HALF_CONST + i][j..])
                        .store_partial(num_columns, &mut a_out[2 * i + 1][j..]);
                }
            }
        }

        impl DCT1D for DCT1DImpl<$n> {
            #[inline(always)]
            fn do_dct<D: SimdDescriptor, const COLUMNS: usize>(
                d: D,
                data: &mut [[f32; COLUMNS]],
                starting_column: usize,
                num_columns: usize,
            ) {
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
                CoeffBundle::<$nhalf, COLUMNS>::add_reverse::<D>(
                    d,
                    &data[0..$nhalf],
                    &data[$nhalf..$n],
                    &mut tmp_buffer[0..$nhalf],
                    starting_column,
                    num_columns,
                );

                // 2. First Recursive Call (do_dct)
                //    first half
                maybe_call_dct!(d, $nhalf,
                    DCT1DImpl::<$nhalf>::do_dct::<D, COLUMNS>(
                        d,
                        &mut tmp_buffer[0..$nhalf],
                        starting_column,
                        num_columns,
                    )
                );

                // 3. SubReverse
                //    Inputs: first N/2 rows of data, second N/2 rows of data
                //    Output: second N/2 rows of tmp_buffer
                CoeffBundle::<$nhalf, COLUMNS>::sub_reverse::<D>(
                    d,
                    &data[0..$nhalf],
                    &data[$nhalf..$n],
                    &mut tmp_buffer[$nhalf..$n],
                    starting_column,
                    num_columns,
                );

                // 4. Multiply(tmp);
                //    Operates on the entire tmp_buffer.
                CoeffBundle::<$n, COLUMNS>::multiply::<D>(
                    d,
                    &mut tmp_buffer,
                    starting_column,
                    num_columns,
                );

                // 5. Second Recursive Call (do_dct)
                //    second half.
                maybe_call_dct!(d, $nhalf,
                    DCT1DImpl::<$nhalf>::do_dct::<D, COLUMNS>(
                        d,
                        &mut tmp_buffer[$nhalf..$n],
                        starting_column,
                        num_columns,
                    )
                );

                // 6. B
                //    Operates on the second N/2 rows of tmp_buffer.
                CoeffBundle::<$nhalf, COLUMNS>::b::<D>(
                    d,
                    &mut tmp_buffer[$nhalf..$n],
                    starting_column,
                    num_columns,
                );

                // 7. InverseEvenOdd
                CoeffBundle::<$n, COLUMNS>::inverse_even_odd::<D>(
                    d,
                    &tmp_buffer,
                    data,
                    starting_column,
                    num_columns,
                );
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
            fn b_transpose<D: SimdDescriptor>(
                d: D,
                coeff: &mut [[f32; SZ]],
                starting_column: usize,
                num_columns: usize,
            ) {
                let j = starting_column;
                for i in (1..$nhalf).rev() {
                    let coeffs_curr = D::F32Vec::load_partial(d, num_columns, &coeff[i][j..]);
                    let coeffs_prev = D::F32Vec::load_partial(d, num_columns, &coeff[i - 1][j..]);
                    (coeffs_curr + coeffs_prev).store_partial(num_columns, &mut coeff[i][j..]);
                }
                let sqrt2 = D::F32Vec::splat(d, SQRT_2 as f32);
                let coeffs = D::F32Vec::load_partial(d, num_columns, &coeff[0][j..]);
                (coeffs * sqrt2).store_partial(num_columns, &mut coeff[0][j..]);
            }
        }

        impl<const SZ: usize> CoeffBundle<$n, SZ> {
            fn forward_even_odd<D: SimdDescriptor>(
                d: D,
                a_in: &[[f32; SZ]],
                a_out: &mut [[f32; SZ]],
                starting_column: usize,
                num_columns: usize,
            ) {
                let j = starting_column;
                for i in 0..($nhalf) {
                    D::F32Vec::load_partial(d, num_columns, &a_in[2 * i][j..])
                        .store_partial(num_columns, &mut a_out[i][j..]);
                }
                for i in ($nhalf)..$n {
                    D::F32Vec::load_partial(d, num_columns, &a_in[2 * (i - $nhalf) + 1][j..])
                        .store_partial(num_columns, &mut a_out[i][j..]);
                }
            }
            fn multiply_and_add<D: SimdDescriptor>(
                d: D,
                coeff: &[[f32; SZ]],
                out: &mut [[f32; SZ]],
                starting_column: usize,
                num_columns: usize,
            ) {
                let j = starting_column;
                for i in 0..($nhalf) {
                    let mul = D::F32Vec::splat(d, WcMultipliers::<$n>::K_MULTIPLIERS[i]);
                    let in1 = D::F32Vec::load_partial(d, num_columns, &coeff[i][j..]);
                    let in2 = D::F32Vec::load_partial(d, num_columns, &coeff[$nhalf + i][j..]);
                    in2.mul_add(mul, in1)
                        .store_partial(num_columns, &mut out[i][j..]);
                    in2.mul_add(mul.neg(), in1)
                        .store_partial(num_columns, &mut out[($n - i - 1)][j..]);
                }
            }
        }

        impl IDCT1D for IDCT1DImpl<$n> {
            fn do_idct<D: SimdDescriptor, const COLUMNS: usize>(
                d: D,
                data: &mut [[f32; COLUMNS]],
                starting_column: usize,
                num_columns: usize,
            ) {
                const { assert!($nhalf * 2 == $n, "N/2 * 2 must be N") }

                // We assume `data` is arranged as a nxCOLUMNS matrix.

                let mut tmp = [[0.0f32; COLUMNS]; $n];

                // 1. ForwardEvenOdd
                CoeffBundle::<$n, COLUMNS>::forward_even_odd(
                    d,
                    data,
                    &mut tmp,
                    starting_column,
                    num_columns,
                );
                // 2. First Recursive Call (IDCT1DImpl::do_idct)
                // first half
                maybe_call_idct!(d, $nhalf,
                    IDCT1DImpl::<$nhalf>::do_idct::<D, COLUMNS>(
                        d,
                        &mut tmp[0..$nhalf],
                        starting_column,
                        num_columns,
                    )
                );
                // 3. BTranspose.
                // only the second half
                CoeffBundle::<$nhalf, COLUMNS>::b_transpose::<D>(
                    d,
                    &mut tmp[$nhalf..$n],
                    starting_column,
                    num_columns,
                );
                // 4. Second Recursive Call (IDCT1DImpl::do_idct)
                // second half
                maybe_call_idct!(d, $nhalf,
                    IDCT1DImpl::<$nhalf>::do_idct::<D, COLUMNS>(
                        d,
                        &mut tmp[$nhalf..$n],
                        starting_column,
                        num_columns,
                    )
                );
                // 5. MultiplyAndAdd.
                CoeffBundle::<$n, COLUMNS>::multiply_and_add(
                    d,
                    &tmp,
                    data,
                    starting_column,
                    num_columns,
                );
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

#[inline(always)]
pub fn dct2d<D: SimdDescriptor, const ROWS: usize, const COLS: usize>(
    d: D,
    data: &mut [f32],
    scratch: &mut [f32],
) where
    DCT1DImpl<ROWS>: DCT1D,
    DCT1DImpl<COLS>: DCT1D,
{
    assert_eq!(data.len(), ROWS * COLS, "Data length mismatch");

    // OPTION 2: Wrap entire loop bodies (with inline always) - two boundaries
    // 1. Row transforms.
    d.call(|d| {
        let temp_rows = data.as_chunks_mut::<COLS>().0;
        let num_full = COLS / D::F32Vec::LEN;
        let remainder = COLS % D::F32Vec::LEN;
        for starting_column in (0..num_full * D::F32Vec::LEN).step_by(D::F32Vec::LEN) {
            DCT1DImpl::<ROWS>::do_dct::<D, COLS>(d, temp_rows, starting_column, D::F32Vec::LEN);
        }
        if remainder != 0 {
            DCT1DImpl::<ROWS>::do_dct::<D, COLS>(
                d,
                temp_rows,
                num_full * D::F32Vec::LEN,
                remainder,
            );
        }
    });

    // 2. Transpose.
    let temp_cols_slice = &mut scratch[..ROWS * COLS];
    d.transpose::<ROWS, COLS>(data, temp_cols_slice);

    // 3. Column transforms.
    d.call(|d| {
        let temp_cols = temp_cols_slice.as_chunks_mut::<ROWS>().0;
        let num_full = ROWS / D::F32Vec::LEN;
        let remainder = ROWS % D::F32Vec::LEN;
        for starting_row in (0..num_full * D::F32Vec::LEN).step_by(D::F32Vec::LEN) {
            DCT1DImpl::<COLS>::do_dct::<D, ROWS>(d, temp_cols, starting_row, D::F32Vec::LEN);
        }
        if remainder != 0 {
            DCT1DImpl::<COLS>::do_dct::<D, ROWS>(
                d,
                temp_cols,
                num_full * D::F32Vec::LEN,
                remainder,
            );
        }
    });

    // 4. Transpose back.
    d.transpose::<COLS, ROWS>(temp_cols_slice, data);
}

#[inline(always)]
pub fn idct2d<D: SimdDescriptor, const ROWS: usize, const COLS: usize>(
    d: D,
    data: &mut [f32],
    scratch: &mut [f32],
) where
    IDCT1DImpl<ROWS>: IDCT1D,
    IDCT1DImpl<COLS>: IDCT1D,
{
    assert_eq!(data.len(), ROWS * COLS, "Data length mismatch");

    // 1. Column IDCTs (on transposed data)
    let temp_cols_slice = &mut scratch[..ROWS * COLS];
    if ROWS < COLS {
        d.transpose::<ROWS, COLS>(data, temp_cols_slice);
    } else {
        temp_cols_slice.copy_from_slice(data);
    }

    let temp_cols = temp_cols_slice.as_chunks_mut::<ROWS>().0;
    let num_full = ROWS / D::F32Vec::LEN;
    let remainder = ROWS % D::F32Vec::LEN;
    for starting_row in (0..num_full * D::F32Vec::LEN).step_by(D::F32Vec::LEN) {
        IDCT1DImpl::<COLS>::do_idct::<D, ROWS>(d, temp_cols, starting_row, D::F32Vec::LEN);
    }
    if remainder != 0 {
        IDCT1DImpl::<COLS>::do_idct::<D, ROWS>(d, temp_cols, num_full * D::F32Vec::LEN, remainder);
    }

    // 2. Transpose back
    d.transpose::<COLS, ROWS>(temp_cols_slice, data);

    // 3. Row IDCTs
    let temp_rows = data.as_chunks_mut::<COLS>().0;
    let num_full = COLS / D::F32Vec::LEN;
    let remainder = COLS % D::F32Vec::LEN;
    for starting_column in (0..num_full * D::F32Vec::LEN).step_by(D::F32Vec::LEN) {
        IDCT1DImpl::<ROWS>::do_idct::<D, COLS>(d, temp_rows, starting_column, D::F32Vec::LEN);
    }
    if remainder != 0 {
        IDCT1DImpl::<ROWS>::do_idct::<D, COLS>(d, temp_rows, num_full * D::F32Vec::LEN, remainder);
    }
}

pub fn compute_scaled_dct<D: SimdDescriptor, const ROWS: usize, const COLS: usize>(
    d: D,
    mut from: [[f32; COLS]; ROWS],
    to: &mut [f32],
) where
    DCT1DImpl<ROWS>: DCT1D,
    DCT1DImpl<COLS>: DCT1D,
{
    let num_full = COLS / D::F32Vec::LEN;
    let remainder = COLS % D::F32Vec::LEN;
    for starting_column in (0..num_full * D::F32Vec::LEN).step_by(D::F32Vec::LEN) {
        DCT1DImpl::<ROWS>::do_dct::<D, COLS>(d, &mut from, starting_column, D::F32Vec::LEN);
    }
    if remainder != 0 {
        DCT1DImpl::<ROWS>::do_dct::<D, COLS>(d, &mut from, num_full * D::F32Vec::LEN, remainder);
    }
    let mut transposed_dct_buffer = [[0.0; ROWS]; COLS];
    d.transpose::<ROWS, COLS>(
        from.as_flattened(),
        transposed_dct_buffer.as_flattened_mut(),
    );
    let num_full = ROWS / D::F32Vec::LEN;
    let remainder = ROWS % D::F32Vec::LEN;
    for starting_row in (0..num_full * D::F32Vec::LEN).step_by(D::F32Vec::LEN) {
        DCT1DImpl::<COLS>::do_dct::<D, ROWS>(
            d,
            &mut transposed_dct_buffer,
            starting_row,
            D::F32Vec::LEN,
        );
    }
    if remainder != 0 {
        DCT1DImpl::<COLS>::do_dct::<D, ROWS>(
            d,
            &mut transposed_dct_buffer,
            num_full * D::F32Vec::LEN,
            remainder,
        );
    }
    let normalization_factor = D::F32Vec::splat(d, 1.0 / (ROWS * COLS) as f32);
    if ROWS >= COLS {
        if ROWS * COLS < D::F32Vec::LEN {
            let coeffs =
                D::F32Vec::load_partial(d, ROWS * COLS, transposed_dct_buffer.as_flattened());
            (coeffs * normalization_factor).store_partial(ROWS * COLS, to);
        } else {
            assert_eq!(ROWS * COLS % D::F32Vec::LEN, 0);
            for i in (0..ROWS * COLS).step_by(D::F32Vec::LEN) {
                let coeffs = D::F32Vec::load(d, transposed_dct_buffer.as_flattened()[i..].as_ref());
                (coeffs * normalization_factor).store(to[i..].as_mut());
            }
        }
    } else {
        d.transpose::<COLS, ROWS>(
            transposed_dct_buffer.as_flattened(),
            to[..ROWS * COLS].as_mut(),
        );
        if ROWS * COLS < D::F32Vec::LEN {
            let coeffs = D::F32Vec::load_partial(d, ROWS * COLS, to);
            (coeffs * normalization_factor).store_partial(ROWS * COLS, to);
        } else {
            assert_eq!(ROWS * COLS % D::F32Vec::LEN, 0);
            for i in (0..ROWS * COLS).step_by(D::F32Vec::LEN) {
                let coeffs = D::F32Vec::load(d, to[i..].as_ref());
                (coeffs * normalization_factor).store(to[i..].as_mut());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        simd::{ScalarDescriptor, SimdDescriptor, test_all_instruction_sets},
        util::test::{assert_all_almost_abs_eq, assert_almost_abs_eq},
        var_dct::{
            dct::{DCT1D, DCT1DImpl, IDCT1D, IDCT1DImpl, compute_scaled_dct, dct2d, idct2d},
            dct_slow::{dct1d, idct1d},
        },
    };
    use test_log::test;

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
                for j in 0..M {
                    DCT1DImpl::<N>::do_dct::<_, M>(d, &mut output, j, 1);
                }

                for i in 0..N {
                    assert_almost_abs_eq(output[i][0], output_matrix_slow[i][0] as f32, $tolerance);
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

                // Generate input data for the reference idct1d.
                // Results in vec![vec![1.0], vec![2.0], ..., vec![N.0]]
                let input_matrix_for_ref: Vec<Vec<f64>> =
                    std::array::from_fn::<f64, NM, _>(|i| (i + 1) as f64)
                        .chunks(M)
                        .map(|row_slice| row_slice.to_vec())
                        .collect();

                let output_matrix_slow: Vec<Vec<f64>> = idct1d(&input_matrix_for_ref);

                // IDCT1DImpl expects input coefficient data in [[f32; M]; N] format.
                let mut input_arr_2d = [[0.0f32; M]; N];
                for r_idx in 0..N {
                    for c_idx in 0..M {
                        input_arr_2d[r_idx][c_idx] = input_matrix_for_ref[r_idx][c_idx] as f32;
                    }
                }

                let mut output = input_arr_2d;
                let d = ScalarDescriptor {};
                for j in 0..M {
                    IDCT1DImpl::<N>::do_idct::<_, M>(d, &mut output, j, 1);
                }

                for i in 0..N {
                    assert_almost_abs_eq(output[i][0], output_matrix_slow[i][0] as f32, $tolerance);
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

        // Initialize an N x M matrix with data from 1.0 to 24.0
        let input_coeffs_matrix_for_ref: Vec<Vec<f64>> =
            std::array::from_fn::<f64, NM, _>(|i| (i + 1) as f64)
                .chunks(M)
                .map(|row_slice| row_slice.to_vec())
                .collect();

        let output_matrix_slow: Vec<Vec<f64>> = idct1d(&input_coeffs_matrix_for_ref);

        // Prepare input for the implementation under test (IDCT1DImpl)
        // IDCT1DImpl expects data in [[f32; M]; N] format.
        let mut input_coeffs_for_fast_impl = [[0.0f32; M]; N];
        for r in 0..N {
            for c in 0..M {
                // Use the same source coefficient values as the reference IDCT
                input_coeffs_for_fast_impl[r][c] = input_coeffs_matrix_for_ref[r][c] as f32;
            }
        }

        // This will be modified in-place by IDCT1DImpl
        let mut output_fast_impl = input_coeffs_for_fast_impl;

        // Call the implementation under test (operates on 2D data)
        let d = ScalarDescriptor {};
        for j in 0..M {
            IDCT1DImpl::<N>::do_idct::<_, M>(d, &mut output_fast_impl, j, 1);
        }

        // Compare results element-wise
        for r_idx in 0..N {
            for c_idx in 0..M {
                assert_almost_abs_eq(
                    output_fast_impl[r_idx][c_idx],
                    output_matrix_slow[r_idx][c_idx] as f32,
                    1e-5,
                );
            }
        }
    }

    #[test]
    fn test_dct1d_8x3_eq_slow() {
        const N: usize = 8;
        const M: usize = 3;
        const NM: usize = N * M; // 24

        // Initialize a 3 x 8 marix with data from 1.0 to 24.0
        let input_matrix_for_ref: Vec<Vec<f64>> =
            std::array::from_fn::<f64, NM, _>(|i| (i + 1) as f64)
                .chunks(M)
                .map(|row_slice| row_slice.to_vec())
                .collect();

        let output_matrix_slow: Vec<Vec<f64>> = dct1d(&input_matrix_for_ref);

        // Prepare input for the implementation under test (DCT1DImpl)
        // DCT1DImpl expects data in [[f32; M]; N] format.
        let mut input_for_fast_impl = [[0.0f32; M]; N];
        for r in 0..N {
            for c in 0..M {
                // Use the same source values as the reference DCT
                input_for_fast_impl[r][c] = input_matrix_for_ref[r][c] as f32;
            }
        }

        // This will be modified in-place by DCT1DImpl
        let mut output_fast_impl = input_for_fast_impl;

        // Call the implementation under test (operates on 2D data)
        let d = ScalarDescriptor {};
        for j in 0..M {
            DCT1DImpl::<N>::do_dct::<_, M>(d, &mut output_fast_impl, j, 1);
        }

        // Compare results element-wise
        for r_freq_idx in 0..N {
            for c_col_idx in 0..M {
                assert_almost_abs_eq(
                    output_fast_impl[r_freq_idx][c_col_idx],
                    output_matrix_slow[r_freq_idx][c_col_idx] as f32,
                    1e-5,
                );
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
                let mut scratch = [0.0f32; M * N];
                let d = ScalarDescriptor {};
                idct2d::<_, N, M>(d, &mut data, &mut scratch);
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
                let mut scratch = [0.0f32; M * N];
                let d = ScalarDescriptor {};
                dct2d::<_, N, M>(d, &mut data, &mut scratch);
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

    #[test]
    fn test_compute_scaled_dct_wide() {
        let input = [
            [86.0, 239.0, 213.0, 36.0, 34.0, 142.0, 248.0, 87.0],
            [128.0, 122.0, 131.0, 72.0, 156.0, 112.0, 248.0, 55.0],
            [120.0, 31.0, 246.0, 177.0, 119.0, 154.0, 176.0, 248.0],
            [21.0, 151.0, 107.0, 101.0, 202.0, 71.0, 246.0, 48.0],
        ];

        let mut output = [0.0; 4 * 8];

        let d = ScalarDescriptor {};
        compute_scaled_dct::<_, 4, 8>(d, input, &mut output);

        assert_all_almost_abs_eq(
            output,
            [
                135.219, -13.1026, 0.573698, -6.19682, -29.5938, 11.5028, -13.3955, 21.9205,
                1.4572, 11.3448, 16.3991, 2.50104, -20.549, 0.363681, 3.94596, -4.05406, -8.21875,
                6.57931, 0.601308, 1.51804, -20.5312, -9.29264, -19.6983, -0.850355, 12.4189,
                -5.0881, 5.82096, -20.1997, 3.87769, 2.80762, 24.6634, -8.93341,
            ],
            1e-3,
        );
    }

    #[test]
    fn test_compute_scaled_dct_tall() {
        let input = [
            [86.0, 239.0, 213.0, 36.0],
            [34.0, 142.0, 248.0, 87.0],
            [128.0, 122.0, 131.0, 72.0],
            [156.0, 112.0, 248.0, 55.0],
            [120.0, 31.0, 246.0, 177.0],
            [119.0, 154.0, 176.0, 248.0],
            [21.0, 151.0, 107.0, 101.0],
            [202.0, 71.0, 246.0, 48.0],
        ];

        let mut output = [0.0; 8 * 4];

        let d = ScalarDescriptor {};
        compute_scaled_dct::<_, 8, 4>(d, input, &mut output);

        assert_all_almost_abs_eq(
            output,
            [
                135.219, -0.899633, -4.54363, 9.7776, 7.65625, -7.7203, 10.5073, -11.9921,
                -8.31418, 5.39457, 11.3896, -17.5006, 11.6535, 12.6257, 9.27026, -0.767252,
                -29.5938, -19.9538, -17.5214, -0.467021, -3.28125, -7.67861, 11.3504, 5.01615,
                24.9226, -4.19572, -7.10474, -16.7029, 24.2961, -16.8923, -3.32708, -4.09777,
            ],
            1e-3,
        );
    }

    fn bench_dct2d<D: SimdDescriptor>(d: D) {
        const ROWS: usize = 8;
        const COLS: usize = 8;
        let mut data = [
            86.0, 239.0, 213.0, 36.0, 34.0, 142.0, 248.0, 87.0, 128.0, 122.0, 131.0, 72.0, 156.0,
            112.0, 248.0, 55.0, 120.0, 31.0, 246.0, 177.0, 119.0, 154.0, 176.0, 248.0, 21.0, 151.0,
            107.0, 101.0, 202.0, 71.0, 246.0, 48.0, 86.0, 239.0, 213.0, 36.0, 34.0, 142.0, 248.0,
            87.0, 128.0, 122.0, 131.0, 72.0, 156.0, 112.0, 248.0, 55.0, 120.0, 31.0, 246.0, 177.0,
            119.0, 154.0, 176.0, 248.0, 21.0, 151.0, 107.0, 101.0, 202.0, 71.0, 246.0, 48.0,
        ];
        let mut scratch = [0.0; ROWS * COLS];

        let iters = std::env::var("DCT2D_BENCH_ITERATIONS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);

        let start = std::time::Instant::now();
        for _ in 0..iters {
            dct2d::<_, ROWS, COLS>(d, &mut data, &mut scratch);
        }
        let elapsed = start.elapsed();
        if iters > 1 {
            println!("dct2d 8x8 ({:?}): {:?} per iteration", d, elapsed / iters);
        }
    }

    test_all_instruction_sets!(bench_dct2d);

    fn bench_idct2d<D: SimdDescriptor>(d: D) {
        const ROWS: usize = 8;
        const COLS: usize = 8;
        let mut data = [
            86.0, 239.0, 213.0, 36.0, 34.0, 142.0, 248.0, 87.0,
            128.0, 122.0, 131.0, 72.0, 156.0, 112.0, 248.0, 55.0,
            120.0, 31.0, 246.0, 177.0, 119.0, 154.0, 176.0, 248.0,
            21.0, 151.0, 107.0, 101.0, 202.0, 71.0, 246.0, 48.0,
            86.0, 239.0, 213.0, 36.0, 34.0, 142.0, 248.0, 87.0,
            128.0, 122.0, 131.0, 72.0, 156.0, 112.0, 248.0, 55.0,
            120.0, 31.0, 246.0, 177.0, 119.0, 154.0, 176.0, 248.0,
            21.0, 151.0, 107.0, 101.0, 202.0, 71.0, 246.0, 48.0,
        ];
        let mut scratch = [0.0; ROWS * COLS];

        let iters = std::env::var("DCT2D_BENCH_ITERATIONS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);

        let start = std::time::Instant::now();
        for _ in 0..iters {
            idct2d::<_, ROWS, COLS>(d, &mut data, &mut scratch);
        }
        let elapsed = start.elapsed();
        if iters > 1 {
            println!("idct2d 8x8 ({:?}): {:?} per iteration", d, elapsed / iters);
        }
    }

    test_all_instruction_sets!(bench_idct2d);

    fn bench_compute_scaled_dct<D: SimdDescriptor>(d: D) {
        const ROWS: usize = 8;
        const COLS: usize = 4;
        let input = [
            [86.0, 239.0, 213.0, 36.0],
            [34.0, 142.0, 248.0, 87.0],
            [128.0, 122.0, 131.0, 72.0],
            [156.0, 112.0, 248.0, 55.0],
            [120.0, 31.0, 246.0, 177.0],
            [119.0, 154.0, 176.0, 248.0],
            [21.0, 151.0, 107.0, 101.0],
            [202.0, 71.0, 246.0, 48.0],
        ];
        let mut output = [0.0; ROWS * COLS];

        let iters = std::env::var("DCT2D_BENCH_ITERATIONS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);

        let start = std::time::Instant::now();
        for _ in 0..iters {
            compute_scaled_dct::<_, ROWS, COLS>(d, input, &mut output);
        }
        let elapsed = start.elapsed();
        if iters > 1 {
            println!(
                "compute_scaled_dct {}x{} ({:?}): {:?} per iteration",
                ROWS,
                COLS,
                d,
                elapsed / iters
            );
        }
    }

    test_all_instruction_sets!(bench_compute_scaled_dct);
}
