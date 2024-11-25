// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

macro_rules! assert_almost_eq {
    ($left:expr, $right:expr, $max_error:expr $(,)?) => {
        match (&$left, &$right, &$max_error) {
            (left_val, right_val, max_error) => {
                let diff = if *left_val > *right_val {
                    *left_val - *right_val
                } else {
                    *right_val - *left_val
                };
                match diff.partial_cmp(max_error) {
                    Some(std::cmp::Ordering::Greater) | None => panic!(
                        "assertion failed: `(left ≈ right)`\n  left: `{:?}`,\n right: `{:?}`,\n max_error: `{:?}`",
                        left_val, right_val, max_error
                    ),
                    _ => {}
                }
            }
        }
    };
}
pub(crate) use assert_almost_eq;

#[cfg(test)]
mod tests {
    use std::panic;

    #[test]
    fn test_with_floats() {
        assert_almost_eq!(1.0000001f64, 1.0000002, 0.000001);
        assert_almost_eq!(1.0, 1.1, 0.2);
    }

    #[test]
    fn test_with_integers() {
        assert_almost_eq!(100, 101, 2);
        assert_almost_eq!(777u32, 770, 7);
        assert_almost_eq!(500i64, 498, 3);
    }

    #[test]
    #[should_panic]
    fn test_panic_float() {
        assert_almost_eq!(1.0, 1.2, 0.1);
    }
    #[test]
    #[should_panic]
    fn test_panic_integer() {
        assert_almost_eq!(100, 105, 2);
    }

    #[test]
    #[should_panic]
    fn test_nan_comparison() {
        assert_almost_eq!(f64::NAN, f64::NAN, 0.1);
    }

    #[test]
    #[should_panic]
    fn test_nan_tolerance() {
        assert_almost_eq!(1.0, 1.0, f64::NAN);
    }

    #[test]
    fn test_infinity_tolerance() {
        assert_almost_eq!(1.0, 1.0, f64::INFINITY);
    }

    #[test]
    #[should_panic]
    fn test_nan_comparison_with_infinity_tolerance() {
        assert_almost_eq!(f32::NAN, f32::NAN, f32::INFINITY);
    }

    #[test]
    #[should_panic]
    fn test_infinity_comparison_with_infinity_tolerance() {
        assert_almost_eq!(f32::INFINITY, f32::INFINITY, f32::INFINITY);
    }
}
