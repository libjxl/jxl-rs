// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#[macro_export]
macro_rules! assert_almost_eq {
    ($left:expr, $right:expr, $max_error:expr $(,)?) => {
        match (&$left, &$right) {
            (left_val, right_val) => {
                let diff = if *left_val > *right_val {
                    *left_val - *right_val
                } else {
                    *right_val - *left_val
                };
                if !(diff <= $max_error) {
                    panic!(
                        "assertion failed: `(left ≈ right)`\n  left: `{:?}`,\n right: `{:?}`,\n max_error: `{:?}`",
                        left_val, right_val, $max_error
                    );
                }
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_panic() {
        use std::panic;
        let result = panic::catch_unwind(|| {
            assert_almost_eq!(1.0, 1.2, 0.1);
        });
        assert!(
            result.is_err(),
            "Expected assert_almost_eq! to panic, but it didn't"
        );

        let result = panic::catch_unwind(|| {
            assert_almost_eq!(100, 105, 2);
        });
        assert!(
            result.is_err(),
            "Expected assert_almost_eq! to panic, but it didn't"
        );
    }
}
