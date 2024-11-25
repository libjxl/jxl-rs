use std::panic;

/// Generic helper function for comparing values with a maximum error margin.
/// Works for types that implement `Sub` and `PartialOrd`
pub fn almost_equal<T>(a: T, b: T, max_error: T) -> bool
where
    T: std::ops::Sub<Output = T> + PartialOrd + Copy,
{
    let diff = if a > b { a - b } else { b - a };
    diff <= max_error
}

#[macro_export]
macro_rules! assert_almost_eq {
    ($a:expr, $b:expr, $max_error:expr $(,)?) => {{
        let left = $a;
        let right = $b;
        let max_error = $max_error;
        if !(almost_equal(left, right, max_error)) {
            panic!(
                "assertion failed: `(left ≈ right)`\n  left: `{:?}`,\n right: `{:?}`,\n max_error: `{:?}`",
                left, right, max_error
            );
        }
    }};
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
