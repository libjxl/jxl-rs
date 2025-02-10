// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    bit_reader::BitReader,
    container::ContainerParser,
    error::Error,
    headers::{encodings::*, frame_header::TocNonserialized, FileHeader, JxlHeader},
};

pub fn abs_delta<T: Num + std::cmp::PartialOrd>(left_val: T, right_val: T) -> T {
    if left_val > right_val {
        left_val - right_val
    } else {
        right_val - left_val
    }
}

macro_rules! assert_almost_eq {
    ($left:expr, $right:expr, $max_error:expr $(,)?) => {
        let (left_val, right_val, max_error) = (&$left, &$right, &$max_error);
        match $crate::util::test::abs_delta(*left_val, *right_val).partial_cmp(max_error) {
            Some(std::cmp::Ordering::Greater) | None => panic!(
                "assertion failed: `(left ≈ right)`\n  left: `{:?}`,\n right: `{:?}`,\n max_error: `{:?}`",
                left_val, right_val, max_error
            ),
            _ => {}
        }
    };
}
pub(crate) use assert_almost_eq;

macro_rules! assert_all_almost_eq {
    ($left:expr, $right:expr, $max_error:expr $(,)?) => {
        let (left_val, right_val, max_error) = (&$left, &$right, &$max_error);
        if left_val.len() != right_val.len() {
            panic!("assertion failed: `(left ≈ right)`\n left.len(): `{}`,\n right.len(): `{}`", left_val.len(), right_val.len());
        }
        for index in 0..left_val.len() {
            match $crate::util::test::abs_delta(left_val[index], right_val[index]).partial_cmp(max_error) {
                Some(std::cmp::Ordering::Greater) | None =>  panic!(
                    "assertion failed: `(left ≈ right)`\n left: `{:?}`,\n right: `{:?}`,\n max_error: `{:?}`,\n left[{}]: `{}`,\n right[{}]: `{}`",
                    left_val, right_val, max_error, index, left_val[index], index, right_val[index]
                ),
                _ => {}
            }
        }
    };
}

pub fn read_all_frameheader_and_toc(image: &[u8]) -> Result<Vec<(FrameHeader, Toc)>, Error> {
    let codestream = ContainerParser::collect_codestream(image).unwrap();
    let mut br = BitReader::new(&codestream);
    let file_header = FileHeader::read(&mut br).unwrap();
    let mut frames_and_tocs = Vec::new();
    // Loop until we can no longer read valid frame data
    loop {
        // Attempt to read the next frame header
        let frame_header = match FrameHeader::read_unconditional(
            &(),
            &mut br,
            &file_header.frame_header_nonserialized(),
        ) {
            Ok(header) => header,
            // If we can't read another valid frame header, break out (no more frames)
            Err(_) => break,
        };

        // Read the table of contents for this frame
        let num_toc_entries = frame_header.num_toc_entries();
        let toc = Toc::read_unconditional(
            &(),
            &mut br,
            &TocNonserialized {
                num_entries: num_toc_entries as u32,
            },
        )?;

        let bytes_until_next_frame = toc.entries.iter().sum::<u32>() as usize;
        frames_and_tocs.push((frame_header, toc));
        // Advance to the next byte boundary before skipping the frame payload
        br.jump_to_byte_boundary()?;

        // Skip the actual frame payload by summing the TOC entries
        br.skip_bits(8 * bytes_until_next_frame)?;
    }

    // Return all frame headers and TOCs
    Ok(frames_and_tocs)
}

pub(crate) use assert_all_almost_eq;
use num_traits::Num;

use crate::headers::frame_header::{FrameHeader, Toc};

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
