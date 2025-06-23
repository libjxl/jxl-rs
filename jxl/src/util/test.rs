// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::{
    io::{BufRead, BufReader, Cursor, Read, Write},
    num::{ParseFloatError, ParseIntError},
};

use crate::{
    bit_reader::BitReader,
    container::ContainerParser,
    error::Error as JXLError,
    headers::{FileHeader, JxlHeader, encodings::*, frame_header::TocNonserialized},
    image::Image,
};

use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Invalid PFM: {0}")]
    InvalidPFM(String),
}

impl From<ParseFloatError> for Error {
    fn from(value: ParseFloatError) -> Self {
        Error::InvalidPFM(value.to_string())
    }
}

impl From<ParseIntError> for Error {
    fn from(value: ParseIntError) -> Self {
        Error::InvalidPFM(value.to_string())
    }
}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Error::InvalidPFM(value.to_string())
    }
}

impl From<JXLError> for Error {
    fn from(value: JXLError) -> Self {
        Error::InvalidPFM(value.to_string())
    }
}

macro_rules! assert_almost_eq {
    ($left:expr, $right:expr, $max_error:expr $(,)?) => {
        let (left_val, right_val, max_error) = ($left as f64, $right as f64, $max_error as f64);
        if matches!((left_val - right_val).abs().partial_cmp(&max_error), Some(std::cmp::Ordering::Greater) | None) {
            panic!(
                "assertion failed: `(left ≈ right)`\n  left: `{:?}`,\n right: `{:?}`,\n max_error: `{:?}`",
                left_val, right_val, max_error
            )
        }
    };
    ($left:expr, $right:expr, $max_error_abs:expr, $max_error_rel:expr $(,)?) => {
        let (left_val, right_val, max_error_abs, max_error_rel) = ($left as f64, $right as f64, $max_error_abs as f64, $max_error_rel as f64);
        let error = (left_val - right_val).abs();
        if matches!(error.partial_cmp(&max_error_abs), Some(std::cmp::Ordering::Greater) | None) {
            panic!(
                "assertion failed: `(left ≈ right)`\n  left: `{:?}`,\n right: `{:?}`,\n max_error_abs: `{:?}`",
                left_val, right_val, max_error_abs
            )
        }
        let actual_error_rel = left_val.abs().min(right_val.abs()) * max_error_rel;
        if matches!(error.partial_cmp(&actual_error_rel), Some(std::cmp::Ordering::Greater) | None) {
            panic!(
                "assertion failed: `(left ≈ right)`\n  left: `{:?}`,\n right: `{:?}`,\n max_error_rel: `{:?}`",
                left_val, right_val, max_error_rel
            )
        }
    };
}
pub(crate) use assert_almost_eq;

macro_rules! assert_all_almost_eq {
    ($left:expr, $right:expr, $max_error:expr $(,)?) => {
        let (left_val, right_val, max_error) = (&$left, &$right, $max_error as f64);
        if left_val.len() != right_val.len() {
            panic!("assertion failed: `(left ≈ right)`\n left.len(): `{}`,\n right.len(): `{}`", left_val.len(), right_val.len());
        }
        for index in 0..left_val.len() {
            if (left_val[index] as f64- right_val[index] as f64).abs() > max_error {
                panic!(
                    "assertion failed: `(left ≈ right)`\n left: `{:?}`,\n right: `{:?}`,\n max_error: `{:?}`,\n left[{}]: `{}`,\n right[{}]: `{}`",
                    left_val, right_val, max_error, index, left_val[index], index, right_val[index]
                )
            }
        }
    };
}

pub fn read_headers_and_toc(image: &[u8]) -> Result<(FileHeader, FrameHeader, Toc), JXLError> {
    let codestream = ContainerParser::collect_codestream(image).unwrap();
    let mut br = BitReader::new(&codestream);
    let file_header = FileHeader::read(&mut br)?;

    let frame_header =
        FrameHeader::read_unconditional(&(), &mut br, &file_header.frame_header_nonserialized())?;
    let num_toc_entries = frame_header.num_toc_entries();
    let toc = Toc::read_unconditional(
        &(),
        &mut br,
        &TocNonserialized {
            num_entries: num_toc_entries as u32,
        },
    )?;
    Ok((file_header, frame_header, toc))
}

pub fn write_pfm(image: Vec<Image<f32>>, mut buf: impl Write) -> Result<(), Error> {
    if image.len() == 1 {
        buf.write_all(b"Pf\n")?;
    } else if image.len() == 3 {
        buf.write_all(b"PF\n")?;
    } else {
        return Err(Error::InvalidPFM(format!(
            "invalid number of channels: {}",
            image.len()
        )));
    }
    let size = image[0].size();
    for c in image.iter().skip(1) {
        assert_eq!(size, c.size());
    }
    buf.write_fmt(format_args!("{} {}\n", size.0, size.1))?;
    buf.write_all(b"1.0\n")?;
    let mut b: [u8; 4];
    for row in 0..size.1 {
        for col in 0..size.0 {
            for c in image.iter() {
                b = c.as_rect().row(size.1 - row - 1)[col].to_be_bytes();
                buf.write_all(&b)?;
            }
        }
    }
    buf.flush()?;
    Ok(())
}

pub fn read_pfm(b: &[u8]) -> Result<Vec<Image<f32>>, Error> {
    let mut bf = BufReader::new(Cursor::new(b));
    let mut line = String::new();
    bf.read_line(&mut line)?;
    let channels = match line.trim() {
        "Pf" => 1,
        "PF" => 3,
        &_ => return Err(Error::InvalidPFM(format!("invalid PFM type header {line}"))),
    };
    line.clear();
    bf.read_line(&mut line)?;
    let mut dims = line.split_whitespace();
    let xres = if let Some(xres_str) = dims.next() {
        xres_str.trim().parse()?
    } else {
        return Err(Error::InvalidPFM(format!(
            "invalid PFM resolution header {line}",
        )));
    };
    let yres = if let Some(yres_str) = dims.next() {
        yres_str.trim().parse()?
    } else {
        return Err(Error::InvalidPFM(format!(
            "invalid PFM resolution header {line}",
        )));
    };
    line.clear();
    bf.read_line(&mut line)?;
    let endianness: f32 = line.trim().parse()?;

    let mut res = Vec::<Image<f32>>::new();
    for _ in 0..channels {
        let img = Image::new((xres, yres))?;
        res.push(img);
    }

    let mut buf = [0u8; 4];
    for row in 0..yres {
        for col in 0..xres {
            for chan in res.iter_mut() {
                bf.read_exact(&mut buf)?;
                chan.as_rect_mut().row(yres - row - 1)[col] = if endianness < 0.0 {
                    f32::from_le_bytes(buf)
                } else {
                    f32::from_be_bytes(buf)
                }
            }
        }
    }

    Ok(res)
}

pub(crate) use assert_all_almost_eq;

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
