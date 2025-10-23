// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use jxl::image::ImageRect;

pub trait ToU8ForWriting {
    fn to_u8_for_writing(self) -> u8;
}

impl ToU8ForWriting for u8 {
    fn to_u8_for_writing(self) -> u8 {
        self
    }
}

impl ToU8ForWriting for u16 {
    fn to_u8_for_writing(self) -> u8 {
        ((self as u32 * 0xff + 0x8000) / 0xffff) as u8
    }
}

impl ToU8ForWriting for f32 {
    fn to_u8_for_writing(self) -> u8 {
        (self * 255.0).clamp(0.0, 255.0).round() as u8
    }
}

impl ToU8ForWriting for u32 {
    fn to_u8_for_writing(self) -> u8 {
        ((self as u64 * 0xff + 0x80000000) / 0xffffffff) as u8
    }
}

impl ToU8ForWriting for half::f16 {
    fn to_u8_for_writing(self) -> u8 {
        self.to_f32().to_u8_for_writing()
    }
}

pub fn to_pgm_as_8bit(img: &ImageRect<'_, f32>) -> Vec<u8> {
    use std::io::Write;
    let mut ret = vec![];
    write!(&mut ret, "P5\n{} {}\n255\n", img.size().0, img.size().1).unwrap();
    ret.extend(
        (0..img.size().1)
            .flat_map(|x| img.row(x).iter())
            .map(|x| x.to_u8_for_writing()),
    );
    ret
}

pub fn to_ppm_as_8bit(img: &[ImageRect<'_, f32>; 3]) -> Vec<u8> {
    use std::io::Write;
    let mut ret = vec![];
    assert_eq!(img[0].size(), img[1].size());
    assert_eq!(img[0].size(), img[2].size());
    write!(
        &mut ret,
        "P6\n{} {}\n255\n",
        img[0].size().0,
        img[0].size().1
    )
    .unwrap();
    ret.extend(
        (0..img[0].size().1)
            .flat_map(|y| {
                (0..img[0].size().0).flat_map(move |x| [0, 1, 2].map(move |c| img[c].row(y)[x]))
            })
            .map(|x| x.to_u8_for_writing()),
    );
    ret
}

#[cfg(test)]
mod test {
    use super::{ToU8ForWriting, to_pgm_as_8bit};
    use jxl::error::Result;
    use jxl::image::Image;

    #[test]
    fn covert_to_pgm() -> Result<()> {
        let image = Image::<f32>::new((32, 32))?;
        assert!(to_pgm_as_8bit(&image.as_rect()).starts_with(b"P5\n32 32\n255\n"));
        Ok(())
    }

    #[test]
    fn u16_to_u8() {
        let mut left_source_u16 = 0xffffu16 / 510;
        for want_u8 in 0x00u8..0xffu8 {
            assert_eq!(left_source_u16.to_u8_for_writing(), want_u8);
            assert_eq!((left_source_u16 + 1).to_u8_for_writing(), want_u8 + 1);
            // Since we have 256 u8 values, but 0x00 and 0xff only have half the
            // range, we actually get whole ranges of size 0xffff / 255.
            left_source_u16 = left_source_u16.wrapping_add(0xffff / 255);
        }
    }

    #[test]
    fn f32_to_u8() {
        let epsilon = 1e-4f32;
        for want_u8 in 0x00u8..0xffu8 {
            let threshold = 1f32 / 510f32 + (1f32 / 255f32) * (want_u8 as f32);
            assert_eq!((threshold - epsilon).to_u8_for_writing(), want_u8);
            assert_eq!((threshold + epsilon).to_u8_for_writing(), want_u8 + 1);
        }
    }

    #[test]
    fn u32_to_u8() {
        let mut left_source_u32 = 0xffffffffu32 / 510;
        for want_u8 in 0x00u8..0xffu8 {
            assert_eq!(left_source_u32.to_u8_for_writing(), want_u8);
            assert_eq!((left_source_u32 + 1).to_u8_for_writing(), want_u8 + 1);
            // Since we have 256 u8 values, but 0x00 and 0xff only have half the
            // range, we actually get whole ranges of size 0xffffffff / 255.
            left_source_u32 = left_source_u32.wrapping_add(0xffffffffu32 / 255);
        }
    }

    #[test]
    fn f16_to_u8() {
        let epsilon = half::f16::from_f32(1e-3f32);
        for want_u8 in 0x00u8..0xffu8 {
            let threshold = half::f16::from_f32(1f32 / 510f32 + (1f32 / 255f32) * (want_u8 as f32));
            assert_eq!((threshold - epsilon).to_u8_for_writing(), want_u8);
            assert_eq!((threshold + epsilon).to_u8_for_writing(), want_u8 + 1);
        }
    }
}
