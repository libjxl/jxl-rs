// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

extern crate jxl_headers_derive;

use jxl_headers_derive::UnconditionalCoder;

use crate::bit_reader::BitReader;
use crate::error::Error;
use crate::headers::encodings::*;

#[derive(UnconditionalCoder, Debug)]
pub struct Size {
    small: bool,
    #[condition(small)]
    #[coder(Bits(5) + 1)]
    ysize_div8: Option<u32>,
    #[condition(!small)]
    #[coder(1 + u2S(Bits(9), Bits(13), Bits(18), Bits(30)))]
    ysize: Option<u32>,
    #[coder(Bits(3))]
    ratio: u32,
    #[condition(small && ratio == 0)]
    #[coder(Bits(5) + 1)]
    xsize_div8: Option<u32>,
    #[condition(!small && ratio == 0)]
    #[coder(1 + u2S(Bits(9), Bits(13), Bits(18), Bits(30)))]
    xsize: Option<u32>,
}

#[derive(UnconditionalCoder, Debug)]
pub struct Preview {
    div8: bool,
    #[condition(div8)]
    #[coder(u2S(16, 32, Bits(5) + 1, Bits(9) + 33))]
    ysize_div8: Option<u32>,
    #[condition(!div8)]
    #[coder(1 + u2S(Bits(6), Bits(8) + 64, Bits(10) + 320, Bits(12) + 1344))]
    ysize: Option<u32>,
    #[coder(Bits(3))]
    ratio: u32,
    #[condition(div8 && ratio == 0)]
    #[coder(u2S(16, 32, Bits(5) + 1, Bits(9) + 33))]
    xsize_div8: Option<u32>,
    #[condition(!div8 && ratio == 0)]
    #[coder(1 + u2S(Bits(6), Bits(8) + 64, Bits(10) + 320, Bits(12) + 1344))]
    xsize: Option<u32>,
}

fn map_aspect_ratio(ysize: u32, ratio: u32) -> u32 {
    match ratio {
        1 => ysize,
        2 => (ysize as u64 * 12 / 10) as u32,
        3 => (ysize as u64 * 4 / 3) as u32,
        4 => (ysize as u64 * 3 / 2) as u32,
        5 => (ysize as u64 * 16 / 9) as u32,
        6 => (ysize as u64 * 5 / 4) as u32,
        7 => ysize * 2,
        _ => panic!("Invalid ratio: {}", ratio),
    }
}

impl Size {
    pub fn ysize(&self) -> u32 {
        if self.small {
            self.ysize_div8.unwrap() * 8
        } else {
            self.ysize.unwrap()
        }
    }

    pub fn xsize(&self) -> u32 {
        if self.ratio == 0 {
            if self.small {
                self.xsize_div8.unwrap() * 8
            } else {
                self.xsize.unwrap()
            }
        } else {
            map_aspect_ratio(self.ysize(), self.ratio)
        }
    }
}

impl Preview {
    pub fn ysize(&self) -> u32 {
        if self.div8 {
            self.ysize_div8.unwrap() * 8
        } else {
            self.ysize.unwrap()
        }
    }

    pub fn xsize(&self) -> u32 {
        if self.ratio == 0 {
            if self.div8 {
                self.xsize_div8.unwrap() * 8
            } else {
                self.xsize.unwrap()
            }
        } else {
            map_aspect_ratio(self.ysize(), self.ratio)
        }
    }
}
