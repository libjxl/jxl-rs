// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use jxl::bit_reader::BitReader;
use jxl::headers::FileHeaders;
use std::env;
use std::fs;

use jxl::headers::JxlHeader;

// TODO(veluca93): BMFF
fn parse_jxl_codestream(data: &[u8]) -> Result<(), jxl::error::Error> {
    let mut br = BitReader::new(data);
    let mut fh = FileHeaders::new();
    fh.read(&mut br)?;
    println!("Image size: {} x {}", fh.size.xsize(), fh.size.ysize());
    println!("Image metadata: {:#?}", fh.image_metadata);
    Ok(())
}

fn main() {
    let args: Vec<String> = env::args().collect();
    assert_eq!(args.len(), 2);
    let file = &args[1];
    let contents = fs::read(file).expect("Something went wrong reading the file");
    let res = parse_jxl_codestream(&contents[..]);
    if let Err(err) = res {
        println!("Error parsing JXL codestream: {}", err)
    }
}
