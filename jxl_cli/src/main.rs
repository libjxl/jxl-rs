// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use jxl::bit_reader::BitReader;
use jxl::container::{ContainerParser, ParseEvent};
use jxl::headers::{
    encodings::UnconditionalCoder,
    frame_header::{FrameHeader, FrameHeaderNonserialized},
    FileHeader,
};
use jxl::icc::read_icc;
use std::env;
use std::fs;
use std::io::Read;

use jxl::headers::JxlHeader;

fn parse_jxl_codestream(data: &[u8]) -> Result<(), jxl::error::Error> {
    let mut br = BitReader::new(data);
    let fh = FileHeader::read(&mut br)?;
    println!("Image size: {} x {}", fh.size.xsize(), fh.size.ysize());
    let _icc = if fh.image_metadata.color_encoding.want_icc {
        let r = read_icc(&mut br)?;
        println!("ICC: {} {:?}", r.len(), r);
        Some(r)
    } else {
        None
    };

    let have_timecode = match fh.image_metadata.animation {
        Some(ref a) => a.have_timecodes,
        None => false,
    };
    let _frame_header = FrameHeader::read_unconditional(
        &(),
        &mut br,
        &FrameHeaderNonserialized {
            xyb_encoded: fh.image_metadata.xyb_encoded,
            num_extra_channels: fh.image_metadata.extra_channel_info.len() as u32,
            extra_channel_info: fh.image_metadata.extra_channel_info,
            have_animation: fh.image_metadata.animation.is_some(),
            have_timecode,
            img_width: fh.size.xsize(),
            img_height: fh.size.ysize(),
        },
    )?;

    Ok(())
}

fn main() {
    #[cfg(feature = "tracing-subscriber")]
    {
        use tracing_subscriber::{fmt, prelude::*, EnvFilter};
        tracing_subscriber::registry()
            .with(fmt::layer())
            .with(EnvFilter::from_default_env())
            .init();
    }

    let args: Vec<String> = env::args().collect();
    assert_eq!(args.len(), 2);
    let file = &args[1];
    let mut file = fs::File::open(file).expect("cannot open file");

    let mut parser = ContainerParser::new();
    let mut buf = vec![0u8; 4096];
    let mut buf_valid = 0usize;
    let mut codestream = Vec::new();
    loop {
        let count = file
            .read(&mut buf[buf_valid..])
            .expect("cannot read data from file");
        if count == 0 {
            break;
        }
        buf_valid += count;

        for event in parser.process_bytes(&buf[..buf_valid]) {
            match event {
                Ok(ParseEvent::BitstreamKind(kind)) => {
                    println!("Bitstream kind: {kind:?}");
                }
                Ok(ParseEvent::Codestream(buf)) => {
                    codestream.extend_from_slice(buf);
                }
                Err(err) => {
                    println!("Error parsing JXL codestream: {err}");
                    return;
                }
            }
        }

        let consumed = parser.previous_consumed_bytes();
        buf.copy_within(consumed..buf_valid, 0);
        buf_valid -= consumed;
    }

    let res = parse_jxl_codestream(&codestream);
    if let Err(err) = res {
        println!("Error parsing JXL codestream: {}", err)
    }
}
