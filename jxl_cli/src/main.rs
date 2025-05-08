// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use clap::Parser;
use jxl::bit_reader::BitReader;
use jxl::container::{ContainerParser, ParseEvent};
use jxl::frame::{DecoderState, Frame, Section};
use jxl::headers::FileHeader;
use jxl::icc::read_icc;
use std::fs;
use std::io::Read;
use std::path::PathBuf;

use jxl::headers::JxlHeader;

fn parse_jxl_codestream(data: &[u8]) -> Result<(), jxl::error::Error> {
    let mut br = BitReader::new(data);
    let file_header = FileHeader::read(&mut br)?;
    println!(
        "Image size: {} x {}",
        file_header.size.xsize(),
        file_header.size.ysize()
    );
    if file_header.image_metadata.color_encoding.want_icc {
        let r = read_icc(&mut br)?;
        println!("found {}-byte ICC", r.len());
    };
    let mut decoder_state = DecoderState::new(file_header);
    loop {
        let mut frame = Frame::new(&mut br, decoder_state)?;
        br.jump_to_byte_boundary()?;

        let mut section_readers = frame.sections(&mut br)?;

        println!("read frame with {} sections", section_readers.len());

        frame.decode_lf_global(&mut section_readers[frame.get_section_idx(Section::LfGlobal)])?;

        for group in 0..frame.header().num_lf_groups() {
            frame.decode_lf_group(
                group,
                &mut section_readers[frame.get_section_idx(Section::Lf { group })],
            )?;
        }

        frame.decode_hf_global(&mut section_readers[frame.get_section_idx(Section::HfGlobal)])?;

        frame.prepare_for_hf()?;

        for pass in 0..frame.header().passes.num_passes as usize {
            for group in 0..frame.header().num_groups() {
                frame.decode_hf_group(
                    group,
                    pass,
                    &mut section_readers[frame.get_section_idx(Section::Hf { group, pass })],
                )?;
            }
        }

        if let Some(state) = frame.finalize()? {
            decoder_state = state;
        } else {
            break;
        }
    }

    Ok(())
}

#[derive(Parser)]
struct Opt {
    input: PathBuf,
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

    let opt = Opt::parse();
    let filename = opt.input;
    let mut file = match fs::File::open(filename) {
        Ok(file) => file,
        Err(err) => {
            println!("Cannot open file: {err}");
            return;
        }
    };

    let mut parser = ContainerParser::new();
    let mut buf = vec![0u8; 4096];
    let mut buf_valid = 0usize;
    let mut codestream = Vec::new();
    loop {
        let chunk_size = match file.read(&mut buf[buf_valid..]) {
            Ok(l) => l,
            Err(err) => {
                println!("Cannot read data from file: {err}");
                return;
            }
        };
        if chunk_size == 0 {
            break;
        }
        buf_valid += chunk_size;

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
