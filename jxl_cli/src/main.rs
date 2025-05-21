// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use clap::Parser;
use jxl::bit_reader::BitReader;
use jxl::container::{ContainerParser, ParseEvent};
use jxl::enc::{ImageData, ImageFrame};
use jxl::error::Error;
use jxl::frame::{DecoderState, Frame, Section};
use jxl::headers::FileHeader;
use jxl::icc::read_icc;
use std::fs;
use std::io::Read;
use std::path::PathBuf;

use jxl::headers::JxlHeader;

fn decode_jxl_codestream(data: &[u8]) -> Result<(ImageData<f32>, Vec<u8>), Error> {
    let mut br = BitReader::new(data);
    let file_header = FileHeader::read(&mut br)?;
    println!(
        "Image size: {} x {}",
        file_header.size.xsize(),
        file_header.size.ysize()
    );
    // TODO(firsching): Make it such that we also write icc bytes in the
    // case where want_icc is false.
    let mut icc_bytes = Vec::<u8>::new();
    if file_header.image_metadata.color_encoding.want_icc {
        let r = read_icc(&mut br)?;
        println!("found {}-byte ICC", r.len());
        icc_bytes = r;
    };

    br.jump_to_byte_boundary()?;
    let mut image_data: ImageData<f32> = ImageData {
        size: (
            file_header.size.xsize() as usize,
            file_header.size.ysize() as usize,
        ),
        frames: vec![],
    };
    let mut decoder_state = DecoderState::new(file_header);
    loop {
        let mut frame = Frame::new(&mut br, decoder_state)?;
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

        let result = frame.finalize()?;
        image_data.frames.push(ImageFrame {
            size: image_data.size,
            channels: result.1,
        });
        if let Some(state) = result.0 {
            decoder_state = state;
        } else {
            break;
        }
    }

    Ok((image_data, icc_bytes))
}

fn save_icc(icc_bytes: Vec<u8>, icc_filename: Option<PathBuf>) -> Result<(), Error> {
    match icc_filename {
        Some(icc_filename) => {
            std::fs::write(icc_filename, icc_bytes).map_err(|_| Error::OutputWriteFailure)
        }
        None => Ok(()),
    }
}

fn save_image(image_data: ImageData<f32>, output_filename: PathBuf) -> Result<(), Error> {
    let fn_str: String = String::from(output_filename.to_string_lossy());
    let mut output_bytes: Vec<u8> = vec![];
    if fn_str.ends_with(".ppm") {
        if image_data.frames.len() == 1 {
            if let [r, g, b] = &image_data.frames[0].channels[..] {
                output_bytes = jxl::enc::to_ppm_as_8bit(&[r.as_rect(), g.as_rect(), b.as_rect()]);
            }
        }
    } else if fn_str.ends_with(".pgm") {
        if image_data.frames.len() == 1 {
            if let [g] = &image_data.frames[0].channels[..] {
                output_bytes = g.as_rect().to_pgm_as_8bit();
            }
        }
    } else if fn_str.ends_with(".npy") {
        output_bytes = jxl::enc::numpy::to_numpy(image_data)?;
    }
    if output_bytes.is_empty() {
        return Err(Error::OutputFormatNotSupported);
    }
    if std::fs::write(output_filename, output_bytes).is_err() {
        Err(Error::OutputWriteFailure)
    } else {
        Ok(())
    }
}

#[derive(Parser)]
struct Opt {
    /// Input JXL file
    input: PathBuf,

    /// Output image file, should end in .ppm, .pgm or .npy
    output: PathBuf,

    ///  If specified, writes the ICC profile of the decoded image
    #[clap(long)]
    icc_out: Option<PathBuf>,
}

fn main() -> Result<(), Error> {
    #[cfg(feature = "tracing-subscriber")]
    {
        use tracing_subscriber::{fmt, prelude::*, EnvFilter};
        tracing_subscriber::registry()
            .with(fmt::layer())
            .with(EnvFilter::from_default_env())
            .init();
    }

    let opt = Opt::parse();
    let input_filename = opt.input;
    let mut file = match fs::File::open(input_filename.clone()) {
        Ok(file) => file,
        Err(err) => {
            println!("Cannot open file: {err}");
            return Err(Error::FileNotFound(input_filename));
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
                return Err(Error::InputReadFailure);
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
                    return Err(err);
                }
            }
        }

        let consumed = parser.previous_consumed_bytes();
        buf.copy_within(consumed..buf_valid, 0);
        buf_valid -= consumed;
    }

    let (image_data, icc_bytes) = decode_jxl_codestream(&codestream)?;
    save_image(image_data, opt.output)?;
    save_icc(icc_bytes, opt.icc_out)
}
