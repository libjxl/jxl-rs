// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    bit_reader::BitReader,
    error::Error,
    frame::{DecoderState, Frame, Section},
    headers::{FileHeader, JxlHeader},
    icc::read_icc,
    image::{Image, ImageDataType},
    util::tracing_wrappers::*,
};

#[allow(clippy::type_complexity)]
pub struct DecodeOptions<FC> {
    xyb_output_linear: bool,
    enable_output: bool,
    frame_callback: Option<FC>,
}

impl DecodeOptions<fn(&Frame) -> Result<(), Error>> {
    pub fn new() -> Self {
        Self {
            xyb_output_linear: true,
            enable_output: true,
            frame_callback: None,
        }
    }
}

impl<FC> DecodeOptions<FC> {
    pub fn set_frame_callback<NFC>(self, frame_callback: NFC) -> DecodeOptions<NFC> {
        DecodeOptions {
            xyb_output_linear: self.xyb_output_linear,
            enable_output: self.enable_output,
            frame_callback: Some(frame_callback),
        }
    }

    pub fn set_xyb_output_linear(mut self, xyb_output_linear: bool) -> Self {
        self.xyb_output_linear = xyb_output_linear;
        self
    }
}

impl Default for DecodeOptions<fn(&Frame) -> Result<(), Error>> {
    fn default() -> Self {
        Self::new()
    }
}

pub struct ImageFrame<T: ImageDataType> {
    pub size: (usize, usize),
    pub channels: Vec<Image<T>>,
}

pub struct ImageData<T: ImageDataType> {
    pub size: (usize, usize),
    pub frames: Vec<ImageFrame<T>>,
}

pub fn decode_jxl_codestream<FC>(
    mut options: DecodeOptions<FC>,
    data: &[u8],
) -> Result<(ImageData<f32>, Vec<u8>), Error>
where
    FC: FnMut(&Frame) -> Result<(), Error>,
{
    let mut br = BitReader::new(data);
    let file_header = FileHeader::read(&mut br)?;
    let input_xsize = file_header.size.xsize();
    let input_ysize = file_header.size.ysize();
    let (output_xsize, output_ysize) = if file_header.image_metadata.orientation.is_transposing() {
        (input_ysize, input_xsize)
    } else {
        (input_xsize, input_ysize)
    };
    info!("Image size: {} x {}", output_xsize, output_ysize);
    let icc_bytes = if file_header.image_metadata.color_encoding.want_icc {
        let r = read_icc(&mut br)?;
        println!("found {}-byte ICC", r.len());
        r
    } else {
        // TODO: handle potential error here?
        file_header
            .image_metadata
            .color_encoding
            .maybe_create_profile()?
            .unwrap()
    };

    br.jump_to_byte_boundary()?;
    let mut image_data: ImageData<f32> = ImageData {
        size: (output_xsize as usize, output_ysize as usize),
        frames: vec![],
    };
    let mut decoder_state = DecoderState::new(file_header);
    decoder_state.xyb_output_linear = options.xyb_output_linear;
    decoder_state.enable_output = options.enable_output;
    loop {
        let mut frame = Frame::new(&mut br, decoder_state)?;
        let mut section_readers = frame.sections(&mut br)?;

        info!("read frame with {} sections", section_readers.len());

        frame.decode_lf_global(&mut section_readers[frame.get_section_idx(Section::LfGlobal)])?;

        for group in 0..frame.header().num_lf_groups() {
            frame.decode_lf_group(
                group,
                &mut section_readers[frame.get_section_idx(Section::Lf { group })],
            )?;
        }

        frame.decode_hf_global(&mut section_readers[frame.get_section_idx(Section::HfGlobal)])?;

        for pass in 0..frame.header().passes.num_passes as usize {
            for group in 0..frame.header().num_groups() {
                frame.decode_hf_group(
                    group,
                    pass,
                    &mut section_readers[frame.get_section_idx(Section::Hf { group, pass })],
                )?;
            }
        }

        if let Some(ref mut callback) = options.frame_callback {
            callback(&frame)?;
        }
        let result = frame.finalize()?;
        if let Some(channels) = result.channels {
            image_data.frames.push(ImageFrame {
                size: channels[0].size(),
                channels,
            });
        }
        if let Some(state) = result.decoder_state {
            decoder_state = state;
        } else {
            break;
        }
    }

    Ok((image_data, icc_bytes))
}

#[cfg(test)]
mod test {
    use super::{decode_jxl_codestream, DecodeOptions};
    use crate::{container::ContainerParser, error::Error};
    use jxl_macros::for_each_test_file;
    use std::path::Path;

    fn read_file_from_path(path: &Path) -> Result<(), Error> {
        let data = std::fs::read(path).unwrap();
        let codestream = ContainerParser::collect_codestream(data.as_slice()).unwrap();
        let mut options = DecodeOptions::new();
        options.enable_output = false;
        decode_jxl_codestream(options, &codestream)?;
        Ok(())
    }

    for_each_test_file!(read_file_from_path);
}
