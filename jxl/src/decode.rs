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
pub struct DecodeOptions<'a> {
    pub xyb_output_linear: bool,
    pub frame_callback: Option<&'a mut dyn FnMut(&Frame) -> Result<(), Error>>,
}

impl<'a> DecodeOptions<'a> {
    pub fn new() -> DecodeOptions<'a> {
        DecodeOptions {
            xyb_output_linear: true,
            frame_callback: None,
        }
    }
}

impl<'a> Default for DecodeOptions<'a> {
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

#[allow(unused_mut)]
pub fn decode_jxl_codestream(
    mut options: DecodeOptions,
    data: &[u8],
) -> Result<(ImageData<f32>, Vec<u8>), Error> {
    let mut br = BitReader::new(data);
    let file_header = FileHeader::read(&mut br)?;
    info!(
        "Image size: {} x {}",
        file_header.size.xsize(),
        file_header.size.ysize()
    );
    // TODO(firsching): Make it such that we also write icc bytes in the
    // case where want_icc is false.
    let mut icc_bytes = Vec::<u8>::new();
    if file_header.image_metadata.color_encoding.want_icc {
        let r = read_icc(&mut br)?;
        info!("found {}-byte ICC", r.len());
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
    decoder_state.xyb_output_linear = options.xyb_output_linear;
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

        if let Some(ref mut callback) = options.frame_callback {
            callback(&frame)?;
        }
        let frame_size = frame.header().size();
        let result = frame.finalize()?;
        image_data.frames.push(ImageFrame {
            size: frame_size,
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

#[cfg(test)]
mod test {
    use super::decode_jxl_codestream;
    use crate::{container::ContainerParser, error::Error};
    use jxl_macros::for_each_test_file;
    use std::path::Path;

    fn read_file_from_path(path: &Path) -> Result<(), Error> {
        let data = std::fs::read(path).unwrap();
        let codestream = ContainerParser::collect_codestream(data.as_slice()).unwrap();
        decode_jxl_codestream(Default::default(), &codestream)?;
        Ok(())
    }

    for_each_test_file!(read_file_from_path);
}
