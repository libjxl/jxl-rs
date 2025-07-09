// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    api::{JxlColorEncoding, JxlPrimaries, JxlTransferFunction, JxlWhitePoint},
    bit_reader::BitReader,
    error::Error,
    frame::{DecoderState, Frame, Section},
    headers::{
        FileHeader, JxlHeader,
        bit_depth::BitDepth,
        color_encoding::{ColorSpace, RenderingIntent},
    },
    icc::IncrementalIccReader,
    image::{Image, ImageDataType},
    util::tracing_wrappers::*,
};

#[allow(clippy::type_complexity)]
pub struct DecodeOptions<'a> {
    pub xyb_output_linear: bool,
    pub render_spotcolors: bool,
    enable_output: bool,
    pub frame_callback: Option<&'a mut dyn FnMut(&Frame) -> Result<(), Error>>,
}

impl<'a> DecodeOptions<'a> {
    pub fn new() -> DecodeOptions<'a> {
        DecodeOptions {
            xyb_output_linear: true,
            enable_output: true,
            render_spotcolors: true,
            frame_callback: None,
        }
    }
}

impl Default for DecodeOptions<'_> {
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

pub struct DecodeResult {
    pub image_data: ImageData<f32>,
    pub bit_depth: BitDepth,
    pub original_icc: Vec<u8>,
    // None means sRGB.
    pub data_icc: Option<Vec<u8>>,
}

pub fn decode_jxl_codestream(
    mut options: DecodeOptions,
    data: &[u8],
) -> Result<DecodeResult, Error> {
    let mut br = BitReader::new(data);
    let file_header = FileHeader::read(&mut br)?;
    let bit_depth = file_header.image_metadata.bit_depth;
    let input_xsize = file_header.size.xsize();
    let input_ysize = file_header.size.ysize();
    let (output_xsize, output_ysize) = if file_header.image_metadata.orientation.is_transposing() {
        (input_ysize, input_xsize)
    } else {
        (input_xsize, input_ysize)
    };
    info!("Image size: {} x {}", output_xsize, output_ysize);
    let original_icc_bytes = if file_header.image_metadata.color_encoding.want_icc {
        let mut r = IncrementalIccReader::new(&mut br)?;
        r.read_all(&mut br)?;
        let icc = r.finalize()?;
        println!("found {}-byte ICC", icc.len());
        icc
    } else {
        // TODO: handle potential error here?
        JxlColorEncoding::from_internal(&file_header.image_metadata.color_encoding)?
            .maybe_create_profile()?
            .unwrap()
    };
    let data_icc_bytes = if file_header.image_metadata.xyb_encoded {
        if options.xyb_output_linear {
            let color_encoding =
                if file_header.image_metadata.color_encoding.color_space == ColorSpace::Gray {
                    JxlColorEncoding::GrayscaleColorSpace {
                        white_point: JxlWhitePoint::D65,
                        transfer_function: JxlTransferFunction::Linear,
                        rendering_intent: RenderingIntent::Relative,
                    }
                } else {
                    JxlColorEncoding::RgbColorSpace {
                        white_point: JxlWhitePoint::D65,
                        primaries: JxlPrimaries::SRGB,
                        transfer_function: JxlTransferFunction::Linear,
                        rendering_intent: RenderingIntent::Relative,
                    }
                };
            Some(color_encoding.maybe_create_profile()?.unwrap())
        } else {
            // Regular (non-linear) sRGB.
            None
        }
    } else {
        Some(original_icc_bytes.clone())
    };

    br.jump_to_byte_boundary()?;
    let mut image_data: ImageData<f32> = ImageData {
        size: (output_xsize as usize, output_ysize as usize),
        frames: vec![],
    };
    let mut decoder_state = DecoderState::new(file_header);
    decoder_state.xyb_output_linear = options.xyb_output_linear;
    decoder_state.enable_output = options.enable_output;
    decoder_state.render_spotcolors = options.render_spotcolors;
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

        frame.prepare_render_pipeline()?;
        frame.finalize_lf()?;

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

    Ok(DecodeResult {
        image_data,
        bit_depth,
        original_icc: original_icc_bytes,
        data_icc: data_icc_bytes,
    })
}

#[cfg(test)]
mod test {
    use super::{DecodeOptions, decode_jxl_codestream};
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
