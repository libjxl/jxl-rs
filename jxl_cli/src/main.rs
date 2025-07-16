// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use clap::Parser;
use jxl::api::{JxlColorProfile, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer};
use jxl::decode::{ImageData, ImageFrame};
use jxl::error::Error;
use jxl::headers::bit_depth::BitDepth;
use jxl::image::Image;
use jxl::util::NewWithCapacity;
use std::fs;
use std::io::Read;
use std::mem::MaybeUninit;
use std::path::PathBuf;

pub mod enc;

fn save_icc(icc_bytes: &[u8], icc_filename: Option<PathBuf>) -> Result<(), Error> {
    match icc_filename {
        Some(icc_filename) => {
            std::fs::write(icc_filename, icc_bytes).map_err(|_| Error::OutputWriteFailure)
        }
        None => Ok(()),
    }
}

fn save_image(
    image_data: ImageData<f32>,
    bit_depth: BitDepth,
    color_profile: &JxlColorProfile,
    output_filename: PathBuf,
) -> Result<(), Error> {
    let fn_str: String = String::from(output_filename.to_string_lossy());
    let mut output_bytes: Vec<u8> = vec![];
    if fn_str.ends_with(".ppm") {
        if image_data.frames.len() == 1 {
            assert_eq!(image_data.frames[0].size, image_data.size);
            if let [r, g, b] = &image_data.frames[0].channels[..] {
                output_bytes = enc::pnm::to_ppm_as_8bit(&[r.as_rect(), g.as_rect(), b.as_rect()]);
            }
        }
    } else if fn_str.ends_with(".pgm") {
        if image_data.frames.len() == 1 {
            assert_eq!(image_data.frames[0].size, image_data.size);
            if let [g] = &image_data.frames[0].channels[..] {
                output_bytes = enc::pnm::to_pgm_as_8bit(&g.as_rect());
            }
        }
    } else if fn_str.ends_with(".npy") {
        output_bytes = enc::numpy::to_numpy(image_data)?;
    } else if fn_str.ends_with(".png") {
        output_bytes = enc::png::to_png(image_data, bit_depth, color_profile)?;
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

    /// Output image file, should end in .ppm, .pgm, .png or .npy
    output: PathBuf,

    ///  If specified, writes the ICC profile of the decoded image
    #[clap(long)]
    icc_out: Option<PathBuf>,

    ///  Likewise but for the ICC profile of the original colorspace
    #[clap(long)]
    original_icc_out: Option<PathBuf>,

    /// If specified, takes precedence over the bit depth in the input metadata
    #[clap(long)]
    override_bitdepth: Option<u32>,
}

fn main() -> Result<(), Error> {
    #[cfg(feature = "tracing-subscriber")]
    {
        use tracing_subscriber::{EnvFilter, fmt, prelude::*};
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

    let numpy_output = String::from(opt.output.to_string_lossy()).ends_with(".npy");
    let mut options = JxlDecoderOptions::default();
    options.xyb_output_linear = numpy_output;
    options.render_spot_colors = !numpy_output;
    let decoder = JxlDecoder::<jxl::api::states::Initialized>::new(options);
    let mut input_bytes = Vec::<u8>::new();
    file.read_to_end(&mut input_bytes)?;
    let mut input_buffer = input_bytes.as_slice();

    // TODO(sboukortt): somehow factor out these functions?
    fn feed_initialized_decoder(
        initialized_decoder: JxlDecoder<jxl::api::states::Initialized>,
        input_buffer: &mut &[u8],
    ) -> Result<JxlDecoder<jxl::api::states::WithImageInfo>, Error> {
        match initialized_decoder.process(input_buffer)? {
            jxl::api::ProcessingResult::Complete { result: decoder } => Ok(decoder),
            jxl::api::ProcessingResult::NeedsMoreInput {
                fallback: decoder, ..
            } => {
                if input_buffer.is_empty() {
                    println!("Not enough data.");
                    Err(Error::FileTruncated)
                } else {
                    feed_initialized_decoder(decoder, input_buffer)
                }
            }
        }
    }

    fn feed_decoder_for_frame_info(
        decoder_with_image_info: JxlDecoder<jxl::api::states::WithImageInfo>,
        input_buffer: &mut &[u8],
    ) -> Result<JxlDecoder<jxl::api::states::WithFrameInfo>, Error> {
        match decoder_with_image_info.process(input_buffer)? {
            jxl::api::ProcessingResult::Complete { result: decoder } => Ok(decoder),
            jxl::api::ProcessingResult::NeedsMoreInput {
                fallback: decoder, ..
            } => {
                if input_buffer.is_empty() {
                    println!("Not enough data.");
                    Err(Error::FileTruncated)
                } else {
                    feed_decoder_for_frame_info(decoder, input_buffer)
                }
            }
        }
    }

    fn feed_decoder_for_image_info(
        initialized_decoder: JxlDecoder<jxl::api::states::WithFrameInfo>,
        input_buffer: &mut &[u8],
        outputs: &mut Vec<JxlOutputBuffer<'_>>,
    ) -> Result<JxlDecoder<jxl::api::states::WithImageInfo>, Error> {
        match initialized_decoder.process(input_buffer, outputs)? {
            jxl::api::ProcessingResult::Complete { result: decoder } => Ok(decoder),
            jxl::api::ProcessingResult::NeedsMoreInput {
                fallback: decoder, ..
            } => {
                if input_buffer.is_empty() {
                    println!("Not enough data.");
                    Err(Error::FileTruncated)
                } else {
                    feed_decoder_for_image_info(decoder, input_buffer, outputs)
                }
            }
        }
    }

    let with_image_info = feed_initialized_decoder(decoder, &mut input_buffer)?;
    let embedded_profile = with_image_info.embedded_color_profile();
    let output_profile = with_image_info.output_color_profile().clone();
    let original_bit_depth = with_image_info.basic_info().bit_depth;
    let num_channels = with_image_info
        .current_pixel_format()
        .color_type
        .samples_per_pixel();

    let original_icc_result = save_icc(embedded_profile.as_icc().as_slice(), opt.original_icc_out);
    let data_icc = output_profile.as_icc();
    let data_icc_result = save_icc(data_icc.as_slice(), opt.icc_out);

    let with_frame_info = feed_decoder_for_frame_info(with_image_info, &mut input_buffer)?;

    let (xsize, ysize) = with_frame_info.frame_header().size();

    let mut output_buffers: Vec<Vec<MaybeUninit<u8>>> = Vec::new_with_capacity(num_channels)?;
    for _ in 0..num_channels {
        output_buffers.push(vec![MaybeUninit::uninit(); xsize * ysize * 4]);
    }

    let image_from_vec = |vec: &Vec<MaybeUninit<u8>>, size| {
        let mut image = Image::<f32>::new(size)?;
        let mut rect = image.as_rect_mut();
        for y in 0..size.1 {
            let row = rect.row(y);
            for x in 0..size.0 {
                row[x] = f32::from_ne_bytes(unsafe {
                    [
                        vec[4 * (y * size.0 + x)].assume_init(),
                        vec[4 * (y * size.0 + x) + 1].assume_init(),
                        vec[4 * (y * size.0 + x) + 2].assume_init(),
                        vec[4 * (y * size.0 + x) + 3].assume_init(),
                    ]
                })
            }
        }
        Ok::<Image<f32>, Error>(image)
    };

    let mut outputs = output_buffers
        .as_mut_slice()
        .iter_mut()
        .map(|buffer| JxlOutputBuffer::new_uninit(buffer.as_mut_slice(), ysize, 4 * xsize))
        .collect::<Vec<JxlOutputBuffer<'_>>>();

    let mut image_data = ImageData {
        size: (xsize, ysize),
        frames: Vec::new(),
    };

    let mut with_image_info =
        feed_decoder_for_image_info(with_frame_info, &mut input_buffer, &mut outputs)?;

    let mut image_frame = ImageFrame {
        size: (xsize, ysize),
        channels: Vec::new(),
    };
    for vec in output_buffers.iter() {
        image_frame
            .channels
            .push(image_from_vec(vec, (xsize, ysize))?);
    }
    image_data.frames.push(image_frame);

    while !input_buffer.is_empty() {
        let with_frame_info = feed_decoder_for_frame_info(with_image_info, &mut input_buffer)?;

        let (xsize, ysize) = with_frame_info.frame_header().size();

        output_buffers.clear();
        output_buffers.reserve(num_channels);
        for _ in 0..num_channels {
            output_buffers.push(vec![MaybeUninit::uninit(); xsize * ysize * 4]);
        }

        let mut outputs = output_buffers
            .as_mut_slice()
            .iter_mut()
            .map(|buffer| JxlOutputBuffer::new_uninit(buffer.as_mut_slice(), ysize, 4 * xsize))
            .collect::<Vec<JxlOutputBuffer<'_>>>();

        with_image_info =
            feed_decoder_for_image_info(with_frame_info, &mut input_buffer, &mut outputs)?;

        let mut image_frame = ImageFrame {
            size: (xsize, ysize),
            channels: Vec::new(),
        };
        for vec in output_buffers.iter() {
            image_frame
                .channels
                .push(image_from_vec(vec, (xsize, ysize))?);
        }
        image_data.frames.push(image_frame);
    }

    let output_bit_depth = match opt.override_bitdepth {
        None => original_bit_depth,
        Some(num_bits) => BitDepth::integer_samples(num_bits),
    };
    let image_result = save_image(image_data, output_bit_depth, &output_profile, opt.output);

    if let Err(ref err) = original_icc_result {
        println!("Failed to save original ICC profile: {err}");
    }
    if let Err(ref err) = data_icc_result {
        println!("Failed to save data ICC profile: {err}");
    }
    if let Err(ref err) = image_result {
        println!("Failed to save image: {err}");
    }

    original_icc_result?;
    data_icc_result?;
    image_result?;

    Ok(())
}
