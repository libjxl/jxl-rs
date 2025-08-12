// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use clap::Parser;
use jxl::api::{
    JxlColorEncoding, JxlColorProfile, JxlColorType, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
};
use jxl::container::{ContainerParser, ParseEvent};
use jxl::decode::{DecodeOptions, DecodeResult, ImageData, ImageFrame};
use jxl::error::{Error, Result};
use jxl::headers::bit_depth::BitDepth;
use jxl::headers::extra_channels::ExtraChannel;
use jxl::image::Image;
use std::fs;
use std::io::Read;
use std::path::PathBuf;

pub mod enc;

fn save_icc(icc_bytes: &[u8], icc_filename: Option<PathBuf>) -> Result<()> {
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

    // Use the library decoder in a one shot buffer instead of the
    // public API.
    // TODO(zond): Remove this option when the rate of API bugs approaches 0.
    #[clap(long, short, action)]
    without_api: bool,
}

fn f32_from_bytes(vec: &[u8]) -> f32 {
    f32::from_ne_bytes(vec.try_into().unwrap())
}

fn image_from_vec(vec: &[u8], size: (usize, usize)) -> Result<Image<f32>> {
    let mut image = Image::<f32>::new(size)?;
    let mut rect = image.as_rect_mut();
    for y in 0..size.1 {
        let row = rect.row(y);
        for (x, v) in row.iter_mut().enumerate().take(size.0) {
            let base_idx = 4 * (y * size.0 + x);
            *v = f32_from_bytes(&vec[base_idx..base_idx + 4]);
        }
    }
    Ok(image)
}

// Extract RGB channels from interleaved RGB buffer
fn images_from_rgb_vec(vec: &[u8], size: (usize, usize)) -> Result<Vec<Image<f32>>> {
    let mut r_image = Image::<f32>::new(size)?;
    let mut g_image = Image::<f32>::new(size)?;
    let mut b_image = Image::<f32>::new(size)?;

    let mut r_rect = r_image.as_rect_mut();
    let mut g_rect = g_image.as_rect_mut();
    let mut b_rect = b_image.as_rect_mut();

    for y in 0..size.1 {
        let r_row = r_rect.row(y);
        let g_row = g_rect.row(y);
        let b_row = b_rect.row(y);
        for x in 0..size.0 {
            let base_idx = 12 * (y * size.0 + x);
            r_row[x] = f32_from_bytes(&vec[base_idx..base_idx + 4]);
            g_row[x] = f32_from_bytes(&vec[base_idx + 4..base_idx + 8]);
            b_row[x] = f32_from_bytes(&vec[base_idx + 8..base_idx + 12]);
        }
    }
    Ok(vec![r_image, g_image, b_image])
}

fn with_lib(opt: Opt) -> Result<()> {
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
                return Err(Error::InputReadFailure(err));
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

    let numpy_output = String::from(opt.output.to_string_lossy()).ends_with(".npy");
    let mut options = DecodeOptions::new();
    options.xyb_output_linear = numpy_output;
    options.render_spotcolors = !numpy_output;
    let DecodeResult {
        image_data,
        bit_depth,
        original_icc,
        data_icc,
    } = jxl::decode::decode_jxl_codestream(options, &codestream)?;

    let original_icc_result = save_icc(original_icc.as_slice(), opt.original_icc_out);
    let srgb;
    let grayscale = image_data.frames[0].channels.len() < 3;
    let data_icc_result = save_icc(
        match data_icc.as_ref() {
            Some(data_icc) => data_icc.as_slice(),
            None => {
                srgb = JxlColorEncoding::srgb(grayscale)
                    .maybe_create_profile()?
                    .unwrap();
                srgb.as_slice()
            }
        },
        opt.icc_out,
    );
    let bit_depth = match opt.override_bitdepth {
        None => bit_depth,
        Some(num_bits) => BitDepth::integer_samples(num_bits),
    };
    let color_profile = match data_icc {
        Some(vec) => JxlColorProfile::Icc(vec),
        None => JxlColorProfile::Simple(JxlColorEncoding::srgb(grayscale)),
    };
    let image_result = save_image(image_data, bit_depth, &color_profile, opt.output);

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

fn with_api(opt: Opt) -> Result<()> {
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
    let mut input_bytes = Vec::<u8>::new();
    file.read_to_end(&mut input_bytes)?;
    let mut input_buffer = input_bytes.as_slice();

    let mut initialized_decoder = JxlDecoder::<jxl::api::states::Initialized>::new(options);

    // Process until we have image info
    let mut decoder_with_image_info = loop {
        match initialized_decoder.process(&mut input_buffer).unwrap() {
            jxl::api::ProcessingResult::Complete { result } => break Ok(result),
            jxl::api::ProcessingResult::NeedsMoreInput { fallback, .. } => {
                if input_buffer.is_empty() {
                    println!("Not enough data.");
                    break Err(Error::FileTruncated);
                }
                initialized_decoder = fallback;
            }
        }
    }?;

    let embedded_profile = decoder_with_image_info.embedded_color_profile();
    let output_profile = decoder_with_image_info.output_color_profile().clone();
    let alpha_channel_index = decoder_with_image_info
        .basic_info()
        .extra_channels
        .iter()
        .position(|info| info.ec_type == ExtraChannel::Alpha);
    let original_bit_depth = decoder_with_image_info.basic_info().bit_depth;
    let pixel_format = decoder_with_image_info.current_pixel_format().clone();
    let color_type = pixel_format.color_type;
    // TODO(zond): This is the way the API works right now, let's improve it when the API is cleverer.
    let samples_per_pixel = if color_type == JxlColorType::Grayscale {
        1
    } else {
        3
    };

    let original_icc_result = save_icc(embedded_profile.as_icc().as_slice(), opt.original_icc_out);
    let data_icc = output_profile.as_icc();
    let data_icc_result = save_icc(data_icc.as_slice(), opt.icc_out);

    let (untransposed_w, untransposed_h) = decoder_with_image_info.basic_info().size;
    let mut image_data = ImageData {
        size: if decoder_with_image_info
            .basic_info()
            .orientation
            .is_transposing()
        {
            (untransposed_h, untransposed_w)
        } else {
            (untransposed_w, untransposed_h)
        },
        frames: Vec::new(),
    };

    loop {
        let mut decoder_with_frame_info = loop {
            match decoder_with_image_info.process(&mut input_buffer).unwrap() {
                jxl::api::ProcessingResult::Complete { result } => break Ok(result),
                jxl::api::ProcessingResult::NeedsMoreInput { fallback, .. } => {
                    if input_buffer.is_empty() {
                        println!("Not enough data.");
                        break Err(Error::FileTruncated);
                    }
                    decoder_with_image_info = fallback;
                }
            }
        }?;

        let num_pixels = image_data.size.0 * image_data.size.1;

        let mut output_vecs = vec![vec![0u8; num_pixels * samples_per_pixel * 4]];

        if let Some(alpha_index) = alpha_channel_index {
            assert!(
                alpha_index == 0,
                "alpha channel isn't first channel after color channels"
            );
            output_vecs.push(vec![0u8; image_data.size.0 * image_data.size.1 * 4]);
        }

        let mut output_bufs: Vec<JxlOutputBuffer<'_>> = output_vecs
            .iter_mut()
            .map(|v| {
                let len = v.len();
                JxlOutputBuffer::new(v, image_data.size.1, len / image_data.size.1)
            })
            .collect();

        decoder_with_image_info = loop {
            match decoder_with_frame_info
                .process(&mut input_buffer, &mut output_bufs)
                .unwrap()
            {
                jxl::api::ProcessingResult::Complete { result } => break Ok(result),
                jxl::api::ProcessingResult::NeedsMoreInput { fallback, .. } => {
                    if input_buffer.is_empty() {
                        println!("Not enough data.");
                        break Err(Error::FileTruncated);
                    }
                    decoder_with_frame_info = fallback;
                }
            }
        }?;

        let mut image_frame = ImageFrame {
            size: image_data.size,
            channels: Vec::new(),
        };

        // Handle RGB vs grayscale buffer layout
        if pixel_format.color_type == JxlColorType::Grayscale {
            // Each buffer contains a single channel
            for vec in output_vecs.iter() {
                image_frame
                    .channels
                    .push(image_from_vec(vec, (image_data.size.0, image_data.size.1))?);
            }
        } else {
            // First buffer contains interleaved RGB
            let rgb_channels =
                images_from_rgb_vec(&output_vecs[0], (image_data.size.0, image_data.size.1))?;
            image_frame.channels.extend(rgb_channels);

            // Additional buffers contain extra channels (e.g., alpha)
            for vec in output_vecs.iter().skip(1) {
                image_frame
                    .channels
                    .push(image_from_vec(vec, (image_data.size.0, image_data.size.1))?);
            }
        }

        image_data.frames.push(image_frame);

        if !decoder_with_image_info.has_more_frames() {
            break;
        }
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
    if opt.without_api {
        with_lib(opt)
    } else {
        with_api(opt)
    }
}
