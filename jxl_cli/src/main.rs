// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use clap::Parser;
use color_eyre::eyre::{Result, WrapErr, eyre};
use jxl::api::{JxlColorProfile, JxlColorType, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer};
use jxl::image::{Image, ImageDataType};
use std::fs;
use std::io::Read;
use std::path::PathBuf;
use std::time::Instant;

pub mod enc;

fn save_icc(icc_bytes: &[u8], icc_filename: Option<PathBuf>) -> Result<()> {
    icc_filename.map_or(Ok(()), |path| {
        std::fs::write(&path, icc_bytes)
            .wrap_err_with(|| format!("Failed to write ICC profile to {:?}", path))
    })
}

pub struct ImageFrame<T: ImageDataType> {
    pub size: (usize, usize),
    pub channels: Vec<Image<T>>,
}

pub struct ImageData<T: ImageDataType> {
    pub size: (usize, usize),
    pub frames: Vec<ImageFrame<T>>,
}

fn save_image(
    image_data: ImageData<f32>,
    bit_depth: u32,
    color_profile: &JxlColorProfile,
    output_filename: &PathBuf,
) -> Result<()> {
    let fn_str = output_filename.to_string_lossy();
    let mut output_bytes: Vec<u8> = vec![];
    if fn_str.ends_with(".exr") {
        output_bytes = enc::exr::to_exr(image_data, bit_depth, color_profile)?;
    } else if fn_str.ends_with(".ppm") {
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
        return Err(eyre!("Output format {:?} not supported", output_filename));
    }
    std::fs::write(output_filename, output_bytes)
        .wrap_err_with(|| format!("Failed to write decoded image to {:?}", &output_filename))
}

#[derive(Parser)]
struct Opt {
    /// Input JXL file
    input: PathBuf,

    /// Output image file, should end in .ppm, .pgm, .png or .npy
    #[clap(required_unless_present = "speedtest")]
    output: Option<PathBuf>,

    /// Print measured decoding speed..
    #[clap(long, short, action)]
    speedtest: bool,

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

// Extract RGB channels from interleaved RGB buffer
fn planes_from_interleaved(interleaved: &Image<f32>) -> Result<Vec<Image<f32>>> {
    let size = interleaved.size();
    let size = (size.0 / 3, size.1);
    let mut r_image = Image::<f32>::new(size)?;
    let mut g_image = Image::<f32>::new(size)?;
    let mut b_image = Image::<f32>::new(size)?;

    let mut r_rect = r_image.as_rect_mut();
    let mut g_rect = g_image.as_rect_mut();
    let mut b_rect = b_image.as_rect_mut();
    let interleaved_rect = interleaved.as_rect();

    for y in 0..size.1 {
        let r_row = r_rect.row(y);
        let g_row = g_rect.row(y);
        let b_row = b_rect.row(y);
        let src_row = interleaved_rect.row(y);
        for x in 0..size.0 {
            r_row[x] = src_row[3 * x];
            g_row[x] = src_row[3 * x + 1];
            b_row[x] = src_row[3 * x + 2];
        }
    }
    Ok(vec![r_image, g_image, b_image])
}

fn main() -> Result<()> {
    #[cfg(feature = "tracing-subscriber")]
    {
        use tracing_subscriber::{EnvFilter, fmt, prelude::*};
        tracing_subscriber::registry()
            .with(fmt::layer())
            .with(EnvFilter::from_default_env())
            .init();
    }

    let opt = Opt::parse();
    let mut file = fs::File::open(opt.input.clone())
        .wrap_err_with(|| format!("Failed to read source image from {:?}", opt.input))?;

    let (numpy_output, exr_output) = match &opt.output.as_ref().map(|p| p.to_string_lossy()) {
        Some(path) => (path.ends_with(".npy"), path.ends_with(".exr")),
        None => (false, false),
    };
    let mut options = JxlDecoderOptions::default();
    options.xyb_output_linear = numpy_output || exr_output;
    options.render_spot_colors = !numpy_output;
    let mut input_bytes = Vec::<u8>::new();
    file.read_to_end(&mut input_bytes)?;
    let mut input_buffer = input_bytes.as_slice();

    let start = Instant::now();

    let mut initialized_decoder = JxlDecoder::<jxl::api::states::Initialized>::new(options);

    // Process until we have image info
    let mut decoder_with_image_info = loop {
        match initialized_decoder.process(&mut input_buffer).unwrap() {
            jxl::api::ProcessingResult::Complete { result } => break Ok(result),
            jxl::api::ProcessingResult::NeedsMoreInput { fallback, .. } => {
                if input_buffer.is_empty() {
                    break Err(eyre!("Source file {:?} truncated", opt.input));
                }
                initialized_decoder = fallback;
            }
        }
    }?;

    let embedded_profile = decoder_with_image_info.embedded_color_profile();
    let output_profile = decoder_with_image_info.output_color_profile().clone();
    let info = decoder_with_image_info.basic_info();
    let num_pixels = info.size.0 * info.size.1;
    let extra_channels = info.extra_channels.len();
    let original_bit_depth = info.bit_depth.clone();
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

    let (untransposed_w, untransposed_h) = info.size;
    let mut image_data = ImageData {
        size: if info.orientation.is_transposing() {
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
                        break Err(eyre!("Source file {:?} truncated", opt.input));
                    }
                    decoder_with_image_info = fallback;
                }
            }
        }?;

        let mut outputs = vec![Image::<f32>::new((
            image_data.size.0 * samples_per_pixel,
            image_data.size.1,
        ))?];

        for _ in 0..extra_channels {
            outputs.push(Image::<f32>::new(image_data.size)?);
        }

        let mut output_bufs: Vec<JxlOutputBuffer<'_>> = outputs
            .iter_mut()
            .map(JxlOutputBuffer::from_image)
            .collect();

        decoder_with_image_info = loop {
            match decoder_with_frame_info
                .process(&mut input_buffer, &mut output_bufs)
                .unwrap()
            {
                jxl::api::ProcessingResult::Complete { result } => break Ok(result),
                jxl::api::ProcessingResult::NeedsMoreInput { fallback, .. } => {
                    if input_buffer.is_empty() {
                        break Err(eyre!("Source file {:?} truncated", opt.input));
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
        if color_type == JxlColorType::Grayscale {
            // Each buffer contains a single channel
            image_frame.channels = outputs;
        } else {
            // First buffer contains interleaved RGB
            let rgb_channels = planes_from_interleaved(&outputs[0])?;
            image_frame.channels.extend(rgb_channels);

            // Additional buffers contain extra channels (e.g., alpha)
            image_frame.channels.extend(outputs.into_iter().skip(1));
        }

        image_data.frames.push(image_frame);

        if !decoder_with_image_info.has_more_frames() {
            break;
        }
    }

    if opt.speedtest {
        let duration = start.elapsed().as_nanos() as f64 / 1e9;
        println!(
            "Decoded {} pixels in {} seconds: {} pixels/s",
            num_pixels,
            duration,
            num_pixels as f64 / duration
        );
    }

    let image_result: Option<Result<()>> = opt.output.map(|path| {
        let output_bit_depth = match opt.override_bitdepth {
            None => original_bit_depth.bits_per_sample(),
            Some(num_bits) => num_bits,
        };
        let image_result = save_image(image_data, output_bit_depth, &output_profile, &path);

        if let Err(ref err) = original_icc_result {
            println!("Failed to save original ICC profile: {err}");
        }
        if let Err(ref err) = data_icc_result {
            println!("Failed to save data ICC profile: {err}");
        }
        if let Err(ref err) = image_result {
            println!("Failed to save image: {err}");
        }
        image_result
    });

    original_icc_result?;
    data_icc_result?;

    image_result.unwrap_or(Ok(()))?;

    Ok(())
}
