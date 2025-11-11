// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use clap::Parser;
use color_eyre::eyre::{Result, WrapErr, eyre};
use jxl::api::{
    JxlAnimation, JxlBitDepth, JxlColorProfile, JxlColorType, JxlDecoder, JxlDecoderOptions,
    JxlOutputBuffer,
};
use jxl::image::{Image, ImageDataType, Rect};
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};
use std::{fs, mem};

pub mod enc;

fn save_icc(icc_bytes: &[u8], icc_filename: Option<&PathBuf>) -> Result<()> {
    icc_filename.map_or(Ok(()), |path| {
        std::fs::write(path, icc_bytes)
            .wrap_err_with(|| format!("Failed to write ICC profile to {:?}", path))
    })
}

pub struct ImageFrame<T: ImageDataType> {
    pub channels: Vec<Image<T>>,
    pub duration: f64,
    pub color_type: JxlColorType,
}

pub struct DecodeOutput<T: ImageDataType> {
    pub size: (usize, usize),
    pub frames: Vec<ImageFrame<T>>,
    pub original_bit_depth: JxlBitDepth,
    pub output_profile: JxlColorProfile,
    pub jxl_animation: Option<JxlAnimation>,
    pub original_icc_result: Result<()>,
    pub data_icc_result: Result<()>,
}

fn save_image(
    image_data: &DecodeOutput<f32>,
    bit_depth: u32,
    output_filename: &PathBuf,
) -> Result<()> {
    let fn_str = output_filename.to_string_lossy();
    let mut writer = BufWriter::new(File::create(output_filename)?);
    if fn_str.ends_with(".exr") {
        enc::exr::to_exr(image_data, bit_depth, &mut writer)?;
    } else if fn_str.ends_with(".ppm") {
        if image_data.frames.len() == 1
            && let [r, g, b] = &image_data.frames[0].channels[..]
        {
            enc::pnm::to_ppm_as_8bit([r, g, b], &mut writer)?;
        }
    } else if fn_str.ends_with(".pgm") {
        if image_data.frames.len() == 1
            && let [g] = &image_data.frames[0].channels[..]
        {
            enc::pnm::to_pgm_as_8bit(g, &mut writer)?;
        }
    } else if fn_str.ends_with(".npy") {
        enc::numpy::to_numpy(image_data, &mut writer)?;
    } else if fn_str.ends_with(".png") {
        enc::png::to_png(image_data, bit_depth, &mut writer)?;
    } else {
        return Err(eyre!(
            "Output format not supported for {:?}",
            output_filename
        ));
    }
    writer
        .flush()
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

    /// Number of times to repeat the decoding.
    #[clap(long, short)]
    num_reps: Option<u32>,

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

    for y in 0..size.1 {
        let r_row = r_image.row_mut(y);
        let g_row = g_image.row_mut(y);
        let b_row = b_image.row_mut(y);
        let src_row = interleaved.row(y);
        for x in 0..size.0 {
            r_row[x] = src_row[3 * x];
            g_row[x] = src_row[3 * x + 1];
            b_row[x] = src_row[3 * x + 2];
        }
    }
    Ok(vec![r_image, g_image, b_image])
}

fn decode_bytes(
    mut input_buffer: &[u8],
    decoder_options: JxlDecoderOptions,
    cli_opt: &Opt,
) -> Result<(DecodeOutput<f32>, Duration)> {
    let start = Instant::now();

    let mut initialized_decoder = JxlDecoder::<jxl::api::states::Initialized>::new(decoder_options);

    // Process until we have image info
    let mut decoder_with_image_info = loop {
        match initialized_decoder.process(&mut input_buffer)? {
            jxl::api::ProcessingResult::Complete { result } => break Ok(result),
            jxl::api::ProcessingResult::NeedsMoreInput { fallback, .. } => {
                if input_buffer.is_empty() {
                    break Err(eyre!("Source file {:?} truncated", cli_opt.input));
                }
                initialized_decoder = fallback;
            }
        }
    }?;

    let info = decoder_with_image_info.basic_info();
    let embedded_profile = decoder_with_image_info.embedded_color_profile();
    let output_profile = decoder_with_image_info.output_color_profile().clone();
    let data_icc_result = save_icc(output_profile.as_icc().as_slice(), cli_opt.icc_out.as_ref());

    let mut image_data = DecodeOutput {
        size: info.size,
        frames: Vec::new(),
        original_bit_depth: info.bit_depth.clone(),
        output_profile,
        jxl_animation: info.animation.clone(),
        original_icc_result: save_icc(
            embedded_profile.as_icc().as_slice(),
            cli_opt.original_icc_out.as_ref(),
        ),
        data_icc_result,
    };

    let extra_channels = info.extra_channels.len();
    let pixel_format = decoder_with_image_info.current_pixel_format().clone();
    let color_type = pixel_format.color_type;
    // TODO(zond): This is the way the API works right now, let's improve it when the API is cleverer.
    let samples_per_pixel = if color_type == JxlColorType::Grayscale {
        1
    } else {
        3
    };

    loop {
        let mut decoder_with_frame_info = loop {
            match decoder_with_image_info.process(&mut input_buffer)? {
                jxl::api::ProcessingResult::Complete { result } => break Ok(result),
                jxl::api::ProcessingResult::NeedsMoreInput { fallback, .. } => {
                    if input_buffer.is_empty() {
                        break Err(eyre!("Source file {:?} truncated", cli_opt.input));
                    }
                    decoder_with_image_info = fallback;
                }
            }
        }?;

        let frame_header = decoder_with_frame_info.frame_header();

        let mut outputs = vec![Image::<f32>::new((
            image_data.size.0 * samples_per_pixel,
            image_data.size.1,
        ))?];

        for _ in 0..extra_channels {
            outputs.push(Image::<f32>::new(image_data.size)?);
        }

        let mut output_bufs: Vec<JxlOutputBuffer<'_>> = outputs
            .iter_mut()
            .map(|x| {
                let rect = Rect {
                    size: x.size(),
                    origin: (0, 0),
                };
                JxlOutputBuffer::from_image_rect_mut(x.get_rect_mut(rect).into_raw())
            })
            .collect();

        decoder_with_image_info = loop {
            match decoder_with_frame_info.process(&mut input_buffer, &mut output_bufs)? {
                jxl::api::ProcessingResult::Complete { result } => break Ok(result),
                jxl::api::ProcessingResult::NeedsMoreInput { fallback, .. } => {
                    if input_buffer.is_empty() {
                        break Err(eyre!("Source file {:?} truncated", cli_opt.input));
                    }
                    decoder_with_frame_info = fallback;
                }
            }
        }?;

        image_data.frames.push(ImageFrame {
            duration: frame_header.duration.unwrap_or(0.0),
            channels: outputs,
            color_type,
        });

        if !decoder_with_image_info.has_more_frames() {
            break;
        }
    }

    Ok((image_data, start.elapsed()))
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
    let options = || {
        let mut options = JxlDecoderOptions::default();
        options.xyb_output_linear = numpy_output || exr_output;
        options.render_spot_colors = !numpy_output;
        options
    };
    let mut input_bytes = Vec::<u8>::new();
    file.read_to_end(&mut input_bytes)?;
    let input_buffer = input_bytes.as_slice();

    let reps = opt.num_reps.unwrap_or(1);
    let mut duration_sum = Duration::new(0, 0);
    let mut image_data = (0..reps)
        .try_fold(None, |_, _| -> Result<Option<DecodeOutput<f32>>> {
            let (iteration_image_data, iteration_duration) =
                decode_bytes(input_buffer, options(), &opt)?;
            duration_sum += iteration_duration;
            Ok(Some(iteration_image_data))
        })?
        .unwrap();

    for frame in image_data.frames.iter_mut() {
        if frame.color_type != JxlColorType::Grayscale {
            let mut new_channels = planes_from_interleaved(&frame.channels[0])?;
            new_channels.extend(mem::take(&mut frame.channels).into_iter().skip(1));
            frame.channels = new_channels;
        }
    }

    if opt.speedtest {
        let num_pixels = image_data.size.0 * image_data.size.1;
        let duration_seconds = duration_sum.as_nanos() as f64 / 1e9;
        let avg_seconds = duration_seconds / reps as f64;
        println!(
            "Decoded {} pixels in {} seconds: {} pixels/s",
            reps as usize * num_pixels,
            duration_seconds,
            num_pixels as f64 / avg_seconds
        );
    }

    let image_result: Option<Result<()>> = opt.output.map(|path| {
        let output_bit_depth = match opt.override_bitdepth {
            None => image_data.original_bit_depth.bits_per_sample(),
            Some(num_bits) => num_bits,
        };
        let image_result = save_image(&image_data, output_bit_depth, &path);

        if let Err(err) = &image_data.original_icc_result {
            println!("Failed to save original ICC profile: {err}");
        }
        if let Err(err) = &image_data.data_icc_result {
            println!("Failed to save data ICC profile: {err}");
        }
        if let Err(ref err) = image_result {
            println!("Failed to save image: {err}");
        }
        image_result
    });

    image_data.original_icc_result?;
    image_data.data_icc_result?;

    image_result.unwrap_or(Ok(()))?;

    Ok(())
}
