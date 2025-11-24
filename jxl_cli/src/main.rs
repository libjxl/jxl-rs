// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use clap::Parser;
use color_eyre::eyre::{Result, WrapErr, eyre};
use jxl::api::{JxlColorType, JxlDecoderOptions};
use jxl::image::Image;
use jxl_cli::{dec, enc};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, Write};
use std::path::PathBuf;
use std::time::Duration;
use std::{fs, mem};

fn save_icc(icc_bytes: &[u8], icc_filename: Option<&PathBuf>) -> Result<()> {
    icc_filename.map_or(Ok(()), |path| {
        std::fs::write(path, icc_bytes)
            .wrap_err_with(|| format!("Failed to write ICC profile to {:?}", path))
    })
}

fn save_image(
    image_data: &dec::DecodeOutput<f32>,
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
    #[clap(required_unless_present_any = ["speedtest", "info"])]
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

    /// Extract the preview frame instead of the main image
    #[clap(long, action)]
    preview: bool,

    /// Print image information without decoding
    #[clap(long, short, action)]
    info: bool,
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
    let options = |skip_preview: bool| {
        let mut options = JxlDecoderOptions::default();
        options.xyb_output_linear = numpy_output || exr_output;
        options.render_spot_colors = !numpy_output;
        options.skip_preview = skip_preview;
        options
    };

    // Handle --info flag: print image info and exit
    if opt.info {
        let mut reader = BufReader::new(&mut file);
        let decoder = dec::decode_header(&mut reader, options(true))?;
        let info = decoder.basic_info();
        println!("Image size: {}x{}", info.size.0, info.size.1);
        println!("Bit depth: {:?}", info.bit_depth);
        println!("Orientation: {:?}", info.orientation);
        if let Some(preview_size) = info.preview_size {
            println!("Preview size: {}x{}", preview_size.0, preview_size.1);
        } else {
            println!("Preview: none");
        }
        if let Some(anim) = &info.animation {
            println!(
                "Animation: {} loops, {}/{} tps",
                anim.num_loops, anim.tps_numerator, anim.tps_denominator
            );
        }
        println!("Extra channels: {}", info.extra_channels.len());
        return Ok(());
    }

    // Handle --preview flag: check if preview exists
    if opt.preview {
        let mut reader = BufReader::new(&mut file);
        let decoder = dec::decode_header(&mut reader, options(true))?;
        let info = decoder.basic_info();
        if info.preview_size.is_none() {
            return Err(eyre!("This file does not contain a preview frame"));
        }
        // Seek back to start for actual decoding
        file.seek(std::io::SeekFrom::Start(0))?;
    }

    let reps = opt.num_reps.unwrap_or(1);
    let mut duration_sum = Duration::new(0, 0);
    // When extracting preview, don't skip it; otherwise skip preview by default
    let skip_preview = !opt.preview;

    let mut image_data = if reps > 1 {
        // For multiple repetitions (benchmarking), read into memory to avoid I/O variability
        let mut input_bytes = Vec::<u8>::new();
        file.read_to_end(&mut input_bytes)?;
        (0..reps)
            .try_fold(None, |_, _| -> Result<Option<dec::DecodeOutput<f32>>> {
                let mut input = input_bytes.as_slice();
                let (mut iteration_image_data, iteration_duration) =
                    dec::decode_frames(&mut input, options(skip_preview))?;
                duration_sum += iteration_duration;
                // When extracting preview, only keep the first frame (the preview)
                if opt.preview {
                    iteration_image_data.frames.truncate(1);
                    if let Some(frame) = iteration_image_data.frames.first() {
                        let samples = if frame.color_type == JxlColorType::Grayscale {
                            1
                        } else {
                            3
                        };
                        let (w, h) = frame.channels[0].size();
                        iteration_image_data.size = (w / samples, h);
                    }
                }
                Ok(Some(iteration_image_data))
            })?
            .unwrap()
    } else {
        // For single decode, stream from file
        let mut reader = BufReader::new(file);
        let (mut image_data, duration) = dec::decode_frames(&mut reader, options(skip_preview))?;
        duration_sum = duration;
        // When extracting preview, only keep the first frame (the preview)
        if opt.preview {
            image_data.frames.truncate(1);
            if let Some(frame) = image_data.frames.first() {
                let samples = if frame.color_type == JxlColorType::Grayscale {
                    1
                } else {
                    3
                };
                let (w, h) = frame.channels[0].size();
                image_data.size = (w / samples, h);
            }
        }
        image_data
    };

    let data_icc_result = save_icc(
        image_data.output_profile.as_icc().as_slice(),
        opt.icc_out.as_ref(),
    );
    let original_icc_result = save_icc(
        image_data.embedded_profile.as_icc().as_slice(),
        opt.original_icc_out.as_ref(),
    );

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

        if let Err(err) = &original_icc_result {
            println!("Failed to save original ICC profile: {err}");
        }
        if let Err(err) = &data_icc_result {
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
