// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use clap::Parser;
use color_eyre::eyre::{Result, WrapErr, eyre};
use jxl::api::{JxlDecoderOptions, JxlMetadataBox, JxlMetadataCaptureOptions, ProcessingResult};
use jxl_cli::dec::OutputDataType;
use jxl_cli::enc::OutputFormat;
use jxl_cli::metadata::{compress_metadata_boxes, print_metadata_info, save_metadata_boxes};
use jxl_cli::{cms::Lcms2Cms, dec};
use std::fs;
use std::io::{BufReader, Read, Seek};
use std::path::PathBuf;
use std::time::Duration;

const VERSION_STRING: &str = concat!(
    env!("VERGEN_GIT_DESCRIBE"),
    " (rustc ",
    env!("VERGEN_RUSTC_SEMVER"),
    ")"
);

#[derive(Parser)]
#[command(version = VERSION_STRING)]
struct Opt {
    /// Input JXL file
    input: PathBuf,

    /// Output image file, should end in .ppm, .pgm, .png, .apng or .npy
    /// (optional with --speedtest or --info)
    #[clap(required_unless_present_any = ["speedtest", "info", "exif_out", "xmp_out", "jumbf_out", "metadata_out", "compress_metadata"])]
    output: Option<PathBuf>,

    /// Print measured decoding speed.
    #[clap(long, short, action)]
    speedtest: bool,

    /// Number of times to repeat the decoding (only valid with --speedtest).
    #[clap(long, short, default_value_t = 1, requires = "speedtest")]
    num_reps: usize,

    /// Number of warmup decodes before measuring (only valid with --speedtest).
    #[clap(long, default_value_t = 1, requires = "speedtest")]
    warmup_reps: usize,

    ///  If specified, writes the ICC profile of the decoded image
    #[clap(long)]
    icc_out: Option<PathBuf>,

    ///  Likewise but for the ICC profile of the original colorspace
    #[clap(long)]
    original_icc_out: Option<PathBuf>,

    /// If specified, writes EXIF metadata boxes to files based on this path
    #[clap(long)]
    exif_out: Option<PathBuf>,

    /// If specified, writes XMP metadata boxes to files based on this path
    #[clap(long)]
    xmp_out: Option<PathBuf>,

    /// If specified, writes JUMBF metadata boxes to files based on this path
    #[clap(long)]
    jumbf_out: Option<PathBuf>,

    /// If specified, writes all metadata boxes to the given directory
    #[clap(long)]
    metadata_out: Option<PathBuf>,

    /// If specified, takes precedence over the bit depth in the input metadata
    #[clap(long)]
    override_bitdepth: Option<usize>,

    /// Extract the preview frame instead of the main image
    #[clap(long, action)]
    preview: bool,

    /// Print image information without decoding
    #[clap(long, short, action)]
    info: bool,

    /// Use high precision mode for decoding
    #[clap(long)]
    high_precision: bool,

    /// Output data type for decoder (u8, u16, f16, f32). Used for benchmarking
    /// the decoder's conversion pipeline. Default: pick based on bit depth
    /// and output format.
    #[clap(long)]
    data_type: Option<OutputDataType>,

    /// Allow partial files (flush pixels on EOF)
    #[clap(long)]
    allow_partial_files: bool,

    /// Force a partial render every `render_interval` bytes.
    #[clap(long)]
    render_interval: Option<usize>,

    /// Brotli-compress metadata boxes (Exif, XMP, JUMBF) and write to output path
    #[clap(long)]
    compress_metadata: Option<PathBuf>,
}

fn save_icc(icc_bytes: &[u8], icc_filename: Option<&PathBuf>) -> Result<()> {
    icc_filename.map_or(Ok(()), |path| {
        std::fs::write(path, icc_bytes)
            .wrap_err_with(|| format!("Failed to write ICC profile to {:?}", path))
    })
}

/// Save all metadata boxes according to CLI options.
fn save_all_metadata(
    opt: &Opt,
    exif: &Option<Vec<JxlMetadataBox>>,
    xmp: &Option<Vec<JxlMetadataBox>>,
    jumbf: &Option<Vec<JxlMetadataBox>>,
) -> Result<()> {
    save_metadata_boxes(opt.exif_out.as_ref(), exif, true)?;
    save_metadata_boxes(opt.xmp_out.as_ref(), xmp, false)?;
    save_metadata_boxes(opt.jumbf_out.as_ref(), jumbf, false)?;

    if let Some(dir) = &opt.metadata_out {
        fs::create_dir_all(dir)
            .wrap_err_with(|| format!("Failed to create metadata output directory {:?}", dir))?;
        save_metadata_boxes(Some(&dir.join("metadata_exif.exif")), exif, true)?;
        save_metadata_boxes(Some(&dir.join("metadata_xmp.xmp")), xmp, false)?;
        save_metadata_boxes(Some(&dir.join("metadata_jumbf.bin")), jumbf, false)?;
    }
    Ok(())
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

    // Handle --compress-metadata mode (no decoding needed)
    if let Some(output_path) = &opt.compress_metadata {
        return compress_metadata_boxes(&opt.input, output_path);
    }

    let mut file = fs::File::open(opt.input.clone())
        .wrap_err_with(|| format!("Failed to read source image from {:?}", opt.input))?;

    let output_format = opt
        .output
        .as_ref()
        .map(|f| OutputFormat::from_output_filename(&f.to_string_lossy()))
        .transpose()?;

    let high_precision = opt.high_precision;
    let wants_metadata = opt.metadata_out.is_some()
        || opt.exif_out.is_some()
        || opt.xmp_out.is_some()
        || opt.jumbf_out.is_some();

    let metadata_capture = JxlMetadataCaptureOptions {
        capture_exif: wants_metadata || opt.exif_out.is_some(),
        capture_xmp: wants_metadata || opt.xmp_out.is_some(),
        capture_jumbf: wants_metadata || opt.jumbf_out.is_some(),
        ..JxlMetadataCaptureOptions::capture_all()
    };
    let options = |skip_preview: bool| {
        let mut options = JxlDecoderOptions::default();
        options.render_spot_colors = !matches!(output_format, Some(OutputFormat::Npy));
        options.skip_preview = skip_preview;
        options.high_precision = high_precision;
        options.cms = Some(Box::new(Lcms2Cms));
        options.metadata_capture = metadata_capture.clone();
        options
    };

    // Handle --info flag: print image info and exit (unless metadata extraction is also requested)
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
        if !wants_metadata {
            return Ok(());
        }
        // Seek back to start so the metadata-only path can re-read the file
        file.seek(std::io::SeekFrom::Start(0))?;
    }

    // Fast path: metadata extraction without pixel decoding
    if opt.output.is_none() && !opt.speedtest && wants_metadata {
        let mut reader = BufReader::new(file);
        let mut decoder = dec::decode_header(&mut reader, options(true))?;

        // Skip through all frames to consume the codestream and discover
        // any trailing metadata boxes (e.g. EXIF/XMP after codestream data)
        loop {
            let frame_decoder = match decoder.process(&mut reader)? {
                ProcessingResult::Complete { result } => result,
                ProcessingResult::NeedsMoreInput { .. } => {
                    return Err(eyre!("Source file truncated"));
                }
            };
            decoder = match frame_decoder.skip_frame(&mut reader)? {
                ProcessingResult::Complete { result } => result,
                ProcessingResult::NeedsMoreInput { .. } => {
                    return Err(eyre!("Source file truncated"));
                }
            };
            if !decoder.has_more_frames() {
                break;
            }
        }
        let exif = decoder.exif_boxes().map(|b| b.to_vec());
        let xmp = decoder.xmp_boxes().map(|b| b.to_vec());
        let jumbf = decoder.jumbf_boxes().map(|b| b.to_vec());

        if opt.info {
            if metadata_capture.capture_exif {
                print_metadata_info("EXIF", &exif);
            }
            if metadata_capture.capture_xmp {
                print_metadata_info("XMP", &xmp);
            }
            if metadata_capture.capture_jumbf {
                print_metadata_info("JUMBF", &jumbf);
            }
        }

        save_all_metadata(&opt, &exif, &xmp, &jumbf)?;

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

    let mut duration_sum = Duration::new(0, 0);
    // When extracting preview, don't skip it; otherwise skip preview by default
    let skip_preview = !opt.preview;

    macro_rules! run_decoder {
        ($input: expr) => {{
            #[cfg(feature = "exr")]
            let linear_output = matches!(output_format, Some(OutputFormat::Exr));
            #[cfg(not(feature = "exr"))]
            let linear_output = false;
            let (mut output, duration) = dec::decode_frames(
                $input,
                options(skip_preview),
                opt.override_bitdepth,
                opt.data_type,
                output_format
                    .map(|x| x.supported_output_data_types())
                    .unwrap_or(OutputDataType::ALL),
                output_format.is_none_or(|x| x.should_fold_alpha()),
                linear_output,
                opt.render_interval,
                opt.allow_partial_files,
            )?;
            if opt.preview {
                output.frames.truncate(1);
                let ctype = output.frames[0].color_type;
                let bsize = output.frames[0].channels[0].byte_size();
                let bytes_per_pixel =
                    ctype.samples_per_pixel() * output.data_type.bits_per_sample() / 8;
                output.size = (bsize.0 / bytes_per_pixel, bsize.1);
            }
            (output, duration)
        }};
    }

    // For benchmarking, always read into memory to avoid I/O variability
    let output = if opt.speedtest {
        let mut input_bytes = Vec::<u8>::new();
        file.read_to_end(&mut input_bytes)?;

        for _ in 0..opt.warmup_reps {
            run_decoder!(&mut input_bytes.as_slice());
        }

        let mut last_output = None;

        for _ in 0..opt.num_reps {
            let (output, duration) = run_decoder!(&mut input_bytes.as_slice());
            duration_sum += duration;
            last_output = Some(output);
        }
        last_output.unwrap()
    } else if opt.render_interval.is_some() {
        let mut input_bytes = Vec::<u8>::new();
        file.read_to_end(&mut input_bytes)?;
        run_decoder!(&mut input_bytes.as_slice()).0
    } else {
        // For single decode without speedtest, stream from file
        run_decoder!(&mut BufReader::new(file)).0
    };

    // Get metadata from typed output before converting
    let output_icc = output.output_profile.as_icc().to_vec();
    let embedded_icc = output.embedded_profile.as_icc().to_vec();
    let image_size = output.size;

    if opt.speedtest {
        let num_pixels = image_size.0 * image_size.1;
        let duration_seconds = duration_sum.as_secs_f64();
        let avg_seconds = duration_seconds / opt.num_reps as f64;
        println!(
            "Decoded {} pixels in {:.3} seconds: {:.3} MP/s",
            opt.num_reps * num_pixels,
            duration_seconds,
            (num_pixels as f64 / avg_seconds) / 1e6
        );
    }

    if let Some(output_format) = output_format {
        output_format.save_image(&output, opt.output.as_ref().unwrap())?;
    }

    save_icc(&output_icc, opt.icc_out.as_ref())?;
    save_icc(&embedded_icc, opt.original_icc_out.as_ref())?;

    save_all_metadata(
        &opt,
        &output.exif_boxes,
        &output.xmp_boxes,
        &output.jumbf_boxes,
    )?;

    Ok(())
}
