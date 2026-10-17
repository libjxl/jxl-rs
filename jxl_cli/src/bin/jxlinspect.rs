// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use clap::{Arg, ArgAction, Command};
use color_eyre::eyre::{Result, eyre};
use jxl::api::{
    Event, ExtraChannel, JxlBitDepth, JxlColorEncoding, JxlColorProfile, JxlDecoder,
    JxlDecoderOptions, TocEntry, TocGroupKind,
};

fn parse_jxl(path: &Path) -> Result<()> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let mut options = JxlDecoderOptions::default();
    options.scan_frames_only = true;
    let mut decoder = JxlDecoder::new(options);

    if decoder.process(&mut reader, None, None)? != Event::BasicInfo {
        return Err(eyre!("Source file {:?} truncated", path));
    }

    let info = decoder.basic_info().unwrap().clone();

    let how_lossy = if info.uses_original_profile {
        "(possibly) lossless"
    } else {
        "lossy"
    };
    let color_space = format!("{}", decoder.embedded_color_profile().unwrap());
    let alpha_info = if info
        .extra_channels
        .iter()
        .any(|c| c.ec_type == ExtraChannel::Alpha)
    {
        "+Alpha"
    } else {
        ""
    };
    let image_or_animation = if info.animation.is_some() {
        "Animation"
    } else {
        "Image"
    };
    print!(
        "JPEG XL {}, {}x{}, {}, {}-bit, {}{}",
        image_or_animation,
        info.size.0,
        info.size.1,
        how_lossy,
        info.bit_depth.bits_per_sample(),
        color_space,
        alpha_info,
    );
    if let JxlBitDepth::Float {
        bits_per_sample: _,
        exponent_bits_per_sample: ebps,
    } = info.bit_depth
    {
        print!(", float ({} exponent bits)", ebps);
    }
    println!();
    match decoder.output_color_profile().unwrap() {
        JxlColorProfile::Icc(icc) => match moxcms::ColorProfile::new_from_slice(icc.as_slice()) {
            Err(_) => println!("with unparseable ICC profile"),
            Ok(profile) => {
                let description = match &profile.description {
                    Some(moxcms::ProfileText::PlainString(text)) => Some(text.as_str()),
                    Some(moxcms::ProfileText::Description(text)) => {
                        Some(if text.unicode_string.is_empty() {
                            text.ascii_string.as_str()
                        } else {
                            text.unicode_string.as_str()
                        })
                    }
                    Some(moxcms::ProfileText::Localizable(texts)) => {
                        texts.first().map(|text| text.value.as_str())
                    }
                    None => None,
                };
                match description {
                    None | Some("") => println!("with undescribed {}-byte ICC profile", icc.len()),
                    Some(description) => {
                        println!(
                            "with {}-byte ICC profile (description: {})",
                            icc.len(),
                            description
                        )
                    }
                }
            }
        },
        JxlColorProfile::Simple(color_encoding) => match color_encoding {
            JxlColorEncoding::GrayscaleColorSpace {
                white_point,
                transfer_function,
                rendering_intent,
            } => {
                println!(
                    "White point: {}, Transfer function: {}, Rendering intent: {}",
                    white_point, transfer_function, rendering_intent
                );
            }
            JxlColorEncoding::RgbColorSpace {
                white_point,
                primaries,
                transfer_function,
                rendering_intent,
            } => {
                println!(
                    "White point: {}, Primaries: {}, Transfer function: {}, Rendering intent: {}",
                    white_point, primaries, transfer_function, rendering_intent
                );
            }
            JxlColorEncoding::XYB { rendering_intent } => {
                println!("Rendering intent: {}", rendering_intent);
            }
        },
    }

    if let Some(animation) = info.animation {
        let mut num_frames = 0;
        let mut total_seconds = 0.0;

        loop {
            match decoder.process(&mut reader, None, None)? {
                Event::FrameHeader => {
                    let duration = decoder.frame_header().unwrap().duration.unwrap();
                    total_seconds += duration;
                    println!("Frame {}, duration {}ms", num_frames, duration);
                    num_frames += 1;
                }
                Event::FrameComplete { .. } => {}
                Event::Complete => break,
                _ => {
                    return Err(eyre!("Source file {:?} truncated", path));
                }
            }
        }

        print!(
            "Animation length: {} frames in {} seconds",
            num_frames,
            total_seconds / 1000.0
        );
        if animation.have_timecodes {
            print!(" with (potentially) individual timecodes");
        }
        if animation.num_loops < 1 {
            println!(" (looping indefinitely)");
        } else if animation.num_loops > 1 {
            println!(
                " ({} loops, in total {} seconds)",
                animation.num_loops,
                animation.num_loops as f64 * total_seconds
            );
        }
    }
    Ok(())
}

struct FrameToc {
    data_start: u64,
    entries: Vec<TocEntry>,
}

/// File offset one past the last byte of `[start, start + size)`, in codestream
/// bytes. The end position itself may lie in a box that doesn't exist (or
/// before a box header), so map the last byte instead.
fn file_end(decoder: &JxlDecoder, start: u64, size: u64) -> Result<u64> {
    let last = (start + size)
        .checked_sub(1)
        .ok_or_else(|| eyre!("empty range at the codestream start"))?;
    decoder
        .codestream_offset_to_file_offset(last)
        .map(|o| o + 1)
        .ok_or_else(|| eyre!("codestream offset {last} is not in any parsed box"))
}

/// Prints, per frame, the file offsets of the section data and of each TOC
/// entry, plus where the LF and each HF pass are complete: the byte count
/// after which a progressive reader has that much detail.
fn print_toc(path: &Path) -> Result<()> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut decoder = JxlDecoder::new(JxlDecoderOptions::default());

    // Frames are only inspectable between FrameHeader and FrameComplete, but
    // offsets are resolved at the end, once every box has been parsed (the
    // codestream may be split over several jxlp boxes). Entries are kept in
    // codestream coordinates until then.
    let mut frames = Vec::new();
    loop {
        match decoder.process(&mut reader, None, None)? {
            Event::FrameHeader => {
                frames.push(FrameToc {
                    data_start: decoder.frame_data_codestream_offset(),
                    entries: (0..decoder.toc_num_entries())
                        .map(|i| decoder.toc_entry(i).unwrap())
                        .collect(),
                });
            }
            Event::Complete => break,
            Event::NeedMoreInput { .. } => return Err(eyre!("Source file {:?} truncated", path)),
            Event::BasicInfo | Event::FrameComplete { .. } => {}
        }
    }

    for (idx, frame) in frames.iter().enumerate() {
        let to_file = |pos: u64| {
            decoder
                .codestream_offset_to_file_offset(pos)
                .ok_or_else(|| eyre!("codestream offset {pos} is not in any parsed box"))
        };
        println!(
            "Frame {idx}: section data starts at byte {}",
            to_file(frame.data_start)?
        );

        let mut lf_global_end = None;
        let mut lf_end = None;
        let mut hf_global_end = None;
        let mut pass_ends: Vec<Option<u64>> = Vec::new();
        let max = |cur: &mut Option<u64>, end: u64| *cur = Some(cur.map_or(end, |c| c.max(end)));
        for e in &frame.entries {
            let start = frame.data_start + e.offset;
            let end = file_end(&decoder, start, e.size as u64)?;
            let kind = match e.kind {
                TocGroupKind::All => "All".to_string(),
                TocGroupKind::LfGlobal => "LfGlobal".to_string(),
                TocGroupKind::LfGroup(g) => format!("LfGroup {g}"),
                TocGroupKind::HfGlobal => "HfGlobal".to_string(),
                TocGroupKind::GroupPass {
                    pass_idx,
                    group_idx,
                } => format!("Pass {pass_idx} group {group_idx}"),
            };
            println!(
                "  {kind}: bytes {} to {end} ({} B)",
                end - e.size as u64,
                e.size
            );
            match e.kind {
                TocGroupKind::All | TocGroupKind::LfGlobal => max(&mut lf_global_end, end),
                TocGroupKind::LfGroup(_) => max(&mut lf_end, end),
                TocGroupKind::HfGlobal => max(&mut hf_global_end, end),
                TocGroupKind::GroupPass { pass_idx, .. } => {
                    let pass_idx = pass_idx as usize;
                    if pass_ends.len() <= pass_idx {
                        pass_ends.resize(pass_idx + 1, None);
                    }
                    max(&mut pass_ends[pass_idx], end);
                }
            }
        }

        let print_end = |label: &str, end: Option<u64>| {
            if let Some(end) = end {
                println!("  {label}: ends at byte {end}");
            }
        };
        print_end("LfGlobal", lf_global_end);
        print_end("LF groups", lf_end);
        print_end("HfGlobal", hf_global_end);
        for (pass, end) in pass_ends.into_iter().enumerate() {
            print_end(&format!("HF pass {pass}"), end);
        }
    }
    Ok(())
}

fn main() {
    #[cfg(feature = "tracing-subscriber")]
    {
        use tracing_subscriber::prelude::*;
        use tracing_subscriber::{EnvFilter, fmt};
        tracing_subscriber::registry()
            .with(fmt::layer())
            .with(EnvFilter::from_default_env())
            .init();
    }

    let matches = Command::new("jxlinspect")
        .about("Provides info about a JXL file")
        .arg(
            Arg::new("filename")
                .help("The JXL file to analyze")
                .required(true)
                .index(1),
        )
        .arg(
            Arg::new("toc")
                .long("toc")
                .help("Also print the file offset of each frame section, and where the LF and each HF pass end")
                .action(ArgAction::SetTrue),
        )
        .get_matches();

    let filename = Path::new(matches.get_one::<String>("filename").unwrap());

    let mut res = parse_jxl(filename);
    if res.is_ok() && matches.get_flag("toc") {
        res = print_toc(filename);
    }
    if let Err(err) = res {
        println!("Error parsing JXL codestream: {err}");
    }
}
