//! Inspect a finite, uncompressed `jhgm` box and decode its embedded image.
//!
//! This example reads the complete input into memory. It deliberately reports
//! an EOF-sized gain-map box as unsupported because completing that box would
//! require joining the decoder's buffered prefix with the remaining input.

// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::error::Error;
use std::fs;
use std::path::Path;

use jxl::api::{
    Event, JxlAuxBoxType, JxlColorProfile, JxlColorType, JxlDataFormat, JxlDecoder,
    JxlDecoderOptions, JxlOutputBuffer, JxlPixelFormat,
};
use jxl_gainmap::JxlGainMap;

type ExampleResult<T> = Result<T, Box<dyn Error>>;

fn read_gain_map_box(data: &[u8]) -> ExampleResult<Vec<u8>> {
    let mut input = data;
    let mut options = JxlDecoderOptions::default();
    options.request_aux_boxes = vec![JxlAuxBoxType::HDR_GAIN_MAP];
    options.scan_frames_only = true;
    let mut decoder = JxlDecoder::new(options);
    decoder.close_input();
    loop {
        match decoder.process(&mut input, None, None)? {
            Event::Complete => break,
            Event::NeedMoreInput { size_hint } => {
                return Err(std::io::Error::other(format!(
                    "complete input unexpectedly needs {size_hint} more bytes while searching for a finite jhgm gain-map box"
                ))
                .into());
            }
            Event::BasicInfo | Event::FrameHeader | Event::FrameComplete { .. } => {}
        }
    }

    if decoder.trailing_box().is_some() {
        return Err(std::io::Error::other(
            "EOF-sized jhgm gain-map boxes are not supported by this example",
        )
        .into());
    }
    let box_data = decoder
        .aux_boxes(JxlAuxBoxType::HDR_GAIN_MAP)
        .first()
        .ok_or_else(|| std::io::Error::other("input contains no finite jhgm gain-map box"))?;
    if box_data.is_compressed() {
        return Err(std::io::Error::other(
            "Brotli-compressed jhgm data is not enabled in this example",
        )
        .into());
    }
    Ok(box_data.raw_data().to_vec())
}

fn decode_gain_map(data: &[u8]) -> ExampleResult<(usize, usize, usize, u64)> {
    let mut input = data;
    let mut decoder = JxlDecoder::new(JxlDecoderOptions::default());
    match decoder.process(&mut input, None, None)? {
        Event::BasicInfo => {}
        Event::NeedMoreInput { size_hint } => {
            return Err(std::io::Error::other(format!(
                "complete gain-map image unexpectedly needs {size_hint} more bytes"
            ))
            .into());
        }
        Event::FrameHeader | Event::FrameComplete { .. } | Event::Complete => {
            return Err(std::io::Error::other("gain-map image ended before BasicInfo").into());
        }
    }
    let basic_info = decoder
        .basic_info()
        .ok_or_else(|| std::io::Error::other("decoder emitted BasicInfo without image details"))?;
    if !basic_info.extra_channels.is_empty() {
        return Err(std::io::Error::other("gain-map image has unsupported extra channels").into());
    }
    let (width, height) = basic_info.size;
    let pixel_count = width
        .checked_mul(height)
        .and_then(|count| count.checked_mul(3))
        .ok_or_else(|| std::io::Error::other("gain-map image size overflow"))?;
    decoder.set_pixel_format(JxlPixelFormat {
        color_type: JxlColorType::Rgb,
        color_data_format: Some(JxlDataFormat::U8 { bit_depth: 8 }),
        extra_channel_format: Vec::new(),
    })?;
    let mut pixels = vec![0; pixel_count];
    {
        let mut output = JxlOutputBuffer::new(&mut pixels, height, width * 3);
        loop {
            match decoder.process(&mut input, Some(std::slice::from_mut(&mut output)), None)? {
                Event::FrameComplete { .. } => break,
                Event::NeedMoreInput { size_hint } => {
                    return Err(std::io::Error::other(format!(
                        "complete gain-map image unexpectedly needs {size_hint} more bytes"
                    ))
                    .into());
                }
                Event::FrameHeader | Event::BasicInfo => {}
                Event::Complete => {
                    return Err(
                        std::io::Error::other("gain-map image ended before its frame").into(),
                    );
                }
            }
        }
    }
    let checksum = pixels.iter().map(|&sample| u64::from(sample)).sum();
    Ok((width, height, pixels.len(), checksum))
}

fn run(path: &Path) -> ExampleResult<()> {
    let input = fs::read(path)?;
    let raw_bundle = read_gain_map_box(&input)?;
    let bundle = JxlGainMap::parse(&raw_bundle)?;
    let (color, decoded_icc_bytes) = match bundle.color_profile.as_ref() {
        None => ("baseline".to_owned(), 0),
        Some(JxlColorProfile::Icc(icc)) => ("ICC".to_owned(), icc.len()),
        Some(JxlColorProfile::Simple(encoding)) => (format!("structured {encoding:?}"), 0),
    };
    let (width, height, pixel_bytes, checksum) = decode_gain_map(bundle.gain_map)?;
    println!(
        "jhgm version={} metadata={} color={} decoded_icc={} gain_map={} bytes; nested image={}x{} RGB bytes={} checksum={}",
        bundle.version,
        bundle.metadata.len(),
        color,
        decoded_icc_bytes,
        bundle.gain_map.len(),
        width,
        height,
        pixel_bytes,
        checksum,
    );
    println!("gain-map application is not performed by this example");
    Ok(())
}

fn main() -> ExampleResult<()> {
    let path = std::env::args_os().nth(1).ok_or_else(|| {
        std::io::Error::other("usage: cargo run -p jxl-gainmap --example gain_map -- FILE.jxl")
    })?;
    run(Path::new(&path))
}
