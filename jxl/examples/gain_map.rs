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

use jxl::api::states::{Initialized, WithFrameInfo, WithImageInfo};
use jxl::api::{
    JxlAuxBoxType, JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlGainMapBundle,
    JxlGainMapColorEncoding, JxlOutputBuffer, JxlPixelFormat, ProcessingResult,
};

type ExampleResult<T> = Result<T, Box<dyn Error>>;

fn complete<T, U>(result: ProcessingResult<T, U>) -> ExampleResult<T> {
    match result {
        ProcessingResult::Complete { result } => Ok(result),
        ProcessingResult::NeedsMoreInput { size_hint, .. } => Err(std::io::Error::other(format!(
            "complete input unexpectedly needs {size_hint} more bytes"
        ))
        .into()),
    }
}

fn read_gain_map_box(data: &[u8]) -> ExampleResult<Vec<u8>> {
    let mut input = data;
    let mut options = JxlDecoderOptions::default();
    options.request_aux_boxes = vec![JxlAuxBoxType::GAIN_MAP];
    options.scan_frames_only = true;
    let decoder = JxlDecoder::<Initialized>::new(options);
    let mut image_info: JxlDecoder<WithImageInfo> = complete(decoder.process(&mut input, None)?)?;

    while image_info.has_more_frames() {
        let frame: JxlDecoder<WithFrameInfo> = complete(image_info.process(&mut input, None)?)?;
        image_info = complete(frame.skip_frame(&mut input)?)?;
    }

    let mut trailing = complete(image_info.process_trailing_data(&mut input)?)?;
    while trailing.aux_boxes(JxlAuxBoxType::GAIN_MAP).is_empty()
        && trailing.trailing_box().is_none()
        && !input.is_empty()
    {
        let input_length = input.len();
        complete(trailing.process_trailing_data(&mut input)?)?;
        if input.len() == input_length {
            return Err(std::io::Error::other(
                "trailing data made no progress while searching for a finite jhgm gain-map box",
            )
            .into());
        }
    }

    if trailing.trailing_box().is_some() {
        return Err(std::io::Error::other(
            "EOF-sized jhgm gain-map boxes are not supported by this example",
        )
        .into());
    }
    let box_data = trailing
        .aux_boxes(JxlAuxBoxType::GAIN_MAP)
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
    let decoder = JxlDecoder::<Initialized>::new(JxlDecoderOptions::default());
    let mut image_info: JxlDecoder<WithImageInfo> = complete(decoder.process(&mut input, None)?)?;
    let basic_info = image_info.basic_info().clone();
    if !basic_info.extra_channels.is_empty() {
        return Err(std::io::Error::other("gain-map image has unsupported extra channels").into());
    }
    let (width, height) = basic_info.size;
    let pixel_count = width
        .checked_mul(height)
        .and_then(|count| count.checked_mul(3))
        .ok_or_else(|| std::io::Error::other("gain-map image size overflow"))?;
    image_info.set_pixel_format(JxlPixelFormat {
        color_type: JxlColorType::Rgb,
        color_data_format: Some(JxlDataFormat::U8 { bit_depth: 8 }),
        extra_channel_format: Vec::new(),
    })?;
    let frame: JxlDecoder<WithFrameInfo> = complete(image_info.process(&mut input, None)?)?;
    let mut pixels = vec![0; pixel_count];
    {
        let mut output = JxlOutputBuffer::new(&mut pixels, height, width * 3);
        complete(frame.process(&mut input, std::slice::from_mut(&mut output), None)?)?;
    }
    let checksum = pixels.iter().map(|&sample| u64::from(sample)).sum();
    Ok((width, height, pixels.len(), checksum))
}

fn run(path: &Path) -> ExampleResult<()> {
    let input = fs::read(path)?;
    let raw_bundle = read_gain_map_box(&input)?;
    let bundle = JxlGainMapBundle::parse(&raw_bundle)?;
    let color = match bundle.decode_color_encoding()? {
        None => "absent".to_owned(),
        Some(JxlGainMapColorEncoding::IccRequired) => "ICC required".to_owned(),
        Some(JxlGainMapColorEncoding::Structured(encoding)) => format!("structured {encoding:?}"),
    };
    let alternate_icc_bytes = bundle.decode_alternate_icc()?.map_or(0, |icc| icc.len());
    let (width, height, pixel_bytes, checksum) = decode_gain_map(bundle.gain_map)?;
    println!(
        "jhgm version={} metadata={} color={} alternate_icc={} gain_map={} bytes; nested image={}x{} RGB bytes={} checksum={}",
        bundle.version,
        bundle.metadata.len(),
        color,
        alternate_icc_bytes,
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
        std::io::Error::other("usage: cargo run -p jxl --example gain_map -- FILE.jxl")
    })?;
    run(Path::new(&path))
}
