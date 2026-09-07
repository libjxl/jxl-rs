// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use jxl::api::states::{Initialized, WithFrameInfo, WithImageInfo};
use jxl::api::{
    JxlAuxBoxType, JxlColorEncoding, JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions,
    JxlGainMapBundle, JxlGainMapColorEncoding, JxlOutputBuffer, JxlPixelFormat, ProcessingResult,
};
use jxl::error::Error;
use jxl::headers::color_encoding::RenderingIntent;

const EXPECTED_PIXELS: [u8; 12] = [0, 64, 128, 255, 254, 253, 32, 96, 160, 200, 100, 50];

const METADATA: &[u8] = include_bytes!("testdata/gain_map/combined.jhgm");
const ICC: &[u8] = include_bytes!("testdata/gain_map/srgb.icc");
const NAKED_JXL: &[u8] = include_bytes!("testdata/gain_map/synthetic.jxl");

fn bundle(name: &str) -> JxlGainMapBundle<'static> {
    let data = match name {
        "structured" => include_bytes!("testdata/gain_map/structured.jhgm") as &[u8],
        "icc" => include_bytes!("testdata/gain_map/icc.jhgm") as &[u8],
        "combined" => include_bytes!("testdata/gain_map/combined.jhgm") as &[u8],
        "embedded-container" => {
            include_bytes!("testdata/gain_map/embedded-container.jhgm") as &[u8]
        }
        "want-icc" => include_bytes!("testdata/gain_map/want_icc.jhgm") as &[u8],
        _ => panic!("unknown fixture {name}"),
    };
    JxlGainMapBundle::parse(data).unwrap()
}

fn raw_bundle(metadata: &[u8], color: &[u8], icc: &[u8], gain_map: &[u8]) -> Vec<u8> {
    let mut data = Vec::with_capacity(
        1 + 2 + metadata.len() + 1 + color.len() + 4 + icc.len() + gain_map.len(),
    );
    data.push(0);
    data.extend_from_slice(&(metadata.len() as u16).to_be_bytes());
    data.extend_from_slice(metadata);
    data.push(color.len() as u8);
    data.extend_from_slice(color);
    data.extend_from_slice(&(icc.len() as u32).to_be_bytes());
    data.extend_from_slice(icc);
    data.extend_from_slice(gain_map);
    data
}

fn complete<T, U>(result: ProcessingResult<T, U>) -> T {
    match result {
        ProcessingResult::Complete { result } => result,
        ProcessingResult::NeedsMoreInput { size_hint, .. } => {
            panic!("complete fixture unexpectedly needs {size_hint} more bytes")
        }
    }
}

fn decode_pixels(data: &[u8]) -> Vec<u8> {
    let mut input = data;
    let decoder = JxlDecoder::<Initialized>::new(JxlDecoderOptions::default());
    let mut image_info: JxlDecoder<WithImageInfo> =
        complete(decoder.process(&mut input, None).unwrap());
    assert_eq!(image_info.basic_info().size, (2, 2));
    assert!(image_info.basic_info().extra_channels.is_empty());
    image_info
        .set_pixel_format(JxlPixelFormat {
            color_type: JxlColorType::Rgb,
            color_data_format: Some(JxlDataFormat::U8 { bit_depth: 8 }),
            extra_channel_format: Vec::new(),
        })
        .unwrap();
    let frame: JxlDecoder<WithFrameInfo> = complete(image_info.process(&mut input, None).unwrap());
    let mut pixels = vec![0; EXPECTED_PIXELS.len()];
    {
        let mut output = JxlOutputBuffer::new(&mut pixels, 2, 2 * 3);
        let _image_info: JxlDecoder<WithImageInfo> = complete(
            frame
                .process(&mut input, std::slice::from_mut(&mut output), None)
                .unwrap(),
        );
    }
    pixels
}

fn capture_gain_map(data: &[u8]) -> Vec<u8> {
    let mut input = data;
    let mut options = JxlDecoderOptions::default();
    options.request_aux_boxes = vec![JxlAuxBoxType::GAIN_MAP];
    options.scan_frames_only = true;
    let decoder = JxlDecoder::<Initialized>::new(options);
    let mut image_info: JxlDecoder<WithImageInfo> =
        complete(decoder.process(&mut input, None).unwrap());
    while image_info.has_more_frames() {
        let frame: JxlDecoder<WithFrameInfo> =
            complete(image_info.process(&mut input, None).unwrap());
        image_info = complete(frame.skip_frame(&mut input).unwrap());
    }
    let mut trailing = complete(image_info.process_trailing_data(&mut input).unwrap());
    while trailing.aux_boxes(JxlAuxBoxType::GAIN_MAP).is_empty() && !input.is_empty() {
        complete(trailing.process_trailing_data(&mut input).unwrap());
    }
    let box_data = trailing
        .aux_boxes(JxlAuxBoxType::GAIN_MAP)
        .first()
        .expect("finite jhgm box was not captured");
    assert!(!box_data.is_compressed());
    box_data.raw_data().to_vec()
}

#[test]
fn parses_independent_fields_and_color_variants() {
    let structured = bundle("structured");
    assert_eq!(structured.version, 0);
    assert_eq!(structured.metadata, &METADATA[3..144]);
    assert_eq!(structured.color_encoding, Some(&[0x50, 0xb4, 0][..]));
    assert!(structured.compressed_icc.is_empty());
    assert_eq!(structured.gain_map, NAKED_JXL);
    assert_eq!(
        structured.decode_color_encoding().unwrap(),
        Some(JxlGainMapColorEncoding::Structured(
            JxlColorEncoding::RgbColorSpace {
                white_point: jxl::api::JxlWhitePoint::D65,
                primaries: jxl::api::JxlPrimaries::SRGB,
                transfer_function: jxl::api::JxlTransferFunction::Linear,
                rendering_intent: RenderingIntent::Relative,
            },
        ))
    );
    assert_eq!(structured.decode_alternate_icc().unwrap(), None);

    let icc = bundle("icc");
    assert!(icc.color_encoding.is_none());
    assert_eq!(icc.decode_color_encoding().unwrap(), None);
    assert_eq!(icc.decode_alternate_icc().unwrap().as_deref(), Some(ICC));

    let combined = bundle("combined");
    assert!(combined.color_encoding.is_some());
    assert_eq!(
        combined.decode_color_encoding().unwrap(),
        structured.decode_color_encoding().unwrap()
    );
    assert_eq!(
        combined.decode_alternate_icc().unwrap().as_deref(),
        Some(ICC)
    );

    let want_icc = bundle("want-icc");
    assert_eq!(want_icc.color_encoding, Some(&[0x02][..]));
    assert_eq!(
        want_icc.decode_color_encoding().unwrap(),
        Some(JxlGainMapColorEncoding::IccRequired)
    );
    assert_eq!(
        want_icc.decode_alternate_icc().unwrap().as_deref(),
        Some(ICC)
    );
}

#[test]
fn want_icc_ignores_inapplicable_unknown_and_xyb_color_values() {
    let valid_icc = bundle("icc");
    for color_byte in [0x02, 0x06, 0x0a, 0x1a] {
        let data = raw_bundle(
            &[],
            &[color_byte],
            valid_icc.compressed_icc,
            valid_icc.gain_map,
        );
        let parsed = JxlGainMapBundle::parse(&data).unwrap();
        assert_eq!(parsed.color_encoding, Some(&[color_byte][..]));
        assert_eq!(
            parsed.decode_color_encoding().unwrap(),
            Some(JxlGainMapColorEncoding::IccRequired)
        );
        assert_eq!(parsed.decode_alternate_icc().unwrap().as_deref(), Some(ICC));
    }

    for color_byte in [0x2a, 0x0e] {
        let data = raw_bundle(
            &[],
            &[color_byte],
            valid_icc.compressed_icc,
            valid_icc.gain_map,
        );
        let parsed = JxlGainMapBundle::parse(&data).unwrap();
        assert!(parsed.decode_color_encoding().is_err());
        assert_eq!(parsed.decode_alternate_icc().unwrap().as_deref(), Some(ICC));
    }

    for color_byte in [0x08, 0x18] {
        let data = raw_bundle(
            &[],
            &[color_byte],
            valid_icc.compressed_icc,
            valid_icc.gain_map,
        );
        let parsed = JxlGainMapBundle::parse(&data).unwrap();
        assert!(parsed.decode_color_encoding().is_err());
        assert_eq!(parsed.decode_alternate_icc().unwrap().as_deref(), Some(ICC));
    }
}

#[test]
fn rejects_truncated_and_invalid_fields_without_losing_raw_slices() {
    for data in [
        vec![],
        vec![0],
        vec![0, 0],
        vec![0, 0, 1],
        vec![0, 0, 0],
        vec![0, 0, 0, 1],
        vec![0, 0, 0, 0, 0, 0, 0],
        vec![0, 0, 0, 0, 0, 0, 0, 1],
    ] {
        assert!(matches!(
            JxlGainMapBundle::parse(&data),
            Err(Error::SectionTooShort)
        ));
    }

    let invalid_color = raw_bundle(&[], &[0x50], &[], &[0xb4, 0x00]);
    let parsed_color = JxlGainMapBundle::parse(&invalid_color).unwrap();
    assert_eq!(parsed_color.color_encoding, Some(&[0x50][..]));
    assert_eq!(parsed_color.gain_map, &[0xb4, 0x00]);
    assert!(parsed_color.decode_color_encoding().is_err());

    let valid_icc = bundle("icc");
    let invalid_icc = raw_bundle(
        &[],
        &[],
        &valid_icc.compressed_icc[..1],
        &valid_icc.compressed_icc[1..],
    );
    let parsed_icc = JxlGainMapBundle::parse(&invalid_icc).unwrap();
    assert_eq!(parsed_icc.compressed_icc, &valid_icc.compressed_icc[..1]);
    assert_eq!(parsed_icc.gain_map, &valid_icc.compressed_icc[1..]);
    assert!(parsed_icc.decode_alternate_icc().is_err());

    let mut unknown_version = include_bytes!("testdata/gain_map/structured.jhgm").to_vec();
    unknown_version[0] = 0xff;
    let parsed_unknown = JxlGainMapBundle::parse(&unknown_version).unwrap();
    assert_eq!(parsed_unknown.version, 0xff);
    assert!(parsed_unknown.decode_color_encoding().is_ok());

    for color in [
        [0x01].as_slice(),
        [0x81].as_slice(),
        [0x01, 0xa5].as_slice(),
    ] {
        let data = raw_bundle(&[], color, &[], &[]);
        let parsed = JxlGainMapBundle::parse(&data).unwrap();
        assert!(parsed.decode_color_encoding().is_ok());
    }
}

#[test]
fn captures_finite_aux_box_and_decodes_bare_and_container_gain_maps() {
    let outer = include_bytes!("testdata/gain_map/synthetic-container.jxl");
    let captured = capture_gain_map(outer);
    assert_eq!(captured, include_bytes!("testdata/gain_map/combined.jhgm"));
    let captured_bundle = JxlGainMapBundle::parse(&captured).unwrap();
    assert_eq!(decode_pixels(captured_bundle.gain_map), EXPECTED_PIXELS);

    assert_eq!(decode_pixels(NAKED_JXL), EXPECTED_PIXELS);
    let embedded = bundle("embedded-container");
    assert_eq!(embedded.gain_map.len(), 86);
    assert_eq!(decode_pixels(embedded.gain_map), EXPECTED_PIXELS);
}
