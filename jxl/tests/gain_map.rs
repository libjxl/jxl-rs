// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use jxl::api::states::{Initialized, WithFrameInfo, WithImageInfo};
use jxl::api::{
    JxlAuxBox, JxlAuxBoxType, JxlColorEncoding, JxlColorProfile, JxlColorType, JxlDataFormat,
    JxlDecoder, JxlDecoderOptions, JxlGainMapBundle, JxlOutputBuffer, JxlPixelFormat,
    ProcessingResult,
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

fn with_captured_gain_map<T>(data: &[u8], callback: impl FnOnce(&JxlAuxBox) -> T) -> T {
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
    callback(box_data)
}

fn capture_gain_map(data: &[u8]) -> Vec<u8> {
    with_captured_gain_map(data, |box_data| {
        assert!(!box_data.is_compressed());
        box_data.raw_data().to_vec()
    })
}

#[test]
fn parses_independent_fields_and_color_variants() {
    let structured = bundle("structured");
    assert_eq!(structured.version, 0);
    assert_eq!(structured.metadata, &METADATA[3..144]);
    assert_eq!(structured.color_encoding, Some(&[0x50, 0xb4, 0][..]));
    assert!(structured.compressed_icc.is_empty());
    assert_eq!(structured.gain_map, NAKED_JXL);
    let expected_encoding = JxlColorEncoding::RgbColorSpace {
        white_point: jxl::api::JxlWhitePoint::D65,
        primaries: jxl::api::JxlPrimaries::SRGB,
        transfer_function: jxl::api::JxlTransferFunction::Linear,
        rendering_intent: RenderingIntent::Relative,
    };
    assert!(matches!(
        structured.decode_color_encoding().unwrap(),
        Some(JxlColorProfile::Simple(encoding)) if encoding == expected_encoding
    ));
    assert_eq!(structured.decode_alternate_icc().unwrap(), None);

    let icc = bundle("icc");
    assert!(icc.color_encoding.is_none());
    assert!(icc.decode_color_encoding().unwrap().is_none());
    assert_eq!(icc.decode_alternate_icc().unwrap().as_deref(), Some(ICC));

    let combined = bundle("combined");
    assert!(combined.color_encoding.is_some());
    assert!(
        combined.decode_color_encoding().unwrap() == structured.decode_color_encoding().unwrap()
    );
    assert_eq!(
        combined.decode_alternate_icc().unwrap().as_deref(),
        Some(ICC)
    );

    let want_icc = bundle("want-icc");
    assert_eq!(want_icc.color_encoding, Some(&[0x02][..]));
    assert!(matches!(
        want_icc.decode_color_encoding().unwrap(),
        Some(JxlColorProfile::Icc(profile)) if profile.as_slice() == ICC
    ));
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
        assert!(matches!(
            parsed.decode_color_encoding().unwrap(),
            Some(JxlColorProfile::Icc(profile)) if profile.as_slice() == ICC
        ));
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
fn rejects_nonzero_padding_and_trailing_color_bytes() {
    let structured = bundle("structured");
    let mut nonzero_padding = structured.color_encoding.unwrap().to_vec();
    *nonzero_padding.last_mut().unwrap() |= 2;
    let data = raw_bundle(&[], &nonzero_padding, &[], &[]);
    let parsed = JxlGainMapBundle::parse(&data).unwrap();
    assert!(matches!(
        parsed.decode_color_encoding(),
        Err(Error::NonZeroPadding)
    ));

    let mut trailing_bytes = structured.color_encoding.unwrap().to_vec();
    trailing_bytes.push(0);
    let data = raw_bundle(&[], &trailing_bytes, &[], &[]);
    let parsed = JxlGainMapBundle::parse(&data).unwrap();
    assert!(matches!(
        parsed.decode_color_encoding(),
        Err(Error::InvalidColorEncoding)
    ));

    let icc = bundle("icc");
    let mut want_icc_nonzero_padding = vec![0x02];
    want_icc_nonzero_padding[0] |= 0x10;
    let data = raw_bundle(&[], &want_icc_nonzero_padding, icc.compressed_icc, &[]);
    let parsed = JxlGainMapBundle::parse(&data).unwrap();
    assert!(matches!(
        parsed.decode_color_encoding(),
        Err(Error::NonZeroPadding)
    ));

    let data = raw_bundle(&[], &[0x02, 0], icc.compressed_icc, &[]);
    let parsed = JxlGainMapBundle::parse(&data).unwrap();
    assert!(matches!(
        parsed.decode_color_encoding(),
        Err(Error::InvalidColorEncoding)
    ));
}

#[test]
fn rejects_nonzero_padding_and_trailing_icc_bytes() {
    let valid = bundle("icc");

    let mut nonzero_padding = valid.compressed_icc.to_vec();
    *nonzero_padding.last_mut().unwrap() |= 0x02;
    let data = raw_bundle(&[], &[], &nonzero_padding, &[]);
    let parsed = JxlGainMapBundle::parse(&data).unwrap();
    assert!(matches!(
        parsed.decode_alternate_icc(),
        Err(Error::NonZeroPadding)
    ));

    let mut trailing_bytes = valid.compressed_icc.to_vec();
    trailing_bytes.push(0);
    let data = raw_bundle(&[], &[], &trailing_bytes, &[]);
    let parsed = JxlGainMapBundle::parse(&data).unwrap();
    assert!(matches!(
        parsed.decode_alternate_icc(),
        Err(Error::InvalidIccStream)
    ));
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

    let structured_with_invalid_icc = raw_bundle(&[], &[0x50, 0xb4, 0], &[0], &[]);
    let parsed_structured_with_invalid_icc =
        JxlGainMapBundle::parse(&structured_with_invalid_icc).unwrap();
    assert!(matches!(
        parsed_structured_with_invalid_icc
            .decode_color_encoding()
            .unwrap(),
        Some(JxlColorProfile::Simple(_))
    ));
    assert!(
        parsed_structured_with_invalid_icc
            .decode_alternate_icc()
            .is_err()
    );

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

    let want_icc_without_profile = raw_bundle(&[], &[0x02], &[], &[]);
    let parsed_want_icc_without_profile =
        JxlGainMapBundle::parse(&want_icc_without_profile).unwrap();
    assert!(matches!(
        parsed_want_icc_without_profile.decode_color_encoding(),
        Err(Error::InvalidColorEncoding)
    ));

    let want_icc_with_invalid_profile = raw_bundle(&[], &[0x02], &[0], &[]);
    let parsed_want_icc_with_invalid_profile =
        JxlGainMapBundle::parse(&want_icc_with_invalid_profile).unwrap();
    assert!(
        parsed_want_icc_with_invalid_profile
            .decode_color_encoding()
            .is_err()
    );

    let mut unknown_version = include_bytes!("testdata/gain_map/structured.jhgm").to_vec();
    unknown_version[0] = 0xff;
    let parsed_unknown = JxlGainMapBundle::parse(&unknown_version).unwrap();
    assert_eq!(parsed_unknown.version, 0xff);
    assert!(parsed_unknown.decode_color_encoding().is_ok());

    let valid_default = raw_bundle(&[], &[0x01], &[], &[]);
    let parsed_default = JxlGainMapBundle::parse(&valid_default).unwrap();
    assert!(parsed_default.decode_color_encoding().is_ok());

    let nonzero_padding = raw_bundle(&[], &[0x81], &[], &[]);
    let parsed_nonzero_padding = JxlGainMapBundle::parse(&nonzero_padding).unwrap();
    assert!(matches!(
        parsed_nonzero_padding.decode_color_encoding(),
        Err(Error::NonZeroPadding)
    ));

    let trailing_color_bytes = raw_bundle(&[], &[0x01, 0xa5], &[], &[]);
    let parsed_trailing_color_bytes = JxlGainMapBundle::parse(&trailing_color_bytes).unwrap();
    assert!(matches!(
        parsed_trailing_color_bytes.decode_color_encoding(),
        Err(Error::InvalidColorEncoding)
    ));
}

#[test]
fn arbitrary_bundles_reach_both_decode_helpers_with_bounded_fields() {
    // Exercise arbitrary raw envelope bytes and valid envelopes with arbitrary
    // field contents, including up to 256 bytes of ICC data. Keep the search
    // bounded so malformed ICC streams exercise the normal decoder limits for
    // a short, reproducible test run.
    arbtest::arbtest(|u| {
        let raw_len = usize::from(u.int_in_range::<u8>(0..=64)?);
        let mut raw = vec![0; raw_len];
        u.fill_buffer(&mut raw)?;
        let _ = JxlGainMapBundle::parse(&raw);

        let metadata_len = usize::from(u.int_in_range::<u8>(0..=16)?);
        let mut metadata = vec![0; metadata_len];
        u.fill_buffer(&mut metadata)?;

        let color_len = usize::from(u.int_in_range::<u8>(1..=8)?);
        let mut color = vec![0; color_len];
        u.fill_buffer(&mut color)?;

        let gain_map_len = usize::from(u.int_in_range::<u8>(0..=16)?);
        let mut gain_map = vec![0; gain_map_len];
        u.fill_buffer(&mut gain_map)?;

        let icc_len = usize::from(u.int_in_range::<u16>(0..=256)?);
        let mut compressed_icc = vec![0; icc_len];
        u.fill_buffer(&mut compressed_icc)?;
        let data = raw_bundle(&metadata, &color, &compressed_icc, &gain_map);
        let parsed = JxlGainMapBundle::parse(&data).unwrap();
        let _ = parsed.decode_color_encoding();
        let _ = parsed.decode_alternate_icc();
        Ok(())
    })
    .size_min(128)
    .size_max(512)
    .budget_ms(100);
}

#[test]
fn captures_finite_aux_box_and_decodes_bare_gain_map() {
    let outer = include_bytes!("testdata/gain_map/synthetic-container.jxl");
    let captured = capture_gain_map(outer);
    assert_eq!(captured, include_bytes!("testdata/gain_map/combined.jhgm"));
    let captured_bundle = JxlGainMapBundle::parse(&captured).unwrap();
    assert_eq!(decode_pixels(captured_bundle.gain_map), EXPECTED_PIXELS);

    assert_eq!(decode_pixels(NAKED_JXL), EXPECTED_PIXELS);
}

#[cfg(feature = "brotli")]
#[test]
fn captures_brotli_gain_map_and_decodes_bundle() {
    let outer = include_bytes!("testdata/gain_map/synthetic-container-brob.jxl");
    let captured = with_captured_gain_map(outer, |box_data| {
        assert!(box_data.is_compressed());
        box_data.data(&[]).unwrap().into_owned()
    });
    assert_eq!(captured, include_bytes!("testdata/gain_map/combined.jhgm"));
    let captured_bundle = JxlGainMapBundle::parse(&captured).unwrap();
    assert_eq!(decode_pixels(captured_bundle.gain_map), EXPECTED_PIXELS);
}

#[test]
fn preserves_embedded_container_bytes_for_compatibility() {
    // This synthetic case checks byte preservation for consumers that accept a
    // container-form embedded image. It does not assert ISO validity or imply
    // real-world container usage.
    let source = include_bytes!("testdata/gain_map/embedded-container.jhgm");
    let embedded = bundle("embedded-container");
    assert_eq!(embedded.gain_map.len(), 86);
    assert_eq!(
        embedded.gain_map,
        &source[source.len() - embedded.gain_map.len()..]
    );
    assert!(embedded.gain_map.starts_with(b"\0\0\0\x0cJXL \r\n\x87\n"));
}
