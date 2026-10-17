// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use jxl::api::{JxlColorEncoding, JxlColorProfile};
use jxl::error::Error;
use jxl_gainmap::JxlGainMap;

const METADATA: &[u8] = include_bytes!("testdata/gain_map/combined.jhgm");
const GAIN_MAP_IMAGE: [u8; 46] = [
    0xff, 0x0a, 0x08, 0x10, 0x10, 0x50, 0x5c, 0x08, 0x08, 0x00, 0x01, 0x00, 0x80, 0x00, 0x4b, 0x12,
    0xc5, 0x82, 0x85, 0x24, 0x56, 0x00, 0xe7, 0x15, 0x1c, 0x00, 0xc0, 0x3f, 0x8a, 0x5f, 0x20, 0x08,
    0x00, 0x2d, 0x3c, 0x84, 0x23, 0x44, 0x73, 0xc0, 0x29, 0xd6, 0x1b, 0xa8, 0x91, 0x01,
];

fn fixture(name: &str) -> &'static [u8] {
    match name {
        "structured" => include_bytes!("testdata/gain_map/structured.jhgm"),
        "icc" => include_bytes!("testdata/gain_map/icc.jhgm"),
        "combined" => include_bytes!("testdata/gain_map/combined.jhgm"),
        "want-icc" => include_bytes!("testdata/gain_map/want_icc.jhgm"),
        "malformed-color" => include_bytes!("testdata/gain_map/malformed_color.jhgm"),
        "missing-icc" => include_bytes!("testdata/gain_map/missing_icc.jhgm"),
        "unsupported-version" => include_bytes!("testdata/gain_map/unsupported_version.jhgm"),
        "malformed-unused-icc" => {
            include_bytes!("testdata/gain_map/malformed_unused_icc.jhgm")
        }
        _ => panic!("unknown fixture {name}"),
    }
}

fn bundle(name: &str) -> JxlGainMap<'static> {
    JxlGainMap::parse(fixture(name)).unwrap()
}

fn srgb_profile() -> Vec<u8> {
    JxlColorEncoding::srgb(false)
        .maybe_create_profile()
        .unwrap()
        .unwrap()
}

#[test]
fn decodes_selected_structured_and_icc_profiles() {
    let expected_structured = Some(JxlColorProfile::Simple(JxlColorEncoding::linear_srgb(
        false,
    )));

    let structured = bundle("structured");
    assert_eq!(structured.version, 0);
    assert_eq!(structured.metadata, &METADATA[3..144]);
    assert_eq!(structured.gain_map, GAIN_MAP_IMAGE.as_slice());
    assert_eq!(structured.color_profile, expected_structured);

    let icc = bundle("icc");
    let expected_icc = srgb_profile();
    assert_eq!(
        icc.color_profile,
        Some(JxlColorProfile::Icc(expected_icc.clone()))
    );

    let combined = bundle("combined");
    assert_eq!(combined.color_profile, expected_structured);
    assert_eq!(combined.gain_map, GAIN_MAP_IMAGE.as_slice());

    let want_icc = bundle("want-icc");
    assert_eq!(
        want_icc.color_profile,
        Some(JxlColorProfile::Icc(expected_icc.clone()))
    );
}

#[test]
fn handles_absent_and_invalid_selected_profiles() {
    let expected_structured = Some(JxlColorProfile::Simple(JxlColorEncoding::linear_srgb(
        false,
    )));

    let mut empty_data = vec![0; 8];
    empty_data.extend_from_slice(&GAIN_MAP_IMAGE);
    let empty = JxlGainMap::parse(&empty_data).unwrap();
    assert_eq!(empty.color_profile, None);
    assert_eq!(empty.gain_map, GAIN_MAP_IMAGE.as_slice());

    // The fixture has want_icc=true and a nonzero padding bit.
    assert!(JxlGainMap::parse(fixture("malformed-color")).is_err());
    assert!(JxlGainMap::parse(fixture("missing-icc")).is_err());

    let malformed_unused_icc = JxlGainMap::parse(fixture("malformed-unused-icc")).unwrap();
    assert_eq!(malformed_unused_icc.color_profile, expected_structured);
}

#[test]
fn owns_decoded_icc_profile() {
    let expected = JxlColorProfile::Icc(srgb_profile());
    let profile = {
        let mut data = fixture("icc").to_vec();
        let profile = JxlGainMap::parse(&data).unwrap().color_profile.unwrap();
        data.fill(0);
        profile
    };

    assert_eq!(profile, expected);
}

#[test]
fn rejects_truncated_jhgm_fields() {
    for fixture in [
        include_bytes!("testdata/gain_map/truncated_metadata.jhgm").as_slice(),
        include_bytes!("testdata/gain_map/truncated_color.jhgm").as_slice(),
        include_bytes!("testdata/gain_map/truncated_icc.jhgm").as_slice(),
    ] {
        assert!(matches!(
            JxlGainMap::parse(fixture),
            Err(Error::SectionTooShort)
        ));
    }
}

#[test]
fn rejects_unknown_versions_before_reading_fields() {
    let error = JxlGainMap::parse(fixture("unsupported-version")).unwrap_err();
    match error {
        Error::InvalidEnum(value, field) => {
            assert_eq!(value, 0xff);
            assert_eq!(field, "jhgm version");
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn accepts_empty_images() {
    for name in ["structured", "icc", "combined", "want-icc"] {
        let data = fixture(name);
        let envelope_length = data.len() - GAIN_MAP_IMAGE.len();
        let parsed = JxlGainMap::parse(&data[..envelope_length]).unwrap();
        assert!(parsed.gain_map.is_empty());
    }
}
