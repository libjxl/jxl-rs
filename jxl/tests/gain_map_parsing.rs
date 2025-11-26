// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Integration test for gain map parsing

use jxl::api::GainMapBundle;

#[test]
fn test_parse_gain_map_bundle_from_libjxl_format() {
    // This is the exact format used in libjxl's gain_map_test.cc
    // GoldenTestGainMap with no color encoding, no ICC
    let mut bundle_data = Vec::new();

    // jhgm_version
    bundle_data.push(0x00);

    // gain_map_metadata_size (88 bytes = 0x0058 in BE)
    bundle_data.extend_from_slice(&[0x00, 0x58]);

    // metadata (88 bytes)
    let metadata = b"placeholder gain map metadata, fill with actual example after (ISO 21496-1) is finalized";
    bundle_data.extend_from_slice(metadata);

    // color_encoding_size (0 = not present)
    bundle_data.push(0x00);

    // ICC size (0 = not present)
    bundle_data.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);

    // gain_map codestream
    let codestream = b"placeholder for an actual naked JPEG XL codestream";
    bundle_data.extend_from_slice(codestream);

    // Parse the bundle
    let result = GainMapBundle::from_bytes(&bundle_data);
    assert!(result.is_ok(), "Failed to parse gain map bundle");

    let bundle = result.unwrap();

    // Verify all fields
    assert_eq!(bundle.jhgm_version, 0);
    assert_eq!(bundle.gain_map_metadata.len(), 88);
    assert_eq!(bundle.gain_map_metadata, metadata);
    assert!(bundle.color_encoding.is_none());
    assert_eq!(bundle.alt_icc.len(), 0);
    assert_eq!(bundle.gain_map, codestream);
}

#[test]
fn test_gain_map_round_trip() {
    // Create a bundle
    let original = GainMapBundle {
        jhgm_version: 0,
        gain_map_metadata: b"test metadata for ISO 21496-1".to_vec(),
        color_encoding: None,
        alt_icc: vec![],
        gain_map: vec![0xff, 0x0a, 0x01, 0x02, 0x03],  // Fake JXL codestream
    };

    // Serialize
    let bytes = original.write_to_bytes().expect("Failed to serialize");

    // Deserialize
    let deserialized = GainMapBundle::from_bytes(&bytes).expect("Failed to deserialize");

    // Verify
    assert_eq!(original.jhgm_version, deserialized.jhgm_version);
    assert_eq!(original.gain_map_metadata, deserialized.gain_map_metadata);
    assert_eq!(original.alt_icc, deserialized.alt_icc);
    assert_eq!(original.gain_map, deserialized.gain_map);
}

#[test]
fn test_gain_map_with_icc() {
    // Test with ICC profile
    let icc_data = vec![0x00, 0x01, 0x02, 0x03, 0x04, 0x05]; // Fake ICC
    let original = GainMapBundle {
        jhgm_version: 0,
        gain_map_metadata: b"metadata".to_vec(),
        color_encoding: None,
        alt_icc: icc_data.clone(),
        gain_map: vec![0xff, 0x0a],
    };

    let bytes = original.write_to_bytes().unwrap();
    let deserialized = GainMapBundle::from_bytes(&bytes).unwrap();

    assert_eq!(deserialized.alt_icc, icc_data);
}
