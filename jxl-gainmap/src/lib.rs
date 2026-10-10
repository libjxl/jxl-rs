// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Helpers for inspecting a JPEG XL `jhgm` gain-map payload.
//!
//! [`JxlGainMap`] borrows metadata and image fields while decoding and owning
//! the selected optional color profile during parsing. It does not validate
//! gain-map metadata, capture or decompress boxes, decode the embedded image,
//! or apply the gain map.

use jxl::api::JxlColorProfile;
use jxl::error::{Error, Result};

/// A view of the fields in a `jhgm` gain-map payload.
///
/// Metadata and the embedded image remain in the input buffer. The selected
/// color profile is decoded during parsing and owns any ICC bytes.
#[derive(Clone, Debug, PartialEq)]
pub struct JxlGainMap<'a> {
    /// The `jhgm` bundle version.
    pub version: u8,
    /// ISO 21496-1 gain-map metadata bytes.
    pub metadata: &'a [u8],
    /// The selected alternate color profile, or `None` to reuse the baseline.
    pub color_profile: Option<JxlColorProfile>,
    /// The embedded JPEG XL image bytes.
    pub gain_map: &'a [u8],
}

impl<'a> JxlGainMap<'a> {
    /// Parses a complete, uncompressed `jhgm` payload. The metadata and
    /// embedded image borrow from `data`; the selected color profile is decoded
    /// and owned by the returned value. The outer box header is excluded;
    /// decompress Brotli box data before calling this method. Unsupported
    /// versions are rejected before the remaining fields are read, while an
    /// empty trailing image is accepted for the caller to validate.
    pub fn parse(data: &'a [u8]) -> Result<Self> {
        let mut position = 0;
        let version = take(data, &mut position, 1)?[0];
        if version != 0 {
            return Err(Error::InvalidEnum(
                u32::from(version),
                "jhgm version".to_owned(),
            ));
        }
        let metadata_length = u16::from_be_bytes(take(data, &mut position, 2)?.try_into().unwrap());
        let metadata = take(data, &mut position, usize::from(metadata_length))?;
        let color_length = usize::from(take(data, &mut position, 1)?[0]);
        let color_data = take(data, &mut position, color_length)?;
        let color_encoding = (color_length != 0).then_some(color_data);
        let icc_length = u32::from_be_bytes(take(data, &mut position, 4)?.try_into().unwrap());
        let icc_length = usize::try_from(icc_length).map_err(|_| Error::SizeOverflow)?;
        let compressed_icc = take(data, &mut position, icc_length)?;
        let color_profile = if color_encoding.is_none() && compressed_icc.is_empty() {
            None
        } else {
            Some(JxlColorProfile::from_raw_bytes(
                color_encoding,
                compressed_icc,
            )?)
        };

        Ok(Self {
            version,
            metadata,
            color_profile,
            gain_map: &data[position..],
        })
    }
}

fn take<'a>(data: &'a [u8], position: &mut usize, length: usize) -> Result<&'a [u8]> {
    let end = position.checked_add(length).ok_or(Error::SizeOverflow)?;
    let field = data.get(*position..end).ok_or(Error::SectionTooShort)?;
    *position = end;
    Ok(field)
}
