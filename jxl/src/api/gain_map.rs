// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use super::{JxlColorEncoding, JxlColorProfile};
use crate::bit_reader::BitReader;
use crate::error::{Error, Result};
use crate::headers::color_encoding::{ColorEncoding, ColorSpace};
use crate::headers::encodings::{Empty, UnconditionalCoder};
use crate::icc::IncrementalIccReader;

/// A borrowed view of the fields in a `jhgm` gain-map bundle.
///
/// The view keeps each field in the input buffer. It does not validate the ISO
/// gain-map metadata or decode the embedded JPEG XL image. Use the helper
/// methods to decode the alternate color encoding and ICC profile; callers
/// interpret the metadata and decode the image separately.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JxlGainMapBundle<'a> {
    /// The `jhgm` bundle version.
    pub version: u8,
    /// ISO 21496-1 gain-map metadata bytes.
    pub metadata: &'a [u8],
    /// Raw structured alternate color encoding bytes, when present.
    pub color_encoding: Option<&'a [u8]>,
    /// Compressed alternate ICC profile bytes, when present.
    pub compressed_icc: &'a [u8],
    /// The embedded JPEG XL image bytes.
    pub gain_map: &'a [u8],
}

impl<'a> JxlGainMapBundle<'a> {
    /// Parses a complete, uncompressed `jhgm` payload and borrows its fields
    /// from `data`. The outer box header is not part of `data`; Brotli
    /// compressed box data must be decompressed before calling this method.
    ///
    /// The envelope consists of a one-byte version, a big-endian `u16`
    /// metadata length, metadata, a one-byte structured-color length, color
    /// bytes, a big-endian `u32` ICC length, ICC bytes, and the remaining gain
    /// map image bytes. Unknown versions and empty trailing image fields are
    /// retained for the caller to decide how to handle.
    pub fn parse(data: &'a [u8]) -> Result<Self> {
        let mut position = 0;
        let version = take(data, &mut position, 1)?[0];
        let metadata_length = u16::from_be_bytes(take(data, &mut position, 2)?.try_into().unwrap());
        let metadata = take(data, &mut position, usize::from(metadata_length))?;
        let color_length = usize::from(take(data, &mut position, 1)?[0]);
        let color_data = take(data, &mut position, color_length)?;
        let color_encoding = (color_length != 0).then_some(color_data);
        let icc_length = u32::from_be_bytes(take(data, &mut position, 4)?.try_into().unwrap());
        let icc_length = usize::try_from(icc_length).map_err(|_| Error::SizeOverflow)?;
        let compressed_icc = take(data, &mut position, icc_length)?;

        Ok(Self {
            version,
            metadata,
            color_encoding,
            compressed_icc,
            gain_map: &data[position..],
        })
    }

    /// Decodes the optional alternate color encoding.
    ///
    /// `None` means that the structured color field is absent, even when an
    /// independent alternate ICC field is present. When the structured field
    /// requests ICC, this resolves that reference through
    /// [`Self::decode_alternate_icc`]. The two fields remain independently
    /// available through the raw slices and the ICC helper; this method does
    /// not choose precedence between them. A required ICC field that is
    /// absent or cannot be decoded is reported as an error.
    pub fn decode_color_encoding(&self) -> Result<Option<JxlColorProfile>> {
        let Some(data) = self.color_encoding else {
            return Ok(None);
        };

        // When want_icc is set, the structured color-space value is still
        // serialized but is intentionally ignored. Read just the applicable
        // prefix so values unsupported by structured-color conversion do not
        // reject a usable ICC profile. Keep enum and bounds validation for the
        // color-space prefix.
        let mut prefix_reader = BitReader::new(data);
        let all_default = bool::read_unconditional(&(), &mut prefix_reader, &Empty {})?;
        if !all_default {
            let want_icc = bool::read_unconditional(&(), &mut prefix_reader, &Empty {})?;
            if want_icc {
                ColorSpace::read_unconditional(&(), &mut prefix_reader, &Empty {})?;
                prefix_reader.check_for_error()?;
                let profile = self
                    .decode_alternate_icc()?
                    .ok_or(Error::InvalidColorEncoding)?;
                return Ok(Some(JxlColorProfile::Icc(profile)));
            }
        }

        let mut reader = BitReader::new(data);
        let encoding = ColorEncoding::read_unconditional(&(), &mut reader, &Empty {})?;
        reader.check_for_error()?;

        Ok(Some(JxlColorProfile::Simple(
            JxlColorEncoding::from_internal(&encoding)?,
        )))
    }

    /// Decodes the optional JPEG XL-compressed alternate ICC profile.
    pub fn decode_alternate_icc(&self) -> Result<Option<Vec<u8>>> {
        if self.compressed_icc.is_empty() {
            return Ok(None);
        }

        let mut reader = BitReader::new(self.compressed_icc);
        let mut icc = IncrementalIccReader::new(&mut reader)?;
        icc.read_all(&mut reader)?;
        let profile = icc.finalize(&mut reader)?;
        reader.check_for_error()?;
        Ok(Some(profile))
    }
}

fn take<'a>(data: &'a [u8], position: &mut usize, length: usize) -> Result<&'a [u8]> {
    let end = position.checked_add(length).ok_or(Error::SizeOverflow)?;
    let field = data.get(*position..end).ok_or(Error::SectionTooShort)?;
    *position = end;
    Ok(field)
}
