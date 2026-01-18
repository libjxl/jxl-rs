// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Support for ISO 21496-1 gain maps in JPEG XL.
//!
//! Gain maps allow encoding HDR images with an SDR fallback, where the gain map
//! describes how to convert from SDR to HDR representation.

use crate::bit_reader::BitReader;
use crate::error::{Error, Result};
use crate::headers::color_encoding::ColorEncoding;
use crate::headers::encodings::{Empty, UnconditionalCoder};
use crate::util::tracing_wrappers::warn;

/// A gain map bundle as defined by ISO 21496-1.
///
/// This structure contains all data from a `jhgm` box in a JPEG XL container,
/// including the gain map metadata, optional color encoding, ICC profile, and
/// the gain map codestream itself.
#[derive(Debug, Clone)]
pub struct GainMapBundle {
    /// Version of the gain map format (currently always 0).
    pub jhgm_version: u8,

    /// ISO 21496-1 metadata blob (binary format).
    ///
    /// This contains parameters like gain_map_min, gain_map_max, gamma, offset, etc.
    /// The format is defined by the ISO 21496-1 standard.
    pub gain_map_metadata: Vec<u8>,

    /// Optional color encoding for the gain map.
    ///
    /// If present, this describes the color space of the gain map image.
    /// If None, the gain map uses the same color encoding as the base image.
    pub color_encoding: Option<ColorEncoding>,

    /// Alternative ICC profile for the gain map.
    ///
    /// This uses the same JXL-specific ICC compression as in the image header
    /// (not Brotli). The `alt_icc` field stores the already-compressed
    /// representation of the ICC profile.
    ///
    /// This is used when the color encoding cannot be fully described by the
    /// JPEG XL ColorEncoding structure.
    pub alt_icc: Vec<u8>,

    /// The gain map image data as a JPEG XL codestream or container.
    ///
    /// This can be either a naked JPEG XL codestream or a full JPEG XL container
    /// (but it is not allowed to itself contain a gain map box). Using a container
    /// allows the gain map to include `jbrd` for JPEG bitstream reconstruction.
    pub gain_map: Vec<u8>,
}

impl GainMapBundle {
    /// Calculate the total size of this bundle when serialized to binary format.
    ///
    /// This is useful for allocating buffers before calling [`write_to_bytes`](Self::write_to_bytes).
    pub fn bundle_size(&self) -> usize {
        let mut size = 0;

        // jhgm_version (1 byte)
        size += 1;

        // gain_map_metadata_size (2 bytes BE) + metadata
        size += 2 + self.gain_map_metadata.len();

        // color_encoding_size (1 byte)
        // Note: color_encoding serialization not implemented, always written as 0
        size += 1;

        // alt_icc_size (4 bytes BE) + alt_icc
        size += 4 + self.alt_icc.len();

        // gain_map (remaining bytes)
        size += self.gain_map.len();

        size
    }

    /// Serialize this gain map bundle to bytes.
    ///
    /// The binary format matches libjxl's implementation:
    /// - 1 byte: jhgm_version
    /// - 2 bytes BE: gain_map_metadata_size
    /// - N bytes: gain_map_metadata
    /// - 1 byte: color_encoding_size
    /// - M bytes: color_encoding (if size > 0)
    /// - 4 bytes BE: alt_icc_size
    /// - K bytes: alt_icc
    /// - Remaining: gain_map codestream
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Metadata is too large (> 65535 bytes)
    /// - ICC profile is too large (> 2^32 - 1 bytes)
    ///
    /// Note: ColorEncoding serialization is not yet implemented (requires BitWriter).
    /// If `color_encoding` is set, it will be silently dropped (written as size 0).
    pub fn write_to_bytes(&self) -> Result<Vec<u8>> {
        if self.gain_map_metadata.len() > u16::MAX as usize {
            return Err(Error::InvalidBox);
        }

        if self.alt_icc.len() > u32::MAX as usize {
            return Err(Error::InvalidBox);
        }

        let mut output = Vec::with_capacity(self.bundle_size());

        // Write jhgm_version
        output.push(self.jhgm_version);

        // Write gain_map_metadata_size and metadata
        let metadata_size = self.gain_map_metadata.len() as u16;
        output.extend_from_slice(&metadata_size.to_be_bytes());
        output.extend_from_slice(&self.gain_map_metadata);

        // Write color_encoding
        // Note: ColorEncoding serialization requires a BitWriter which is not yet implemented.
        // For now, we write 0 to indicate no color encoding. Parsing works fine.
        if self.color_encoding.is_some() {
            warn!("ColorEncoding serialization not implemented, dropping color_encoding");
        }
        output.push(0);

        // Write alt_icc_size and alt_icc
        let icc_size = self.alt_icc.len() as u32;
        output.extend_from_slice(&icc_size.to_be_bytes());
        output.extend_from_slice(&self.alt_icc);

        // Write gain_map codestream
        output.extend_from_slice(&self.gain_map);

        Ok(output)
    }

    /// Deserialize a gain map bundle from bytes.
    ///
    /// Parses the binary format produced by [`write_to_bytes`](Self::write_to_bytes).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Input buffer is too small
    /// - Sizes in the header are invalid
    /// - Color encoding parsing fails
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let mut offset = 0;

        // Read jhgm_version
        if data.len() < offset + 1 {
            return Err(Error::OutOfBounds(1));
        }
        let jhgm_version = data[offset];
        offset += 1;

        // Read gain_map_metadata_size and metadata
        if data.len() < offset + 2 {
            return Err(Error::OutOfBounds(2));
        }
        let metadata_size = u16::from_be_bytes([data[offset], data[offset + 1]]) as usize;
        offset += 2;

        if data.len() < offset + metadata_size {
            return Err(Error::OutOfBounds(metadata_size));
        }
        let gain_map_metadata = data[offset..offset + metadata_size].to_vec();
        offset += metadata_size;

        // Read color_encoding_size
        if data.len() < offset + 1 {
            return Err(Error::OutOfBounds(1));
        }
        let color_encoding_size = data[offset] as usize;
        offset += 1;

        // Read color_encoding if present
        let color_encoding = if color_encoding_size > 0 {
            if data.len() < offset + color_encoding_size {
                return Err(Error::OutOfBounds(color_encoding_size));
            }

            let color_encoding_bytes = &data[offset..offset + color_encoding_size];
            let mut br = BitReader::new(color_encoding_bytes);
            let parsed =
                ColorEncoding::read_unconditional(&(), &mut br, &Empty {}).map_err(|_| {
                    // If parsing fails, this is an invalid color encoding
                    Error::InvalidBox
                })?;
            offset += color_encoding_size;
            Some(parsed)
        } else {
            None
        };

        // Read alt_icc_size and alt_icc
        if data.len() < offset + 4 {
            return Err(Error::OutOfBounds(4));
        }
        let icc_size = u32::from_be_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;
        offset += 4;

        if data.len() < offset + icc_size {
            return Err(Error::OutOfBounds(icc_size));
        }
        let alt_icc = data[offset..offset + icc_size].to_vec();
        offset += icc_size;

        // Remaining bytes are the gain_map codestream
        let gain_map = data[offset..].to_vec();

        Ok(Self {
            jhgm_version,
            gain_map_metadata,
            color_encoding,
            alt_icc,
            gain_map,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_trip_minimal() {
        let bundle = GainMapBundle {
            jhgm_version: 0,
            gain_map_metadata: vec![1, 2, 3, 4],
            color_encoding: None,
            alt_icc: vec![],
            gain_map: vec![0xff, 0x0a], // Minimal JXL signature
        };

        let bytes = bundle.write_to_bytes().unwrap();
        let decoded = GainMapBundle::from_bytes(&bytes).unwrap();

        assert_eq!(bundle.jhgm_version, decoded.jhgm_version);
        assert_eq!(bundle.gain_map_metadata, decoded.gain_map_metadata);
        assert_eq!(bundle.alt_icc, decoded.alt_icc);
        assert_eq!(bundle.gain_map, decoded.gain_map);
    }

    #[test]
    fn test_round_trip_with_icc() {
        let bundle = GainMapBundle {
            jhgm_version: 0,
            gain_map_metadata: b"test metadata".to_vec(),
            color_encoding: None,
            alt_icc: b"fake ICC profile".to_vec(),
            gain_map: vec![0xff, 0x0a, 0x00, 0x01],
        };

        let bytes = bundle.write_to_bytes().unwrap();
        let decoded = GainMapBundle::from_bytes(&bytes).unwrap();

        assert_eq!(bundle.jhgm_version, decoded.jhgm_version);
        assert_eq!(bundle.gain_map_metadata, decoded.gain_map_metadata);
        assert_eq!(bundle.alt_icc, decoded.alt_icc);
        assert_eq!(bundle.gain_map, decoded.gain_map);
    }

    #[test]
    fn test_metadata_too_large() {
        let bundle = GainMapBundle {
            jhgm_version: 0,
            gain_map_metadata: vec![0; 70000], // > u16::MAX
            color_encoding: None,
            alt_icc: vec![],
            gain_map: vec![0xff, 0x0a],
        };

        assert!(bundle.write_to_bytes().is_err());
    }

    #[test]
    fn test_truncated_input() {
        // Create valid bundle with non-empty gain_map
        let bundle = GainMapBundle {
            jhgm_version: 0,
            gain_map_metadata: vec![1, 2, 3],
            color_encoding: None,
            alt_icc: vec![],
            gain_map: vec![0xff, 0x0a],
        };

        let bytes = bundle.write_to_bytes().unwrap();

        // Test various truncations that should fail
        assert!(
            GainMapBundle::from_bytes(&bytes[..0]).is_err(),
            "Empty input should fail"
        );
        assert!(
            GainMapBundle::from_bytes(&bytes[..1]).is_err(),
            "Missing metadata size"
        );
        assert!(
            GainMapBundle::from_bytes(&bytes[..3]).is_err(),
            "Missing metadata"
        );

        // Full parse should succeed
        assert!(
            GainMapBundle::from_bytes(&bytes).is_ok(),
            "Full input should succeed"
        );
    }

    #[test]
    fn test_parse_color_encoding() {
        use crate::headers::color_encoding::{ColorSpace, RenderingIntent, TransferFunction};

        // Test data from libjxl's gain_map_test.cc:
        // color_encoding = {0x50, 0xb4, 0x00} which represents a valid ColorEncoding
        // This is the bitstream encoding for sRGB color space
        let color_encoding_bytes = vec![0x50, 0xb4, 0x00];

        // Create a bundle with the color encoding
        let mut bundle_bytes = vec![];
        bundle_bytes.push(0x00); // jhgm_version
        bundle_bytes.extend_from_slice(&[0x00, 0x04]); // metadata_size = 4
        bundle_bytes.extend_from_slice(&[0x01, 0x02, 0x03, 0x04]); // metadata
        bundle_bytes.push(color_encoding_bytes.len() as u8); // color_encoding_size = 3
        bundle_bytes.extend_from_slice(&color_encoding_bytes); // color_encoding
        bundle_bytes.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // icc_size = 0
        bundle_bytes.extend_from_slice(&[0xff, 0x0a]); // gain_map (minimal JXL signature)

        let bundle = GainMapBundle::from_bytes(&bundle_bytes).unwrap();

        // Verify the color encoding was parsed
        assert!(bundle.color_encoding.is_some());
        let ce = bundle.color_encoding.unwrap();

        // The bytes 0x50 0xb4 0x00 decode to a valid ColorEncoding
        // The exact values depend on bit packing, but we verify the parsing works
        assert_eq!(ce.color_space, ColorSpace::RGB);
        assert!(!ce.want_icc);
        assert_eq!(ce.tf.transfer_function, TransferFunction::Linear);
        assert_eq!(ce.rendering_intent, RenderingIntent::Relative);
    }
}
