// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! JPEG reconstruction data structures and parsing.
//!
//! This module handles parsing of the `jbrd` (JPEG Bitstream Reconstruction Data) box
//! which contains the information needed to reconstruct the original JPEG file
//! bit-for-bit from a JXL-recompressed JPEG.

use crate::bit_reader::BitReader;
use crate::error::{Error, Result};

/// Type of APP marker in JPEG file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum AppMarkerType {
    /// Unknown APP marker type
    #[default]
    Unknown = 0,
    /// ICC color profile (APP2)
    Icc = 1,
    /// EXIF metadata (APP1)
    Exif = 2,
    /// XMP metadata (APP1)
    Xmp = 3,
}

impl TryFrom<u8> for AppMarkerType {
    type Error = Error;

    fn try_from(value: u8) -> Result<Self> {
        match value {
            0 => Ok(AppMarkerType::Unknown),
            1 => Ok(AppMarkerType::Icc),
            2 => Ok(AppMarkerType::Exif),
            3 => Ok(AppMarkerType::Xmp),
            _ => Err(Error::InvalidJpegReconstructionData),
        }
    }
}

/// JPEG quantization table.
#[derive(Debug, Clone)]
pub struct JpegQuantTable {
    /// Precision (0 = 8-bit, 1 = 16-bit)
    pub precision: u8,
    /// Table index (0-3)
    pub index: u8,
    /// Whether this table is the last one before SOS
    pub is_last: bool,
    /// Quantization values (64 entries in zigzag order)
    pub values: [u16; 64],
}

impl Default for JpegQuantTable {
    fn default() -> Self {
        Self {
            precision: 0,
            index: 0,
            is_last: false,
            values: [0u16; 64],
        }
    }
}

/// JPEG component information.
#[derive(Debug, Clone, Default)]
pub struct JpegComponent {
    /// Component ID
    pub id: u8,
    /// Horizontal sampling factor
    pub h_samp_factor: u8,
    /// Vertical sampling factor
    pub v_samp_factor: u8,
    /// Quantization table index
    pub quant_idx: u8,
}

/// JPEG Huffman code.
#[derive(Debug, Clone)]
pub struct JpegHuffmanCode {
    /// Table class (0 = DC, 1 = AC)
    pub table_class: u8,
    /// Table slot (0-3)
    pub slot_id: u8,
    /// Whether this is the last DHT segment
    pub is_last: bool,
    /// Number of codes for each length (1-16)
    pub counts: [u8; 16],
    /// Symbol values
    pub values: Vec<u8>,
}

impl Default for JpegHuffmanCode {
    fn default() -> Self {
        Self {
            table_class: 0,
            slot_id: 0,
            is_last: false,
            counts: [0u8; 16],
            values: Vec::new(),
        }
    }
}

/// Reset point for progressive scan.
#[derive(Debug, Clone, Default)]
pub struct JpegResetPoint {
    /// MCU index where reset occurs
    pub mcu: u32,
    /// Last DC coefficient values per component
    pub last_dc: Vec<i16>,
}

/// Information about a single JPEG scan.
#[derive(Debug, Clone)]
pub struct JpegScanInfo {
    /// Number of components in this scan
    pub num_components: u8,
    /// Component indices
    pub component_idx: [u8; 4],
    /// DC Huffman table index per component
    pub dc_tbl_idx: [u8; 4],
    /// AC Huffman table index per component
    pub ac_tbl_idx: [u8; 4],
    /// Spectral selection start
    pub ss: u8,
    /// Spectral selection end
    pub se: u8,
    /// Successive approximation high bit
    pub ah: u8,
    /// Successive approximation low bit
    pub al: u8,
    /// Reset points for error recovery
    pub reset_points: Vec<JpegResetPoint>,
    /// Number of extra zero runs (for progressive encoding)
    pub extra_zero_runs: Vec<(u32, u32)>,
}

impl Default for JpegScanInfo {
    fn default() -> Self {
        Self {
            num_components: 0,
            component_idx: [0u8; 4],
            dc_tbl_idx: [0u8; 4],
            ac_tbl_idx: [0u8; 4],
            ss: 0,
            se: 0,
            ah: 0,
            al: 0,
            reset_points: Vec::new(),
            extra_zero_runs: Vec::new(),
        }
    }
}

/// JPEG reconstruction data from a jbrd box.
///
/// This structure contains all the information needed to reconstruct
/// the original JPEG file bit-for-bit from JXL-recompressed data.
#[derive(Debug, Clone, Default)]
pub struct JpegReconstructionData {
    /// Image width
    pub width: u32,
    /// Image height
    pub height: u32,
    /// Restart interval (in MCUs)
    pub restart_interval: u32,

    /// Quantization tables
    pub quant_tables: Vec<JpegQuantTable>,
    /// Huffman codes
    pub huffman_codes: Vec<JpegHuffmanCode>,
    /// Image components
    pub components: Vec<JpegComponent>,
    /// Scan information
    pub scan_info: Vec<JpegScanInfo>,

    /// APP marker data (decompressed)
    pub app_data: Vec<Vec<u8>>,
    /// APP marker types
    pub app_marker_types: Vec<AppMarkerType>,
    /// COM (comment) marker data (decompressed)
    pub com_data: Vec<Vec<u8>>,

    /// Whether there are zero padding bits
    pub has_zero_padding_bit: bool,
    /// Padding bits data
    pub padding_bits: Vec<u8>,
    /// Order of markers in original file
    pub marker_order: Vec<u8>,
    /// Data between markers
    pub inter_marker_data: Vec<Vec<u8>>,
    /// Trailing data after EOI
    pub tail_data: Vec<u8>,
}

impl JpegReconstructionData {
    /// Parse jbrd box data into JPEG reconstruction data.
    ///
    /// Note: This is a partial implementation. Full parsing requires
    /// Brotli decompression for marker data.
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.is_empty() {
            return Err(Error::InvalidJpegReconstructionData);
        }

        let mut reader = BitReader::new(data);
        let mut result = JpegReconstructionData::default();

        // Parse the Bundle structure (see libjxl jpeg_data.h Fields)
        // The format uses variable-length encoding for most fields

        // Read dimensions
        result.width = Self::read_u32(&mut reader)?;
        result.height = Self::read_u32(&mut reader)?;

        // Read restart interval
        result.restart_interval = Self::read_u32(&mut reader)?;

        // Read number of APP markers
        let num_app_markers = Self::read_u32(&mut reader)? as usize;
        result.app_marker_types = Vec::with_capacity(num_app_markers);
        for _ in 0..num_app_markers {
            let marker_type = reader.read(2)? as u8;
            result
                .app_marker_types
                .push(AppMarkerType::try_from(marker_type)?);
        }

        // Read number of components
        let num_components = Self::read_u32(&mut reader)? as usize;
        result.components = Vec::with_capacity(num_components);
        for _ in 0..num_components {
            let component = JpegComponent {
                id: reader.read(8)? as u8,
                h_samp_factor: (reader.read(4)? as u8).max(1),
                v_samp_factor: (reader.read(4)? as u8).max(1),
                quant_idx: reader.read(2)? as u8,
            };
            result.components.push(component);
        }

        // Read quantization tables
        let num_quant_tables = Self::read_u32(&mut reader)? as usize;
        result.quant_tables = Vec::with_capacity(num_quant_tables);
        for _ in 0..num_quant_tables {
            let mut table = JpegQuantTable::default();
            table.precision = reader.read(1)? as u8;
            table.index = reader.read(2)? as u8;
            table.is_last = reader.read(1)? != 0;
            for i in 0..64 {
                table.values[i] = if table.precision == 0 {
                    reader.read(8)? as u16
                } else {
                    reader.read(16)? as u16
                };
            }
            result.quant_tables.push(table);
        }

        // Read Huffman codes
        let num_huffman_codes = Self::read_u32(&mut reader)? as usize;
        result.huffman_codes = Vec::with_capacity(num_huffman_codes);
        for _ in 0..num_huffman_codes {
            let mut code = JpegHuffmanCode::default();
            code.table_class = reader.read(1)? as u8;
            code.slot_id = reader.read(2)? as u8;
            code.is_last = reader.read(1)? != 0;
            let mut total_count = 0u32;
            for i in 0..16 {
                code.counts[i] = reader.read(8)? as u8;
                total_count += code.counts[i] as u32;
            }
            code.values = Vec::with_capacity(total_count as usize);
            for _ in 0..total_count {
                code.values.push(reader.read(8)? as u8);
            }
            result.huffman_codes.push(code);
        }

        // Read scan info
        let num_scans = Self::read_u32(&mut reader)? as usize;
        result.scan_info = Vec::with_capacity(num_scans);
        for _ in 0..num_scans {
            let mut scan = JpegScanInfo::default();
            scan.num_components = reader.read(2)? as u8 + 1;
            for i in 0..scan.num_components as usize {
                scan.component_idx[i] = reader.read(2)? as u8;
                scan.dc_tbl_idx[i] = reader.read(2)? as u8;
                scan.ac_tbl_idx[i] = reader.read(2)? as u8;
            }
            scan.ss = reader.read(6)? as u8;
            scan.se = reader.read(6)? as u8;
            scan.ah = reader.read(4)? as u8;
            scan.al = reader.read(4)? as u8;

            let num_reset_points = Self::read_u32(&mut reader)? as usize;
            scan.reset_points = Vec::with_capacity(num_reset_points);
            for _ in 0..num_reset_points {
                let mcu = Self::read_u32(&mut reader)?;
                let num_dc = scan.num_components as usize;
                let mut last_dc = Vec::with_capacity(num_dc);
                for _ in 0..num_dc {
                    last_dc.push(reader.read(16)? as i16);
                }
                scan.reset_points.push(JpegResetPoint { mcu, last_dc });
            }

            let num_extra_zeros = Self::read_u32(&mut reader)? as usize;
            scan.extra_zero_runs = Vec::with_capacity(num_extra_zeros);
            for _ in 0..num_extra_zeros {
                let block_idx = Self::read_u32(&mut reader)?;
                let num_zeros = Self::read_u32(&mut reader)?;
                scan.extra_zero_runs.push((block_idx, num_zeros));
            }

            result.scan_info.push(scan);
        }

        // Read marker order
        let num_markers = Self::read_u32(&mut reader)? as usize;
        result.marker_order = Vec::with_capacity(num_markers);
        for _ in 0..num_markers {
            result.marker_order.push(reader.read(8)? as u8);
        }

        // Read flags
        result.has_zero_padding_bit = reader.read(1)? != 0;

        // Skip to byte boundary for Brotli-compressed data
        reader.jump_to_byte_boundary()?;

        // The remaining data is Brotli-compressed marker data (APP, COM, inter-marker, tail)
        // For now, we store the raw compressed data
        // Full implementation would decompress using brotli crate
        let remaining_pos = reader.total_bits_read() / 8;
        if remaining_pos < data.len() {
            // Store remaining compressed data for later decompression
            // This includes: app_data, com_data, inter_marker_data, tail_data
            // All Brotli-compressed
        }

        Ok(result)
    }

    /// Read a variable-length u32 value.
    fn read_u32(reader: &mut BitReader) -> Result<u32> {
        // JXL uses a variable-length encoding for integers
        // First read the selector bits
        let selector = reader.read(2)?;
        match selector {
            0 => Ok(0),
            1 => Ok(reader.read(4)? as u32 + 1),
            2 => Ok(reader.read(8)? as u32 + 17),
            3 => Ok(reader.read(12)? as u32 + 273),
            _ => unreachable!(),
        }
    }

    /// Check if this structure contains valid JPEG reconstruction data.
    pub fn is_valid(&self) -> bool {
        self.width > 0 && self.height > 0 && !self.components.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_app_marker_type_conversion() {
        assert_eq!(AppMarkerType::try_from(0).unwrap(), AppMarkerType::Unknown);
        assert_eq!(AppMarkerType::try_from(1).unwrap(), AppMarkerType::Icc);
        assert_eq!(AppMarkerType::try_from(2).unwrap(), AppMarkerType::Exif);
        assert_eq!(AppMarkerType::try_from(3).unwrap(), AppMarkerType::Xmp);
        assert!(AppMarkerType::try_from(4).is_err());
    }
}
