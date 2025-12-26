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
#[derive(Debug, Clone, Default)]
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
    pub values: Vec<u16>,
}

impl JpegHuffmanCode {
    fn dht_counts_and_values_len(&self) -> ([u8; 16], usize) {
        let total_count: usize = self.counts.iter().map(|&c| c as usize).sum();
        let has_sentinel = total_count > 0
            && self.values.last() == Some(&256)
            && self.values.len() == total_count;
        let mut counts = self.counts;
        let mut values_len = self.values.len();
        if has_sentinel {
            values_len = values_len.saturating_sub(1);
            if let Some(max_idx) = (0..counts.len()).rev().find(|&i| counts[i] != 0) {
                counts[max_idx] = counts[max_idx].saturating_sub(1);
            }
        }
        (counts, values_len)
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
#[derive(Debug, Clone, Default)]
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
    /// Whether this is a grayscale image
    pub is_gray: bool,
    /// Whether the jbrd box uses all-default values (metadata from codestream)
    pub is_all_default: bool,

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
    /// Stored DCT coefficients for bit-exact JPEG reconstruction.
    /// These are the quantized DCT coefficients extracted from the JXL decoder.
    /// Each component has a Vec of i16 coefficients in block order.
    pub dct_coefficients: Option<JpegDctCoefficients>,
}

/// Storage for JPEG DCT coefficients extracted from JXL decoder.
/// Used for bit-exact JPEG reconstruction.
#[derive(Debug, Clone, Default)]
pub struct JpegDctCoefficients {
    /// Image width in pixels
    pub width: usize,
    /// Image height in pixels
    pub height: usize,
    /// Number of components (1 for grayscale, 3 for color)
    pub num_components: usize,
    /// DCT coefficients for each component.
    /// Each component's coefficients are stored in raster order of 8x8 blocks,
    /// with each block containing 64 coefficients in zigzag order.
    pub coefficients: Vec<Vec<i16>>,
    /// Block dimensions per component (in 8x8 blocks).
    pub blocks_x: Vec<usize>,
    pub blocks_y: Vec<usize>,
    /// Quantization table index for each component
    pub quant_indices: Vec<u8>,
}

impl JpegDctCoefficients {
    /// Create a new coefficient storage for the given dimensions.
    pub fn new(width: usize, height: usize, component_blocks: &[(usize, usize)]) -> Self {
        let num_components = component_blocks.len();
        let mut coefficients = Vec::with_capacity(num_components);
        let mut blocks_x = Vec::with_capacity(num_components);
        let mut blocks_y = Vec::with_capacity(num_components);

        for &(bx, by) in component_blocks {
            blocks_x.push(bx);
            blocks_y.push(by);
            coefficients.push(vec![0i16; bx * by * 64]);
        }

        Self {
            width,
            height,
            num_components,
            coefficients,
            blocks_x,
            blocks_y,
            quant_indices: vec![0; num_components],
        }
    }

    /// Store AC coefficients for a block (skips DC at index 0).
    ///
    /// DC coefficients are stored separately via `store_dc()` because they come
    /// from a different decoding path (LF group) than AC coefficients (HF group).
    ///
    /// - `component`: Component index (0=Y, 1=Cb, 2=Cr for color; 0=Gray for grayscale)
    /// - `bx`, `by`: Block coordinates
    /// - `coeffs`: 64 DCT coefficients in natural order (will be converted to zigzag)
    pub fn store_block(&mut self, component: usize, bx: usize, by: usize, coeffs: &[i32]) {
        if component >= self.num_components || coeffs.len() < 64 {
            return;
        }

        let blocks_x = self.blocks_x[component];
        let block_idx = by * blocks_x + bx;
        let offset = block_idx * 64;

        if offset + 64 > self.coefficients[component].len() {
            return;
        }

        // Store AC coefficients only (skip index 0 which is DC)
        // DC is stored separately from LF group via store_dc()
        for i in 1..64 {
            let zigzag_idx = JPEG_NATURAL_ORDER[i];
            let x = zigzag_idx % 8;
            let y = zigzag_idx / 8;
            let transposed_idx = x * 8 + y;
            self.coefficients[component][offset + i] =
                coeffs[transposed_idx].clamp(-32768, 32767) as i16;
        }

    }

    /// Get coefficients for a block in zigzag order.
    pub fn get_block(&self, component: usize, bx: usize, by: usize) -> Option<&[i16]> {
        if component >= self.num_components {
            return None;
        }

        let blocks_x = self.blocks_x[component];
        let block_idx = by * blocks_x + bx;
        let offset = block_idx * 64;

        if offset + 64 > self.coefficients[component].len() {
            return None;
        }

        Some(&self.coefficients[component][offset..offset + 64])
    }

    /// Store just the DC coefficient for a block.
    /// DC is always at index 0 in zigzag order.
    ///
    /// - `component`: Component index (0=Y, 1=Cb, 2=Cr for color)
    /// - `bx`, `by`: Block coordinates
    /// - `dc_value`: The DC coefficient value
    pub fn store_dc(&mut self, component: usize, bx: usize, by: usize, dc_value: i32) {
        if component >= self.num_components {
            return;
        }

        let blocks_x = self.blocks_x[component];
        let block_idx = by * blocks_x + bx;
        let offset = block_idx * 64;

        if offset >= self.coefficients[component].len() {
            return;
        }

        // DC coefficient is at index 0 in zigzag order
        self.coefficients[component][offset] = dc_value.clamp(-32768, 32767) as i16;
    }

    /// Check if we have stored any coefficients
    pub fn has_stored_coefficients(&self) -> bool {
        self.coefficients.iter().any(|c| c.iter().any(|&v| v != 0))
    }
}

impl JpegReconstructionData {
    /// Create a simple representation showing that jbrd data is present.
    /// This is used when full parsing isn't required.
    #[allow(dead_code)]
    pub fn from_raw(data: &[u8]) -> Result<Self> {
        if data.is_empty() {
            return Err(Error::InvalidJpegReconstructionData);
        }
        Ok(JpegReconstructionData {
            width: 1, // Mark as having data
            height: data.len() as u32,
            ..Default::default()
        })
    }

    /// Parse jbrd box data into JPEG reconstruction data.
    ///
    /// The jbrd box uses JXL's Bundle format with an "all_default" check,
    /// followed by marker-by-marker encoding as per libjxl's JPEGData::VisitFields.
    #[allow(clippy::field_reassign_with_default)]
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.is_empty() {
            return Err(Error::InvalidJpegReconstructionData);
        }

        let mut reader = BitReader::new(data);
        let mut result = JpegReconstructionData::default();

        // NOTE: JPEGData does NOT use Bundle's AllDefault pattern!
        // There is no all_default bit. Parsing starts directly with is_gray.

        // Parse following libjxl's JPEGData::VisitFields order exactly:
        // 1. is_gray (Bool with default=false) at bit 0
        result.is_gray = reader.read(1)? != 0;

        // 2. marker_order - read markers via VisitMarker until EOI (0xD9)
        // Each marker is encoded as 6 bits: value = bits + 0xC0
        // SOI (0xD8) is implicit and not included in marker_order
        result.marker_order = Vec::new();
        let mut marker_count = 0;
        loop {
            let marker_bits = reader.read(6)? as u8;
            let marker = marker_bits.wrapping_add(0xC0);
            result.marker_order.push(marker);
            marker_count += 1;
            if marker == 0xD9 {
                // EOI marker - stop reading
                break;
            }
            if marker_count > 16384 {
                // Too many markers - likely parsing error
                return Err(Error::InvalidJpegReconstructionData);
            }
        }

        // Count APP and COM markers from the marker_order
        let num_app_markers = result.marker_order.iter()
            .filter(|&&m| (0xE0..=0xEF).contains(&m))
            .count();
        let num_com_markers = result.marker_order.iter()
            .filter(|&&m| m == 0xFE)
            .count();

        // 3. For each APP marker: read type AND length together
        // libjxl loops: for each app { read type; read 16-bit length }
        result.app_marker_types = Vec::with_capacity(num_app_markers);
        result.app_data = Vec::with_capacity(num_app_markers);
        for i in 0..num_app_markers {
            // Type: U32(Val(0), Val(1), BitsOffset(1, 2), BitsOffset(2, 4))
            let bits_before = reader.total_bits_read();
            let marker_type = Self::read_u32_app_type(&mut reader)?;
            result.app_marker_types.push(AppMarkerType::try_from(marker_type as u8)?);

            // Length: 16 bits (stored as length - 1)
            let len = reader.read(16)? as usize + 1;
            let _ = (i, marker_type, len, bits_before); // silence unused warnings

            // Initialize empty app_data with correct size (will be filled later from Brotli data)
            result.app_data.push(vec![0u8; len]);
        }

        // 4. For each COM marker: read 16-bit length
        result.com_data = Vec::with_capacity(num_com_markers);
        for i in 0..num_com_markers {
            let bits_before = reader.total_bits_read();
            let len = reader.read(16)? as usize + 1;
            let _ = (i, len, bits_before); // silence unused warnings
            result.com_data.push(vec![0u8; len]);
        }

        // 5. num_quant_tables - U32(Val(1), Val(2), Val(3), Val(4))
        let bits_before = reader.total_bits_read();
        let num_quant_tables = Self::read_u32_quant(&mut reader)? as usize;
        let _ = bits_before; // silence unused warning
        // NOTE: Quant table VALUES are NOT stored in jbrd - only metadata.
        // The actual 64 values come from the VarDCT codestream during decoding.
        result.quant_tables = Vec::with_capacity(num_quant_tables);
        for q in 0..num_quant_tables {
            let mut table = JpegQuantTable::default();
            table.precision = reader.read(1)? as u8;
            table.index = reader.read(2)? as u8;
            table.is_last = reader.read(1)? != 0;
            let _ = q; // silence unused warning
            // Values are filled later from codestream, not from jbrd
            result.quant_tables.push(table);
        }

        // 6. component_type (2 bits) then components
        // libjxl enum: kGray=0, kYCbCr=1, kRGB=2, kCustom=3
        let _bits_before = reader.total_bits_read();
        let component_type = reader.read(2)? as u8;

        // Determine number of components
        let num_components = match component_type {
            0 => 1,  // kGray
            1 | 2 => 3,  // kYCbCr or kRGB
            3 => {  // kCustom
                let n = Self::read_u32_general(&mut reader)? as usize;
                if n != 1 && n != 3 {
                    return Err(Error::InvalidJpegReconstructionData);
                }
                n
            }
            _ => return Err(Error::InvalidJpegReconstructionData),
        };

        // For kCustom, read 8-bit IDs
        let mut custom_ids = Vec::new();
        if component_type == 3 {
            for _ in 0..num_components {
                custom_ids.push(reader.read(8)? as u8);
            }
        }

        // Build components - only quant_idx is read from bitstream
        // Sampling factors are NOT stored in jbrd, they default to 1
        // and are determined from the JPEG frame header during reconstruction
        result.components = Vec::with_capacity(num_components);
        for i in 0..num_components {
            // Determine component ID based on type
            let id = match component_type {
                0 => 1,  // kGray
                1 => (i + 1) as u8,  // kYCbCr: 1, 2, 3
                2 => [b'R', b'G', b'B'][i],  // kRGB
                3 => custom_ids[i],  // kCustom
                _ => return Err(Error::InvalidJpegReconstructionData),
            };

            // Read quant index only (2 bits)
            let quant_idx = reader.read(2)? as u8;

            let component = JpegComponent {
                id,
                h_samp_factor: 1,  // Default, set from JPEG header during reconstruction
                v_samp_factor: 1,  // Default, set from JPEG header during reconstruction
                quant_idx,
            };
            let _ = i; // silence unused warning
            result.components.push(component);
        }

        // 7. huffman_code - U32(Val(4), BitsOffset(3, 2), BitsOffset(4, 10), BitsOffset(6, 26))
        let _bits_before = reader.total_bits_read();
        let num_huffman_codes = Self::read_u32_huffman(&mut reader)? as usize;
        result.huffman_codes = Vec::with_capacity(num_huffman_codes);
        for h in 0..num_huffman_codes {
            let mut code = JpegHuffmanCode::default();
            // libjxl: is_ac (Bool), id (2 bits)
            let is_ac = reader.read(1)? != 0;
            let id = reader.read(2)? as u8;
            code.slot_id = id;  // slot_id is just the 2-bit id, not combined with table_class
            code.table_class = if is_ac { 1 } else { 0 };
            code.is_last = reader.read(1)? != 0;

            // libjxl: 17 count values (indices 0-16), each using U32(Val(0), Val(1), BitsOffset(3,2), Bits(8))
            // Looking at comparison with djxl output:
            // Index 0 in jbrd is for 0-bit codes (always 0) and should be skipped
            // Indices 1-16 map to DHT counts[0-15] for bit lengths 1-16
            let mut num_symbols = 0usize;
            for j in 0..17 {
                let count = Self::read_u32_huffman_count(&mut reader)? as u8;
                if j > 0 && j <= 16 {
                    code.counts[j - 1] = count;  // jbrd index j -> DHT counts[j-1]
                    num_symbols += count as usize;
                }
            }

            // If no symbols, skip values (represents empty DHT marker)
            if num_symbols == 0 {
                result.huffman_codes.push(code);
                continue;
            }

            // libjxl: values use U32(Bits(2), BitsOffset(2, 4), BitsOffset(4, 8), BitsOffset(8, 1))
            code.values = Vec::with_capacity(num_symbols);
            for _ in 0..num_symbols {
                let val = Self::read_u32_huffman_value(&mut reader)?;
                if val > 256 {
                    return Err(Error::InvalidJpegReconstructionData);
                }
                code.values.push(val as u16);
            }
            let _ = h; // silence unused warning
            result.huffman_codes.push(code);
        }

        // 8. scan_info - num_scans is NOT serialized, it's counted from marker_order
        // Count DA (0xDA) markers to determine num_scans
        let num_scans = result.marker_order.iter().filter(|&&m| m == 0xDA).count();
        result.scan_info = Vec::with_capacity(num_scans);

        // First loop: read scan metadata (following libjxl order)
        for s in 0..num_scans {
            let mut scan = JpegScanInfo::default();
            let bits_before = reader.total_bits_read();

            // num_components: U32(Val(1), Val(2), Val(3), Val(4))
            scan.num_components = Self::read_u32_num_components(&mut reader)? as u8;

            // Ss, Se, Al, Ah come BEFORE component info in libjxl
            scan.ss = reader.read(6)? as u8;
            scan.se = reader.read(6)? as u8;
            scan.al = reader.read(4)? as u8;
            scan.ah = reader.read(4)? as u8;
            let _ = (s, bits_before); // silence unused warnings

            // Component info: comp_idx, ac_tbl_idx, dc_tbl_idx (note: AC before DC!)
            for i in 0..scan.num_components as usize {
                scan.component_idx[i] = reader.read(2)? as u8;
                scan.ac_tbl_idx[i] = reader.read(2)? as u8;
                scan.dc_tbl_idx[i] = reader.read(2)? as u8;
            }

            // last_needed_pass: U32(Val(0), Val(1), Val(2), BitsOffset(3, 3))
            let _last_needed_pass = Self::read_u32_last_pass(&mut reader)?;

            result.scan_info.push(scan);
        }

        // Second loop: reset_points (separate from scan metadata in libjxl)
        for s in 0..num_scans {
            let num_reset_points = Self::read_u32_reset_count(&mut reader)? as usize;
            result.scan_info[s].reset_points = Vec::with_capacity(num_reset_points);
            let mut last_block_idx: i32 = -1;
            for _ in 0..num_reset_points {
                let delta = Self::read_u32_block_idx(&mut reader)?;
                let block_idx = (last_block_idx + 1) as u32 + delta;
                last_block_idx = block_idx as i32;
                result.scan_info[s].reset_points.push(JpegResetPoint { mcu: block_idx, last_dc: Vec::new() });
            }
        }

        // Third loop: extra_zero_runs (also separate)
        for s in 0..num_scans {
            let num_extra_zeros = Self::read_u32_reset_count(&mut reader)? as usize;
            result.scan_info[s].extra_zero_runs = Vec::with_capacity(num_extra_zeros);
            let mut last_block_idx: i32 = -1;
            for _ in 0..num_extra_zeros {
                let num_zeros = Self::read_u32_extra_zeros(&mut reader)?;
                let delta = Self::read_u32_block_idx(&mut reader)?;
                let block_idx = (last_block_idx + 1) as u32 + delta;
                last_block_idx = block_idx as i32;
                result.scan_info[s].extra_zero_runs.push((block_idx, num_zeros));
            }
        }

        // 9. restart_interval - only read if has_dri marker (DRI = 0xDD)
        // Check if any marker is DRI (0xDD)
        let has_dri = result.marker_order.iter().any(|&m| m == 0xDD);
        if has_dri {
            result.restart_interval = reader.read(16)? as u32;
        } else {
            result.restart_interval = 0;
        }

        // 10. inter_marker_data sizes
        // In libjxl: num_intermarker counts fake 0xff markers used for intermarker data
        // We count these from marker_order (each 0xFF entry marks intermarker data)
        let num_inter_marker = result.marker_order.iter().filter(|&&m| m == 0xFF).count();
        let mut inter_marker_sizes = Vec::with_capacity(num_inter_marker);
        for _ in 0..num_inter_marker {
            // Each size is Bits(16)
            let size = reader.read(16)? as usize;
            inter_marker_sizes.push(size);
        }

        // 11. tail_data size - U32(Val(0), BitsOffset(8, 1), BitsOffset(16, 257), BitsOffset(22, 65793))
        let tail_size = Self::read_u32_tail(&mut reader)? as usize;

        // 12. padding_bits - has_zero_padding_bit then conditional 24-bit length
        result.has_zero_padding_bit = reader.read(1)? != 0;
        let padding_bits_size = if result.has_zero_padding_bit {
            // libjxl uses Bits(24) for padding_bits length
            reader.read(24)? as usize
        } else {
            0
        };

        // Note: width and height are NOT stored in jbrd - they come from the codestream
        // We'll set them from the decoded image later

        // Skip to byte boundary for Brotli-compressed data
        reader.jump_to_byte_boundary()?;

        // Get remaining compressed data
        let remaining_pos = reader.total_bits_read() / 8;
        let compressed_data = if remaining_pos < data.len() {
            &data[remaining_pos..]
        } else {
            &[]
        };

        // Extract COM lengths from pre-sized vectors (set up during parsing)
        let com_lengths: Vec<usize> = result.com_data.iter().map(|v| v.len()).collect();

        // Decompress marker data using Brotli
        Self::decompress_marker_data_v2(
            &mut result,
            compressed_data,
            com_lengths,
            inter_marker_sizes,
            tail_size,
            padding_bits_size,
        )?;

        Ok(result)
    }

    /// Decompress marker data using libjxl's format (v2).
    /// APP data lengths come from the Brotli stream itself.
    /// COM lengths were read from the bitstream.
    fn decompress_marker_data_v2(
        result: &mut JpegReconstructionData,
        compressed_data: &[u8],
        com_lengths: Vec<usize>,
        inter_marker_sizes: Vec<usize>,
        tail_size: usize,
        padding_bits_size: usize,
    ) -> Result<()> {
        let num_app_markers = result.app_marker_types.len();
        let num_com_markers = com_lengths.len();

        if compressed_data.is_empty() {
            result.app_data = vec![Vec::new(); num_app_markers];
            result.com_data = vec![Vec::new(); num_com_markers];
            result.inter_marker_data = inter_marker_sizes.into_iter().map(|_| Vec::new()).collect();
            result.tail_data = Vec::new();
            result.padding_bits = Vec::new();
            return Ok(());
        }

        // Decompress all data at once
        let decompressed = match Self::brotli_decompress(compressed_data) {
            Ok(d) => d,
            Err(e) => {
                return Err(e);
            }
        };
        let mut offset = 0;

        // Read APP marker data from Brotli stream
        // IMPORTANT: Only "Unknown" type APP markers have data in Brotli stream
        // ICC/EXIF/XMP data comes from codestream, not jbrd
        let app_sizes: Vec<usize> = result.app_data.iter().map(|v| v.len()).collect();
        result.app_data = Vec::with_capacity(num_app_markers);
        for (i, size) in app_sizes.iter().enumerate() {
            let marker_type = result.app_marker_types.get(i).copied().unwrap_or_default();

            let final_data = match marker_type {
                AppMarkerType::Unknown => {
                    // Unknown type: data is in Brotli stream
                    if offset + *size > decompressed.len() {
                        return Err(Error::InvalidJpegReconstructionData);
                    }
                    let marker_data = decompressed[offset..offset + *size].to_vec();
                    offset += *size;
                    marker_data
                }
                AppMarkerType::Icc | AppMarkerType::Exif | AppMarkerType::Xmp => {
                    // ICC/EXIF/XMP: data comes from codestream, placeholder here
                    // These will be filled in later from the decoded image's metadata
                    let _ = (i, marker_type); // silence unused warnings
                    vec![0u8; *size]
                }
            };
            result.app_data.push(final_data);
        }

        // Read COM marker data using pre-computed lengths
        result.com_data = Vec::with_capacity(num_com_markers);
        for size in com_lengths {
            if offset + size > decompressed.len() {
                return Err(Error::InvalidJpegReconstructionData);
            }
            result.com_data.push(decompressed[offset..offset + size].to_vec());
            offset += size;
        }

        // Read inter-marker data
        result.inter_marker_data = Vec::with_capacity(inter_marker_sizes.len());
        for size in inter_marker_sizes {
            if size == 0 {
                result.inter_marker_data.push(Vec::new());
            } else {
                if offset + size > decompressed.len() {
                    return Err(Error::InvalidJpegReconstructionData);
                }
                result.inter_marker_data.push(decompressed[offset..offset + size].to_vec());
                offset += size;
            }
        }

        // Read tail data
        if tail_size > 0 {
            if offset + tail_size > decompressed.len() {
                return Err(Error::InvalidJpegReconstructionData);
            }
            result.tail_data = decompressed[offset..offset + tail_size].to_vec();
            offset += tail_size;
        }

        // Read padding bits
        if padding_bits_size > 0 {
            if offset + padding_bits_size > decompressed.len() {
                return Err(Error::InvalidJpegReconstructionData);
            }
            result.padding_bits = decompressed[offset..offset + padding_bits_size].to_vec();
        }

        Ok(())
    }

    /// Decompress data using Brotli.
    fn brotli_decompress(data: &[u8]) -> Result<Vec<u8>> {
        use brotli::Decompressor;
        use std::io::Read;

        let mut decompressor = Decompressor::new(data, 4096);
        let mut decompressed = Vec::new();
        decompressor
            .read_to_end(&mut decompressed)
            .map_err(|_| Error::InvalidJpegReconstructionData)?;
        Ok(decompressed)
    }

    /// Read U32(Val(0), Val(1), BitsOffset(1, 2), BitsOffset(2, 4)) for app_marker_type
    fn read_u32_app_type(reader: &mut BitReader) -> Result<u32> {
        let selector = reader.read(2)?;
        match selector {
            0 => Ok(0),
            1 => Ok(1),
            2 => Ok(reader.read(1)? as u32 + 2),
            3 => Ok(reader.read(2)? as u32 + 4),
            _ => unreachable!(),
        }
    }

    /// Read U32(Val(1), Val(2), Val(3), Val(4)) for num_quant_tables
    fn read_u32_quant(reader: &mut BitReader) -> Result<u32> {
        let selector = reader.read(2)?;
        match selector {
            0 => Ok(1),
            1 => Ok(2),
            2 => Ok(3),
            3 => Ok(4),
            _ => unreachable!(),
        }
    }

    /// Read U32(Val(4), BitsOffset(3, 2), BitsOffset(4, 10), BitsOffset(6, 26)) for num_huffman
    fn read_u32_huffman(reader: &mut BitReader) -> Result<u32> {
        let selector = reader.read(2)?;
        match selector {
            0 => Ok(4),
            1 => Ok(reader.read(3)? as u32 + 2),
            2 => Ok(reader.read(4)? as u32 + 10),
            3 => Ok(reader.read(6)? as u32 + 26),
            _ => unreachable!(),
        }
    }

    /// Read U32(Val(0), Val(1), BitsOffset(3, 2), Bits(8)) for Huffman counts
    fn read_u32_huffman_count(reader: &mut BitReader) -> Result<u32> {
        let selector = reader.read(2)?;
        match selector {
            0 => Ok(0),
            1 => Ok(1),
            2 => Ok(reader.read(3)? as u32 + 2),
            3 => Ok(reader.read(8)? as u32),
            _ => unreachable!(),
        }
    }

    /// Read U32(Bits(2), BitsOffset(2, 4), BitsOffset(4, 8), BitsOffset(8, 1)) for Huffman values
    fn read_u32_huffman_value(reader: &mut BitReader) -> Result<u32> {
        let selector = reader.read(2)?;
        match selector {
            0 => Ok(reader.read(2)? as u32),
            1 => Ok(reader.read(2)? as u32 + 4),
            2 => Ok(reader.read(4)? as u32 + 8),
            3 => Ok(reader.read(8)? as u32 + 1),
            _ => unreachable!(),
        }
    }


    /// Read U32(Val(1), Val(2), Val(3), Val(4)) for num_components
    fn read_u32_num_components(reader: &mut BitReader) -> Result<u32> {
        let selector = reader.read(2)?;
        match selector {
            0 => Ok(1),
            1 => Ok(2),
            2 => Ok(3),
            3 => Ok(4),
            _ => unreachable!(),
        }
    }

    /// Read U32(Val(0), Val(1), Val(2), BitsOffset(3, 3)) for last_needed_pass
    fn read_u32_last_pass(reader: &mut BitReader) -> Result<u32> {
        let selector = reader.read(2)?;
        match selector {
            0 => Ok(0),
            1 => Ok(1),
            2 => Ok(2),
            3 => Ok(reader.read(3)? as u32 + 3),
            _ => unreachable!(),
        }
    }

    /// Read U32(Val(0), BitsOffset(2, 1), BitsOffset(4, 4), BitsOffset(16, 20)) for reset point count
    fn read_u32_reset_count(reader: &mut BitReader) -> Result<u32> {
        let selector = reader.read(2)?;
        match selector {
            0 => Ok(0),
            1 => Ok(reader.read(2)? as u32 + 1),
            2 => Ok(reader.read(4)? as u32 + 4),
            3 => Ok(reader.read(16)? as u32 + 20),
            _ => unreachable!(),
        }
    }

    /// Read U32(Val(0), BitsOffset(3, 1), BitsOffset(5, 9), BitsOffset(28, 41)) for block index delta
    fn read_u32_block_idx(reader: &mut BitReader) -> Result<u32> {
        let selector = reader.read(2)?;
        match selector {
            0 => Ok(0),
            1 => Ok(reader.read(3)? as u32 + 1),
            2 => Ok(reader.read(5)? as u32 + 9),
            3 => Ok(reader.read(28)? as u32 + 41),
            _ => unreachable!(),
        }
    }

    /// Read U32(Val(1), BitsOffset(2, 2), BitsOffset(4, 5), BitsOffset(8, 20)) for extra zero runs
    fn read_u32_extra_zeros(reader: &mut BitReader) -> Result<u32> {
        let selector = reader.read(2)?;
        match selector {
            0 => Ok(1),
            1 => Ok(reader.read(2)? as u32 + 2),
            2 => Ok(reader.read(4)? as u32 + 5),
            3 => Ok(reader.read(8)? as u32 + 20),
            _ => unreachable!(),
        }
    }

    /// Read U32(Val(0), Bits(4), BitsOffset(8, 16), Bits(16)) - general purpose
    fn read_u32_general(reader: &mut BitReader) -> Result<u32> {
        let selector = reader.read(2)?;
        match selector {
            0 => Ok(0),
            1 => Ok(reader.read(4)? as u32),
            2 => Ok(reader.read(8)? as u32 + 16),
            3 => Ok(reader.read(16)? as u32),
            _ => unreachable!(),
        }
    }

    /// Read U32(Val(0), BitsOffset(8, 1), BitsOffset(16, 257), BitsOffset(22, 65793)) for tail_data_len
    fn read_u32_tail(reader: &mut BitReader) -> Result<u32> {
        let selector = reader.read(2)?;
        match selector {
            0 => Ok(0),
            1 => Ok(reader.read(8)? as u32 + 1),
            2 => Ok(reader.read(16)? as u32 + 257),
            3 => Ok(reader.read(22)? as u32 + 65793),
            _ => unreachable!(),
        }
    }


    /// Check if this structure contains valid JPEG reconstruction data.
    /// Note: width/height come from the codestream, so we don't check them here.
    pub fn is_valid(&self) -> bool {
        !self.components.is_empty() && !self.marker_order.is_empty()
    }

    pub fn update_quant_tables_from_raw(
        &mut self,
        qtable: &[i32],
        qtable_den: f32,
        do_ycbcr: bool,
    ) -> Result<()> {
        let expected_den = 1.0 / (8.0 * 255.0);
        if (qtable_den - expected_den).abs() > 1e-8 {
            return Err(Error::InvalidJpegReconstructionData);
        }
        if qtable.len() < 3 * 64 {
            return Err(Error::InvalidJpegReconstructionData);
        }

        let num_components = self.components.len();
        let is_gray = self.is_gray || num_components == 1;
        let jpeg_c_map = if is_gray {
            [0usize, 0, 0]
        } else if do_ycbcr {
            [1usize, 0, 2]
        } else {
            [0usize, 1, 2]
        };

        let mut qt_set = 0u32;
        for c in 0..num_components.min(3) {
            let quant_c = if is_gray { 1 } else { c };
            let mapped_comp = jpeg_c_map[c];
            if mapped_comp >= self.components.len() {
                return Err(Error::InvalidJpegReconstructionData);
            }
            let qpos = self.components[mapped_comp].quant_idx as usize;
            if qpos >= self.quant_tables.len() {
                return Err(Error::InvalidJpegReconstructionData);
            }
            qt_set |= 1u32 << qpos;

            for x in 0..8 {
                for y in 0..8 {
                    let src = qtable[quant_c * 64 + y * 8 + x];
                    if src <= 0 || src > u16::MAX as i32 {
                        return Err(Error::InvalidJpegReconstructionData);
                    }
                    self.quant_tables[qpos].values[x * 8 + y] = src as u16;
                }
            }
        }

        for i in 0..self.quant_tables.len() {
            if (qt_set & (1u32 << i)) != 0 {
                continue;
            }
            if i == 0 {
                return Err(Error::InvalidJpegReconstructionData);
            }
            self.quant_tables[i].values = self.quant_tables[i - 1].values;
        }

        Ok(())
    }

    pub fn fill_icc_app_markers(&mut self, icc: &[u8]) -> Result<()> {
        let mut icc_pos = 0usize;
        let mut num_icc = 0u8;
        for (marker_type, marker) in self
            .app_marker_types
            .iter()
            .copied()
            .zip(self.app_data.iter_mut())
        {
            if marker_type != AppMarkerType::Icc {
                continue;
            }
            if marker.len() < 17 {
                return Err(Error::InvalidJpegReconstructionData);
            }

            let size_minus_1 = marker.len() - 1;
            marker[0] = 0xE2;
            marker[1] = (size_minus_1 >> 8) as u8;
            marker[2] = (size_minus_1 & 0xFF) as u8;
            marker[3..15].copy_from_slice(b"ICC_PROFILE\0");

            num_icc = num_icc.saturating_add(1);
            marker[15] = num_icc;

            let payload_len = marker.len() - 17;
            if icc_pos + payload_len > icc.len() {
                return Err(Error::InvalidJpegReconstructionData);
            }
            marker[17..17 + payload_len].copy_from_slice(&icc[icc_pos..icc_pos + payload_len]);
            icc_pos += payload_len;
        }

        if num_icc > 0 {
            for (marker_type, marker) in self
                .app_marker_types
                .iter()
                .copied()
                .zip(self.app_data.iter_mut())
            {
                if marker_type == AppMarkerType::Icc && marker.len() >= 17 {
                    marker[16] = num_icc;
                }
            }
        }

        if icc_pos != icc.len() && icc_pos != 0 {
            return Err(Error::InvalidJpegReconstructionData);
        }

        Ok(())
    }

    /// Populate with default JPEG tables for the given dimensions.
    /// Used when is_all_default is true and we need to create JPEG from decoded pixels.
    pub fn populate_defaults(&mut self, width: u32, height: u32, is_gray: bool) {
        self.width = width;
        self.height = height;
        self.is_gray = is_gray;

        // Create standard quantization tables
        let mut lum_quant = JpegQuantTable::default();
        lum_quant.index = 0;
        lum_quant.is_last = is_gray;
        lum_quant.values = STD_LUMINANCE_QUANT_TBL;
        self.quant_tables.push(lum_quant);

        if !is_gray {
            let mut chrom_quant = JpegQuantTable::default();
            chrom_quant.index = 1;
            chrom_quant.is_last = true;
            chrom_quant.values = STD_CHROMINANCE_QUANT_TBL;
            self.quant_tables.push(chrom_quant);
        }

        // Create standard Huffman codes
        // DC Luminance
        let mut dc_lum = JpegHuffmanCode::default();
        dc_lum.table_class = 0;
        dc_lum.slot_id = 0;
        dc_lum.is_last = false;
        dc_lum.counts = STD_DC_LUMINANCE_NRCODES;
        dc_lum.values = STD_DC_LUMINANCE_VALUES.iter().map(|&v| v as u16).collect();
        self.huffman_codes.push(dc_lum);

        // AC Luminance
        let mut ac_lum = JpegHuffmanCode::default();
        ac_lum.table_class = 1;
        ac_lum.slot_id = 0;
        ac_lum.is_last = is_gray;
        ac_lum.counts = STD_AC_LUMINANCE_NRCODES;
        ac_lum.values = STD_AC_LUMINANCE_VALUES.iter().map(|&v| v as u16).collect();
        self.huffman_codes.push(ac_lum);

        if !is_gray {
            // DC Chrominance
            let mut dc_chrom = JpegHuffmanCode::default();
            dc_chrom.table_class = 0;
            dc_chrom.slot_id = 1;
            dc_chrom.is_last = false;
            dc_chrom.counts = STD_DC_CHROMINANCE_NRCODES;
            dc_chrom.values = STD_DC_CHROMINANCE_VALUES.iter().map(|&v| v as u16).collect();
            self.huffman_codes.push(dc_chrom);

            // AC Chrominance
            let mut ac_chrom = JpegHuffmanCode::default();
            ac_chrom.table_class = 1;
            ac_chrom.slot_id = 1;
            ac_chrom.is_last = true;
            ac_chrom.counts = STD_AC_CHROMINANCE_NRCODES;
            ac_chrom.values = STD_AC_CHROMINANCE_VALUES.iter().map(|&v| v as u16).collect();
            self.huffman_codes.push(ac_chrom);
        }

        // Create components
        if is_gray {
            self.components.push(JpegComponent {
                id: 1,
                h_samp_factor: 1,
                v_samp_factor: 1,
                quant_idx: 0,
            });
        } else {
            // YCbCr with 4:4:4 sampling (no subsampling for simplicity)
            self.components.push(JpegComponent {
                id: 1,
                h_samp_factor: 1,
                v_samp_factor: 1,
                quant_idx: 0,
            });
            self.components.push(JpegComponent {
                id: 2,
                h_samp_factor: 1,
                v_samp_factor: 1,
                quant_idx: 1,
            });
            self.components.push(JpegComponent {
                id: 3,
                h_samp_factor: 1,
                v_samp_factor: 1,
                quant_idx: 1,
            });
        }

        // Create scan info
        let mut scan = JpegScanInfo::default();
        scan.num_components = if is_gray { 1 } else { 3 };
        for i in 0..scan.num_components as usize {
            scan.component_idx[i] = i as u8;
            scan.dc_tbl_idx[i] = if i == 0 { 0 } else { 1 };
            scan.ac_tbl_idx[i] = if i == 0 { 0 } else { 1 };
        }
        scan.ss = 0;
        scan.se = 63;
        scan.ah = 0;
        scan.al = 0;
        self.scan_info.push(scan);

        // Set standard marker order: DQT, SOF0, DHT, SOS
        self.marker_order = vec![0xDB, 0xC0, 0xC4, 0xDA];
    }

    /// Reconstruct the original JPEG file from this data and the decoded DCT coefficients.
    ///
    /// This method produces a bit-exact reconstruction of the original JPEG file.
    /// The `coefficients` parameter should contain the DCT coefficients for each component,
    /// in the order they appear in `self.components`.
    pub fn reconstruct_jpeg(&self, coefficients: &[Vec<i16>]) -> Result<Vec<u8>> {
        let mut writer = JpegWriter::new();
        writer.write_jpeg(self, coefficients)
    }

    /// Reconstruct the original JPEG file using stored DCT coefficients.
    ///
    /// This method uses the DCT coefficients stored in `self.dct_coefficients`
    /// for bit-exact reconstruction. Returns an error if no coefficients are stored.
    pub fn reconstruct_jpeg_from_stored(&self) -> Result<Vec<u8>> {
        let coeffs = self.dct_coefficients.as_ref()
            .ok_or(Error::InvalidJpegReconstructionData)?;

        if coeffs.coefficients.is_empty() {
            return Err(Error::InvalidJpegReconstructionData);
        }

        // Create a modified copy with dimensions from the coefficients if needed
        let mut data = self.clone();
        if data.width == 0 || data.height == 0 {
            data.width = coeffs.width as u32;
            data.height = coeffs.height as u32;
        }

        let mut writer = JpegWriter::new();
        writer.write_jpeg(&data, &coeffs.coefficients)
    }

    /// Check if this structure has stored DCT coefficients for reconstruction.
    pub fn has_stored_coefficients(&self) -> bool {
        self.dct_coefficients.as_ref()
            .is_some_and(|c| !c.coefficients.is_empty())
    }

    /// Encode pixel data to JPEG format.
    ///
    /// Takes grayscale or RGB pixel data (as f32 values in 0.0-1.0 range) and encodes to JPEG.
    /// For grayscale, pixels should be a single slice.
    /// For RGB, pixels should be interleaved RGB values.
    pub fn encode_from_pixels(&self, pixels: &[f32], width: usize, height: usize) -> Result<Vec<u8>> {
        let mut encoder = JpegEncoder::new(self);
        encoder.encode(pixels, width, height)
    }
}

/// Natural order (zigzag) for JPEG quantization tables.
const JPEG_NATURAL_ORDER: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// Standard JPEG luminance quantization table.
const STD_LUMINANCE_QUANT_TBL: [u16; 64] = [
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
];

/// Standard JPEG chrominance quantization table.
const STD_CHROMINANCE_QUANT_TBL: [u16; 64] = [
    17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
];

/// Standard JPEG DC luminance Huffman table - bit counts.
const STD_DC_LUMINANCE_NRCODES: [u8; 16] = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
/// Standard JPEG DC luminance Huffman table - values.
const STD_DC_LUMINANCE_VALUES: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

/// Standard JPEG DC chrominance Huffman table - bit counts.
const STD_DC_CHROMINANCE_NRCODES: [u8; 16] = [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0];
/// Standard JPEG DC chrominance Huffman table - values.
const STD_DC_CHROMINANCE_VALUES: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

/// Standard JPEG AC luminance Huffman table - bit counts.
const STD_AC_LUMINANCE_NRCODES: [u8; 16] = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 125];
/// Standard JPEG AC luminance Huffman table - values.
const STD_AC_LUMINANCE_VALUES: [u8; 162] = [
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
    0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
    0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
    0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa,
];

/// Standard JPEG AC chrominance Huffman table - bit counts.
const STD_AC_CHROMINANCE_NRCODES: [u8; 16] = [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 119];
/// Standard JPEG AC chrominance Huffman table - values.
const STD_AC_CHROMINANCE_VALUES: [u8; 162] = [
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
    0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
    0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
    0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
    0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
    0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
    0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa,
];

/// JPEG bitstream writer for reconstructing JPEG files.
struct JpegWriter {
    output: Vec<u8>,
    /// Bit buffer for entropy-coded data
    bit_buffer: u32,
    /// Number of bits in the buffer
    bit_count: u8,
    /// Huffman encoding tables built from jbrd data
    huff_tables: Vec<[(u16, u8); 256]>,
}

impl JpegWriter {
    fn new() -> Self {
        Self {
            output: Vec::new(),
            bit_buffer: 0,
            bit_count: 0,
            huff_tables: Vec::new(),
        }
    }

    /// Build Huffman encoding tables from jbrd data.
    /// Tables are stored in fixed order:
    /// [0] = DC luminance (class=0, slot=0)
    /// [1] = DC chrominance (class=0, slot=1)
    /// [2] = AC luminance (class=1, slot=0)
    /// [3] = AC chrominance (class=1, slot=1)
    fn build_huffman_tables(&mut self, data: &JpegReconstructionData) {
        // Initialize with empty tables
        self.huff_tables = vec![[(0u16, 0u8); 256]; 4];

        // Fill tables based on class and slot_id
        for code in &data.huffman_codes {
            let idx = match (code.table_class, code.slot_id) {
                (0, 0) => 0, // DC luminance
                (0, 1) => 1, // DC chrominance
                (1, 0) => 2, // AC luminance
                (1, 1) => 3, // AC chrominance
                _ => continue,
            };
            self.huff_tables[idx] = Self::build_single_huffman_table(&code.counts, &code.values);
        }
    }

    fn build_single_huffman_table(counts: &[u8; 16], values: &[u16]) -> [(u16, u8); 256] {
        let mut table = [(0u16, 0u8); 256];
        let mut code = 0u32;  // Use u32 to avoid overflow
        let mut val_idx = 0;
        let values_len = match values.last() {
            Some(&256) => values.len().saturating_sub(1),
            _ => values.len(),
        };

        for (bits, &count) in counts.iter().enumerate() {
            let bits = bits as u8 + 1;
            for _ in 0..count {
                if val_idx < values_len && code <= 0xFFFF {
                    let value = values[val_idx] as usize;
                    if value < table.len() {
                        table[value] = (code as u16, bits);
                    }
                    val_idx += 1;
                }
                code += 1;
            }
            code <<= 1;
        }
        table
    }

    /// Write bits to the output with byte stuffing
    fn write_bits(&mut self, value: u16, bits: u8) {
        self.bit_buffer = (self.bit_buffer << bits) | (value as u32);
        self.bit_count += bits;

        while self.bit_count >= 8 {
            self.bit_count -= 8;
            let byte = ((self.bit_buffer >> self.bit_count) & 0xFF) as u8;
            self.output.push(byte);
            // Byte stuffing for 0xFF
            if byte == 0xFF {
                self.output.push(0x00);
            }
        }
    }

    /// Flush remaining bits, padding with 1s
    fn flush_bits(&mut self) {
        if self.bit_count > 0 {
            let remaining = 8 - self.bit_count;
            self.bit_buffer = (self.bit_buffer << remaining) | ((1 << remaining) - 1);
            let byte = (self.bit_buffer & 0xFF) as u8;
            self.output.push(byte);
            if byte == 0xFF {
                self.output.push(0x00);
            }
            self.bit_count = 0;
            self.bit_buffer = 0;
        }
    }

    /// Get the number of bits needed to represent a value and its encoded form
    fn get_value_bits(value: i16) -> (u8, i16) {
        if value == 0 {
            return (0, 0);
        }
        let abs_val = value.unsigned_abs();
        let size = 16 - abs_val.leading_zeros() as u8;
        let encoded = if value < 0 {
            value + (1 << size) - 1
        } else {
            value
        };
        (size, encoded)
    }

    /// Write a complete JPEG file from reconstruction data.
    fn write_jpeg(&mut self, data: &JpegReconstructionData, coefficients: &[Vec<i16>]) -> Result<Vec<u8>> {
        // Build Huffman tables from jbrd data
        self.build_huffman_tables(data);

        // Write SOI marker
        self.write_marker(0xD8);

        // Process markers in order
        let mut app_idx = 0;
        let mut com_idx = 0;
        let mut dqt_idx = 0;
        let mut dht_idx = 0;
        let mut scan_idx = 0;
        let mut inter_marker_idx = 0;

        for &marker in &data.marker_order {
            // Write any inter-marker data before this marker
            if inter_marker_idx < data.inter_marker_data.len() {
                let inter_data = &data.inter_marker_data[inter_marker_idx];
                if !inter_data.is_empty() {
                    self.output.extend_from_slice(inter_data);
                }
                inter_marker_idx += 1;
            }

            match marker {
                0xE0..=0xEF => {
                    // APP markers
                    if app_idx < data.app_data.len() {
                        let app_data = &data.app_data[app_idx];
                        // app_data format from Brotli: [marker_type_byte, length_hi, length_lo, content...]
                        // Skip markers with all-zero content (placeholder data not filled)
                        let has_content = app_data.len() > 3 && app_data[3..].iter().any(|&b| b != 0);
                        if has_content {
                            self.write_marker(marker);
                            // Skip the marker type byte and write: [length_hi, length_lo, content...]
                            if app_data.len() > 1 {
                                self.output.extend_from_slice(&app_data[1..]);
                            }
                        }
                        app_idx += 1;
                    }
                }
                0xFE => {
                    // COM marker
                    if com_idx < data.com_data.len() {
                        self.write_marker(0xFE);
                        let com_data = &data.com_data[com_idx];
                        let len = (com_data.len() + 2) as u16;
                        self.output.push((len >> 8) as u8);
                        self.output.push(len as u8);
                        self.output.extend_from_slice(com_data);
                        com_idx += 1;
                    }
                }
                0xDB => {
                    // DQT marker
                    self.write_dqt(data, &mut dqt_idx);
                }
                0xC4 => {
                    // DHT marker
                    self.write_dht(data, &mut dht_idx);
                }
                0xC0..=0xC3 | 0xC5..=0xC7 | 0xC9..=0xCB | 0xCD..=0xCF => {
                    // SOF marker
                    self.write_sof(data, marker);
                }
                0xDA => {
                    // SOS marker
                    if scan_idx < data.scan_info.len() {
                        self.write_sos(data, scan_idx, coefficients)?;
                        scan_idx += 1;
                    }
                }
                0xDD => {
                    // DRI marker
                    if data.restart_interval > 0 {
                        self.write_marker(0xDD);
                        self.output.push(0x00);
                        self.output.push(0x04);
                        self.output.push((data.restart_interval >> 8) as u8);
                        self.output.push(data.restart_interval as u8);
                    }
                }
                0xD9 => {
                    // EOI marker - written at the end
                    break;
                }
                _ => {
                    // Other markers - just write the marker
                    self.write_marker(marker);
                }
            }
        }

        // Write EOI marker
        self.write_marker(0xD9);

        // Write tail data
        if !data.tail_data.is_empty() {
            self.output.extend_from_slice(&data.tail_data);
        }

        Ok(std::mem::take(&mut self.output))
    }

    fn write_marker(&mut self, marker: u8) {
        self.output.push(0xFF);
        self.output.push(marker);
    }

    fn write_dqt(&mut self, data: &JpegReconstructionData, idx: &mut usize) {
        // Find all consecutive DQT tables
        let start_idx = *idx;
        while *idx < data.quant_tables.len() {
            let is_last = data.quant_tables[*idx].is_last;
            *idx += 1;
            if is_last {
                break;
            }
        }

        if start_idx >= data.quant_tables.len() {
            return;
        }

        self.write_marker(0xDB);

        // Calculate length
        let mut len = 2usize;
        for i in start_idx..*idx {
            let table = &data.quant_tables[i];
            len += 1 + if table.precision == 0 { 64 } else { 128 };
        }
        self.output.push((len >> 8) as u8);
        self.output.push(len as u8);

        // Write tables
        for i in start_idx..*idx {
            let table = &data.quant_tables[i];
            let pq_tq = (table.precision << 4) | table.index;
            self.output.push(pq_tq);

            // Check if table values are all zeros (not filled from codestream)
            let all_zeros = table.values.iter().all(|&v| v == 0);

            for &k in &JPEG_NATURAL_ORDER {
                let value = if all_zeros {
                    // Use standard JPEG quant tables as fallback
                    if table.index == 0 {
                        STD_LUMINANCE_QUANT_TBL[k] as u16
                    } else {
                        STD_CHROMINANCE_QUANT_TBL[k] as u16
                    }
                } else {
                    table.values[k]
                };

                if table.precision == 0 {
                    self.output.push(value as u8);
                } else {
                    self.output.push((value >> 8) as u8);
                    self.output.push(value as u8);
                }
            }
        }
    }

    fn write_dht(&mut self, data: &JpegReconstructionData, idx: &mut usize) {
        // Find all consecutive DHT tables
        let start_idx = *idx;
        while *idx < data.huffman_codes.len() {
            let is_last = data.huffman_codes[*idx].is_last;
            *idx += 1;
            if is_last {
                break;
            }
        }

        if start_idx >= data.huffman_codes.len() {
            return;
        }

        self.write_marker(0xC4);

        // Calculate length
        let mut len = 2usize;
        for i in start_idx..*idx {
            let code = &data.huffman_codes[i];
            let (_, values_len) = code.dht_counts_and_values_len();
            len += 1 + 16 + values_len;
        }
        self.output.push((len >> 8) as u8);
        self.output.push(len as u8);

        // Write tables
        for i in start_idx..*idx {
            let code = &data.huffman_codes[i];
            let (counts, values_len) = code.dht_counts_and_values_len();
            let tc_th = (code.table_class << 4) | code.slot_id;
            self.output.push(tc_th);
            self.output.extend_from_slice(&counts);
            for value in code.values.iter().take(values_len) {
                self.output.push(*value as u8);
            }
        }
    }

    fn write_sof(&mut self, data: &JpegReconstructionData, marker: u8) {
        self.write_marker(marker);

        let len = 8 + 3 * data.components.len();
        self.output.push((len >> 8) as u8);
        self.output.push(len as u8);

        // Precision (8 bits)
        self.output.push(8);

        // Height
        self.output.push((data.height >> 8) as u8);
        self.output.push(data.height as u8);

        // Width
        self.output.push((data.width >> 8) as u8);
        self.output.push(data.width as u8);

        // Number of components
        self.output.push(data.components.len() as u8);

        // Component info
        for comp in &data.components {
            self.output.push(comp.id);
            self.output.push((comp.h_samp_factor << 4) | comp.v_samp_factor);
            self.output.push(comp.quant_idx);
        }
    }

    fn write_sos(
        &mut self,
        data: &JpegReconstructionData,
        scan_idx: usize,
        coefficients: &[Vec<i16>],
    ) -> Result<()> {
        let scan = &data.scan_info[scan_idx];

        self.write_marker(0xDA);

        let len = 6 + 2 * scan.num_components as usize;
        self.output.push((len >> 8) as u8);
        self.output.push(len as u8);

        self.output.push(scan.num_components);

        for i in 0..scan.num_components as usize {
            let comp_idx = scan.component_idx[i] as usize;
            if comp_idx < data.components.len() {
                self.output.push(data.components[comp_idx].id);
            } else {
                self.output.push((i + 1) as u8);
            }
            self.output.push((scan.dc_tbl_idx[i] << 4) | scan.ac_tbl_idx[i]);
        }

        self.output.push(scan.ss);
        self.output.push(scan.se);
        self.output.push((scan.ah << 4) | scan.al);

        // Encode the entropy-coded data using the DCT coefficients
        self.encode_scan_data(data, scan, coefficients)?;

        Ok(())
    }

    /// Find Huffman table index for a given class and slot.
    /// Returns index into self.huff_tables array:
    /// 0 = DC luminance (class=0, slot=0)
    /// 1 = DC chrominance (class=0, slot=1)
    /// 2 = AC luminance (class=1, slot=0)
    /// 3 = AC chrominance (class=1, slot=1)
    fn find_huff_table(&self, _data: &JpegReconstructionData, table_class: u8, slot_id: u8) -> Option<usize> {
        let idx = match (table_class, slot_id) {
            (0, 0) => 0, // DC luminance
            (0, 1) => 1, // DC chrominance
            (1, 0) => 2, // AC luminance
            (1, 1) => 3, // AC chrominance
            _ => return None,
        };
        Some(idx)
    }

    /// Encode the scan data (entropy-coded segment)
    fn encode_scan_data(
        &mut self,
        data: &JpegReconstructionData,
        scan: &JpegScanInfo,
        coefficients: &[Vec<i16>],
    ) -> Result<()> {
        // Calculate MCU dimensions
        let mut max_h = 1u8;
        let mut max_v = 1u8;
        for comp in &data.components {
            max_h = max_h.max(comp.h_samp_factor);
            max_v = max_v.max(comp.v_samp_factor);
        }

        let mcu_width = max_h as usize * 8;
        let mcu_height = max_v as usize * 8;
        let mcus_x = (data.width as usize + mcu_width - 1) / mcu_width;
        let mcus_y = (data.height as usize + mcu_height - 1) / mcu_height;

        // Track last DC values for differential encoding
        let mut last_dc = vec![0i16; scan.num_components as usize];

        // For baseline JPEG (ss=0, se=63, ah=0, al=0), encode all coefficients
        let is_baseline = scan.ss == 0 && scan.se == 63 && scan.ah == 0 && scan.al == 0;

        if !is_baseline {
            // Progressive JPEG not fully supported yet - just flush bits
            self.flush_bits();
            return Ok(());
        }

        // Encode each MCU
        for mcu_y in 0..mcus_y {
            for mcu_x in 0..mcus_x {
                // Encode each component in the scan
                for scan_comp_idx in 0..scan.num_components as usize {
                    let comp_idx = scan.component_idx[scan_comp_idx] as usize;
                    if comp_idx >= data.components.len() || comp_idx >= coefficients.len() {
                        continue;
                    }

                    let comp = &data.components[comp_idx];
                    let h_factor = comp.h_samp_factor as usize;
                    let v_factor = comp.v_samp_factor as usize;

                    // Get Huffman tables for this component
                    let dc_table_idx = self.find_huff_table(data, 0, scan.dc_tbl_idx[scan_comp_idx]);
                    let ac_table_idx = self.find_huff_table(data, 1, scan.ac_tbl_idx[scan_comp_idx]);

                    if dc_table_idx.is_none() || ac_table_idx.is_none() {
                        continue;
                    }

                    let dc_table_idx = dc_table_idx.unwrap();
                    let ac_table_idx = ac_table_idx.unwrap();

                    // Calculate blocks per row for this component
                    let comp_blocks_x = (data.width as usize * h_factor + max_h as usize * 8 - 1)
                        / (max_h as usize * 8);

                    // Encode each block in the MCU for this component
                    for v in 0..v_factor {
                        for h in 0..h_factor {
                            let block_x = mcu_x * h_factor + h;
                            let block_y = mcu_y * v_factor + v;
                            let block_idx = block_y * comp_blocks_x + block_x;

                            if block_idx * 64 >= coefficients[comp_idx].len() {
                                continue;
                            }

                            let block_coeffs = &coefficients[comp_idx][block_idx * 64..(block_idx + 1) * 64];

                            self.encode_block(
                                block_coeffs,
                                &mut last_dc[scan_comp_idx],
                                dc_table_idx,
                                ac_table_idx,
                            );
                        }
                    }
                }
            }
        }

        self.flush_bits();
        Ok(())
    }

    /// Encode a single 8x8 block of DCT coefficients
    fn encode_block(
        &mut self,
        coeffs: &[i16],
        last_dc: &mut i16,
        dc_table_idx: usize,
        ac_table_idx: usize,
    ) {
        if coeffs.len() < 64 || dc_table_idx >= self.huff_tables.len() || ac_table_idx >= self.huff_tables.len() {
            return;
        }

        // Copy tables to avoid borrow issues
        let dc_table = self.huff_tables[dc_table_idx];
        let ac_table = self.huff_tables[ac_table_idx];

        // Encode DC coefficient (differential)
        let dc = coeffs[0];
        let dc_diff = dc - *last_dc;
        *last_dc = dc;

        let (dc_size, dc_value) = Self::get_value_bits(dc_diff);
        let (dc_code, dc_bits) = dc_table[dc_size as usize];

        if dc_bits > 0 {
            self.write_bits(dc_code, dc_bits);
            if dc_size > 0 {
                self.write_bits(dc_value as u16, dc_size);
            }
        }

        // Encode AC coefficients
        let mut zero_count = 0u8;
        for i in 1..64 {
            let ac = coeffs[i];
            if ac == 0 {
                zero_count += 1;
            } else {
                // Emit ZRL symbols for runs of 16 zeros
                while zero_count >= 16 {
                    let (zrl_code, zrl_bits) = ac_table[0xF0];
                    if zrl_bits > 0 {
                        self.write_bits(zrl_code, zrl_bits);
                    }
                    zero_count -= 16;
                }

                // Encode the non-zero coefficient
                let (ac_size, ac_value) = Self::get_value_bits(ac);
                let symbol = (zero_count << 4) | ac_size;
                let (ac_code, ac_bits) = ac_table[symbol as usize];
                if ac_bits > 0 {
                    self.write_bits(ac_code, ac_bits);
                    self.write_bits(ac_value as u16, ac_size);
                }
                zero_count = 0;
            }
        }

        // If we have trailing zeros, emit EOB
        if zero_count > 0 {
            let (eob_code, eob_bits) = ac_table[0x00];
            if eob_bits > 0 {
                self.write_bits(eob_code, eob_bits);
            }
        }
    }
}

/// JPEG encoder that converts pixels to JPEG format.
struct JpegEncoder<'a> {
    data: &'a JpegReconstructionData,
    output: Vec<u8>,
    bit_buffer: u32,
    bit_count: u8,
    /// Huffman encoding tables (code, length) indexed by symbol
    /// [0] = DC luminance, [1] = DC chrominance, [2] = AC luminance, [3] = AC chrominance
    huff_tables: [[(u16, u8); 256]; 4],
}

impl<'a> JpegEncoder<'a> {
    fn new(data: &'a JpegReconstructionData) -> Self {
        // Try to use jbrd Huffman tables, fall back to standard if not available
        let mut huff_tables = [[(0u16, 0u8); 256]; 4];

        // Try to find tables from jbrd data
        let mut found = [false; 4];
        for code in &data.huffman_codes {
            let idx = match (code.table_class, code.slot_id) {
                (0, 0) => 0, // DC luminance
                (0, 1) => 1, // DC chrominance
                (1, 0) => 2, // AC luminance
                (1, 1) => 3, // AC chrominance
                _ => continue,
            };
            huff_tables[idx] = Self::build_huffman_table_from_jbrd(&code.counts, &code.values);
            found[idx] = true;
        }

        // Fall back to standard tables for any not found
        if !found[0] {
            huff_tables[0] = Self::build_huffman_table(&STD_DC_LUMINANCE_NRCODES, &STD_DC_LUMINANCE_VALUES);
        }
        if !found[1] {
            huff_tables[1] = Self::build_huffman_table(&STD_DC_CHROMINANCE_NRCODES, &STD_DC_CHROMINANCE_VALUES);
        }
        if !found[2] {
            huff_tables[2] = Self::build_huffman_table(&STD_AC_LUMINANCE_NRCODES, &STD_AC_LUMINANCE_VALUES);
        }
        if !found[3] {
            huff_tables[3] = Self::build_huffman_table(&STD_AC_CHROMINANCE_NRCODES, &STD_AC_CHROMINANCE_VALUES);
        }

        Self {
            data,
            output: Vec::new(),
            bit_buffer: 0,
            bit_count: 0,
            huff_tables,
        }
    }

    /// Build a Huffman encoding table from jbrd counts and values.
    fn build_huffman_table_from_jbrd(counts: &[u8; 16], values: &[u16]) -> [(u16, u8); 256] {
        let mut table = [(0u16, 0u8); 256];
        let mut code = 0u16;
        let mut val_idx = 0;
        let values_len = match values.last() {
            Some(&256) => values.len().saturating_sub(1),
            _ => values.len(),
        };

        for (bits, &count) in counts.iter().enumerate() {
            let bits = bits as u8 + 1;
            for _ in 0..count {
                if val_idx < values_len {
                    let value = values[val_idx] as usize;
                    if value < table.len() {
                        table[value] = (code, bits);
                    }
                    val_idx += 1;
                }
                code += 1;
            }
            code <<= 1;
        }
        table
    }

    /// Build a Huffman encoding table from counts and values.
    fn build_huffman_table(counts: &[u8; 16], values: &[u8]) -> [(u16, u8); 256] {
        let mut table = [(0u16, 0u8); 256];
        let mut code = 0u16;
        let mut val_idx = 0;

        for (bits, &count) in counts.iter().enumerate() {
            let bits = bits as u8 + 1;
            for _ in 0..count {
                if val_idx < values.len() {
                    table[values[val_idx] as usize] = (code, bits);
                    val_idx += 1;
                }
                code += 1;
            }
            code <<= 1;
        }
        table
    }

    /// Encode pixels to JPEG.
    fn encode(&mut self, pixels: &[f32], width: usize, height: usize) -> Result<Vec<u8>> {
        // Write SOI
        self.output.extend_from_slice(&[0xFF, 0xD8]);

        // Write DQT markers
        self.write_dqt()?;

        // Write SOF0 marker
        self.write_sof0(width, height)?;

        // Write DHT markers
        self.write_dht()?;

        // Write SOS and encode image data
        self.write_sos_and_data(pixels, width, height)?;

        // Write EOI
        self.output.extend_from_slice(&[0xFF, 0xD9]);

        Ok(std::mem::take(&mut self.output))
    }

    fn write_dqt(&mut self) -> Result<()> {
        for table in &self.data.quant_tables {
            self.output.extend_from_slice(&[0xFF, 0xDB]);
            let len = 2 + 1 + 64;
            self.output.push((len >> 8) as u8);
            self.output.push(len as u8);
            self.output.push((table.precision << 4) | table.index);
            for &k in &JPEG_NATURAL_ORDER {
                self.output.push(table.values[k] as u8);
            }
        }
        Ok(())
    }

    fn write_sof0(&mut self, width: usize, height: usize) -> Result<()> {
        self.output.extend_from_slice(&[0xFF, 0xC0]);
        let len = 8 + 3 * self.data.components.len();
        self.output.push((len >> 8) as u8);
        self.output.push(len as u8);
        self.output.push(8); // 8-bit precision
        self.output.push((height >> 8) as u8);
        self.output.push(height as u8);
        self.output.push((width >> 8) as u8);
        self.output.push(width as u8);
        self.output.push(self.data.components.len() as u8);

        for comp in &self.data.components {
            self.output.push(comp.id);
            self.output.push((comp.h_samp_factor << 4) | comp.v_samp_factor);
            self.output.push(comp.quant_idx);
        }
        Ok(())
    }

    fn write_dht(&mut self) -> Result<()> {
        // Write all Huffman tables
        for code in &self.data.huffman_codes {
            let (counts, values_len) = code.dht_counts_and_values_len();
            self.output.extend_from_slice(&[0xFF, 0xC4]);
            let len = 2 + 1 + 16 + values_len;
            self.output.push((len >> 8) as u8);
            self.output.push(len as u8);
            self.output.push((code.table_class << 4) | code.slot_id);
            self.output.extend_from_slice(&counts);
            for value in code.values.iter().take(values_len) {
                self.output.push(*value as u8);
            }
        }
        Ok(())
    }

    fn write_sos_and_data(&mut self, pixels: &[f32], width: usize, height: usize) -> Result<()> {
        // Write SOS header
        self.output.extend_from_slice(&[0xFF, 0xDA]);
        let num_components = self.data.components.len();
        let len = 6 + 2 * num_components;
        self.output.push((len >> 8) as u8);
        self.output.push(len as u8);
        self.output.push(num_components as u8);

        for (i, comp) in self.data.components.iter().enumerate() {
            self.output.push(comp.id);
            let dc_ac = if i == 0 { 0x00 } else { 0x11 };
            self.output.push(dc_ac);
        }
        self.output.push(0); // Ss
        self.output.push(63); // Se
        self.output.push(0); // Ah/Al

        // Encode image data
        let is_gray = self.data.is_gray;
        let blocks_x = (width + 7) / 8;
        let blocks_y = (height + 7) / 8;

        let mut last_dc = [0i16; 3];

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                if is_gray {
                    let block = self.extract_gray_block(pixels, width, height, bx, by);
                    let coeffs = self.dct_and_quantize(&block, 0);
                    self.encode_block(&coeffs, &mut last_dc[0], true)?;
                } else {
                    let (y_block, cb_block, cr_block) =
                        self.extract_ycbcr_blocks(pixels, width, height, bx, by);

                    let y_coeffs = self.dct_and_quantize(&y_block, 0);
                    let cb_coeffs = self.dct_and_quantize(&cb_block, 1);
                    let cr_coeffs = self.dct_and_quantize(&cr_block, 1);

                    self.encode_block(&y_coeffs, &mut last_dc[0], true)?;
                    self.encode_block(&cb_coeffs, &mut last_dc[1], false)?;
                    self.encode_block(&cr_coeffs, &mut last_dc[2], false)?;
                }
            }
        }

        // Flush remaining bits
        self.flush_bits()?;

        Ok(())
    }

    fn extract_gray_block(&self, pixels: &[f32], width: usize, height: usize, bx: usize, by: usize) -> [f32; 64] {
        let mut block = [0.0f32; 64];
        for y in 0..8 {
            for x in 0..8 {
                let px = (bx * 8 + x).min(width - 1);
                let py = (by * 8 + y).min(height - 1);
                let gray = pixels[py * width + px];
                block[y * 8 + x] = gray * 255.0 - 128.0;
            }
        }
        block
    }

    fn extract_ycbcr_blocks(
        &self,
        pixels: &[f32],
        width: usize,
        height: usize,
        bx: usize,
        by: usize,
    ) -> ([f32; 64], [f32; 64], [f32; 64]) {
        let mut y_block = [0.0f32; 64];
        let mut cb_block = [0.0f32; 64];
        let mut cr_block = [0.0f32; 64];

        for y in 0..8 {
            for x in 0..8 {
                let px = (bx * 8 + x).min(width - 1);
                let py = (by * 8 + y).min(height - 1);
                let idx = (py * width + px) * 3;

                let r = pixels.get(idx).copied().unwrap_or(0.0) * 255.0;
                let g = pixels.get(idx + 1).copied().unwrap_or(0.0) * 255.0;
                let b = pixels.get(idx + 2).copied().unwrap_or(0.0) * 255.0;

                // RGB to YCbCr conversion
                let y_val = 0.299 * r + 0.587 * g + 0.114 * b;
                let cb_val = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0;
                let cr_val = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0;

                let block_idx = y * 8 + x;
                y_block[block_idx] = y_val - 128.0;
                cb_block[block_idx] = cb_val - 128.0;
                cr_block[block_idx] = cr_val - 128.0;
            }
        }

        (y_block, cb_block, cr_block)
    }

    fn dct_and_quantize(&self, block: &[f32; 64], quant_idx: usize) -> [i16; 64] {
        // Apply 2D DCT
        let mut dct_block = [0.0f32; 64];

        for v in 0..8 {
            for u in 0..8 {
                let cu = if u == 0 { 1.0 / 2.0_f32.sqrt() } else { 1.0 };
                let cv = if v == 0 { 1.0 / 2.0_f32.sqrt() } else { 1.0 };

                let mut sum = 0.0f32;
                for y in 0..8 {
                    for x in 0..8 {
                        let cos_x = ((2 * x + 1) as f32 * u as f32 * std::f32::consts::PI / 16.0).cos();
                        let cos_y = ((2 * y + 1) as f32 * v as f32 * std::f32::consts::PI / 16.0).cos();
                        sum += block[y * 8 + x] * cos_x * cos_y;
                    }
                }
                dct_block[v * 8 + u] = 0.25 * cu * cv * sum;
            }
        }

        // Quantize
        let mut result = [0i16; 64];
        let quant_table = if quant_idx < self.data.quant_tables.len() {
            &self.data.quant_tables[quant_idx].values
        } else {
            &STD_LUMINANCE_QUANT_TBL
        };

        for i in 0..64 {
            let zigzag_idx = JPEG_NATURAL_ORDER[i];
            let q = quant_table[zigzag_idx] as f32;
            result[i] = (dct_block[zigzag_idx] / q).round() as i16;
        }

        result
    }

    fn encode_block(&mut self, coeffs: &[i16; 64], last_dc: &mut i16, is_lum: bool) -> Result<()> {
        // Encode DC coefficient
        let dc = coeffs[0];
        let dc_diff = dc - *last_dc;
        *last_dc = dc;

        // Copy the huffman tables to avoid borrow issues
        // [0] = DC lum, [1] = DC chrom, [2] = AC lum, [3] = AC chrom
        let dc_huff = if is_lum {
            self.huff_tables[0]
        } else {
            self.huff_tables[1]
        };
        let ac_huff = if is_lum {
            self.huff_tables[2]
        } else {
            self.huff_tables[3]
        };

        self.encode_dc(dc_diff, &dc_huff)?;

        // Encode AC coefficients
        let mut zero_count = 0u8;
        for i in 1..64 {
            let ac = coeffs[i];
            if ac == 0 {
                zero_count += 1;
            } else {
                while zero_count >= 16 {
                    // ZRL (zero run length) = 0xF0
                    self.write_huffman(0xF0, &ac_huff)?;
                    zero_count -= 16;
                }
                let (size, value) = Self::get_value_bits(ac);
                let symbol = (zero_count << 4) | size;
                self.write_huffman(symbol, &ac_huff)?;
                self.write_bits(value as u16, size)?;
                zero_count = 0;
            }
        }

        if zero_count > 0 {
            // EOB (end of block) = 0x00
            self.write_huffman(0x00, &ac_huff)?;
        }

        Ok(())
    }

    fn encode_dc(&mut self, dc_diff: i16, huff_table: &[(u16, u8); 256]) -> Result<()> {
        let (size, value) = Self::get_value_bits(dc_diff);
        self.write_huffman(size as u8, huff_table)?;
        if size > 0 {
            self.write_bits(value as u16, size)?;
        }
        Ok(())
    }

    fn get_value_bits(value: i16) -> (u8, i16) {
        if value == 0 {
            return (0, 0);
        }

        let abs_val = value.unsigned_abs();
        let size = 16 - abs_val.leading_zeros() as u8;

        let encoded = if value < 0 {
            value + (1 << size) - 1
        } else {
            value
        };

        (size, encoded)
    }

    fn write_huffman(&mut self, symbol: u8, table: &[(u16, u8); 256]) -> Result<()> {
        let (code, bits) = table[symbol as usize];
        if bits == 0 {
            return Err(Error::InvalidJpegReconstructionData);
        }
        self.write_bits(code, bits)
    }

    fn write_bits(&mut self, value: u16, bits: u8) -> Result<()> {
        self.bit_buffer = (self.bit_buffer << bits) | (value as u32);
        self.bit_count += bits;

        while self.bit_count >= 8 {
            self.bit_count -= 8;
            let byte = ((self.bit_buffer >> self.bit_count) & 0xFF) as u8;
            self.output.push(byte);
            // Byte stuffing for 0xFF
            if byte == 0xFF {
                self.output.push(0x00);
            }
        }

        Ok(())
    }

    fn flush_bits(&mut self) -> Result<()> {
        if self.bit_count > 0 {
            // Pad with 1s
            let remaining = 8 - self.bit_count;
            self.bit_buffer = (self.bit_buffer << remaining) | ((1 << remaining) - 1);
            let byte = (self.bit_buffer & 0xFF) as u8;
            self.output.push(byte);
            if byte == 0xFF {
                self.output.push(0x00);
            }
            self.bit_count = 0;
            self.bit_buffer = 0;
        }
        Ok(())
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

    #[test]
    fn test_jpeg_natural_order() {
        // Verify the zigzag order is correct
        assert_eq!(JPEG_NATURAL_ORDER[0], 0);
        assert_eq!(JPEG_NATURAL_ORDER[1], 1);
        assert_eq!(JPEG_NATURAL_ORDER[2], 8);
        assert_eq!(JPEG_NATURAL_ORDER[63], 63);
    }

    #[test]
    fn test_jpeg_reconstruction_data_default() {
        let data = JpegReconstructionData::default();
        assert!(!data.is_valid());
    }
}
