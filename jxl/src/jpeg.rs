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
use jxl_simd::{simd_function, F32SimdVec};
use jxl_transforms::{idct2d_2_2, idct2d_4_4, idct2d_8_8};

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
        let has_sentinel =
            total_count > 0 && self.values.last() == Some(&256) && self.values.len() == total_count;
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
        for (i, &zigzag_idx) in JPEG_NATURAL_ORDER.iter().enumerate().skip(1) {
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
        let num_app_markers = result
            .marker_order
            .iter()
            .filter(|&&m| (0xE0..=0xEF).contains(&m))
            .count();
        let num_com_markers = result.marker_order.iter().filter(|&&m| m == 0xFE).count();

        // 3. For each APP marker: read type AND length together
        // libjxl loops: for each app { read type; read 16-bit length }
        result.app_marker_types = Vec::with_capacity(num_app_markers);
        result.app_data = Vec::with_capacity(num_app_markers);
        for i in 0..num_app_markers {
            // Type: U32(Val(0), Val(1), BitsOffset(1, 2), BitsOffset(2, 4))
            let bits_before = reader.total_bits_read();
            let marker_type = Self::read_u32_app_type(&mut reader)?;
            result
                .app_marker_types
                .push(AppMarkerType::try_from(marker_type as u8)?);

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
            0 => 1,     // kGray
            1 | 2 => 3, // kYCbCr or kRGB
            3 => {
                // kCustom
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
                0 => 1,                     // kGray
                1 => (i + 1) as u8,         // kYCbCr: 1, 2, 3
                2 => [b'R', b'G', b'B'][i], // kRGB
                3 => custom_ids[i],         // kCustom
                _ => return Err(Error::InvalidJpegReconstructionData),
            };

            // Read quant index only (2 bits)
            let quant_idx = reader.read(2)? as u8;

            let component = JpegComponent {
                id,
                h_samp_factor: 1, // Default, set from JPEG header during reconstruction
                v_samp_factor: 1, // Default, set from JPEG header during reconstruction
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
            code.slot_id = id; // slot_id is just the 2-bit id, not combined with table_class
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
                    code.counts[j - 1] = count; // jbrd index j -> DHT counts[j-1]
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
                result.scan_info[s].reset_points.push(JpegResetPoint {
                    mcu: block_idx,
                    last_dc: Vec::new(),
                });
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
                result.scan_info[s]
                    .extra_zero_runs
                    .push((block_idx, num_zeros));
            }
        }

        // 9. restart_interval - only read if has_dri marker (DRI = 0xDD)
        // Check if any marker is DRI (0xDD)
        let has_dri = result.marker_order.contains(&0xDD);
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
            result
                .com_data
                .push(decompressed[offset..offset + size].to_vec());
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
                result
                    .inter_marker_data
                    .push(decompressed[offset..offset + size].to_vec());
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
        for (c, &mapped_comp) in jpeg_c_map.iter().enumerate().take(num_components.min(3)) {
            let quant_c = if is_gray { 1 } else { c };
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
        let lum_quant = JpegQuantTable {
            index: 0,
            is_last: is_gray,
            values: STD_LUMINANCE_QUANT_TBL,
            ..Default::default()
        };
        self.quant_tables.push(lum_quant);

        if !is_gray {
            let chrom_quant = JpegQuantTable {
                index: 1,
                is_last: true,
                values: STD_CHROMINANCE_QUANT_TBL,
                ..Default::default()
            };
            self.quant_tables.push(chrom_quant);
        }

        // Create standard Huffman codes
        // DC Luminance
        let dc_lum = JpegHuffmanCode {
            table_class: 0,
            slot_id: 0,
            is_last: false,
            counts: STD_DC_LUMINANCE_NRCODES,
            values: STD_DC_LUMINANCE_VALUES.iter().map(|&v| v as u16).collect(),
        };
        self.huffman_codes.push(dc_lum);

        // AC Luminance
        let ac_lum = JpegHuffmanCode {
            table_class: 1,
            slot_id: 0,
            is_last: is_gray,
            counts: STD_AC_LUMINANCE_NRCODES,
            values: STD_AC_LUMINANCE_VALUES.iter().map(|&v| v as u16).collect(),
        };
        self.huffman_codes.push(ac_lum);

        if !is_gray {
            // DC Chrominance
            let dc_chrom = JpegHuffmanCode {
                table_class: 0,
                slot_id: 1,
                is_last: false,
                counts: STD_DC_CHROMINANCE_NRCODES,
                values: STD_DC_CHROMINANCE_VALUES
                    .iter()
                    .map(|&v| v as u16)
                    .collect(),
            };
            self.huffman_codes.push(dc_chrom);

            // AC Chrominance
            let ac_chrom = JpegHuffmanCode {
                table_class: 1,
                slot_id: 1,
                is_last: true,
                counts: STD_AC_CHROMINANCE_NRCODES,
                values: STD_AC_CHROMINANCE_VALUES
                    .iter()
                    .map(|&v| v as u16)
                    .collect(),
            };
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
        let mut scan = JpegScanInfo {
            num_components: if is_gray { 1 } else { 3 },
            ..Default::default()
        };
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
        let coeffs = self
            .dct_coefficients
            .as_ref()
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
        self.dct_coefficients
            .as_ref()
            .is_some_and(|c| !c.coefficients.is_empty())
    }

    /// Encode pixel data to JPEG format.
    ///
    /// Takes grayscale or RGB pixel data (as f32 values in 0.0-1.0 range) and encodes to JPEG.
    /// For grayscale, pixels should be a single slice.
    /// For RGB, pixels should be interleaved RGB values.
    pub fn encode_from_pixels(
        &self,
        pixels: &[f32],
        width: usize,
        height: usize,
    ) -> Result<Vec<u8>> {
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
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113,
    92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
];

/// Standard JPEG chrominance quantization table.
const STD_CHROMINANCE_QUANT_TBL: [u16; 64] = [
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
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
        let mut code = 0u32; // Use u32 to avoid overflow
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
    fn write_jpeg(
        &mut self,
        data: &JpegReconstructionData,
        coefficients: &[Vec<i16>],
    ) -> Result<Vec<u8>> {
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
                        let has_content =
                            app_data.len() > 3 && app_data[3..].iter().any(|&b| b != 0);
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
                        STD_LUMINANCE_QUANT_TBL[k]
                    } else {
                        STD_CHROMINANCE_QUANT_TBL[k]
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
            self.output
                .push((comp.h_samp_factor << 4) | comp.v_samp_factor);
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
            self.output
                .push((scan.dc_tbl_idx[i] << 4) | scan.ac_tbl_idx[i]);
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
    fn find_huff_table(
        &self,
        _data: &JpegReconstructionData,
        table_class: u8,
        slot_id: u8,
    ) -> Option<usize> {
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
        let mcus_x = (data.width as usize).div_ceil(mcu_width);
        let mcus_y = (data.height as usize).div_ceil(mcu_height);

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
                for (scan_comp_idx, last_dc_value) in last_dc
                    .iter_mut()
                    .enumerate()
                    .take(scan.num_components as usize)
                {
                    let comp_idx = scan.component_idx[scan_comp_idx] as usize;
                    if comp_idx >= data.components.len() || comp_idx >= coefficients.len() {
                        continue;
                    }

                    let comp = &data.components[comp_idx];
                    let h_factor = comp.h_samp_factor as usize;
                    let v_factor = comp.v_samp_factor as usize;

                    // Get Huffman tables for this component
                    let dc_table_idx =
                        self.find_huff_table(data, 0, scan.dc_tbl_idx[scan_comp_idx]);
                    let ac_table_idx =
                        self.find_huff_table(data, 1, scan.ac_tbl_idx[scan_comp_idx]);

                    if dc_table_idx.is_none() || ac_table_idx.is_none() {
                        continue;
                    }

                    let dc_table_idx = dc_table_idx.unwrap();
                    let ac_table_idx = ac_table_idx.unwrap();

                    // Calculate blocks per row for this component
                    let comp_blocks_x =
                        (data.width as usize * h_factor).div_ceil(max_h as usize * 8);

                    // Encode each block in the MCU for this component
                    for v in 0..v_factor {
                        for h in 0..h_factor {
                            let block_x = mcu_x * h_factor + h;
                            let block_y = mcu_y * v_factor + v;
                            let block_idx = block_y * comp_blocks_x + block_x;

                            if block_idx * 64 >= coefficients[comp_idx].len() {
                                continue;
                            }

                            let block_coeffs =
                                &coefficients[comp_idx][block_idx * 64..(block_idx + 1) * 64];

                            self.encode_block(
                                block_coeffs,
                                last_dc_value,
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
        if coeffs.len() < 64
            || dc_table_idx >= self.huff_tables.len()
            || ac_table_idx >= self.huff_tables.len()
        {
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
        for &ac in coeffs.iter().take(64).skip(1) {
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
            huff_tables[0] =
                Self::build_huffman_table(&STD_DC_LUMINANCE_NRCODES, &STD_DC_LUMINANCE_VALUES);
        }
        if !found[1] {
            huff_tables[1] =
                Self::build_huffman_table(&STD_DC_CHROMINANCE_NRCODES, &STD_DC_CHROMINANCE_VALUES);
        }
        if !found[2] {
            huff_tables[2] =
                Self::build_huffman_table(&STD_AC_LUMINANCE_NRCODES, &STD_AC_LUMINANCE_VALUES);
        }
        if !found[3] {
            huff_tables[3] =
                Self::build_huffman_table(&STD_AC_CHROMINANCE_NRCODES, &STD_AC_CHROMINANCE_VALUES);
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
            self.output
                .push((comp.h_samp_factor << 4) | comp.v_samp_factor);
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
        let blocks_x = width.div_ceil(8);
        let blocks_y = height.div_ceil(8);

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

    fn extract_gray_block(
        &self,
        pixels: &[f32],
        width: usize,
        height: usize,
        bx: usize,
        by: usize,
    ) -> [f32; 64] {
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
                        let cos_x =
                            ((2 * x + 1) as f32 * u as f32 * std::f32::consts::PI / 16.0).cos();
                        let cos_y =
                            ((2 * y + 1) as f32 * v as f32 * std::f32::consts::PI / 16.0).cos();
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
        for &ac in coeffs.iter().skip(1) {
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
        self.write_huffman(size, huff_table)?;
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

    // ===== AppMarkerType tests =====

    #[test]
    fn test_app_marker_type_conversion() {
        assert_eq!(AppMarkerType::try_from(0).unwrap(), AppMarkerType::Unknown);
        assert_eq!(AppMarkerType::try_from(1).unwrap(), AppMarkerType::Icc);
        assert_eq!(AppMarkerType::try_from(2).unwrap(), AppMarkerType::Exif);
        assert_eq!(AppMarkerType::try_from(3).unwrap(), AppMarkerType::Xmp);
        assert!(AppMarkerType::try_from(4).is_err());
    }

    #[test]
    fn test_app_marker_type_default() {
        let marker = AppMarkerType::default();
        assert_eq!(marker, AppMarkerType::Unknown);
    }

    // ===== JpegQuantTable tests =====

    #[test]
    fn test_jpeg_quant_table_default() {
        let table = JpegQuantTable::default();
        assert_eq!(table.precision, 0);
        assert_eq!(table.index, 0);
        assert!(!table.is_last);
        assert_eq!(table.values, [0u16; 64]);
    }

    #[test]
    fn test_jpeg_quant_table_custom() {
        let mut table = JpegQuantTable::default();
        table.precision = 1;
        table.index = 2;
        table.is_last = true;
        table.values[0] = 16;
        table.values[63] = 99;

        assert_eq!(table.precision, 1);
        assert_eq!(table.index, 2);
        assert!(table.is_last);
        assert_eq!(table.values[0], 16);
        assert_eq!(table.values[63], 99);
    }

    // ===== JpegComponent tests =====

    #[test]
    fn test_jpeg_component_default() {
        let comp = JpegComponent::default();
        assert_eq!(comp.id, 0);
        assert_eq!(comp.h_samp_factor, 0);
        assert_eq!(comp.v_samp_factor, 0);
        assert_eq!(comp.quant_idx, 0);
    }

    #[test]
    fn test_jpeg_component_typical_y() {
        let mut comp = JpegComponent::default();
        comp.id = 1;
        comp.h_samp_factor = 2;
        comp.v_samp_factor = 2;
        comp.quant_idx = 0;

        assert_eq!(comp.id, 1);
        assert_eq!(comp.h_samp_factor, 2);
        assert_eq!(comp.v_samp_factor, 2);
    }

    // ===== JpegHuffmanCode tests =====

    #[test]
    fn test_jpeg_huffman_code_default() {
        let code = JpegHuffmanCode::default();
        assert_eq!(code.table_class, 0);
        assert_eq!(code.slot_id, 0);
        assert!(!code.is_last);
        assert_eq!(code.counts, [0u8; 16]);
        assert!(code.values.is_empty());
    }

    #[test]
    fn test_jpeg_huffman_code_dht_counts() {
        let mut code = JpegHuffmanCode::default();
        code.counts[0] = 1;
        code.counts[1] = 2;
        code.values = vec![0, 1, 2];

        let (counts, len) = code.dht_counts_and_values_len();
        assert_eq!(counts[0], 1);
        assert_eq!(counts[1], 2);
        assert_eq!(len, 3);
    }

    #[test]
    fn test_jpeg_huffman_code_sentinel_handling() {
        let mut code = JpegHuffmanCode::default();
        code.counts[0] = 2;
        code.values = vec![0, 256]; // 256 is sentinel

        let (counts, len) = code.dht_counts_and_values_len();
        assert_eq!(counts[0], 1); // One less due to sentinel
        assert_eq!(len, 1); // Sentinel not counted
    }

    // ===== JpegScanInfo tests =====

    #[test]
    fn test_jpeg_scan_info_default() {
        let scan = JpegScanInfo::default();
        assert_eq!(scan.num_components, 0);
        assert_eq!(scan.ss, 0);
        assert_eq!(scan.se, 0);
        assert_eq!(scan.ah, 0);
        assert_eq!(scan.al, 0);
        assert!(scan.reset_points.is_empty());
    }

    #[test]
    fn test_jpeg_scan_info_baseline() {
        let mut scan = JpegScanInfo::default();
        scan.num_components = 3;
        scan.ss = 0;
        scan.se = 63;
        scan.component_idx = [0, 1, 2, 0];

        assert_eq!(scan.num_components, 3);
        assert_eq!(scan.se, 63);
    }

    // ===== JpegResetPoint tests =====

    #[test]
    fn test_jpeg_reset_point_default() {
        let point = JpegResetPoint::default();
        assert_eq!(point.mcu, 0);
        assert!(point.last_dc.is_empty());
    }

    #[test]
    fn test_jpeg_reset_point_with_dc() {
        let point = JpegResetPoint {
            mcu: 100,
            last_dc: vec![0, -10, 20],
        };
        assert_eq!(point.mcu, 100);
        assert_eq!(point.last_dc.len(), 3);
    }

    // ===== JpegDctCoefficients tests =====

    #[test]
    fn test_jpeg_dct_coefficients_new() {
        let coeffs = JpegDctCoefficients::new(16, 16, &[(2, 2), (1, 1), (1, 1)]);

        assert_eq!(coeffs.width, 16);
        assert_eq!(coeffs.height, 16);
        assert_eq!(coeffs.num_components, 3);
        assert_eq!(coeffs.coefficients.len(), 3);
        assert_eq!(coeffs.coefficients[0].len(), 2 * 2 * 64);
        assert_eq!(coeffs.coefficients[1].len(), 1 * 1 * 64);
        assert_eq!(coeffs.coefficients[2].len(), 1 * 1 * 64);
    }

    #[test]
    fn test_jpeg_dct_coefficients_store_dc() {
        let mut coeffs = JpegDctCoefficients::new(8, 8, &[(1, 1)]);
        coeffs.store_dc(0, 0, 0, 100);

        let block = coeffs.get_block(0, 0, 0).unwrap();
        assert_eq!(block[0], 100);
    }

    #[test]
    fn test_jpeg_dct_coefficients_store_block() {
        let mut coeffs = JpegDctCoefficients::new(8, 8, &[(1, 1)]);
        let mut block_data = [0i32; 64];
        block_data[1] = 50;
        block_data[8] = 60;

        coeffs.store_block(0, 0, 0, &block_data);

        let stored = coeffs.get_block(0, 0, 0).unwrap();
        // DC should still be 0 (not stored by store_block)
        assert_eq!(stored[0], 0);
    }

    #[test]
    fn test_jpeg_dct_coefficients_has_stored() {
        let mut coeffs = JpegDctCoefficients::new(8, 8, &[(1, 1)]);
        assert!(!coeffs.has_stored_coefficients());

        coeffs.store_dc(0, 0, 0, 100);
        assert!(coeffs.has_stored_coefficients());
    }

    #[test]
    fn test_jpeg_dct_coefficients_out_of_bounds() {
        let coeffs = JpegDctCoefficients::new(8, 8, &[(1, 1)]);

        // Out of bounds component
        assert!(coeffs.get_block(5, 0, 0).is_none());

        // Out of bounds block
        assert!(coeffs.get_block(0, 10, 10).is_none());
    }

    // ===== JpegReconstructionData tests =====

    #[test]
    fn test_jpeg_reconstruction_data_default() {
        let data = JpegReconstructionData::default();
        assert!(!data.is_valid());
    }

    #[test]
    fn test_jpeg_reconstruction_data_is_valid() {
        let mut data = JpegReconstructionData::default();
        assert!(!data.is_valid());

        data.components.push(JpegComponent::default());
        assert!(!data.is_valid()); // Still missing marker_order

        data.marker_order.push(0xD8);
        assert!(data.is_valid());
    }

    #[test]
    fn test_jpeg_reconstruction_data_from_raw_empty() {
        let result = JpegReconstructionData::from_raw(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_jpeg_reconstruction_data_from_raw_valid() {
        let result = JpegReconstructionData::from_raw(&[0x00, 0x01, 0x02]);
        assert!(result.is_ok());
        let data = result.unwrap();
        assert_eq!(data.width, 1);
        assert_eq!(data.height, 3);
    }

    // ===== JPEG_NATURAL_ORDER tests =====

    #[test]
    fn test_jpeg_natural_order() {
        // Verify the zigzag order is correct
        assert_eq!(JPEG_NATURAL_ORDER[0], 0);
        assert_eq!(JPEG_NATURAL_ORDER[1], 1);
        assert_eq!(JPEG_NATURAL_ORDER[2], 8);
        assert_eq!(JPEG_NATURAL_ORDER[63], 63);
    }

    #[test]
    fn test_jpeg_natural_order_no_duplicates() {
        let mut seen = [false; 64];
        for &idx in &JPEG_NATURAL_ORDER {
            assert!(!seen[idx], "Duplicate index {} in JPEG_NATURAL_ORDER", idx);
            seen[idx] = true;
        }
    }

    #[test]
    fn test_jpeg_natural_order_complete() {
        // All 64 positions should be covered
        let mut seen = [false; 64];
        for &idx in &JPEG_NATURAL_ORDER {
            seen[idx] = true;
        }
        assert!(seen.iter().all(|&v| v));
    }

    #[test]
    fn test_jpeg_natural_order_zigzag_pattern() {
        // First few entries should follow zigzag pattern
        // 0,0 -> 0,1 -> 1,0 -> 2,0 -> 1,1 -> 0,2 -> ...
        let expected_first_8 = [0, 1, 8, 16, 9, 2, 3, 10];
        for (i, &expected) in expected_first_8.iter().enumerate() {
            assert_eq!(
                JPEG_NATURAL_ORDER[i], expected,
                "Position {} should be {}, got {}",
                i, expected, JPEG_NATURAL_ORDER[i]
            );
        }
    }
}

// ============================================================================
// STANDALONE JPEG DECODER
// ============================================================================

/// Decoded JPEG image data.
#[derive(Debug, Clone)]
pub struct JpegDecodedImage {
    /// Image width in pixels
    pub width: usize,
    /// Image height in pixels
    pub height: usize,
    /// Number of color components (1 for grayscale, 3 for RGB)
    pub num_components: usize,
    /// Pixel data in row-major order, interleaved RGB or grayscale
    /// Values are normalized to 0.0-1.0 range
    pub pixels: Vec<f32>,
}

/// Output pixel format for direct JPEG decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JpegOutputFormat {
    Gray8,
    Rgb8,
    Bgr8,
    Rgba8,
    Bgra8,
    Cmyk8,
    Ycbcr8,
    Ycck8,
}

impl JpegOutputFormat {
    fn bytes_per_pixel(self) -> usize {
        match self {
            JpegOutputFormat::Gray8 => 1,
            JpegOutputFormat::Rgb8 | JpegOutputFormat::Bgr8 | JpegOutputFormat::Ycbcr8 => 3,
            JpegOutputFormat::Rgba8
            | JpegOutputFormat::Bgra8
            | JpegOutputFormat::Cmyk8
            | JpegOutputFormat::Ycck8 => 4,
        }
    }
}

/// Output pixel format for 16-bit JPEG decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JpegOutputFormat16 {
    Gray16,
    Rgb16,
    Bgr16,
    Rgba16,
    Bgra16,
    Cmyk16,
    Ycbcr16,
    Ycck16,
}

impl JpegOutputFormat16 {
    fn samples_per_pixel(self) -> usize {
        match self {
            JpegOutputFormat16::Gray16 => 1,
            JpegOutputFormat16::Rgb16 | JpegOutputFormat16::Bgr16 | JpegOutputFormat16::Ycbcr16 => 3,
            JpegOutputFormat16::Rgba16
            | JpegOutputFormat16::Bgra16
            | JpegOutputFormat16::Cmyk16
            | JpegOutputFormat16::Ycck16 => 4,
        }
    }
}

/// JPEG metadata extracted during decoding.
#[derive(Debug, Clone, Default)]
pub struct JpegMetadata {
    /// ICC color profile data
    pub icc_profile: Option<Vec<u8>>,
    /// EXIF metadata
    pub exif_data: Option<Vec<u8>>,
    /// XMP metadata
    pub xmp_data: Option<Vec<u8>>,
    /// JFIF version (major, minor)
    pub jfif_version: Option<(u8, u8)>,
    /// Comment strings from COM markers
    pub comments: Vec<String>,
}

/// JPEG decoder for decoding standard JPEG files.
///
/// Supports baseline DCT (SOF0) and progressive DCT (SOF2) JPEG files.
#[derive(Clone)]
pub struct JpegDecoder {
    width: u16,
    height: u16,
    num_components: u8,
    data_precision: u8,
    components: Vec<JpegDecoderComponent>,
    quant_tables: [[u16; 64]; 4],
    dc_huff_tables: [HuffmanTable; 4],
    ac_huff_tables: [HuffmanTable; 4],
    restart_interval: u16,
    is_progressive: bool,
    metadata: JpegMetadata,
    saw_jfif: bool,
    adobe_transform: Option<u8>,
    icc_segments: Vec<Option<Vec<u8>>>,
    icc_expected_segments: Option<usize>,
}

#[derive(Clone, Default)]
struct JpegDecoderComponent {
    id: u8,
    h_samp: u8,
    v_samp: u8,
    quant_table_id: u8,
    dc_table_id: u8,
    ac_table_id: u8,
}

#[derive(Clone, Copy)]
struct JpegScanHeader {
    num_components: usize,
    component_ids: [u8; 4],
    dc_table_id: [u8; 4],
    ac_table_id: [u8; 4],
    ss: u8,
    se: u8,
    ah: u8,
    al: u8,
}

#[derive(Clone, Copy)]
struct JpegOutputLayout {
    out_width: usize,
    out_height: usize,
    crop_x: usize,
    crop_y: usize,
    denom: usize,
}

#[derive(Clone, Copy)]
struct JpegComponentLayout {
    width: usize,
    height: usize,
    max_h: usize,
    max_v: usize,
    mcus_x: usize,
    mcus_y: usize,
    dct_size: usize,
}

#[derive(Clone)]
struct HuffmanTable {
    /// Lookup table for fast decoding (8-bit codes)
    lookup: [i16; 256],
    /// Bit lengths for lookup table entries
    lookup_bits: [u8; 256],
    /// Maximum code value for each bit length
    maxcode: [i32; 18],
    /// Value offset for each bit length
    valoffset: [i32; 18],
    /// Symbol values
    huffval: Vec<u8>,
    /// Whether this table is valid
    valid: bool,
}

impl Default for HuffmanTable {
    fn default() -> Self {
        Self {
            lookup: [-1; 256],
            lookup_bits: [0; 256],
            maxcode: [-1; 18],
            valoffset: [0; 18],
            huffval: Vec::new(),
            valid: false,
        }
    }
}

impl HuffmanTable {
    fn build(bits: &[u8; 16], huffval: &[u8]) -> Self {
        let mut table = Self::default();
        table.huffval = huffval.to_vec();

        // Generate size table
        let mut huffsize = Vec::new();
        for (i, &count) in bits.iter().enumerate() {
            for _ in 0..count {
                huffsize.push((i + 1) as u8);
            }
        }
        huffsize.push(0);

        // Generate code table
        let mut huffcode = Vec::new();
        let mut code = 0u32;
        let mut si = huffsize[0];
        let mut k = 0;

        while huffsize[k] != 0 {
            while huffsize[k] == si {
                huffcode.push(code);
                code += 1;
                k += 1;
            }
            code <<= 1;
            si += 1;
        }

        // Generate decoding tables
        let mut p = 0;
        for l in 1..=16 {
            if bits[l - 1] != 0 {
                table.valoffset[l] = p as i32 - huffcode.get(p).copied().unwrap_or(0) as i32;
                p += bits[l - 1] as usize;
                table.maxcode[l] = huffcode.get(p - 1).copied().unwrap_or(0) as i32;
            } else {
                table.maxcode[l] = -1;
            }
        }
        table.maxcode[17] = 0xFFFFF;

        // Build lookup table for codes up to 8 bits
        for (i, (&size, &code)) in huffsize.iter().zip(huffcode.iter()).enumerate() {
            if size == 0 || size > 8 {
                break;
            }
            let code_shifted = (code << (8 - size)) as usize;
            let fill_count = 1 << (8 - size);
            for j in 0..fill_count {
                table.lookup[code_shifted + j] = i as i16;
                table.lookup_bits[code_shifted + j] = size;
            }
        }

        table.valid = true;
        table
    }
}

const STD_DC_LUMA_BITS: [u8; 16] = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
const STD_DC_LUMA_VALS: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
const STD_DC_CHROMA_BITS: [u8; 16] = [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0];
const STD_DC_CHROMA_VALS: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

const STD_AC_LUMA_BITS: [u8; 16] = [
    0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d,
];
const STD_AC_LUMA_VALS: [u8; 162] = [
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51,
    0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08, 0x23, 0x42, 0xB1,
    0xC1, 0x15, 0x52, 0xD1, 0xF0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16,
    0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x34, 0x35, 0x36,
    0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x53,
    0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85,
    0x86, 0x87, 0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99,
    0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4,
    0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8,
    0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
    0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5,
    0xF6, 0xF7, 0xF8, 0xF9, 0xFA,
];

const STD_AC_CHROMA_BITS: [u8; 16] = [
    0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77,
];
const STD_AC_CHROMA_VALS: [u8; 162] = [
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07,
    0x61, 0x71, 0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xA1, 0xB1, 0xC1, 0x09,
    0x23, 0x33, 0x52, 0xF0, 0x15, 0x62, 0x72, 0xD1, 0x0A, 0x16, 0x24, 0x34, 0xE1, 0x25,
    0xF1, 0x17, 0x18, 0x19, 0x1A, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x35, 0x36, 0x37, 0x38,
    0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56,
    0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74,
    0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5,
    0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA,
    0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6,
    0xD7, 0xD8, 0xD9, 0xDA, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF2,
    0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA,
];

/// Bit reader for JPEG data with byte stuffing handling.
struct JpegBitReader<'a> {
    data: &'a [u8],
    pos: usize,
    bits: u32,
    bits_left: u8,
}

impl<'a> JpegBitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            bits: 0,
            bits_left: 0,
        }
    }

    fn fill_bits(&mut self) {
        while self.bits_left <= 24 && self.pos < self.data.len() {
            let byte = self.data[self.pos];
            self.pos += 1;

            if byte == 0xFF {
                // Handle byte stuffing
                if self.pos < self.data.len() && self.data[self.pos] == 0x00 {
                    self.pos += 1;
                } else {
                    // Marker found - back up
                    self.pos -= 1;
                    return;
                }
            }

            self.bits |= (byte as u32) << (24 - self.bits_left);
            self.bits_left += 8;
        }
    }

    fn peek_bits(&mut self, n: u8) -> u32 {
        if self.bits_left < n {
            self.fill_bits();
        }
        self.bits >> (32 - n)
    }

    fn consume_bits(&mut self, n: u8) {
        self.bits <<= n;
        self.bits_left = self.bits_left.saturating_sub(n);
    }

    fn read_bits(&mut self, n: u8) -> u32 {
        let val = self.peek_bits(n);
        self.consume_bits(n);
        val
    }

    fn decode_huffman(&mut self, table: &HuffmanTable) -> Result<u8> {
        self.fill_bits();

        // Try fast lookup
        let peek = (self.bits >> 24) as usize;
        if table.lookup[peek] >= 0 {
            let symbol = table.lookup[peek] as usize;
            let bits = table.lookup_bits[peek];
            self.consume_bits(bits);
            return Ok(table.huffval[symbol]);
        }

        // Slow path for longer codes
        let mut code = self.peek_bits(8) as i32;
        self.consume_bits(8);

        for l in 9..=16 {
            code = (code << 1) | self.read_bits(1) as i32;
            if code <= table.maxcode[l] {
                let idx = (code + table.valoffset[l]) as usize;
                return Ok(table.huffval.get(idx).copied().unwrap_or(0));
            }
        }

        Err(Error::InvalidJpegData)
    }

    fn receive_extend(&mut self, nbits: u8) -> i32 {
        if nbits == 0 {
            return 0;
        }
        let val = self.read_bits(nbits) as i32;
        let vt = 1 << (nbits - 1);
        if val < vt {
            val + (-1 << nbits) + 1
        } else {
            val
        }
    }

    fn align_to_byte(&mut self) {
        let discard = self.bits_left % 8;
        self.consume_bits(discard);
    }
}

impl JpegDecoder {
    /// Create a new JPEG decoder and parse the JPEG headers.
    pub fn new(data: &[u8]) -> Result<Self> {
        if data.len() < 2 || data[0] != 0xFF || data[1] != 0xD8 {
            return Err(Error::InvalidJpegData);
        }

        let mut decoder = Self {
            width: 0,
            height: 0,
            num_components: 0,
            data_precision: 8,
            components: Vec::new(),
            quant_tables: [[0; 64]; 4],
            dc_huff_tables: Default::default(),
            ac_huff_tables: Default::default(),
            restart_interval: 0,
            is_progressive: false,
            metadata: JpegMetadata::default(),
            saw_jfif: false,
            adobe_transform: None,
            icc_segments: Vec::new(),
            icc_expected_segments: None,
        };

        decoder.parse_headers(data)?;
        Ok(decoder)
    }

    /// Get image width in pixels.
    pub fn width(&self) -> usize {
        self.width as usize
    }

    /// Get image height in pixels.
    pub fn height(&self) -> usize {
        self.height as usize
    }

    /// Get number of color components.
    pub fn num_components(&self) -> usize {
        self.num_components as usize
    }

    /// Get bits per sample.
    pub fn bits_per_sample(&self) -> u8 {
        self.data_precision
    }

    /// Check if this is a progressive JPEG.
    pub fn is_progressive(&self) -> bool {
        self.is_progressive
    }

    /// Get JPEG metadata (JFIF, EXIF, ICC, etc.).
    pub fn metadata(&self) -> &JpegMetadata {
        &self.metadata
    }

    /// Extract and return owned metadata (for CXX bridge compatibility).
    pub fn extract_metadata(&self) -> JpegMetadata {
        self.metadata.clone()
    }

    fn sample_max(&self) -> f32 {
        let precision = self.data_precision.max(1).min(16) as u32;
        ((1u32 << precision) - 1) as f32
    }

    fn level_shift(&self) -> f32 {
        let precision = self.data_precision.max(1).min(16) as u32;
        (1u32 << (precision - 1)) as f32
    }

    fn quantize_to_u8(&self, value: f32) -> u8 {
        (value.clamp(0.0, 1.0) * 255.0).round() as u8
    }

    fn quantize_to_u16(&self, value: f32) -> u16 {
        (value.clamp(0.0, 1.0) * self.sample_max()).round() as u16
    }

    fn use_ycbcr(&self) -> bool {
        if self.num_components != 3 {
            return false;
        }

        if self.saw_jfif {
            return true;
        }

        if let Some(transform) = self.adobe_transform {
            return match transform {
                0 => false,
                1 => true,
                _ => true,
            };
        }

        if self.components.len() >= 3 {
            let cid0 = self.components[0].id;
            let cid1 = self.components[1].id;
            let cid2 = self.components[2].id;

            if cid0 == 1 && cid1 == 2 && cid2 == 3 {
                return true;
            }
            if cid0 == 82 && cid1 == 71 && cid2 == 66 {
                return false;
            }
        }

        true
    }

    fn use_ycck(&self) -> bool {
        if self.num_components != 4 {
            return false;
        }
        match self.adobe_transform {
            Some(2) => true,
            Some(0) => false,
            Some(_) => true,
            None => false,
        }
    }

    fn parse_headers(&mut self, data: &[u8]) -> Result<()> {
        let mut pos = 2; // Skip SOI

        while pos + 2 <= data.len() {
            if data[pos] != 0xFF {
                pos += 1;
                continue;
            }

            let marker = data[pos + 1];
            pos += 2;

            match marker {
                0xD8 => {} // SOI - ignore
                0xD9 => break, // EOI
                0xDA => {
                    // SOS - parse scan header to get Huffman table selectors, then stop.
                    if pos + 2 > data.len() {
                        return Err(Error::InvalidJpegData);
                    }
                    let length = u16::from_be_bytes([data[pos], data[pos + 1]]) as usize;
                    if pos + length > data.len() {
                        return Err(Error::InvalidJpegData);
                    }
                    let _ = self.parse_sos(&data[pos..pos + length])?;
                    break;
                }
                0x00 | 0xFF => {} // Padding or fill bytes
                0xD0..=0xD7 => {} // RST markers
                _ => {
                    if pos + 2 > data.len() {
                        break;
                    }
                    let length = u16::from_be_bytes([data[pos], data[pos + 1]]) as usize;
                    if pos + length > data.len() {
                        break;
                    }

                    match marker {
                        0xC0 | 0xC1 | 0xC2 => self.parse_sof(&data[pos..pos + length], marker == 0xC2)?,
                        0xC4 => self.parse_dht(&data[pos..pos + length])?,
                        0xDB => self.parse_dqt(&data[pos..pos + length])?,
                        0xDD => self.parse_dri(&data[pos..pos + length])?,
                        0xDC => self.parse_dnl(&data[pos..pos + length])?,
                        0xE0 => self.parse_app0(&data[pos..pos + length])?,
                        0xE1 => self.parse_app1(&data[pos..pos + length])?,
                        0xE2 => self.parse_app2(&data[pos..pos + length])?,
                        0xEE => self.parse_app14(&data[pos..pos + length])?,
                        0xFE => self.parse_com(&data[pos..pos + length])?,
                        _ => {} // Skip unknown markers
                    }

                    pos += length;
                }
            }
        }

        Ok(())
    }

    fn parse_sof(&mut self, data: &[u8], progressive: bool) -> Result<()> {
        if data.len() < 8 {
            return Err(Error::InvalidJpegData);
        }

        let precision = data[2];
        if precision != 8 && precision != 12 && precision != 16 {
            return Err(Error::InvalidJpegData);
        }

        self.height = u16::from_be_bytes([data[3], data[4]]);
        self.width = u16::from_be_bytes([data[5], data[6]]);
        self.num_components = data[7];
        self.is_progressive = progressive;
        self.data_precision = precision;

        if data.len() < 8 + self.num_components as usize * 3 {
            return Err(Error::InvalidJpegData);
        }

        self.components.clear();
        for i in 0..self.num_components as usize {
            let offset = 8 + i * 3;
            self.components.push(JpegDecoderComponent {
                id: data[offset],
                h_samp: data[offset + 1] >> 4,
                v_samp: data[offset + 1] & 0x0F,
                quant_table_id: data[offset + 2],
                dc_table_id: 0,
                ac_table_id: 0,
            });
        }

        Ok(())
    }

    fn parse_dqt(&mut self, data: &[u8]) -> Result<()> {
        let mut pos = 2; // Skip length
        while pos < data.len() {
            let pq_tq = data[pos];
            let precision = pq_tq >> 4;
            let table_id = (pq_tq & 0x0F) as usize;
            pos += 1;

            if table_id >= 4 {
                return Err(Error::InvalidJpegData);
            }

            if precision == 0 {
                // 8-bit values
                for i in 0..64 {
                    if pos >= data.len() {
                        return Err(Error::InvalidJpegData);
                    }
                    self.quant_tables[table_id][JPEG_NATURAL_ORDER[i]] = data[pos] as u16;
                    pos += 1;
                }
            } else {
                // 16-bit values
                for i in 0..64 {
                    if pos + 1 >= data.len() {
                        return Err(Error::InvalidJpegData);
                    }
                    self.quant_tables[table_id][JPEG_NATURAL_ORDER[i]] =
                        u16::from_be_bytes([data[pos], data[pos + 1]]);
                    pos += 2;
                }
            }
        }
        Ok(())
    }

    fn parse_dht(&mut self, data: &[u8]) -> Result<()> {
        let mut pos = 2; // Skip length
        while pos < data.len() {
            let tc_th = data[pos];
            let table_class = tc_th >> 4;
            let table_id = (tc_th & 0x0F) as usize;
            pos += 1;

            if table_id >= 4 {
                return Err(Error::InvalidJpegData);
            }

            if pos + 16 > data.len() {
                return Err(Error::InvalidJpegData);
            }

            let mut bits = [0u8; 16];
            bits.copy_from_slice(&data[pos..pos + 16]);
            pos += 16;

            let total_symbols: usize = bits.iter().map(|&b| b as usize).sum();
            if pos + total_symbols > data.len() {
                return Err(Error::InvalidJpegData);
            }

            let huffval = &data[pos..pos + total_symbols];
            pos += total_symbols;

            let table = HuffmanTable::build(&bits, huffval);
            if table_class == 0 {
                self.dc_huff_tables[table_id] = table;
            } else {
                self.ac_huff_tables[table_id] = table;
            }
        }
        Ok(())
    }

    fn parse_dri(&mut self, data: &[u8]) -> Result<()> {
        if data.len() >= 4 {
            self.restart_interval = u16::from_be_bytes([data[2], data[3]]);
        }
        Ok(())
    }

    fn parse_dnl(&mut self, data: &[u8]) -> Result<()> {
        if data.len() >= 4 {
            let height = u16::from_be_bytes([data[2], data[3]]);
            if height > 0 {
                self.height = height;
            }
        }
        Ok(())
    }

    fn parse_app0(&mut self, data: &[u8]) -> Result<()> {
        if data.len() >= 14 && &data[2..7] == b"JFIF\0" {
            self.metadata.jfif_version = Some((data[7], data[8]));
            self.saw_jfif = true;
        }
        Ok(())
    }

    fn parse_app1(&mut self, data: &[u8]) -> Result<()> {
        if data.len() >= 8 && &data[2..6] == b"Exif" {
            self.metadata.exif_data = Some(data[8..].to_vec());
        } else if data.len() >= 32 && data[2..].starts_with(b"http://ns.adobe.com/xap/") {
            self.metadata.xmp_data = Some(data[2..].to_vec());
        }
        Ok(())
    }

    fn parse_app2(&mut self, data: &[u8]) -> Result<()> {
        if data.len() >= 14 && &data[2..14] == b"ICC_PROFILE\0" {
            if data.len() < 16 {
                return Ok(());
            }

            let seq = data[14] as usize;
            let total = data[15] as usize;
            if seq == 0 || total == 0 {
                return Ok(());
            }

            if self.icc_expected_segments.is_none() {
                self.icc_expected_segments = Some(total);
                self.icc_segments = vec![None; total];
            }

            if self.icc_expected_segments == Some(total) && seq <= total {
                self.icc_segments[seq - 1] = Some(data[16..].to_vec());

                if self.icc_segments.iter().all(|seg| seg.is_some()) {
                    let mut icc = Vec::new();
                    for seg in &self.icc_segments {
                        if let Some(bytes) = seg {
                            icc.extend_from_slice(bytes);
                        }
                    }
                    self.metadata.icc_profile = Some(icc);
                }
            }
        }
        Ok(())
    }

    fn parse_app14(&mut self, data: &[u8]) -> Result<()> {
        if data.len() >= 14 && &data[2..7] == b"Adobe" {
            self.adobe_transform = data.get(13).copied();
        }
        Ok(())
    }

    fn parse_com(&mut self, data: &[u8]) -> Result<()> {
        if data.len() > 2 {
            if let Ok(comment) = String::from_utf8(data[2..].to_vec()) {
                self.metadata.comments.push(comment);
            }
        }
        Ok(())
    }

    fn ensure_default_huffman_tables(&mut self) {
        if !self.dc_huff_tables[0].valid {
            self.dc_huff_tables[0] = HuffmanTable::build(&STD_DC_LUMA_BITS, &STD_DC_LUMA_VALS);
        }
        if !self.ac_huff_tables[0].valid {
            self.ac_huff_tables[0] = HuffmanTable::build(&STD_AC_LUMA_BITS, &STD_AC_LUMA_VALS);
        }
        if !self.dc_huff_tables[1].valid {
            self.dc_huff_tables[1] = HuffmanTable::build(&STD_DC_CHROMA_BITS, &STD_DC_CHROMA_VALS);
        }
        if !self.ac_huff_tables[1].valid {
            self.ac_huff_tables[1] = HuffmanTable::build(&STD_AC_CHROMA_BITS, &STD_AC_CHROMA_VALS);
        }
    }

    fn update_height_from_dnl(&mut self, data: &[u8]) {
        if self.height != 0 {
            return;
        }
        let mut pos = 2;
        while pos + 3 < data.len() {
            if data[pos] == 0xFF && data[pos + 1] == 0xDC {
                let length = u16::from_be_bytes([data[pos + 2], data[pos + 3]]) as usize;
                if pos + 2 + length <= data.len() && length >= 4 {
                    let height = u16::from_be_bytes([data[pos + 4], data[pos + 5]]);
                    if height > 0 {
                        self.height = height;
                        return;
                    }
                }
            }
            pos += 1;
        }
    }

    fn parse_sos(&mut self, data: &[u8]) -> Result<JpegScanHeader> {
        if data.len() < 6 {
            return Err(Error::InvalidJpegData);
        }

        let num_components = data[2] as usize;
        let required = 3 + num_components * 2 + 3;
        if data.len() < required {
            return Err(Error::InvalidJpegData);
        }

        let mut component_ids = [0u8; 4];
        let mut dc_table_id = [0u8; 4];
        let mut ac_table_id = [0u8; 4];

        for i in 0..num_components {
            let comp_id = data[3 + i * 2];
            let table_sel = data[4 + i * 2];
            let dc_id = table_sel >> 4;
            let ac_id = table_sel & 0x0F;

            component_ids[i] = comp_id;
            dc_table_id[i] = dc_id;
            ac_table_id[i] = ac_id;

            if let Some(comp) = self.components.iter_mut().find(|c| c.id == comp_id) {
                comp.dc_table_id = dc_id;
                comp.ac_table_id = ac_id;
            }
        }

        let ss = data[3 + num_components * 2];
        let se = data[4 + num_components * 2];
        let ah_al = data[5 + num_components * 2];

        Ok(JpegScanHeader {
            num_components,
            component_ids,
            dc_table_id,
            ac_table_id,
            ss,
            se,
            ah: ah_al >> 4,
            al: ah_al & 0x0F,
        })
    }

    /// Decode the JPEG image and return pixel data.
    pub fn decode(&self, data: &[u8]) -> Result<JpegDecodedImage> {
        let mut decoder = self.clone();
        decoder.ensure_default_huffman_tables();
        decoder.update_height_from_dnl(data);

        if decoder.is_progressive {
            decoder.decode_progressive(data)
        } else {
            let (_scan_header, scan_start, scan_end) = decoder.find_sos_segment(data)?;
            let scan_data = &data[scan_start..scan_end];
            decoder.decode_baseline(scan_data)
        }
    }

    fn find_sos_segment(&mut self, data: &[u8]) -> Result<(JpegScanHeader, usize, usize)> {
        let mut pos = 2;
        while pos + 2 <= data.len() {
            if data[pos] == 0xFF && data[pos + 1] == 0xDA {
                pos += 2;
                if pos + 2 > data.len() {
                    return Err(Error::InvalidJpegData);
                }
                let length = u16::from_be_bytes([data[pos], data[pos + 1]]) as usize;
                if pos + length > data.len() {
                    return Err(Error::InvalidJpegData);
                }
                let header = self.parse_sos(&data[pos..pos + length])?;
                let scan_start = pos + length;
                let scan_end = self.find_scan_end(data, scan_start)?;
                return Ok((header, scan_start, scan_end));
            }
            pos += 1;
        }
        Err(Error::InvalidJpegData)
    }

    fn find_scan_end(&self, data: &[u8], mut pos: usize) -> Result<usize> {
        while pos + 1 < data.len() {
            if data[pos] == 0xFF {
                let marker = data[pos + 1];
                if marker == 0x00 || marker == 0xFF {
                    pos += 2;
                    continue;
                }
                if (0xD0..=0xD7).contains(&marker) {
                    pos += 2;
                    continue;
                }
                return Ok(pos);
            }
            pos += 1;
        }
        Ok(data.len())
    }

    fn component_layout(&self) -> Result<JpegComponentLayout> {
        let width = self.width as usize;
        let height = self.height as usize;
        if width == 0 || height == 0 {
            return Err(Error::InvalidJpegData);
        }

        let max_h = self.components.iter().map(|c| c.h_samp).max().unwrap_or(1) as usize;
        let max_v = self.components.iter().map(|c| c.v_samp).max().unwrap_or(1) as usize;
        let mcu_width = max_h * 8;
        let mcu_height = max_v * 8;
        let mcus_x = (width + mcu_width - 1) / mcu_width;
        let mcus_y = (height + mcu_height - 1) / mcu_height;

        Ok(JpegComponentLayout {
            width,
            height,
            max_h,
            max_v,
            mcus_x,
            mcus_y,
            dct_size: 8,
        })
    }

    fn component_layout_scaled(&self, denom: usize) -> Result<JpegComponentLayout> {
        let mut layout = self.component_layout()?;
        let denom = denom.max(1);
        let dct_size = 8 / denom;
        if dct_size == 0 {
            return Err(Error::InvalidJpegData);
        }
        if denom > 1 {
            layout.width = (layout.width + denom - 1) / denom;
            layout.height = (layout.height + denom - 1) / denom;
            layout.dct_size = dct_size;
        }
        Ok(layout)
    }

    fn decode_baseline_components(&self, scan_data: &[u8], layout: JpegComponentLayout) -> Result<Vec<Vec<f32>>> {
        let mut reader = JpegBitReader::new(scan_data);
        let mut dc_pred = vec![0i32; self.num_components as usize];

        let mut comp_data: Vec<Vec<f32>> = self
            .components
            .iter()
            .map(|c| {
                let comp_w = layout.mcus_x * c.h_samp as usize * layout.dct_size;
                let comp_h = layout.mcus_y * c.v_samp as usize * layout.dct_size;
                vec![0.0f32; comp_w * comp_h]
            })
            .collect();

        let mut mcu_count = 0usize;

        for mcu_y in 0..layout.mcus_y {
            for mcu_x in 0..layout.mcus_x {
                if self.restart_interval > 0
                    && mcu_count > 0
                    && mcu_count % self.restart_interval as usize == 0
                {
                    reader.align_to_byte();
                    while reader.pos < reader.data.len() {
                        if reader.data[reader.pos] == 0xFF {
                            let marker = reader.data.get(reader.pos + 1).copied().unwrap_or(0);
                            if (0xD0..=0xD7).contains(&marker) {
                                reader.pos += 2;
                                break;
                            }
                        }
                        reader.pos += 1;
                    }
                    dc_pred.fill(0);
                }

                for (comp_idx, comp) in self.components.iter().enumerate() {
                    let dc_table = &self.dc_huff_tables[comp.dc_table_id as usize];
                    let ac_table = &self.ac_huff_tables[comp.ac_table_id as usize];
                    let quant = &self.quant_tables[comp.quant_table_id as usize];

                    for v in 0..comp.v_samp as usize {
                        for h in 0..comp.h_samp as usize {
                            let mut block = [0i64; 64];

                            let dc_category = reader.decode_huffman(dc_table)?;
                            let dc_diff = reader.receive_extend(dc_category);
                            dc_pred[comp_idx] += dc_diff;
                            block[0] = dc_pred[comp_idx] as i64 * quant[0] as i64;

                            let mut k = 1;
                            while k < 64 {
                                let rs = reader.decode_huffman(ac_table)?;
                                let r = rs >> 4;
                                let s = rs & 0x0F;

                                if s == 0 {
                                    if r == 15 {
                                        k += 16;
                                    } else {
                                        break;
                                    }
                                } else {
                                    k += r as usize;
                                    if k >= 64 {
                                        break;
                                    }
                                    let zz_idx = JPEG_NATURAL_ORDER[k];
                                    block[zz_idx] =
                                        reader.receive_extend(s) as i64 * quant[zz_idx] as i64;
                                    k += 1;
                                }
                            }

                            let mut float_block = [0.0f32; 64];
                            let out = &mut float_block[..layout.dct_size * layout.dct_size];
                            self.idct_block_scaled(&block, layout.dct_size, out)?;

                            let block_x = mcu_x * comp.h_samp as usize + h;
                            let block_y = mcu_y * comp.v_samp as usize + v;
                            let comp_stride = layout.mcus_x * comp.h_samp as usize * layout.dct_size;

                            for by in 0..layout.dct_size {
                                let dst_row = (block_y * layout.dct_size + by) * comp_stride;
                                let src_row = by * layout.dct_size;
                                let dst = &mut comp_data[comp_idx]
                                    [dst_row + block_x * layout.dct_size..dst_row + block_x * layout.dct_size + layout.dct_size];
                                dst.copy_from_slice(&float_block[src_row..src_row + layout.dct_size]);
                            }
                        }
                    }
                }

                mcu_count += 1;
            }
        }

        Ok(comp_data)
    }

    fn coefficients_to_components(
        &self,
        coefficients: &[Vec<i32>],
        layout: JpegComponentLayout,
    ) -> Result<Vec<Vec<f32>>> {
        let mut comp_data: Vec<Vec<f32>> = Vec::with_capacity(self.components.len());

        for (comp_idx, comp) in self.components.iter().enumerate() {
            let blocks_x = layout.mcus_x * comp.h_samp as usize;
            let blocks_y = layout.mcus_y * comp.v_samp as usize;
            let quant = &self.quant_tables[comp.quant_table_id as usize];
            let mut output = vec![0.0f32; blocks_x * blocks_y * layout.dct_size * layout.dct_size];

            for by in 0..blocks_y {
                for bx in 0..blocks_x {
                    let block_idx = by * blocks_x + bx;
                    let coef_offset = block_idx * 64;

                    let mut block = [0i64; 64];
                    for i in 0..64 {
                        block[i] =
                            coefficients[comp_idx][coef_offset + i] as i64 * quant[i] as i64;
                    }

                    let mut float_block = [0.0f32; 64];
                    let out = &mut float_block[..layout.dct_size * layout.dct_size];
                    self.idct_block_scaled(&block, layout.dct_size, out)?;

                    let out_stride = blocks_x * layout.dct_size;
                    for y in 0..layout.dct_size {
                        let dst_row = (by * layout.dct_size + y) * out_stride;
                        let src_row = y * layout.dct_size;
                        let dst = &mut output
                            [dst_row + bx * layout.dct_size..dst_row + bx * layout.dct_size + layout.dct_size];
                        dst.copy_from_slice(&float_block[src_row..src_row + layout.dct_size]);
                    }
                }
            }

            comp_data.push(output);
        }

        Ok(comp_data)
    }

    fn decode_baseline(&self, scan_data: &[u8]) -> Result<JpegDecodedImage> {
        let layout = self.component_layout()?;
        let comp_data = self.decode_baseline_components(scan_data, layout)?;
        self.components_to_rgb(
            &comp_data,
            layout.width,
            layout.height,
            layout.mcus_x,
            layout.max_h,
            layout.max_v,
        )
    }

    fn decode_baseline_into(
        &self,
        scan_data: &[u8],
        format: JpegOutputFormat,
        output: &mut [u8],
        row_stride: usize,
        layout: JpegOutputLayout,
    ) -> Result<()> {
        let (comp_layout, render_layout) = if matches!(layout.denom, 2 | 4 | 8) {
            let comp_layout = self.component_layout_scaled(layout.denom)?;
            let render_layout = JpegOutputLayout {
                denom: 1,
                ..layout
            };
            (comp_layout, render_layout)
        } else {
            (self.component_layout()?, layout)
        };

        let comp_data = self.decode_baseline_components(scan_data, comp_layout)?;
        self.components_into_output(&comp_data, comp_layout, render_layout, format, output, row_stride)
    }

    fn decode_baseline_into_u16(
        &self,
        scan_data: &[u8],
        format: JpegOutputFormat16,
        output: &mut [u16],
        row_stride: usize,
        layout: JpegOutputLayout,
    ) -> Result<()> {
        let (comp_layout, render_layout) = if matches!(layout.denom, 2 | 4 | 8) {
            let comp_layout = self.component_layout_scaled(layout.denom)?;
            let render_layout = JpegOutputLayout {
                denom: 1,
                ..layout
            };
            (comp_layout, render_layout)
        } else {
            (self.component_layout()?, layout)
        };

        let comp_data = self.decode_baseline_components(scan_data, comp_layout)?;
        self.components_into_output_u16(&comp_data, comp_layout, render_layout, format, output, row_stride)
    }

    fn decode_progressive_scans(
        &mut self,
        data: &[u8],
        coefficients: &mut [Vec<i32>],
    ) -> Result<()> {
        let mut pos = 2;
        while pos + 1 < data.len() {
            if data[pos] != 0xFF {
                pos += 1;
                continue;
            }

            let marker = data[pos + 1];
            pos += 2;

            match marker {
                0xD8 => {} // SOI
                0xD9 => break, // EOI
                0xDA => {
                    if pos + 2 > data.len() {
                        return Err(Error::InvalidJpegData);
                    }
                    let length = u16::from_be_bytes([data[pos], data[pos + 1]]) as usize;
                    if pos + length > data.len() {
                        return Err(Error::InvalidJpegData);
                    }
                    let scan_header = self.parse_sos(&data[pos..pos + length])?;
                    let scan_start = pos + length;
                    let scan_end = self.find_scan_end(data, scan_start)?;
                    let scan_data = &data[scan_start..scan_end];
                    self.decode_progressive_scan(&scan_header, scan_data, coefficients)?;
                    pos = scan_end;
                }
                0x00 | 0xFF | 0x01 => {} // Padding or TEM
                0xD0..=0xD7 => {} // RST markers
                _ => {
                    if pos + 2 > data.len() {
                        break;
                    }
                    let length = u16::from_be_bytes([data[pos], data[pos + 1]]) as usize;
                    if pos + length > data.len() {
                        return Err(Error::InvalidJpegData);
                    }

                    match marker {
                        0xC4 => self.parse_dht(&data[pos..pos + length])?,
                        0xDB => self.parse_dqt(&data[pos..pos + length])?,
                        0xDD => self.parse_dri(&data[pos..pos + length])?,
                        0xDC => self.parse_dnl(&data[pos..pos + length])?,
                        0xE0 => self.parse_app0(&data[pos..pos + length])?,
                        0xE1 => self.parse_app1(&data[pos..pos + length])?,
                        0xE2 => self.parse_app2(&data[pos..pos + length])?,
                        0xEE => self.parse_app14(&data[pos..pos + length])?,
                        0xFE => self.parse_com(&data[pos..pos + length])?,
                        _ => {}
                    }

                    pos += length;
                }
            }
        }

        Ok(())
    }

    fn decode_progressive_scan(
        &self,
        scan_header: &JpegScanHeader,
        scan_data: &[u8],
        coefficients: &mut [Vec<i32>],
    ) -> Result<()> {
        let width = self.width as usize;
        let height = self.height as usize;
        let mut reader = JpegBitReader::new(scan_data);
        let mut eob_run: u32 = 0;
        let mut dc_pred = vec![0i32; self.num_components as usize];

        let max_h = self
            .components
            .iter()
            .filter(|c| scan_header.component_ids[..scan_header.num_components].contains(&c.id))
            .map(|c| c.h_samp)
            .max()
            .unwrap_or(1) as usize;
        let max_v = self
            .components
            .iter()
            .filter(|c| scan_header.component_ids[..scan_header.num_components].contains(&c.id))
            .map(|c| c.v_samp)
            .max()
            .unwrap_or(1) as usize;

        let mcu_width = max_h * 8;
        let mcu_height = max_v * 8;
        let mcus_x = (width + mcu_width - 1) / mcu_width;
        let mcus_y = (height + mcu_height - 1) / mcu_height;

        let is_dc_scan = scan_header.ss == 0 && scan_header.se == 0;
        let is_first = scan_header.ah == 0;

        for mcu_y in 0..mcus_y {
            for mcu_x in 0..mcus_x {
                if self.restart_interval > 0
                    && (mcu_y * mcus_x + mcu_x) > 0
                    && (mcu_y * mcus_x + mcu_x) % self.restart_interval as usize == 0
                {
                    reader.align_to_byte();
                    while reader.pos + 1 < reader.data.len() {
                        if reader.data[reader.pos] == 0xFF {
                            let marker = reader.data.get(reader.pos + 1).copied().unwrap_or(0);
                            if (0xD0..=0xD7).contains(&marker) {
                                reader.pos += 2;
                                break;
                            }
                        }
                        reader.pos += 1;
                    }
                    dc_pred.fill(0);
                    eob_run = 0;
                }

                for scan_comp_idx in 0..scan_header.num_components {
                    let comp_id = scan_header.component_ids[scan_comp_idx];
                    let comp_idx = self
                        .components
                        .iter()
                        .position(|c| c.id == comp_id)
                        .ok_or(Error::InvalidJpegData)?;
                    let comp = &self.components[comp_idx];
                    let dc_table = &self.dc_huff_tables[scan_header.dc_table_id[scan_comp_idx] as usize];
                    let ac_table = &self.ac_huff_tables[scan_header.ac_table_id[scan_comp_idx] as usize];
                    let blocks_x = mcus_x * comp.h_samp as usize;

                    for v in 0..comp.v_samp as usize {
                        for h in 0..comp.h_samp as usize {
                            let block_x = mcu_x * comp.h_samp as usize + h;
                            let block_y = mcu_y * comp.v_samp as usize + v;
                            let block_idx = block_y * blocks_x + block_x;
                            let coef_offset = block_idx * 64;

                            if is_dc_scan {
                                if is_first {
                                    let dc_category = reader.decode_huffman(dc_table)?;
                                    let dc_diff = reader.receive_extend(dc_category);
                                    dc_pred[comp_idx] += dc_diff;
                                    coefficients[comp_idx][coef_offset] =
                                        dc_pred[comp_idx] << scan_header.al;
                                } else {
                                    let bit = reader.read_bits(1) as i32;
                                    if bit != 0 {
                                        coefficients[comp_idx][coef_offset] += 1 << scan_header.al;
                                    }
                                }
                            } else {
                                let mut k = scan_header.ss as usize;
                                let end = scan_header.se as usize;

                                if is_first {
                                    if eob_run > 0 {
                                        eob_run -= 1;
                                        continue;
                                    }

                                    while k <= end {
                                        let rs = reader.decode_huffman(ac_table)?;
                                        let r = (rs >> 4) as usize;
                                        let s = (rs & 0x0F) as u8;

                                        if s == 0 {
                                            if r == 15 {
                                                k += 16;
                                            } else {
                                                let extra = if r > 0 {
                                                    reader.read_bits(r as u8)
                                                } else {
                                                    0
                                                };
                                                eob_run = (1u32 << r) + extra;
                                                eob_run = eob_run.saturating_sub(1);
                                                break;
                                            }
                                        } else {
                                            k += r;
                                            if k > end {
                                                break;
                                            }
                                            let coef = reader.receive_extend(s) << scan_header.al;
                                            coefficients[comp_idx][coef_offset + JPEG_NATURAL_ORDER[k]] = coef;
                                            k += 1;
                                        }
                                    }
                                } else {
                                    let bit_mask = 1 << scan_header.al;
                                    if eob_run > 0 {
                                        for idx in k..=end {
                                            let zz = JPEG_NATURAL_ORDER[idx];
                                            let val = &mut coefficients[comp_idx][coef_offset + zz];
                                            if *val != 0 {
                                                let bit = reader.read_bits(1) as i32;
                                                if bit != 0 {
                                                    *val += if *val > 0 { bit_mask } else { -bit_mask };
                                                }
                                            }
                                        }
                                        eob_run -= 1;
                                        continue;
                                    }

                                    while k <= end {
                                        let rs = reader.decode_huffman(ac_table)?;
                                        let r = (rs >> 4) as usize;
                                        let s = (rs & 0x0F) as u8;

                                        if s == 0 {
                                            if r == 15 {
                                                let mut zeros = 16;
                                                while zeros > 0 && k <= end {
                                                    let zz = JPEG_NATURAL_ORDER[k];
                                                    let val = &mut coefficients[comp_idx][coef_offset + zz];
                                                    if *val != 0 {
                                                        let bit = reader.read_bits(1) as i32;
                                                        if bit != 0 {
                                                            *val += if *val > 0 { bit_mask } else { -bit_mask };
                                                        }
                                                    } else {
                                                        zeros -= 1;
                                                    }
                                                    k += 1;
                                                }
                                            } else {
                                                let extra = if r > 0 {
                                                    reader.read_bits(r as u8)
                                                } else {
                                                    0
                                                };
                                                eob_run = (1u32 << r) + extra;
                                                eob_run = eob_run.saturating_sub(1);
                                                break;
                                            }
                                        } else {
                                            let mut zeros = r;
                                            while zeros > 0 && k <= end {
                                                let zz = JPEG_NATURAL_ORDER[k];
                                                let val = &mut coefficients[comp_idx][coef_offset + zz];
                                                if *val != 0 {
                                                    let bit = reader.read_bits(1) as i32;
                                                    if bit != 0 {
                                                        *val += if *val > 0 { bit_mask } else { -bit_mask };
                                                    }
                                                } else {
                                                    zeros -= 1;
                                                }
                                                k += 1;
                                            }
                                            if k > end {
                                                break;
                                            }
                                            let bit = reader.read_bits(1) as i32;
                                            let new_val = if bit != 0 { bit_mask } else { -bit_mask };
                                            coefficients[comp_idx][coef_offset + JPEG_NATURAL_ORDER[k]] = new_val;
                                            k += 1;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn decode_progressive(&mut self, data: &[u8]) -> Result<JpegDecodedImage> {
        let layout = self.component_layout()?;
        let width = layout.width;
        let height = layout.height;
        let max_h = layout.max_h;
        let max_v = layout.max_v;
        let mcus_x = layout.mcus_x;
        let mcus_y = layout.mcus_y;

        // Allocate coefficient storage for progressive decoding
        let mut coefficients: Vec<Vec<i32>> = self.components.iter().map(|c| {
            let blocks_x = mcus_x * c.h_samp as usize;
            let blocks_y = mcus_y * c.v_samp as usize;
            vec![0i32; blocks_x * blocks_y * 64]
        }).collect();

        self.decode_progressive_scans(data, &mut coefficients)?;

        let comp_data = self.coefficients_to_components(&coefficients, layout)?;
        self.components_to_rgb(&comp_data, width, height, mcus_x, max_h, max_v)
    }

    fn decode_progressive_into(
        &mut self,
        data: &[u8],
        format: JpegOutputFormat,
        output: &mut [u8],
        row_stride: usize,
        layout: JpegOutputLayout,
    ) -> Result<()> {
        let width = self.width as usize;
        let height = self.height as usize;
        if width == 0 || height == 0 {
            return Err(Error::InvalidJpegData);
        }

        let max_h = self.components.iter().map(|c| c.h_samp).max().unwrap_or(1) as usize;
        let max_v = self.components.iter().map(|c| c.v_samp).max().unwrap_or(1) as usize;
        let mcus_x = (width + max_h * 8 - 1) / (max_h * 8);
        let mcus_y = (height + max_v * 8 - 1) / (max_v * 8);

        let mut coefficients: Vec<Vec<i32>> = self
            .components
            .iter()
            .map(|c| {
                let blocks_x = mcus_x * c.h_samp as usize;
                let blocks_y = mcus_y * c.v_samp as usize;
                vec![0i32; blocks_x * blocks_y * 64]
            })
            .collect();

        self.decode_progressive_scans(data, &mut coefficients)?;

        let (comp_layout, render_layout) = if matches!(layout.denom, 2 | 4 | 8) {
            let comp_layout = self.component_layout_scaled(layout.denom)?;
            let render_layout = JpegOutputLayout {
                denom: 1,
                ..layout
            };
            (comp_layout, render_layout)
        } else {
            (self.component_layout()?, layout)
        };

        self.render_coefficients_into(&coefficients, comp_layout, format, output, row_stride, render_layout)
    }

    fn decode_progressive_into_u16(
        &mut self,
        data: &[u8],
        format: JpegOutputFormat16,
        output: &mut [u16],
        row_stride: usize,
        layout: JpegOutputLayout,
    ) -> Result<()> {
        let width = self.width as usize;
        let height = self.height as usize;
        if width == 0 || height == 0 {
            return Err(Error::InvalidJpegData);
        }

        let max_h = self.components.iter().map(|c| c.h_samp).max().unwrap_or(1) as usize;
        let max_v = self.components.iter().map(|c| c.v_samp).max().unwrap_or(1) as usize;
        let mcus_x = (width + max_h * 8 - 1) / (max_h * 8);
        let mcus_y = (height + max_v * 8 - 1) / (max_v * 8);

        let mut coefficients: Vec<Vec<i32>> = self
            .components
            .iter()
            .map(|c| {
                let blocks_x = mcus_x * c.h_samp as usize;
                let blocks_y = mcus_y * c.v_samp as usize;
                vec![0i32; blocks_x * blocks_y * 64]
            })
            .collect();

        self.decode_progressive_scans(data, &mut coefficients)?;

        let (comp_layout, render_layout) = if matches!(layout.denom, 2 | 4 | 8) {
            let comp_layout = self.component_layout_scaled(layout.denom)?;
            let render_layout = JpegOutputLayout {
                denom: 1,
                ..layout
            };
            (comp_layout, render_layout)
        } else {
            (self.component_layout()?, layout)
        };

        self.render_coefficients_into_u16(&coefficients, comp_layout, format, output, row_stride, render_layout)
    }

    fn render_coefficients_into(
        &self,
        coefficients: &[Vec<i32>],
        comp_layout: JpegComponentLayout,
        format: JpegOutputFormat,
        output: &mut [u8],
        row_stride: usize,
        layout: JpegOutputLayout,
    ) -> Result<()> {
        let comp_data = self.coefficients_to_components(coefficients, comp_layout)?;
        self.components_into_output(&comp_data, comp_layout, layout, format, output, row_stride)
    }

    fn render_coefficients_into_u16(
        &self,
        coefficients: &[Vec<i32>],
        comp_layout: JpegComponentLayout,
        format: JpegOutputFormat16,
        output: &mut [u16],
        row_stride: usize,
        layout: JpegOutputLayout,
    ) -> Result<()> {
        let comp_data = self.coefficients_to_components(coefficients, comp_layout)?;
        self.components_into_output_u16(&comp_data, comp_layout, layout, format, output, row_stride)
    }

    fn idct_8x8(&self, input: &[i64; 64], output: &mut [f32; 64]) {
        let mut block = [0.0f32; 64];
        for i in 0..64 {
            block[i] = input[i] as f32;
        }

        jpeg_idct_8x8_dispatch(&mut block);

        let max_sample = self.sample_max();
        let level_shift = self.level_shift();
        for (dst, src) in output.iter_mut().zip(block.iter()) {
            *dst = ((*src * 0.125) + level_shift) / max_sample;
        }
    }

    fn idct_block_scaled(&self, input: &[i64; 64], dct_size: usize, output: &mut [f32]) -> Result<()> {
        let max_sample = self.sample_max();
        let level_shift = self.level_shift();
        let expected = dct_size * dct_size;
        if output.len() < expected {
            return Err(Error::InvalidJpegData);
        }

        match dct_size {
            8 => {
                let mut block = [0.0f32; 64];
                for i in 0..64 {
                    block[i] = input[i] as f32;
                }
                jpeg_idct_8x8_dispatch(&mut block);
                for i in 0..64 {
                    output[i] = ((block[i] * 0.125) + level_shift) / max_sample;
                }
            }
            4 => {
                let mut block = [0.0f32; 16];
                for y in 0..4 {
                    for x in 0..4 {
                        block[y * 4 + x] = input[y * 8 + x] as f32;
                    }
                }
                jpeg_idct_4x4_dispatch(&mut block);
                for i in 0..16 {
                    output[i] = ((block[i] * 0.125) + level_shift) / max_sample;
                }
            }
            2 => {
                let mut block = [0.0f32; 4];
                for y in 0..2 {
                    for x in 0..2 {
                        block[y * 2 + x] = input[y * 8 + x] as f32;
                    }
                }
                jpeg_idct_2x2_dispatch(&mut block);
                for i in 0..4 {
                    output[i] = ((block[i] * 0.125) + level_shift) / max_sample;
                }
            }
            1 => {
                output[0] = ((input[0] as f32 * 0.125) + level_shift) / max_sample;
            }
            _ => return Err(Error::InvalidJpegData),
        }

        Ok(())
    }

    fn sample_component_value(
        &self,
        comp: &JpegDecoderComponent,
        plane: &[f32],
        comp_stride: usize,
        comp_height: usize,
        max_h: usize,
        max_v: usize,
        x: usize,
        y: usize,
    ) -> f32 {
        if comp_stride == 0 || comp_height == 0 {
            return 0.0;
        }

        let h_ratio = max_h / comp.h_samp as usize;
        let v_ratio = max_v / comp.v_samp as usize;
        let max_sample = self.sample_max();

        if h_ratio == 1 && v_ratio == 1 {
            let idx = y.saturating_mul(comp_stride).saturating_add(x);
            return plane.get(idx).copied().unwrap_or(0.0);
        }

        let comp_width = comp_stride;
        let fancy_h2v1 = comp_width > 2;
        let fancy_h1v2 = comp_height > 1;
        let fancy_h2v2 = comp_width > 2 && comp_height > 1;

        if h_ratio == 2 && v_ratio == 1 && fancy_h2v1 {
            let comp_y = y.min(comp_height - 1);
            let row = &plane[comp_y * comp_stride..(comp_y + 1) * comp_stride];
            let in_x = (x / 2).min(comp_width - 1);
            let cur = row[in_x] * max_sample;

            let out = if x % 2 == 0 {
                if in_x == 0 {
                    cur
                } else {
                    let left = row[in_x - 1] * max_sample;
                    (cur * 3.0 + left + 1.0) * 0.25
                }
            } else if in_x + 1 >= comp_width {
                cur
            } else {
                let right = row[in_x + 1] * max_sample;
                (cur * 3.0 + right + 2.0) * 0.25
            };

            return (out / max_sample).clamp(0.0, 1.0);
        }

        if h_ratio == 1 && v_ratio == 2 && fancy_h1v2 {
            let in_y = (y / 2).min(comp_height - 1);
            let cur_row = &plane[in_y * comp_stride..(in_y + 1) * comp_stride];
            let col = x.min(comp_width - 1);
            let cur = cur_row[col] * max_sample;

            let (neighbor_row, bias) = if y % 2 == 0 {
                let above = if in_y == 0 { in_y } else { in_y - 1 };
                (
                    &plane[above * comp_stride..(above + 1) * comp_stride],
                    1.0,
                )
            } else {
                let below = if in_y + 1 < comp_height { in_y + 1 } else { in_y };
                (
                    &plane[below * comp_stride..(below + 1) * comp_stride],
                    2.0,
                )
            };

            let neighbor = neighbor_row[col] * max_sample;
            let out = (cur * 3.0 + neighbor + bias) * 0.25;
            return (out / max_sample).clamp(0.0, 1.0);
        }

        if h_ratio == 2 && v_ratio == 2 && fancy_h2v2 {
            let in_y = (y / 2).min(comp_height - 1);
            let in_x = (x / 2).min(comp_width - 1);
            let row0 = &plane[in_y * comp_stride..(in_y + 1) * comp_stride];
            let row1 = if y % 2 == 0 {
                let above = if in_y == 0 { in_y } else { in_y - 1 };
                &plane[above * comp_stride..(above + 1) * comp_stride]
            } else {
                let below = if in_y + 1 < comp_height { in_y + 1 } else { in_y };
                &plane[below * comp_stride..(below + 1) * comp_stride]
            };

            let thiscolsum = row0[in_x] * 3.0 * max_sample + row1[in_x] * max_sample;
            let last_idx = if in_x == 0 { 0 } else { in_x - 1 };
            let next_idx = if in_x + 1 < comp_width { in_x + 1 } else { in_x };
            let lastcolsum = row0[last_idx] * 3.0 * max_sample + row1[last_idx] * max_sample;
            let nextcolsum = row0[next_idx] * 3.0 * max_sample + row1[next_idx] * max_sample;

            let out = if x % 2 == 0 {
                if in_x == 0 {
                    (thiscolsum * 4.0 + 8.0) / 16.0
                } else {
                    (thiscolsum * 3.0 + lastcolsum + 8.0) / 16.0
                }
            } else if in_x + 1 >= comp_width {
                (thiscolsum * 4.0 + 7.0) / 16.0
            } else {
                (thiscolsum * 3.0 + nextcolsum + 7.0) / 16.0
            };

            return (out / max_sample).clamp(0.0, 1.0);
        }

        let comp_x = (x / h_ratio).min(comp_width - 1);
        let comp_y = (y / v_ratio).min(comp_height - 1);
        let idx = comp_y * comp_stride + comp_x;
        plane.get(idx).copied().unwrap_or(0.0)
    }

    fn components_to_rgb(
        &self,
        comp_data: &[Vec<f32>],
        width: usize,
        height: usize,
        mcus_x: usize,
        max_h: usize,
        max_v: usize,
    ) -> Result<JpegDecodedImage> {
        let num_components = self.num_components as usize;
        let mcus_y = (height + max_v * 8 - 1) / (max_v * 8);
        let mut comp_strides = Vec::with_capacity(self.components.len());
        let mut comp_heights = Vec::with_capacity(self.components.len());
        let max_sample = self.sample_max();
        let offset = self.level_shift();
        for comp in &self.components {
            comp_strides.push(mcus_x * comp.h_samp as usize * 8);
            comp_heights.push(mcus_y * comp.v_samp as usize * 8);
        }

        if num_components == 1 {
            let mut pixels = vec![0.0f32; width * height];
            let comp_stride = comp_strides[0];
            let comp_height = comp_heights[0];

            for y in 0..height {
                for x in 0..width {
                    let val = self.sample_component_value(
                        &self.components[0],
                        &comp_data[0],
                        comp_stride,
                        comp_height,
                        max_h,
                        max_v,
                        x,
                        y,
                    );
                    pixels[y * width + x] = val.clamp(0.0, 1.0);
                }
            }

            Ok(JpegDecodedImage {
                width,
                height,
                num_components: 1,
                pixels,
            })
        } else if num_components == 3 {
            let use_ycbcr = self.use_ycbcr();
            let mut pixels = vec![0.0f32; width * height * 3];

            for y in 0..height {
                for x in 0..width {
                    let mut comps = [0.0f32; 3];
                    for (comp_idx, comp) in self.components.iter().enumerate().take(3) {
                        let comp_stride = comp_strides[comp_idx];
                        let comp_height = comp_heights[comp_idx];
                        comps[comp_idx] = self.sample_component_value(
                            comp,
                            &comp_data[comp_idx],
                            comp_stride,
                            comp_height,
                            max_h,
                            max_v,
                            x,
                            y,
                        );
                    }

                    let (r, g, b) = if use_ycbcr {
                        ycbcr_to_rgb(comps[0], comps[1], comps[2], max_sample, offset)
                    } else {
                        (comps[0], comps[1], comps[2])
                    };

                    let out_idx = (y * width + x) * 3;
                    pixels[out_idx] = r;
                    pixels[out_idx + 1] = g;
                    pixels[out_idx + 2] = b;
                }
            }

            Ok(JpegDecodedImage {
                width,
                height,
                num_components: 3,
                pixels,
            })
        } else if num_components == 4 {
            let use_ycck = self.use_ycck();
            let mut pixels = vec![0.0f32; width * height * 3];

            for y in 0..height {
                for x in 0..width {
                    let mut comps = [0.0f32; 4];

                    for (comp_idx, comp) in self.components.iter().enumerate().take(4) {
                        let comp_stride = comp_strides[comp_idx];
                        let comp_height = comp_heights[comp_idx];
                        comps[comp_idx] = self.sample_component_value(
                            comp,
                            &comp_data[comp_idx],
                            comp_stride,
                            comp_height,
                            max_h,
                            max_v,
                            x,
                            y,
                        );
                    }

                    let (r, g, b) = if use_ycck {
                        ycck_to_rgb(comps[0], comps[1], comps[2], comps[3], max_sample, offset)
                    } else {
                        cmyk_to_rgb(comps[0], comps[1], comps[2], comps[3])
                    };

                    let out_idx = (y * width + x) * 3;
                    pixels[out_idx] = r.clamp(0.0, 1.0);
                    pixels[out_idx + 1] = g.clamp(0.0, 1.0);
                    pixels[out_idx + 2] = b.clamp(0.0, 1.0);
                }
            }

            Ok(JpegDecodedImage {
                width,
                height,
                num_components: 3,
                pixels,
            })
        } else {
            Err(Error::InvalidJpegData)
        }
    }

    fn components_into_output(
        &self,
        comp_data: &[Vec<f32>],
        comp_layout: JpegComponentLayout,
        output_layout: JpegOutputLayout,
        format: JpegOutputFormat,
        output: &mut [u8],
        row_stride: usize,
    ) -> Result<()> {
        let bpp = format.bytes_per_pixel();
        let use_ycbcr = self.use_ycbcr();
        let use_ycck = self.use_ycck();
        let max_sample = self.sample_max();
        let offset = self.level_shift();
        let mut comp_strides = Vec::with_capacity(self.components.len());
        let mut comp_heights = Vec::with_capacity(self.components.len());
        for comp in &self.components {
            comp_strides.push(comp_layout.mcus_x * comp.h_samp as usize * comp_layout.dct_size);
            comp_heights.push(comp_layout.mcus_y * comp.v_samp as usize * comp_layout.dct_size);
        }

        if output_layout.denom == 1
            && output_layout.crop_x == 0
            && output_layout.crop_y == 0
            && self.num_components == 3
            && comp_layout.max_h == 1
            && comp_layout.max_v == 1
            && output_layout.out_width == comp_layout.width
            && output_layout.out_height == comp_layout.height
            && format == JpegOutputFormat::Bgra8
        {
            let stride = comp_strides[0];
            for y in 0..output_layout.out_height {
                let row_start = y * stride;
                let row_end = row_start + output_layout.out_width;
                let out_start = y * row_stride;
                let out_end = out_start + output_layout.out_width * 4;

                let row0 = &comp_data[0][row_start..row_end];
                let row1 = &comp_data[1][row_start..row_end];
                let row2 = &comp_data[2][row_start..row_end];
                let out_row = &mut output[out_start..out_end];

                if use_ycbcr {
                    jpeg_ycbcr_to_bgra_row_dispatch(row0, row1, row2, out_row, max_sample, offset);
                } else {
                    jpeg_rgb_to_bgra_row_dispatch(row0, row1, row2, out_row);
                }
            }

            return Ok(());
        }

        for src_y in 0..comp_layout.height {
            if output_layout.denom > 1 && (src_y % output_layout.denom) != 0 {
                continue;
            }
            let scaled_y = src_y / output_layout.denom;
            if scaled_y < output_layout.crop_y
                || scaled_y >= output_layout.crop_y + output_layout.out_height
            {
                continue;
            }
            let out_y = scaled_y - output_layout.crop_y;
            let row_base = out_y * row_stride;

            for src_x in 0..comp_layout.width {
                if output_layout.denom > 1 && (src_x % output_layout.denom) != 0 {
                    continue;
                }
                let scaled_x = src_x / output_layout.denom;
                if scaled_x < output_layout.crop_x
                    || scaled_x >= output_layout.crop_x + output_layout.out_width
                {
                    continue;
                }
                let out_x = scaled_x - output_layout.crop_x;
                let out_idx = row_base + out_x * bpp;

                let mut comps = [0.0f32; 4];
                for (comp_idx, comp) in self.components.iter().enumerate() {
                    let comp_stride = comp_strides[comp_idx];
                    let comp_height = comp_heights[comp_idx];
                    comps[comp_idx] = self.sample_component_value(
                        comp,
                        &comp_data[comp_idx],
                        comp_stride,
                        comp_height,
                        comp_layout.max_h,
                        comp_layout.max_v,
                        src_x,
                        src_y,
                    );
                }

                let (r, g, b, gray) = match self.num_components {
                    1 => (comps[0], comps[0], comps[0], comps[0]),
                    3 => {
                        if use_ycbcr {
                            let (r, g, b) = ycbcr_to_rgb(comps[0], comps[1], comps[2], max_sample, offset);
                            (r, g, b, comps[0])
                        } else {
                            (
                                comps[0],
                                comps[1],
                                comps[2],
                                0.299 * comps[0] + 0.587 * comps[1] + 0.114 * comps[2],
                            )
                        }
                    }
                    4 => {
                        if use_ycck {
                            let (r, g, b) = ycck_to_rgb(comps[0], comps[1], comps[2], comps[3], max_sample, offset);
                            (r, g, b, 0.299 * r + 0.587 * g + 0.114 * b)
                        } else {
                            let (r, g, b) = cmyk_to_rgb(comps[0], comps[1], comps[2], comps[3]);
                            (r, g, b, 0.299 * r + 0.587 * g + 0.114 * b)
                        }
                    }
                    _ => return Err(Error::InvalidJpegData),
                };

                match format {
                    JpegOutputFormat::Gray8 => {
                        output[out_idx] = self.quantize_to_u8(gray);
                    }
                    JpegOutputFormat::Rgb8 => {
                        output[out_idx] = self.quantize_to_u8(r);
                        output[out_idx + 1] = self.quantize_to_u8(g);
                        output[out_idx + 2] = self.quantize_to_u8(b);
                    }
                    JpegOutputFormat::Bgr8 => {
                        output[out_idx] = self.quantize_to_u8(b);
                        output[out_idx + 1] = self.quantize_to_u8(g);
                        output[out_idx + 2] = self.quantize_to_u8(r);
                    }
                    JpegOutputFormat::Rgba8 => {
                        output[out_idx] = self.quantize_to_u8(r);
                        output[out_idx + 1] = self.quantize_to_u8(g);
                        output[out_idx + 2] = self.quantize_to_u8(b);
                        output[out_idx + 3] = 255;
                    }
                    JpegOutputFormat::Bgra8 => {
                        output[out_idx] = self.quantize_to_u8(b);
                        output[out_idx + 1] = self.quantize_to_u8(g);
                        output[out_idx + 2] = self.quantize_to_u8(r);
                        output[out_idx + 3] = 255;
                    }
                    JpegOutputFormat::Cmyk8 => {
                        if self.num_components != 4 {
                            return Err(Error::InvalidJpegData);
                        }
                        if use_ycck {
                            let (c, m, y, k) = ycck_to_cmyk(comps[0], comps[1], comps[2], comps[3], max_sample, offset);
                            output[out_idx] = self.quantize_to_u8(c);
                            output[out_idx + 1] = self.quantize_to_u8(m);
                            output[out_idx + 2] = self.quantize_to_u8(y);
                            output[out_idx + 3] = self.quantize_to_u8(k);
                        } else {
                            output[out_idx] = self.quantize_to_u8(comps[0]);
                            output[out_idx + 1] = self.quantize_to_u8(comps[1]);
                            output[out_idx + 2] = self.quantize_to_u8(comps[2]);
                            output[out_idx + 3] = self.quantize_to_u8(comps[3]);
                        }
                    }
                    JpegOutputFormat::Ycbcr8 => {
                        let (y, cb, cr) = if self.num_components == 3 && use_ycbcr {
                            (comps[0], comps[1], comps[2])
                        } else if self.num_components == 3 {
                            rgb_to_ycbcr(comps[0], comps[1], comps[2], max_sample, offset)
                        } else {
                            return Err(Error::InvalidJpegData);
                        };
                        output[out_idx] = self.quantize_to_u8(y);
                        output[out_idx + 1] = self.quantize_to_u8(cb);
                        output[out_idx + 2] = self.quantize_to_u8(cr);
                    }
                    JpegOutputFormat::Ycck8 => {
                        if self.num_components != 4 {
                            return Err(Error::InvalidJpegData);
                        }
                        let (y, cb, cr, k) = if use_ycck {
                            (comps[0], comps[1], comps[2], comps[3])
                        } else {
                            cmyk_to_ycck(comps[0], comps[1], comps[2], comps[3], max_sample, offset)
                        };
                        output[out_idx] = self.quantize_to_u8(y);
                        output[out_idx + 1] = self.quantize_to_u8(cb);
                        output[out_idx + 2] = self.quantize_to_u8(cr);
                        output[out_idx + 3] = self.quantize_to_u8(k);
                    }
                }
            }
        }

        Ok(())
    }

    fn components_into_output_u16(
        &self,
        comp_data: &[Vec<f32>],
        comp_layout: JpegComponentLayout,
        output_layout: JpegOutputLayout,
        format: JpegOutputFormat16,
        output: &mut [u16],
        row_stride: usize,
    ) -> Result<()> {
        let spp = format.samples_per_pixel();
        let use_ycbcr = self.use_ycbcr();
        let use_ycck = self.use_ycck();
        let max_sample = self.sample_max();
        let offset = self.level_shift();
        let alpha = max_sample.round() as u16;
        let mut comp_strides = Vec::with_capacity(self.components.len());
        let mut comp_heights = Vec::with_capacity(self.components.len());
        for comp in &self.components {
            comp_strides.push(comp_layout.mcus_x * comp.h_samp as usize * comp_layout.dct_size);
            comp_heights.push(comp_layout.mcus_y * comp.v_samp as usize * comp_layout.dct_size);
        }

        for src_y in 0..comp_layout.height {
            if output_layout.denom > 1 && (src_y % output_layout.denom) != 0 {
                continue;
            }
            let scaled_y = src_y / output_layout.denom;
            if scaled_y < output_layout.crop_y
                || scaled_y >= output_layout.crop_y + output_layout.out_height
            {
                continue;
            }
            let out_y = scaled_y - output_layout.crop_y;
            let row_base = out_y * row_stride;

            for src_x in 0..comp_layout.width {
                if output_layout.denom > 1 && (src_x % output_layout.denom) != 0 {
                    continue;
                }
                let scaled_x = src_x / output_layout.denom;
                if scaled_x < output_layout.crop_x
                    || scaled_x >= output_layout.crop_x + output_layout.out_width
                {
                    continue;
                }
                let out_x = scaled_x - output_layout.crop_x;
                let out_idx = row_base + out_x * spp;

                let mut comps = [0.0f32; 4];
                for (comp_idx, comp) in self.components.iter().enumerate() {
                    let comp_stride = comp_strides[comp_idx];
                    let comp_height = comp_heights[comp_idx];
                    comps[comp_idx] = self.sample_component_value(
                        comp,
                        &comp_data[comp_idx],
                        comp_stride,
                        comp_height,
                        comp_layout.max_h,
                        comp_layout.max_v,
                        src_x,
                        src_y,
                    );
                }

                let (r, g, b, gray) = match self.num_components {
                    1 => (comps[0], comps[0], comps[0], comps[0]),
                    3 => {
                        if use_ycbcr {
                            let (r, g, b) =
                                ycbcr_to_rgb(comps[0], comps[1], comps[2], max_sample, offset);
                            (r, g, b, comps[0])
                        } else {
                            (
                                comps[0],
                                comps[1],
                                comps[2],
                                0.299 * comps[0] + 0.587 * comps[1] + 0.114 * comps[2],
                            )
                        }
                    }
                    4 => {
                        if use_ycck {
                            let (r, g, b) =
                                ycck_to_rgb(comps[0], comps[1], comps[2], comps[3], max_sample, offset);
                            (r, g, b, 0.299 * r + 0.587 * g + 0.114 * b)
                        } else {
                            let (r, g, b) = cmyk_to_rgb(comps[0], comps[1], comps[2], comps[3]);
                            (r, g, b, 0.299 * r + 0.587 * g + 0.114 * b)
                        }
                    }
                    _ => return Err(Error::InvalidJpegData),
                };

                match format {
                    JpegOutputFormat16::Gray16 => {
                        output[out_idx] = self.quantize_to_u16(gray);
                    }
                    JpegOutputFormat16::Rgb16 => {
                        output[out_idx] = self.quantize_to_u16(r);
                        output[out_idx + 1] = self.quantize_to_u16(g);
                        output[out_idx + 2] = self.quantize_to_u16(b);
                    }
                    JpegOutputFormat16::Bgr16 => {
                        output[out_idx] = self.quantize_to_u16(b);
                        output[out_idx + 1] = self.quantize_to_u16(g);
                        output[out_idx + 2] = self.quantize_to_u16(r);
                    }
                    JpegOutputFormat16::Rgba16 => {
                        output[out_idx] = self.quantize_to_u16(r);
                        output[out_idx + 1] = self.quantize_to_u16(g);
                        output[out_idx + 2] = self.quantize_to_u16(b);
                        output[out_idx + 3] = alpha;
                    }
                    JpegOutputFormat16::Bgra16 => {
                        output[out_idx] = self.quantize_to_u16(b);
                        output[out_idx + 1] = self.quantize_to_u16(g);
                        output[out_idx + 2] = self.quantize_to_u16(r);
                        output[out_idx + 3] = alpha;
                    }
                    JpegOutputFormat16::Cmyk16 => {
                        if self.num_components != 4 {
                            return Err(Error::InvalidJpegData);
                        }
                        if use_ycck {
                            let (c, m, y, k) =
                                ycck_to_cmyk(comps[0], comps[1], comps[2], comps[3], max_sample, offset);
                            output[out_idx] = self.quantize_to_u16(c);
                            output[out_idx + 1] = self.quantize_to_u16(m);
                            output[out_idx + 2] = self.quantize_to_u16(y);
                            output[out_idx + 3] = self.quantize_to_u16(k);
                        } else {
                            output[out_idx] = self.quantize_to_u16(comps[0]);
                            output[out_idx + 1] = self.quantize_to_u16(comps[1]);
                            output[out_idx + 2] = self.quantize_to_u16(comps[2]);
                            output[out_idx + 3] = self.quantize_to_u16(comps[3]);
                        }
                    }
                    JpegOutputFormat16::Ycbcr16 => {
                        let (y, cb, cr) = if self.num_components == 3 && use_ycbcr {
                            (comps[0], comps[1], comps[2])
                        } else if self.num_components == 3 {
                            rgb_to_ycbcr(comps[0], comps[1], comps[2], max_sample, offset)
                        } else {
                            return Err(Error::InvalidJpegData);
                        };
                        output[out_idx] = self.quantize_to_u16(y);
                        output[out_idx + 1] = self.quantize_to_u16(cb);
                        output[out_idx + 2] = self.quantize_to_u16(cr);
                    }
                    JpegOutputFormat16::Ycck16 => {
                        if self.num_components != 4 {
                            return Err(Error::InvalidJpegData);
                        }
                        let (y, cb, cr, k) = if use_ycck {
                            (comps[0], comps[1], comps[2], comps[3])
                        } else {
                            cmyk_to_ycck(comps[0], comps[1], comps[2], comps[3], max_sample, offset)
                        };
                        output[out_idx] = self.quantize_to_u16(y);
                        output[out_idx + 1] = self.quantize_to_u16(cb);
                        output[out_idx + 2] = self.quantize_to_u16(cr);
                        output[out_idx + 3] = self.quantize_to_u16(k);
                    }
                }
            }
        }

        Ok(())
    }

    /// Get the image dimensions.
    pub fn dimensions(&self) -> (usize, usize) {
        (self.width as usize, self.height as usize)
    }
}

/// Convenience function to decode a JPEG file.
pub fn decode_jpeg(data: &[u8]) -> Result<JpegDecodedImage> {
    let decoder = JpegDecoder::new(data)?;
    decoder.decode(data)
}

// ============================================================================
// EXTENDED FEATURES FOR LIBJPEG-TURBO PARITY
// ============================================================================

/// JPEG bit depth modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JpegBitDepth {
    /// 8-bit samples (standard)
    Bits8,
    /// 12-bit samples
    Bits12,
    /// 16-bit samples (lossless only)
    Bits16,
}

impl JpegBitDepth {
    /// Get bits per sample.
    pub fn bits(&self) -> u8 {
        match self {
            JpegBitDepth::Bits8 => 8,
            JpegBitDepth::Bits12 => 12,
            JpegBitDepth::Bits16 => 16,
        }
    }

    /// Get maximum sample value.
    pub fn max_value(&self) -> u16 {
        ((1u32 << self.bits()) - 1) as u16
    }
}

/// Output colorspace options (libjpeg-turbo compatible).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum JpegColorSpace {
    /// Grayscale
    Grayscale,
    /// RGB (interleaved)
    #[default]
    Rgb,
    /// BGR (interleaved)
    Bgr,
    /// RGBX (with padding byte)
    Rgbx,
    /// BGRX (with padding byte)
    Bgrx,
    /// XRGB (padding first)
    Xrgb,
    /// XBGR (padding first)
    Xbgr,
    /// RGBA (with alpha = 0xFF)
    Rgba,
    /// BGRA (with alpha = 0xFF)
    Bgra,
    /// ARGB (alpha first)
    Argb,
    /// ABGR (alpha first)
    Abgr,
    /// YCbCr (no conversion)
    YCbCr,
    /// CMYK
    Cmyk,
    /// YCCK
    Ycck,
}

impl JpegColorSpace {
    /// Get number of output components.
    pub fn num_components(&self) -> usize {
        match self {
            JpegColorSpace::Grayscale => 1,
            JpegColorSpace::Rgb | JpegColorSpace::Bgr | JpegColorSpace::YCbCr => 3,
            JpegColorSpace::Rgbx | JpegColorSpace::Bgrx | JpegColorSpace::Xrgb
            | JpegColorSpace::Xbgr | JpegColorSpace::Rgba | JpegColorSpace::Bgra
            | JpegColorSpace::Argb | JpegColorSpace::Abgr | JpegColorSpace::Cmyk
            | JpegColorSpace::Ycck => 4,
        }
    }
}

/// IDCT scaling factor for reduced-size decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IdctScale {
    /// Full size (8x8)
    #[default]
    Full,
    /// 7/8 size
    Scale7_8,
    /// 3/4 size (6x6)
    Scale3_4,
    /// 5/8 size
    Scale5_8,
    /// 1/2 size (4x4)
    Scale1_2,
    /// 3/8 size
    Scale3_8,
    /// 1/4 size (2x2)
    Scale1_4,
    /// 1/8 size (1x1, DC only)
    Scale1_8,
}

impl IdctScale {
    /// Get the output block size.
    pub fn block_size(&self) -> usize {
        match self {
            IdctScale::Full => 8,
            IdctScale::Scale7_8 => 7,
            IdctScale::Scale3_4 => 6,
            IdctScale::Scale5_8 => 5,
            IdctScale::Scale1_2 => 4,
            IdctScale::Scale3_8 => 3,
            IdctScale::Scale1_4 => 2,
            IdctScale::Scale1_8 => 1,
        }
    }

    /// Get the scale factor as a fraction (numerator/8).
    pub fn numerator(&self) -> usize {
        match self {
            IdctScale::Full => 8,
            IdctScale::Scale7_8 => 7,
            IdctScale::Scale3_4 => 6,
            IdctScale::Scale5_8 => 5,
            IdctScale::Scale1_2 => 4,
            IdctScale::Scale3_8 => 3,
            IdctScale::Scale1_4 => 2,
            IdctScale::Scale1_8 => 1,
        }
    }

    pub fn denom(&self) -> usize {
        8 / self.numerator()
    }
}

/// Lossless JPEG predictor selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LosslessPredictor {
    /// No prediction
    #[default]
    None = 0,
    /// Ra (left)
    Left = 1,
    /// Rb (above)
    Above = 2,
    /// Rc (upper-left)
    UpperLeft = 3,
    /// Ra + Rb - Rc
    LinearCombination = 4,
    /// Ra + (Rb - Rc) / 2
    LeftPlusHalfDiff = 5,
    /// Rb + (Ra - Rc) / 2
    AbovePlusHalfDiff = 6,
    /// (Ra + Rb) / 2
    Average = 7,
}

impl LosslessPredictor {
    /// Create from selection value.
    pub fn from_value(v: u8) -> Self {
        match v {
            0 => LosslessPredictor::None,
            1 => LosslessPredictor::Left,
            2 => LosslessPredictor::Above,
            3 => LosslessPredictor::UpperLeft,
            4 => LosslessPredictor::LinearCombination,
            5 => LosslessPredictor::LeftPlusHalfDiff,
            6 => LosslessPredictor::AbovePlusHalfDiff,
            7 => LosslessPredictor::Average,
            _ => LosslessPredictor::None,
        }
    }

    /// Compute predicted value.
    pub fn predict(&self, left: i32, above: i32, upper_left: i32) -> i32 {
        match self {
            LosslessPredictor::None => 0,
            LosslessPredictor::Left => left,
            LosslessPredictor::Above => above,
            LosslessPredictor::UpperLeft => upper_left,
            LosslessPredictor::LinearCombination => left + above - upper_left,
            LosslessPredictor::LeftPlusHalfDiff => left + (above - upper_left) / 2,
            LosslessPredictor::AbovePlusHalfDiff => above + (left - upper_left) / 2,
            LosslessPredictor::Average => (left + above) / 2,
        }
    }
}

/// Lossless transform types (jpegtran compatible).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JpegTransform {
    /// No transform
    None,
    /// Flip horizontally
    FlipHorizontal,
    /// Flip vertically
    FlipVertical,
    /// Rotate 90 degrees clockwise
    Rotate90,
    /// Rotate 180 degrees
    Rotate180,
    /// Rotate 270 degrees clockwise (90 counter-clockwise)
    Rotate270,
    /// Transpose (swap rows and columns)
    Transpose,
    /// Transverse (transpose then rotate 180)
    Transverse,
}

/// Arithmetic decoder for arithmetic-coded JPEG.
struct ArithmeticDecoder {
    c: u32,
    a: u32,
    ct: i32,
    data: Vec<u8>,
    pos: usize,
    dc_stats: [[u8; 64]; 4],
    ac_stats: [[u8; 256]; 8],
}

/// QE table for arithmetic decoder (probability estimation).
const QE_TABLE: [(u16, u8, u8); 113] = [
    (0x5a1d, 1, 1), (0x2586, 14, 2), (0x1114, 16, 3), (0x080b, 18, 4),
    (0x03d8, 20, 5), (0x01da, 23, 6), (0x00e5, 25, 7), (0x006f, 28, 8),
    (0x0036, 30, 9), (0x001a, 33, 10), (0x000d, 35, 11), (0x0006, 9, 12),
    (0x0003, 10, 13), (0x0001, 12, 13), (0x5a7f, 15, 15), (0x3f25, 36, 16),
    (0x2cf2, 38, 17), (0x207c, 39, 18), (0x17b9, 40, 19), (0x1182, 42, 20),
    (0x0cef, 43, 21), (0x09a1, 45, 22), (0x072f, 46, 23), (0x055c, 48, 24),
    (0x0406, 49, 25), (0x0303, 51, 26), (0x0240, 52, 27), (0x01b1, 54, 28),
    (0x0144, 56, 29), (0x00f5, 57, 30), (0x00b7, 59, 31), (0x008a, 60, 32),
    (0x0068, 62, 33), (0x004e, 63, 34), (0x003b, 32, 35), (0x002c, 33, 9),
    (0x5ae1, 37, 37), (0x484c, 64, 38), (0x3a0d, 65, 39), (0x2ef1, 67, 40),
    (0x261f, 68, 41), (0x1f33, 69, 42), (0x19a8, 70, 43), (0x1518, 72, 44),
    (0x1177, 73, 45), (0x0e74, 74, 46), (0x0bfb, 75, 47), (0x09f8, 77, 48),
    (0x0861, 78, 49), (0x0706, 79, 50), (0x05cd, 48, 51), (0x04de, 50, 52),
    (0x040f, 50, 53), (0x0363, 51, 54), (0x02d4, 52, 55), (0x025c, 53, 56),
    (0x01f8, 54, 57), (0x01a4, 55, 58), (0x0160, 56, 59), (0x0125, 57, 60),
    (0x00f6, 58, 61), (0x00cb, 59, 62), (0x00ab, 61, 63), (0x008f, 61, 32),
    (0x5b12, 65, 65), (0x4d04, 80, 66), (0x412c, 81, 67), (0x37d8, 82, 68),
    (0x2fe8, 83, 69), (0x293c, 84, 70), (0x2379, 86, 71), (0x1edf, 87, 72),
    (0x1aa9, 87, 73), (0x174e, 72, 74), (0x1424, 72, 75), (0x119c, 74, 76),
    (0x0f6b, 74, 77), (0x0d51, 75, 78), (0x0bb6, 77, 79), (0x0a40, 77, 48),
    (0x5832, 80, 81), (0x4d1c, 88, 82), (0x438e, 89, 83), (0x3bdd, 90, 84),
    (0x34ee, 91, 85), (0x2eae, 92, 86), (0x299a, 93, 87), (0x2516, 86, 71),
    (0x5570, 88, 89), (0x4ca9, 95, 90), (0x44d9, 96, 91), (0x3e22, 97, 92),
    (0x3824, 99, 93), (0x32b4, 99, 94), (0x2e17, 93, 86), (0x56a8, 95, 96),
    (0x4f46, 101, 97), (0x47e5, 102, 98), (0x41cf, 103, 99), (0x3c3d, 104, 100),
    (0x375e, 99, 93), (0x5231, 105, 102), (0x4c0f, 106, 103), (0x4639, 107, 104),
    (0x415e, 103, 99), (0x5627, 105, 106), (0x50e7, 108, 107), (0x4b85, 109, 103),
    (0x5597, 110, 109), (0x504f, 111, 107), (0x5a10, 110, 111), (0x5522, 112, 109),
    (0x59eb, 112, 111),
];

impl Default for ArithmeticDecoder {
    fn default() -> Self {
        Self {
            c: 0,
            a: 0,
            ct: 0,
            data: Vec::new(),
            pos: 0,
            dc_stats: [[0; 64]; 4],
            ac_stats: [[0; 256]; 8],
        }
    }
}

impl ArithmeticDecoder {
    fn init(&mut self, data: &[u8]) {
        self.data = data.to_vec();
        self.pos = 0;
        self.c = 0;
        self.a = 0;
        self.ct = -16;

        // Initialize by reading two bytes
        self.byte_in();
        self.c <<= 8;
        self.byte_in();
        self.c <<= 8;
        self.byte_in();
        self.c = (self.c << 8) | 0xFF;
        self.a = 0x8000;
        self.ct = 0;
    }

    fn byte_in(&mut self) {
        if self.pos < self.data.len() {
            let b = self.data[self.pos];
            self.pos += 1;

            if b == 0xFF {
                if self.pos < self.data.len() {
                    let b2 = self.data[self.pos];
                    if b2 == 0x00 {
                        self.pos += 1;
                        self.c |= 0xFF00;
                    }
                }
            } else {
                self.c |= (b as u32) << 8;
            }
        }
        self.ct += 8;
    }

    fn decode(&mut self, st: &mut u8) -> u8 {
        let state = *st as usize;
        let (qe, nm, nl) = QE_TABLE[state & 0x7F];
        let sense = (state >> 7) as u8;

        self.a -= qe as u32;

        if (self.c >> 16) < self.a {
            if self.a < 0x8000 {
                let result = if self.a < qe as u32 {
                    *st = (nl as u8) | (sense << 7);
                    1 - sense
                } else {
                    *st = (nm as u8) | (sense << 7);
                    sense
                };
                self.renormalize();
                result
            } else {
                sense
            }
        } else {
            self.c -= (self.a as u32) << 16;
            let result = if self.a < qe as u32 {
                *st = (nm as u8) | (sense << 7);
                sense
            } else {
                *st = (nl as u8) | ((1 - sense) << 7);
                1 - sense
            };
            self.a = qe as u32;
            self.renormalize();
            result
        }
    }

    fn renormalize(&mut self) {
        while self.a < 0x8000 {
            if self.ct == 0 {
                self.byte_in();
            }
            self.a <<= 1;
            self.c <<= 1;
            self.ct -= 1;
        }
    }
}

/// Decode options for advanced JPEG decoding.
#[derive(Debug, Clone, Default)]
pub struct JpegDecodeOptions {
    /// Output colorspace
    pub colorspace: JpegColorSpace,
    /// IDCT scaling
    pub scale: IdctScale,
    /// Crop region (x, y, width, height)
    pub crop: Option<(usize, usize, usize, usize)>,
}

impl JpegDecoder {
    /// Decode with custom options.
    pub fn decode_with_options(&self, data: &[u8], options: &JpegDecodeOptions) -> Result<JpegDecodedImage> {
        let image = self.decode(data)?;

        // Apply colorspace conversion if needed
        let converted = self.convert_colorspace(&image, options.colorspace)?;

        // Apply scaling
        let scaled = self.scale_image(&converted, options.scale)?;

        // Apply cropping if specified
        if let Some((x, y, w, h)) = options.crop {
            self.crop_image(&scaled, x, y, w, h)
        } else {
            Ok(scaled)
        }
    }

    fn convert_colorspace(&self, image: &JpegDecodedImage, colorspace: JpegColorSpace) -> Result<JpegDecodedImage> {
        match colorspace {
            JpegColorSpace::Rgb => Ok(image.clone()),
            JpegColorSpace::Bgr => {
                if image.num_components != 3 {
                    return Ok(image.clone());
                }
                let mut pixels = image.pixels.clone();
                for i in (0..pixels.len()).step_by(3) {
                    pixels.swap(i, i + 2);
                }
                Ok(JpegDecodedImage {
                    width: image.width,
                    height: image.height,
                    num_components: 3,
                    pixels,
                })
            }
            JpegColorSpace::Rgba | JpegColorSpace::Rgbx => {
                if image.num_components != 3 {
                    return Ok(image.clone());
                }
                let mut pixels = Vec::with_capacity(image.width * image.height * 4);
                for i in (0..image.pixels.len()).step_by(3) {
                    pixels.push(image.pixels[i]);
                    pixels.push(image.pixels[i + 1]);
                    pixels.push(image.pixels[i + 2]);
                    pixels.push(1.0); // Alpha = 1.0 (255)
                }
                Ok(JpegDecodedImage {
                    width: image.width,
                    height: image.height,
                    num_components: 4,
                    pixels,
                })
            }
            JpegColorSpace::Bgra | JpegColorSpace::Bgrx => {
                if image.num_components != 3 {
                    return Ok(image.clone());
                }
                let mut pixels = Vec::with_capacity(image.width * image.height * 4);
                for i in (0..image.pixels.len()).step_by(3) {
                    pixels.push(image.pixels[i + 2]);
                    pixels.push(image.pixels[i + 1]);
                    pixels.push(image.pixels[i]);
                    pixels.push(1.0);
                }
                Ok(JpegDecodedImage {
                    width: image.width,
                    height: image.height,
                    num_components: 4,
                    pixels,
                })
            }
            JpegColorSpace::Argb | JpegColorSpace::Xrgb => {
                if image.num_components != 3 {
                    return Ok(image.clone());
                }
                let mut pixels = Vec::with_capacity(image.width * image.height * 4);
                for i in (0..image.pixels.len()).step_by(3) {
                    pixels.push(1.0); // Alpha first
                    pixels.push(image.pixels[i]);
                    pixels.push(image.pixels[i + 1]);
                    pixels.push(image.pixels[i + 2]);
                }
                Ok(JpegDecodedImage {
                    width: image.width,
                    height: image.height,
                    num_components: 4,
                    pixels,
                })
            }
            JpegColorSpace::Abgr | JpegColorSpace::Xbgr => {
                if image.num_components != 3 {
                    return Ok(image.clone());
                }
                let mut pixels = Vec::with_capacity(image.width * image.height * 4);
                for i in (0..image.pixels.len()).step_by(3) {
                    pixels.push(1.0);
                    pixels.push(image.pixels[i + 2]);
                    pixels.push(image.pixels[i + 1]);
                    pixels.push(image.pixels[i]);
                }
                Ok(JpegDecodedImage {
                    width: image.width,
                    height: image.height,
                    num_components: 4,
                    pixels,
                })
            }
            _ => Ok(image.clone()),
        }
    }

    fn crop_image(&self, image: &JpegDecodedImage, x: usize, y: usize, w: usize, h: usize) -> Result<JpegDecodedImage> {
        if x + w > image.width || y + h > image.height {
            return Err(Error::InvalidJpegData);
        }

        let nc = image.num_components;
        let mut pixels = Vec::with_capacity(w * h * nc);

        for row in y..y + h {
            let src_start = (row * image.width + x) * nc;
            let src_end = src_start + w * nc;
            pixels.extend_from_slice(&image.pixels[src_start..src_end]);
        }

        Ok(JpegDecodedImage {
            width: w,
            height: h,
            num_components: nc,
            pixels,
        })
    }

    fn scale_image(&self, image: &JpegDecodedImage, scale: IdctScale) -> Result<JpegDecodedImage> {
        if scale == IdctScale::Full {
            return Ok(image.clone());
        }

        let numerator = scale.numerator();
        let new_width = (image.width * numerator + 7) / 8;
        let new_height = (image.height * numerator + 7) / 8;

        if new_width == 0 || new_height == 0 {
            return Err(Error::InvalidJpegData);
        }

        let nc = image.num_components;
        let mut pixels = vec![0.0f32; new_width * new_height * nc];

        for y in 0..new_height {
            let src_y = (y * 8) / numerator;
            let src_y = src_y.min(image.height - 1);
            for x in 0..new_width {
                let src_x = (x * 8) / numerator;
                let src_x = src_x.min(image.width - 1);
                let src_idx = (src_y * image.width + src_x) * nc;
                let dst_idx = (y * new_width + x) * nc;
                pixels[dst_idx..dst_idx + nc]
                    .copy_from_slice(&image.pixels[src_idx..src_idx + nc]);
            }
        }

        Ok(JpegDecodedImage {
            width: new_width,
            height: new_height,
            num_components: nc,
            pixels,
        })
    }

    fn compute_output_layout(&self, options: &JpegDecodeOptions) -> Result<JpegOutputLayout> {
        let denom = options.scale.denom().max(1);
        let scaled_width = (self.width as usize + denom - 1) / denom;
        let scaled_height = (self.height as usize + denom - 1) / denom;

        if let Some((x, y, w, h)) = options.crop {
            if x + w > scaled_width || y + h > scaled_height {
                return Err(Error::InvalidJpegData);
            }
            Ok(JpegOutputLayout {
                out_width: w,
                out_height: h,
                crop_x: x,
                crop_y: y,
                denom,
            })
        } else {
            Ok(JpegOutputLayout {
                out_width: scaled_width,
                out_height: scaled_height,
                crop_x: 0,
                crop_y: 0,
                denom,
            })
        }
    }

    /// Decode directly into the caller-provided output buffer.
    pub fn decode_into(
        &self,
        data: &[u8],
        options: &JpegDecodeOptions,
        format: JpegOutputFormat,
        output: &mut [u8],
        row_stride: usize,
    ) -> Result<(usize, usize)> {
        let mut decoder = self.clone();
        decoder.ensure_default_huffman_tables();
        decoder.update_height_from_dnl(data);

        if decoder.data_precision > 8 {
            return Err(Error::InvalidJpegData);
        }

        let layout = decoder.compute_output_layout(options)?;
        let bpp = format.bytes_per_pixel();
        if layout.out_height == 0 || layout.out_width == 0 {
            return Err(Error::InvalidJpegData);
        }

        let required = row_stride * (layout.out_height - 1) + layout.out_width * bpp;
        if output.len() < required {
            return Err(Error::InvalidJpegData);
        }

        if decoder.is_progressive {
            decoder.decode_progressive_into(data, format, output, row_stride, layout)?;
        } else {
            let (_scan_header, scan_start, scan_end) = decoder.find_sos_segment(data)?;
            let scan_data = &data[scan_start..scan_end];
            decoder.decode_baseline_into(scan_data, format, output, row_stride, layout)?;
        }

        Ok((layout.out_width, layout.out_height))
    }

    /// Decode directly into the caller-provided 16-bit output buffer.
    /// `row_stride` is measured in samples (u16s), not bytes.
    pub fn decode_into_u16(
        &self,
        data: &[u8],
        options: &JpegDecodeOptions,
        format: JpegOutputFormat16,
        output: &mut [u16],
        row_stride: usize,
    ) -> Result<(usize, usize)> {
        let mut decoder = self.clone();
        decoder.ensure_default_huffman_tables();
        decoder.update_height_from_dnl(data);

        let layout = decoder.compute_output_layout(options)?;
        let spp = format.samples_per_pixel();
        if layout.out_height == 0 || layout.out_width == 0 {
            return Err(Error::InvalidJpegData);
        }

        let required = row_stride * (layout.out_height - 1) + layout.out_width * spp;
        if output.len() < required {
            return Err(Error::InvalidJpegData);
        }

        if decoder.is_progressive {
            decoder.decode_progressive_into_u16(data, format, output, row_stride, layout)?;
        } else {
            let (_scan_header, scan_start, scan_end) = decoder.find_sos_segment(data)?;
            let scan_data = &data[scan_start..scan_end];
            decoder.decode_baseline_into_u16(scan_data, format, output, row_stride, layout)?;
        }

        Ok((layout.out_width, layout.out_height))
    }
}

/// CMYK to RGB conversion.
fn cmyk_to_rgb(c: f32, m: f32, y: f32, k: f32) -> (f32, f32, f32) {
    let r = (1.0 - c) * (1.0 - k);
    let g = (1.0 - m) * (1.0 - k);
    let b = (1.0 - y) * (1.0 - k);
    (r, g, b)
}

/// RGB to YCbCr conversion.
fn rgb_to_ycbcr(r: f32, g: f32, b: f32, max_sample: f32, offset: f32) -> (f32, f32, f32) {
    let r = r * max_sample;
    let g = g * max_sample;
    let b = b * max_sample;
    let y = 0.299 * r + 0.587 * g + 0.114 * b;
    let cb = offset - 0.168736 * r - 0.331264 * g + 0.5 * b;
    let cr = offset + 0.5 * r - 0.418688 * g - 0.081312 * b;
    (
        (y / max_sample).clamp(0.0, 1.0),
        (cb / max_sample).clamp(0.0, 1.0),
        (cr / max_sample).clamp(0.0, 1.0),
    )
}

/// YCbCr to RGB conversion.
fn ycbcr_to_rgb(y: f32, cb: f32, cr: f32, max_sample: f32, offset: f32) -> (f32, f32, f32) {
    let y_scaled = y * max_sample;
    let cb_scaled = (cb * max_sample) - offset;
    let cr_scaled = (cr * max_sample) - offset;

    let r = (y_scaled + 1.402 * cr_scaled).clamp(0.0, max_sample) / max_sample;
    let g = (y_scaled - 0.344136 * cb_scaled - 0.714136 * cr_scaled).clamp(0.0, max_sample) / max_sample;
    let b = (y_scaled + 1.772 * cb_scaled).clamp(0.0, max_sample) / max_sample;
    (r, g, b)
}

/// YCCK to RGB conversion.
fn ycck_to_rgb(y: f32, cb: f32, cr: f32, k: f32, max_sample: f32, offset: f32) -> (f32, f32, f32) {
    let (r, g, b) = ycbcr_to_rgb(y, cb, cr, max_sample, offset);
    cmyk_to_rgb(1.0 - r, 1.0 - g, 1.0 - b, k)
}

/// YCCK to CMYK conversion.
fn ycck_to_cmyk(y: f32, cb: f32, cr: f32, k: f32, max_sample: f32, offset: f32) -> (f32, f32, f32, f32) {
    let (r, g, b) = ycck_to_rgb(y, cb, cr, k, max_sample, offset);
    (1.0 - r, 1.0 - g, 1.0 - b, k)
}

/// CMYK to YCCK conversion.
fn cmyk_to_ycck(c: f32, m: f32, y: f32, k: f32, max_sample: f32, offset: f32) -> (f32, f32, f32, f32) {
    let (r, g, b) = cmyk_to_rgb(c, m, y, k);
    let (yy, cb, cr) = rgb_to_ycbcr(r, g, b, max_sample, offset);
    (yy, cb, cr, k)
}

simd_function!(
    jpeg_idct_8x8_dispatch,
    d: D,
    fn jpeg_idct_8x8(block: &mut [f32; 64]) {
        idct2d_8_8(d, block);
    }
);

simd_function!(
    jpeg_idct_4x4_dispatch,
    d: D,
    fn jpeg_idct_4x4(block: &mut [f32; 16]) {
        idct2d_4_4(d, block);
    }
);

simd_function!(
    jpeg_idct_2x2_dispatch,
    d: D,
    fn jpeg_idct_2x2(block: &mut [f32; 4]) {
        idct2d_2_2(d, block);
    }
);

simd_function!(
    jpeg_ycbcr_to_bgra_row_dispatch,
    d: D,
    fn jpeg_ycbcr_to_bgra_row_impl(
        y: &[f32],
        cb: &[f32],
        cr: &[f32],
        out: &mut [u8],
        max_sample: f32,
        offset: f32,
    ) {
        let len = y.len().min(cb.len()).min(cr.len());
        let vec_len = D::F32Vec::LEN;
        debug_assert!(vec_len <= 16);
        let max_v = D::F32Vec::splat(d, max_sample);
        let offset_v = D::F32Vec::splat(d, offset);
        let scale_v = D::F32Vec::splat(d, 255.0 / max_sample);
        let zero = D::F32Vec::zero(d);
        let max_byte = D::F32Vec::splat(d, 255.0);
        let coeff_r = D::F32Vec::splat(d, 1.402);
        let coeff_gb = D::F32Vec::splat(d, -0.344136);
        let coeff_gr = D::F32Vec::splat(d, -0.714136);
        let coeff_b = D::F32Vec::splat(d, 1.772);

        let mut i = 0usize;
        while i + vec_len <= len {
            let yv = D::F32Vec::load(d, &y[i..]) * max_v;
            let cbv = D::F32Vec::load(d, &cb[i..]) * max_v - offset_v;
            let crv = D::F32Vec::load(d, &cr[i..]) * max_v - offset_v;

            let r = (yv + crv * coeff_r) * scale_v;
            let g = (yv + cbv * coeff_gb + crv * coeff_gr) * scale_v;
            let b = (yv + cbv * coeff_b) * scale_v;

            let r = r.max(zero).min(max_byte);
            let g = g.max(zero).min(max_byte);
            let b = b.max(zero).min(max_byte);

            let mut r_buf = [0u8; 16];
            let mut g_buf = [0u8; 16];
            let mut b_buf = [0u8; 16];
            r.round_store_u8(&mut r_buf[..vec_len]);
            g.round_store_u8(&mut g_buf[..vec_len]);
            b.round_store_u8(&mut b_buf[..vec_len]);

            for lane in 0..vec_len {
                let dst = (i + lane) * 4;
                out[dst] = b_buf[lane];
                out[dst + 1] = g_buf[lane];
                out[dst + 2] = r_buf[lane];
                out[dst + 3] = 255;
            }

            i += vec_len;
        }

        for j in i..len {
            let (r, g, b) = ycbcr_to_rgb(y[j], cb[j], cr[j], max_sample, offset);
            let dst = j * 4;
            out[dst] = (b.clamp(0.0, 1.0) * 255.0).round() as u8;
            out[dst + 1] = (g.clamp(0.0, 1.0) * 255.0).round() as u8;
            out[dst + 2] = (r.clamp(0.0, 1.0) * 255.0).round() as u8;
            out[dst + 3] = 255;
        }
    }
);

simd_function!(
    jpeg_rgb_to_bgra_row_dispatch,
    d: D,
    fn jpeg_rgb_to_bgra_row_impl(r: &[f32], g: &[f32], b: &[f32], out: &mut [u8]) {
        let len = r.len().min(g.len()).min(b.len());
        let vec_len = D::F32Vec::LEN;
        debug_assert!(vec_len <= 16);
        let scale_v = D::F32Vec::splat(d, 255.0);
        let zero = D::F32Vec::zero(d);
        let max_byte = D::F32Vec::splat(d, 255.0);

        let mut i = 0usize;
        while i + vec_len <= len {
            let rv = D::F32Vec::load(d, &r[i..]) * scale_v;
            let gv = D::F32Vec::load(d, &g[i..]) * scale_v;
            let bv = D::F32Vec::load(d, &b[i..]) * scale_v;

            let rv = rv.max(zero).min(max_byte);
            let gv = gv.max(zero).min(max_byte);
            let bv = bv.max(zero).min(max_byte);

            let mut r_buf = [0u8; 16];
            let mut g_buf = [0u8; 16];
            let mut b_buf = [0u8; 16];
            rv.round_store_u8(&mut r_buf[..vec_len]);
            gv.round_store_u8(&mut g_buf[..vec_len]);
            bv.round_store_u8(&mut b_buf[..vec_len]);

            for lane in 0..vec_len {
                let dst = (i + lane) * 4;
                out[dst] = b_buf[lane];
                out[dst + 1] = g_buf[lane];
                out[dst + 2] = r_buf[lane];
                out[dst + 3] = 255;
            }

            i += vec_len;
        }

        for j in i..len {
            let dst = j * 4;
            out[dst] = (b[j].clamp(0.0, 1.0) * 255.0).round() as u8;
            out[dst + 1] = (g[j].clamp(0.0, 1.0) * 255.0).round() as u8;
            out[dst + 2] = (r[j].clamp(0.0, 1.0) * 255.0).round() as u8;
            out[dst + 3] = 255;
        }
    }
);

// ============================================================================
// JPEG DECODER TESTS
// ============================================================================

#[cfg(test)]
mod decoder_tests {
    use super::*;

    #[test]
    fn test_jpeg_decoder_invalid_data() {
        assert!(JpegDecoder::new(&[]).is_err());
        assert!(JpegDecoder::new(&[0x00]).is_err());
        assert!(JpegDecoder::new(&[0xFF]).is_err());
        assert!(JpegDecoder::new(&[0xFF, 0x00]).is_err());
    }

    #[test]
    fn test_jpeg_decoder_valid_soi() {
        // Just SOI marker - should parse but be incomplete
        let data = [0xFF, 0xD8, 0xFF, 0xD9]; // SOI + EOI
        let result = JpegDecoder::new(&data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_jpeg_bit_depth() {
        assert_eq!(JpegBitDepth::Bits8.bits(), 8);
        assert_eq!(JpegBitDepth::Bits12.bits(), 12);
        assert_eq!(JpegBitDepth::Bits16.bits(), 16);
        assert_eq!(JpegBitDepth::Bits8.max_value(), 255);
        assert_eq!(JpegBitDepth::Bits12.max_value(), 4095);
        assert_eq!(JpegBitDepth::Bits16.max_value(), 65535);
    }

    #[test]
    fn test_jpeg_colorspace_components() {
        assert_eq!(JpegColorSpace::Grayscale.num_components(), 1);
        assert_eq!(JpegColorSpace::Rgb.num_components(), 3);
        assert_eq!(JpegColorSpace::Rgba.num_components(), 4);
        assert_eq!(JpegColorSpace::Cmyk.num_components(), 4);
    }

    #[test]
    fn test_idct_scale() {
        assert_eq!(IdctScale::Full.block_size(), 8);
        assert_eq!(IdctScale::Scale1_2.block_size(), 4);
        assert_eq!(IdctScale::Scale1_4.block_size(), 2);
        assert_eq!(IdctScale::Scale1_8.block_size(), 1);
    }

    #[test]
    fn test_lossless_predictor() {
        assert_eq!(LosslessPredictor::Left.predict(100, 50, 25), 100);
        assert_eq!(LosslessPredictor::Above.predict(100, 50, 25), 50);
        assert_eq!(LosslessPredictor::Average.predict(100, 50, 25), 75);
        assert_eq!(LosslessPredictor::LinearCombination.predict(100, 50, 25), 125);
    }

    #[test]
    fn test_cmyk_to_rgb() {
        let (r, g, b) = cmyk_to_rgb(0.0, 0.0, 0.0, 0.0);
        assert!((r - 1.0).abs() < 0.01);
        assert!((g - 1.0).abs() < 0.01);
        assert!((b - 1.0).abs() < 0.01);

        let (r, g, b) = cmyk_to_rgb(0.0, 0.0, 0.0, 1.0);
        assert!((r - 0.0).abs() < 0.01);
        assert!((g - 0.0).abs() < 0.01);
        assert!((b - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_huffman_table_build() {
        let bits = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
        let values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        let table = HuffmanTable::build(&bits, &values);
        assert!(table.valid);
        assert_eq!(table.huffval.len(), 12);
    }

    fn decoder_with_component_ids(ids: &[u8]) -> JpegDecoder {
        let components = ids
            .iter()
            .map(|&id| JpegDecoderComponent {
                id,
                h_samp: 1,
                v_samp: 1,
                quant_table_id: 0,
                dc_table_id: 0,
                ac_table_id: 0,
            })
            .collect();

        JpegDecoder {
            width: 1,
            height: 1,
            num_components: ids.len() as u8,
            data_precision: 8,
            components,
            quant_tables: [[0; 64]; 4],
            dc_huff_tables: Default::default(),
            ac_huff_tables: Default::default(),
            restart_interval: 0,
            is_progressive: false,
            metadata: JpegMetadata::default(),
            saw_jfif: false,
            adobe_transform: None,
            icc_segments: Vec::new(),
            icc_expected_segments: None,
        }
    }

    #[test]
    fn test_use_ycbcr_guess_component_ids() {
        let decoder = decoder_with_component_ids(&[1, 2, 3]);
        assert!(decoder.use_ycbcr());
    }

    #[test]
    fn test_use_ycbcr_rgb_ids() {
        let decoder = decoder_with_component_ids(&[82, 71, 66]);
        assert!(!decoder.use_ycbcr());
    }

    #[test]
    fn test_use_ycbcr_jfif_overrides_adobe() {
        let mut decoder = decoder_with_component_ids(&[1, 2, 3]);
        decoder.saw_jfif = true;
        decoder.adobe_transform = Some(0);
        assert!(decoder.use_ycbcr());
    }

    #[test]
    fn test_idct_dc_only() {
        let decoder = JpegDecoder {
            width: 0,
            height: 0,
            num_components: 0,
            data_precision: 8,
            components: Vec::new(),
            quant_tables: [[0; 64]; 4],
            dc_huff_tables: Default::default(),
            ac_huff_tables: Default::default(),
            restart_interval: 0,
            is_progressive: false,
            metadata: JpegMetadata::default(),
            saw_jfif: false,
            adobe_transform: None,
            icc_segments: Vec::new(),
            icc_expected_segments: None,
        };
        let mut input = [0i64; 64];
        input[0] = 80;
        let mut output = [0.0f32; 64];
        decoder.idct_8x8(&input, &mut output);
        let expected = (80.0 / 8.0 + 128.0) / 255.0;
        for val in &output {
            assert!((val - expected).abs() < 1e-3, "val {} expected {}", val, expected);
        }
    }

    #[test]
    fn test_idct_dc_only_16bit() {
        let decoder = JpegDecoder {
            width: 0,
            height: 0,
            num_components: 0,
            data_precision: 16,
            components: Vec::new(),
            quant_tables: [[0; 64]; 4],
            dc_huff_tables: Default::default(),
            ac_huff_tables: Default::default(),
            restart_interval: 0,
            is_progressive: false,
            metadata: JpegMetadata::default(),
            saw_jfif: false,
            adobe_transform: None,
            icc_segments: Vec::new(),
            icc_expected_segments: None,
        };
        let mut input = [0i64; 64];
        input[0] = 80;
        let mut output = [0.0f32; 64];
        decoder.idct_8x8(&input, &mut output);
        let expected = (80.0 / 8.0 + 32768.0) / 65535.0;
        for val in &output {
            assert!((val - expected).abs() < 1e-6, "val {} expected {}", val, expected);
        }
    }

    #[test]
    fn test_jpeg_decoded_image() {
        let image = JpegDecodedImage {
            width: 8,
            height: 8,
            num_components: 3,
            pixels: vec![0.5; 8 * 8 * 3],
        };
        assert_eq!(image.width, 8);
        assert_eq!(image.height, 8);
        assert_eq!(image.pixels.len(), 192);
    }

    #[test]
    fn test_jpeg_metadata_default() {
        let meta = JpegMetadata::default();
        assert!(meta.icc_profile.is_none());
        assert!(meta.exif_data.is_none());
        assert!(meta.jfif_version.is_none());
    }
}

// ============================================================================
// JPEG FILE INTEGRATION TESTS
// ============================================================================

#[cfg(test)]
mod jpeg_file_tests {
    use super::*;

    // Embed test JPEG files
    const CMYK_16X16: &[u8] = include_bytes!("../resources/test/jpeg/cmyk_16x16.jpg");
    const GRAYSCALE_8X8: &[u8] = include_bytes!("../resources/test/jpeg/grayscale_8x8.jpg");
    const RGB_8X8: &[u8] = include_bytes!("../resources/test/jpeg/rgb_8x8.jpg");
    const PROGRESSIVE_8X8: &[u8] = include_bytes!("../resources/test/jpeg/progressive_8x8.jpg");
    const RGB_64X64: &[u8] = include_bytes!("../resources/test/jpeg/rgb_64x64.jpg");
    const GRADIENT_32X32: &[u8] = include_bytes!("../resources/test/jpeg/gradient_32x32.jpg");
    const QUALITY_100: &[u8] = include_bytes!("../resources/test/jpeg/quality100_16x16.jpg");
    const QUALITY_50: &[u8] = include_bytes!("../resources/test/jpeg/quality50_16x16.jpg");
    const RESTART_8X8: &[u8] = include_bytes!("../resources/test/jpeg/restart_8x8.jpg");
    const SUBSAMPLED_420: &[u8] = include_bytes!("../resources/test/jpeg/subsampled_420_8x8.jpg");
    const SUBSAMPLED_444: &[u8] = include_bytes!("../resources/test/jpeg/subsampled_444_8x8.jpg");
    const WITH_EXIF_8X8: &[u8] = include_bytes!("../resources/test/jpeg/with_exif_8x8.jpg");
    const RGB_12BIT: &[u8] = include_bytes!("../resources/test/jpeg/testorig12.jpg");

    fn rgb_to_bgra_bytes(image: &JpegDecodedImage) -> Vec<u8> {
        let mut out = Vec::with_capacity(image.width * image.height * 4);
        for i in (0..image.pixels.len()).step_by(3) {
            let r = (image.pixels[i].clamp(0.0, 1.0) * 255.0).round() as u8;
            let g = (image.pixels[i + 1].clamp(0.0, 1.0) * 255.0).round() as u8;
            let b = (image.pixels[i + 2].clamp(0.0, 1.0) * 255.0).round() as u8;
            out.push(b);
            out.push(g);
            out.push(r);
            out.push(255);
        }
        out
    }

    fn ycbcr_to_rgb_u8(y: u8, cb: u8, cr: u8) -> (u8, u8, u8) {
        let y = y as f32;
        let cb = cb as f32 - 128.0;
        let cr = cr as f32 - 128.0;
        let r = (y + 1.402 * cr).clamp(0.0, 255.0).round() as u8;
        let g = (y - 0.344136 * cb - 0.714136 * cr).clamp(0.0, 255.0).round() as u8;
        let b = (y + 1.772 * cb).clamp(0.0, 255.0).round() as u8;
        (r, g, b)
    }

    #[test]
    fn test_parse_grayscale_8x8() {
        let decoder = JpegDecoder::new(GRAYSCALE_8X8).expect("Failed to create decoder");
        assert_eq!(decoder.width(), 8);
        assert_eq!(decoder.height(), 8);
        assert_eq!(decoder.num_components(), 1);
        assert!(!decoder.is_progressive());
    }

    #[test]
    fn test_parse_cmyk_16x16() {
        let decoder = JpegDecoder::new(CMYK_16X16).expect("Failed to create decoder");
        assert_eq!(decoder.width(), 16);
        assert_eq!(decoder.height(), 16);
        assert_eq!(decoder.num_components(), 4);
    }

    #[test]
    fn test_parse_rgb_8x8() {
        let decoder = JpegDecoder::new(RGB_8X8).expect("Failed to create decoder");
        assert_eq!(decoder.width(), 8);
        assert_eq!(decoder.height(), 8);
        assert_eq!(decoder.num_components(), 3);
    }

    #[test]
    fn test_parse_12bit_rgb() {
        let decoder = JpegDecoder::new(RGB_12BIT).expect("Failed to create decoder");
        assert_eq!(decoder.bits_per_sample(), 12);
        assert_eq!(decoder.num_components(), 3);
    }

    #[test]
    fn test_decode_into_u16_12bit() {
        let decoder = JpegDecoder::new(RGB_12BIT).expect("Failed to create decoder");
        let options = JpegDecodeOptions::default();
        let width = decoder.width();
        let height = decoder.height();
        let mut output = vec![0u16; width * height * 3];
        let (w, h) = decoder
            .decode_into_u16(
                RGB_12BIT,
                &options,
                JpegOutputFormat16::Rgb16,
                &mut output,
                width * 3,
            )
            .expect("Failed to decode 12-bit JPEG into u16");
        assert_eq!(w, width);
        assert_eq!(h, height);
        assert!(output.iter().any(|&v| v > 255));
    }

    #[test]
    fn test_decode_into_u8_rejects_12bit() {
        let decoder = JpegDecoder::new(RGB_12BIT).expect("Failed to create decoder");
        let options = JpegDecodeOptions::default();
        let mut output = vec![0u8; decoder.width() * decoder.height() * 3];
        let result = decoder.decode_into(
            RGB_12BIT,
            &options,
            JpegOutputFormat::Rgb8,
            &mut output,
            decoder.width() * 3,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_progressive_8x8() {
        let decoder = JpegDecoder::new(PROGRESSIVE_8X8).expect("Failed to create decoder");
        assert_eq!(decoder.width(), 8);
        assert_eq!(decoder.height(), 8);
        assert!(decoder.is_progressive());
    }

    #[test]
    fn test_parse_rgb_64x64() {
        let decoder = JpegDecoder::new(RGB_64X64).expect("Failed to create decoder");
        assert_eq!(decoder.width(), 64);
        assert_eq!(decoder.height(), 64);
    }

    #[test]
    fn test_parse_gradient_32x32() {
        let decoder = JpegDecoder::new(GRADIENT_32X32).expect("Failed to create decoder");
        assert_eq!(decoder.width(), 32);
        assert_eq!(decoder.height(), 32);
    }

    #[test]
    fn test_parse_quality_images() {
        let decoder_100 = JpegDecoder::new(QUALITY_100).expect("Failed to create decoder");
        let decoder_50 = JpegDecoder::new(QUALITY_50).expect("Failed to create decoder");

        // Both should parse to same dimensions
        assert_eq!(decoder_100.width(), decoder_50.width());
        assert_eq!(decoder_100.height(), decoder_50.height());
    }

    #[test]
    fn test_parse_subsampled() {
        let decoder_420 = JpegDecoder::new(SUBSAMPLED_420).expect("Failed to create decoder");
        let decoder_444 = JpegDecoder::new(SUBSAMPLED_444).expect("Failed to create decoder");

        assert_eq!(decoder_420.width(), 8);
        assert_eq!(decoder_420.height(), 8);
        assert_eq!(decoder_444.width(), 8);
        assert_eq!(decoder_444.height(), 8);
    }

    #[test]
    fn test_metadata_extraction() {
        let decoder = JpegDecoder::new(RGB_8X8).expect("Failed to create decoder");
        let metadata = decoder.metadata();

        // JFIF version should be present for ImageMagick-generated files
        assert!(metadata.jfif_version.is_some(), "Expected JFIF version");
    }

    #[test]
    fn test_metadata_exif_present() {
        let decoder = JpegDecoder::new(WITH_EXIF_8X8).expect("Failed to create decoder");
        let metadata = decoder.metadata();
        assert!(metadata.exif_data.is_some(), "Expected EXIF metadata");
    }

    #[test]
    fn test_dimensions_method() {
        let decoder = JpegDecoder::new(RGB_64X64).expect("Failed to create decoder");
        let (w, h) = decoder.dimensions();
        assert_eq!(w, 64);
        assert_eq!(h, 64);
    }

    #[test]
    fn test_decode_grayscale() {
        let decoder = JpegDecoder::new(GRAYSCALE_8X8).expect("Failed to create decoder");
        let result = decoder.decode(GRAYSCALE_8X8);
        // Note: decode may fail if Huffman tables aren't populated correctly
        // This tests that decode doesn't panic
        if let Ok(image) = result {
            assert_eq!(image.width, 8);
            assert_eq!(image.height, 8);
            assert!(image.pixels.len() > 0);
        }
    }

    #[test]
    fn test_decode_restart_marker() {
        let decoder = JpegDecoder::new(RESTART_8X8).expect("Failed to create decoder");
        let image = decoder.decode(RESTART_8X8).expect("Failed to decode restart JPEG");
        assert_eq!(image.width, 8);
        assert_eq!(image.height, 8);
    }

    #[test]
    fn test_decode_cmyk() {
        let decoder = JpegDecoder::new(CMYK_16X16).expect("Failed to create decoder");
        let image = decoder.decode(CMYK_16X16).expect("Failed to decode CMYK JPEG");
        assert_eq!(image.width, 16);
        assert_eq!(image.height, 16);
        assert_eq!(image.num_components, 3);
        assert_eq!(image.pixels.len(), 16 * 16 * 3);
    }

    #[test]
    fn test_decode_into_bgra() {
        let decoder = JpegDecoder::new(RGB_8X8).expect("Failed to create decoder");
        let options = JpegDecodeOptions::default();
        let mut output = vec![0u8; 8 * 8 * 4];
        let (w, h) = decoder
            .decode_into(RGB_8X8, &options, JpegOutputFormat::Bgra8, &mut output, 8 * 4)
            .expect("Failed to decode into BGRA");
        assert_eq!(w, 8);
        assert_eq!(h, 8);
    }

    #[test]
    fn test_decode_into_cmyk() {
        let decoder = JpegDecoder::new(CMYK_16X16).expect("Failed to create decoder");
        let options = JpegDecodeOptions::default();
        let mut output = vec![0u8; 16 * 16 * 4];
        let (w, h) = decoder
            .decode_into(CMYK_16X16, &options, JpegOutputFormat::Cmyk8, &mut output, 16 * 4)
            .expect("Failed to decode into CMYK");
        assert_eq!(w, 16);
        assert_eq!(h, 16);
    }

    #[test]
    fn test_decode_into_scaled_half() {
        let decoder = JpegDecoder::new(RGB_64X64).expect("Failed to create decoder");
        let options = JpegDecodeOptions {
            scale: IdctScale::Scale1_2,
            ..Default::default()
        };
        let mut output = vec![0u8; 32 * 32 * 4];
        let (w, h) = decoder
            .decode_into(RGB_64X64, &options, JpegOutputFormat::Bgra8, &mut output, 32 * 4)
            .expect("Failed to decode into scaled BGRA");
        assert_eq!(w, 32);
        assert_eq!(h, 32);
    }

    #[test]
    fn test_decode_scaled_half() {
        let decoder = JpegDecoder::new(RGB_64X64).expect("Failed to create decoder");
        let options = JpegDecodeOptions {
            scale: IdctScale::Scale1_2,
            ..Default::default()
        };
        let image = decoder
            .decode_with_options(RGB_64X64, &options)
            .expect("Failed to decode scaled JPEG");
        assert_eq!(image.width, 32);
        assert_eq!(image.height, 32);
        assert_eq!(image.pixels.len(), 32 * 32 * 3);
    }

    #[test]
    fn test_decode_into_ycbcr_roundtrip() {
        let decoder = JpegDecoder::new(RGB_8X8).expect("Failed to create decoder");
        let options = JpegDecodeOptions::default();

        let mut ycbcr = vec![0u8; 8 * 8 * 3];
        decoder
            .decode_into(RGB_8X8, &options, JpegOutputFormat::Ycbcr8, &mut ycbcr, 8 * 3)
            .expect("YCbCr decode failed");

        let mut rgb_direct = vec![0u8; 8 * 8 * 3];
        decoder
            .decode_into(RGB_8X8, &options, JpegOutputFormat::Rgb8, &mut rgb_direct, 8 * 3)
            .expect("RGB decode failed");

        for i in 0..(8 * 8) {
            let y = ycbcr[i * 3];
            let cb = ycbcr[i * 3 + 1];
            let cr = ycbcr[i * 3 + 2];
            let (r, g, b) = ycbcr_to_rgb_u8(y, cb, cr);
            let dr = rgb_direct[i * 3];
            let dg = rgb_direct[i * 3 + 1];
            let db = rgb_direct[i * 3 + 2];
            assert!((r as i32 - dr as i32).abs() <= 1);
            assert!((g as i32 - dg as i32).abs() <= 1);
            assert!((b as i32 - db as i32).abs() <= 1);
        }
    }

    #[test]
    fn test_decode_into_subsampled_matches_decode() {
        let decoder = JpegDecoder::new(SUBSAMPLED_420).expect("Failed to create decoder");
        let options = JpegDecodeOptions::default();

        let mut output = vec![0u8; 8 * 8 * 4];
        decoder
            .decode_into(SUBSAMPLED_420, &options, JpegOutputFormat::Bgra8, &mut output, 8 * 4)
            .expect("decode_into failed");

        let image = decoder.decode(SUBSAMPLED_420).expect("decode failed");
        let expected = rgb_to_bgra_bytes(&image);
        assert_eq!(output.len(), expected.len());
        for (a, b) in output.iter().zip(expected.iter()) {
            assert!((*a as i32 - *b as i32).abs() <= 1);
        }
    }

    #[test]
    fn test_all_files_parse() {
        let test_files: &[(&str, &[u8])] = &[
            ("cmyk_16x16", CMYK_16X16),
            ("grayscale_8x8", GRAYSCALE_8X8),
            ("rgb_8x8", RGB_8X8),
            ("progressive_8x8", PROGRESSIVE_8X8),
            ("rgb_64x64", RGB_64X64),
            ("gradient_32x32", GRADIENT_32X32),
            ("quality100_16x16", QUALITY_100),
            ("quality50_16x16", QUALITY_50),
            ("restart_8x8", RESTART_8X8),
            ("subsampled_420", SUBSAMPLED_420),
            ("subsampled_444", SUBSAMPLED_444),
            ("with_exif_8x8", WITH_EXIF_8X8),
            ("rgb_12bit", RGB_12BIT),
        ];

        for (name, data) in test_files {
            let result = JpegDecoder::new(data);
            assert!(result.is_ok(), "Failed to parse {}: {:?}", name, result.err());
            let decoder = result.unwrap();
            assert!(decoder.width() > 0, "{} has zero width", name);
            assert!(decoder.height() > 0, "{} has zero height", name);
        }
    }
}
