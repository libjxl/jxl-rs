//! Minimal CXX wrapper for jxl-rs JPEG decoder.
//!
//! This thin wrapper provides C++-compatible types for JPEG decoding.
//! Designed to be a drop-in replacement for libjpeg-turbo in Chromium.

use jxl::jpeg::{IdctScale, JpegDecodeOptions, JpegDecoder, JpegMetadata, JpegOutputFormat};

#[cxx::bridge(namespace = "blink::jpeg_rs")]
mod ffi {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum JpegRsStatus {
        Success = 0,
        Error = 1,
        NeedMoreInput = 2,
    }

    #[derive(Debug, Clone, Copy)]
    enum JpegRsPixelFormat {
        Rgb8 = 0,
        Rgba8 = 1,
        Bgra8 = 2,
        Gray8 = 3,
        RgbF32 = 4,
    }

    #[derive(Debug, Clone, Copy)]
    enum JpegRsColorSpace {
        Unknown = 0,
        Grayscale = 1,
        Rgb = 2,
        YCbCr = 3,
        Cmyk = 4,
        Ycck = 5,
    }

    #[derive(Debug, Clone)]
    struct JpegRsBasicInfo {
        width: u32,
        height: u32,
        num_components: u32,
        bits_per_sample: u32,
        is_progressive: bool,
        color_space: JpegRsColorSpace,
        has_icc_profile: bool,
        has_exif: bool,
    }

    /// Result of a process call.
    #[derive(Debug, Clone)]
    struct JpegRsProcessResult {
        status: JpegRsStatus,
        bytes_consumed: usize,
    }

    extern "Rust" {
        type JpegRsDecoder;

        /// Create a new JPEG decoder.
        fn jpeg_rs_decoder_create(pixel_limit: u64) -> Box<JpegRsDecoder>;

        /// Check if data starts with JPEG signature (0xFFD8).
        fn jpeg_rs_signature_check(data: &[u8]) -> bool;

        /// Reset decoder state for reuse.
        fn reset(self: &mut JpegRsDecoder);

        /// Set the output pixel format.
        fn set_pixel_format(self: &mut JpegRsDecoder, format: JpegRsPixelFormat);

        /// Parse JPEG headers to get basic info.
        /// Returns Success when headers are fully parsed.
        fn parse_headers(
            self: &mut JpegRsDecoder,
            data: &[u8],
            all_input: bool,
        ) -> JpegRsProcessResult;

        /// Decode image pixels into the provided buffer.
        /// Buffer must be width * height * bytes_per_pixel.
        fn decode_image(
            self: &mut JpegRsDecoder,
            data: &[u8],
            all_input: bool,
            buffer: &mut [u8],
        ) -> JpegRsProcessResult;

        /// Decode image with custom row stride (for direct frame buffer decoding).
        fn decode_image_with_stride(
            self: &mut JpegRsDecoder,
            data: &[u8],
            all_input: bool,
            buffer: &mut [u8],
            row_stride: usize,
        ) -> JpegRsProcessResult;

        /// Get basic info (valid after parse_headers succeeds).
        fn get_basic_info(self: &JpegRsDecoder) -> JpegRsBasicInfo;

        /// Get ICC profile data (valid after parse_headers succeeds).
        fn get_icc_profile(self: &JpegRsDecoder) -> &[u8];

        /// Get EXIF data (valid after parse_headers succeeds).
        fn get_exif_data(self: &JpegRsDecoder) -> &[u8];

        /// Get decoded width (may differ from original if scaled).
        fn get_output_width(self: &JpegRsDecoder) -> u32;

        /// Get decoded height (may differ from original if scaled).
        fn get_output_height(self: &JpegRsDecoder) -> u32;

        /// Set scale denominator for DCT scaling (1, 2, 4, or 8).
        /// Must be called before decode_image.
        fn set_scale_denom(self: &mut JpegRsDecoder, denom: u32);
    }
}

use ffi::*;

/// JPEG decoder wrapper for Chromium integration.
pub struct JpegRsDecoder {
    /// Accumulated input data (JPEG requires full file for progressive)
    input_buffer: Vec<u8>,
    /// Parsed decoder (created after headers are available)
    decoder: Option<JpegDecoder>,
    /// Extracted metadata
    metadata: Option<JpegMetadata>,
    /// ICC profile cache
    icc_profile: Vec<u8>,
    /// EXIF data cache
    exif_data: Vec<u8>,
    /// Output pixel format
    pixel_format: JpegRsPixelFormat,
    /// Pixel limit for DoS protection
    pixel_limit: u64,
    /// Scale denominator (1, 2, 4, or 8)
    scale_denom: u32,
    /// Whether headers have been parsed
    headers_parsed: bool,
    /// Whether image has been decoded
    image_decoded: bool,
    /// Basic info cache
    basic_info: JpegRsBasicInfo,
}

fn jpeg_rs_decoder_create(pixel_limit: u64) -> Box<JpegRsDecoder> {
    Box::new(JpegRsDecoder {
        input_buffer: Vec::new(),
        decoder: None,
        metadata: None,
        icc_profile: Vec::new(),
        exif_data: Vec::new(),
        pixel_format: JpegRsPixelFormat::Bgra8,
        pixel_limit,
        scale_denom: 1,
        headers_parsed: false,
        image_decoded: false,
        basic_info: JpegRsBasicInfo::default(),
    })
}

fn jpeg_rs_signature_check(data: &[u8]) -> bool {
    // JPEG starts with SOI marker: 0xFF 0xD8
    data.len() >= 2 && data[0] == 0xFF && data[1] == 0xD8
}

impl JpegRsDecoder {
    fn reset(&mut self) {
        self.input_buffer.clear();
        self.decoder = None;
        self.metadata = None;
        self.icc_profile.clear();
        self.exif_data.clear();
        self.headers_parsed = false;
        self.image_decoded = false;
        self.basic_info = JpegRsBasicInfo::default();
    }

    fn set_pixel_format(&mut self, format: JpegRsPixelFormat) {
        self.pixel_format = format;
    }

    fn set_scale_denom(&mut self, denom: u32) {
        // Only valid denominators are 1, 2, 4, 8
        self.scale_denom = match denom {
            1 | 2 | 4 | 8 => denom,
            _ => 1,
        };
    }

    fn parse_headers(
        &mut self,
        data: &[u8],
        all_input: bool,
    ) -> JpegRsProcessResult {
        if self.headers_parsed {
            return JpegRsProcessResult {
                status: JpegRsStatus::Success,
                bytes_consumed: 0,
            };
        }

        // Accumulate input data
        self.input_buffer.extend_from_slice(data);

        // Try to parse JPEG headers
        // We need enough data to find SOF marker (Start of Frame)
        match self.try_parse_headers() {
            Ok(true) => {
                self.headers_parsed = true;
                JpegRsProcessResult {
                    status: JpegRsStatus::Success,
                    bytes_consumed: data.len(),
                }
            }
            Ok(false) => {
                if all_input {
                    JpegRsProcessResult {
                        status: JpegRsStatus::Error,
                        bytes_consumed: 0,
                    }
                } else {
                    JpegRsProcessResult {
                        status: JpegRsStatus::NeedMoreInput,
                        bytes_consumed: data.len(),
                    }
                }
            }
            Err(_) => JpegRsProcessResult {
                status: JpegRsStatus::Error,
                bytes_consumed: 0,
            },
        }
    }

    fn try_parse_headers(&mut self) -> Result<bool, ()> {
        // Need at least SOI + some markers
        if self.input_buffer.len() < 20 {
            return Ok(false);
        }

        // Check SOI
        if self.input_buffer[0] != 0xFF || self.input_buffer[1] != 0xD8 {
            return Err(());
        }

        // Scan for SOF marker to get dimensions
        let mut pos = 2;
        while pos + 4 < self.input_buffer.len() {
            if self.input_buffer[pos] != 0xFF {
                pos += 1;
                continue;
            }

            let marker = self.input_buffer[pos + 1];
            pos += 2;

            // Skip markers without length
            if marker == 0x00 || marker == 0x01 || (0xD0..=0xD9).contains(&marker) {
                continue;
            }

            // Get marker length
            if pos + 2 > self.input_buffer.len() {
                return Ok(false); // Need more data
            }
            let len = u16::from_be_bytes([
                self.input_buffer[pos],
                self.input_buffer[pos + 1],
            ]) as usize;

            // SOF markers (Start of Frame)
            if (0xC0..=0xC3).contains(&marker) || (0xC5..=0xC7).contains(&marker) ||
               (0xC9..=0xCB).contains(&marker) || (0xCD..=0xCF).contains(&marker) {
                if pos + len > self.input_buffer.len() {
                    return Ok(false); // Need more data
                }

                // Parse SOF: precision(1) + height(2) + width(2) + num_components(1)
                if len < 8 {
                    return Err(());
                }

                let precision = self.input_buffer[pos + 2];
                let height = u16::from_be_bytes([
                    self.input_buffer[pos + 3],
                    self.input_buffer[pos + 4],
                ]) as u32;
                let width = u16::from_be_bytes([
                    self.input_buffer[pos + 5],
                    self.input_buffer[pos + 6],
                ]) as u32;
                let num_components = self.input_buffer[pos + 7] as u32;

                // Check pixel limit
                if self.pixel_limit > 0 && (width as u64 * height as u64) > self.pixel_limit {
                    return Err(());
                }

                let is_progressive = marker == 0xC2 || marker == 0xC6 ||
                                    marker == 0xCA || marker == 0xCE;

                let color_space = match num_components {
                    1 => JpegRsColorSpace::Grayscale,
                    3 => JpegRsColorSpace::YCbCr,
                    4 => JpegRsColorSpace::Cmyk,
                    _ => JpegRsColorSpace::Unknown,
                };

                // Check for ICC and EXIF by scanning APP markers
                let (has_icc, has_exif) = self.scan_app_markers();

                self.basic_info = JpegRsBasicInfo {
                    width,
                    height,
                    num_components,
                    bits_per_sample: precision as u32,
                    is_progressive,
                    color_space,
                    has_icc_profile: has_icc,
                    has_exif,
                };

                return Ok(true);
            }

            pos += len;
        }

        Ok(false) // Need more data to find SOF
    }

    fn scan_app_markers(&self) -> (bool, bool) {
        let mut has_icc = false;
        let mut has_exif = false;
        let mut pos = 2;

        while pos + 4 < self.input_buffer.len() {
            if self.input_buffer[pos] != 0xFF {
                pos += 1;
                continue;
            }

            let marker = self.input_buffer[pos + 1];
            pos += 2;

            if marker == 0x00 || marker == 0x01 || (0xD0..=0xD9).contains(&marker) {
                continue;
            }

            if pos + 2 > self.input_buffer.len() {
                break;
            }
            let len = u16::from_be_bytes([
                self.input_buffer[pos],
                self.input_buffer[pos + 1],
            ]) as usize;

            // APP1 - EXIF
            if marker == 0xE1 && pos + 8 < self.input_buffer.len() {
                if &self.input_buffer[pos + 2..pos + 8] == b"Exif\x00\x00" {
                    has_exif = true;
                }
            }

            // APP2 - ICC
            if marker == 0xE2 && pos + 14 < self.input_buffer.len() {
                if &self.input_buffer[pos + 2..pos + 14] == b"ICC_PROFILE\x00" {
                    has_icc = true;
                }
            }

            pos += len;
        }

        (has_icc, has_exif)
    }

    fn decode_image(
        &mut self,
        data: &[u8],
        all_input: bool,
        buffer: &mut [u8],
    ) -> JpegRsProcessResult {
        let width = self.get_output_width() as usize;
        let row_stride = width * self.bytes_per_pixel();
        self.decode_image_with_stride(data, all_input, buffer, row_stride)
    }

    fn decode_image_with_stride(
        &mut self,
        data: &[u8],
        all_input: bool,
        buffer: &mut [u8],
        row_stride: usize,
    ) -> JpegRsProcessResult {
        if self.image_decoded {
            return JpegRsProcessResult {
                status: JpegRsStatus::Success,
                bytes_consumed: self.input_buffer.len(),
            };
        }

        // Accumulate input
        self.input_buffer.extend_from_slice(data);

        if !all_input {
            // JPEG decoding typically needs the full file
            // For progressive display, we could implement partial decoding
            return JpegRsProcessResult {
                status: JpegRsStatus::NeedMoreInput,
                bytes_consumed: data.len(),
            };
        }

        // Create decoder and decode
        match JpegDecoder::new(&self.input_buffer) {
            Ok(decoder) => {
                self.metadata = Some(decoder.extract_metadata());
                if let Some(ref meta) = self.metadata {
                    if let Some(ref icc) = meta.icc_profile {
                        self.icc_profile = icc.clone();
                    }
                    if let Some(ref exif) = meta.exif_data {
                        self.exif_data = exif.clone();
                    }
                }

                let options = JpegDecodeOptions {
                    scale: self.idct_scale(),
                    ..Default::default()
                };

                let output_format = match self.output_format() {
                    Some(format) => format,
                    None => {
                        return JpegRsProcessResult {
                            status: JpegRsStatus::Error,
                            bytes_consumed: 0,
                        };
                    }
                };
                match decoder.decode_into(&self.input_buffer, &options, output_format, buffer, row_stride) {
                    Ok(_) => {
                        self.image_decoded = true;
                        self.decoder = Some(decoder);
                        JpegRsProcessResult {
                            status: JpegRsStatus::Success,
                            bytes_consumed: self.input_buffer.len(),
                        }
                    }
                    Err(_) => JpegRsProcessResult {
                        status: JpegRsStatus::Error,
                        bytes_consumed: 0,
                    },
                }
            }
            Err(_) => JpegRsProcessResult {
                status: JpegRsStatus::Error,
                bytes_consumed: 0,
            },
        }
    }

    fn get_basic_info(&self) -> JpegRsBasicInfo {
        self.basic_info.clone()
    }

    fn get_icc_profile(&self) -> &[u8] {
        &self.icc_profile
    }

    fn get_exif_data(&self) -> &[u8] {
        &self.exif_data
    }

    fn get_output_width(&self) -> u32 {
        Self::scaled_dimension(self.basic_info.width, self.scale_denom)
    }

    fn get_output_height(&self) -> u32 {
        Self::scaled_dimension(self.basic_info.height, self.scale_denom)
    }

    fn idct_scale(&self) -> IdctScale {
        match self.scale_denom {
            2 => IdctScale::Scale1_2,
            4 => IdctScale::Scale1_4,
            8 => IdctScale::Scale1_8,
            _ => IdctScale::Full,
        }
    }

    fn scaled_dimension(size: u32, denom: u32) -> u32 {
        if denom == 0 {
            return size;
        }
        (size + denom - 1) / denom
    }

    fn output_format(&self) -> Option<JpegOutputFormat> {
        match self.pixel_format {
            JpegRsPixelFormat::Rgb8 => Some(JpegOutputFormat::Rgb8),
            JpegRsPixelFormat::Rgba8 => Some(JpegOutputFormat::Rgba8),
            JpegRsPixelFormat::Bgra8 => Some(JpegOutputFormat::Bgra8),
            JpegRsPixelFormat::Gray8 => Some(JpegOutputFormat::Gray8),
            JpegRsPixelFormat::RgbF32 => None,
        }
    }
}

impl Default for JpegRsBasicInfo {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            num_components: 0,
            bits_per_sample: 8,
            is_progressive: false,
            color_space: JpegRsColorSpace::Unknown,
            has_icc_profile: false,
            has_exif: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signature_check() {
        assert!(jpeg_rs_signature_check(&[0xFF, 0xD8, 0xFF, 0xE0]));
        assert!(!jpeg_rs_signature_check(&[0x89, 0x50, 0x4E, 0x47])); // PNG
        assert!(!jpeg_rs_signature_check(&[0xFF])); // Too short
    }

    #[test]
    fn test_decoder_create() {
        let decoder = jpeg_rs_decoder_create(1024 * 1024);
        assert!(!decoder.headers_parsed);
        assert!(!decoder.image_decoded);
    }

    #[test]
    fn test_decoder_reset() {
        let mut decoder = jpeg_rs_decoder_create(1024 * 1024);
        decoder.input_buffer.push(0xFF);
        decoder.headers_parsed = true;

        decoder.reset();

        assert!(decoder.input_buffer.is_empty());
        assert!(!decoder.headers_parsed);
    }
}
