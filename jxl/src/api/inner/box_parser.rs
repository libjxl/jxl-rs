// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::error::{Error, Result};
use std::io::IoSliceMut;

use crate::api::{
    JxlBitstreamInput, JxlSignatureType, MetadataCaptureOptions, check_signature_internal,
    inner::process::SmallBuffer, JxlMetadataBox,
};

/// Type of metadata box being captured.
#[derive(Clone, Copy)]
enum MetadataBoxType {
    Exif,
    Xml,
    Jumbf,
}

#[derive(Clone)]
enum ParseState {
    SignatureNeeded,
    BoxNeeded,
    CodestreamBox(u64),
    SkippableBox(u64),
    /// Reading metadata box content (EXIF, XML, or JUMBF)
    MetadataBox {
        box_type: MetadataBoxType,
        bytes_left: u64,
        buffer: Vec<u8>,
        is_brotli_compressed: bool,
    },
    /// Reading brob header (4-byte inner type) before deciding what to do with content
    BrotliBoxHeader {
        /// Total content length remaining (including the 4-byte inner type)
        bytes_left: u64,
    },
}

enum CodestreamBoxType {
    None,
    Jxlc,
    Jxlp(u32),
    LastJxlp,
}

pub(super) struct BoxParser {
    pub(super) box_buffer: SmallBuffer,
    state: ParseState,
    box_type: CodestreamBoxType,

    // Captured metadata boxes
    pub(super) exif_boxes: Vec<JxlMetadataBox>,
    pub(super) xml_boxes: Vec<JxlMetadataBox>,
    pub(super) jumbf_boxes: Vec<JxlMetadataBox>,

    // Aggregate sizes for limit tracking
    exif_total_size: u64,
    xml_total_size: u64,
    jumbf_total_size: u64,

    // Capture options
    capture_exif: bool,
    capture_xml: bool,
    capture_jumbf: bool,
    exif_size_limit: Option<u64>,
    xml_size_limit: Option<u64>,
    jumbf_size_limit: Option<u64>,
}

impl BoxParser {
    pub(super) fn new(opts: &MetadataCaptureOptions) -> Self {
        BoxParser {
            box_buffer: SmallBuffer::new(128),
            state: ParseState::SignatureNeeded,
            box_type: CodestreamBoxType::None,
            exif_boxes: Vec::new(),
            xml_boxes: Vec::new(),
            jumbf_boxes: Vec::new(),
            exif_total_size: 0,
            xml_total_size: 0,
            jumbf_total_size: 0,
            capture_exif: opts.capture_exif,
            capture_xml: opts.capture_xml,
            capture_jumbf: opts.capture_jumbf,
            exif_size_limit: opts.exif_size_limit,
            xml_size_limit: opts.xml_size_limit,
            jumbf_size_limit: opts.jumbf_size_limit,
        }
    }

    // Reads input until the next byte of codestream is available.
    // This function might over-read bytes. Thus, the contents of self.box_buffer should always be
    // read after this function call.
    // Returns the number of codestream bytes that will be available to be read after this call,
    // including any bytes in self.box_buffer.
    // Might return `u64::MAX`, indicating that the rest of the file is codestream.
    pub(super) fn get_more_codestream(&mut self, input: &mut dyn JxlBitstreamInput) -> Result<u64> {
        loop {
            match self.state.clone() {
                ParseState::SignatureNeeded => {
                    self.box_buffer.refill(|b| input.read(b), None)?;
                    match check_signature_internal(&self.box_buffer)? {
                        None => return Err(Error::InvalidSignature),
                        Some(JxlSignatureType::Codestream) => {
                            self.state = ParseState::CodestreamBox(u64::MAX);
                            return Ok(u64::MAX);
                        }
                        Some(JxlSignatureType::Container) => {
                            self.box_buffer
                                .consume(JxlSignatureType::Container.signature().len());
                            self.state = ParseState::BoxNeeded;
                        }
                    }
                }
                ParseState::CodestreamBox(b) => {
                    return Ok(b);
                }
                ParseState::SkippableBox(mut s) => {
                    let num = s.min(usize::MAX as u64) as usize;
                    let skipped = if !self.box_buffer.is_empty() {
                        self.box_buffer.consume(num)
                    } else {
                        input.skip(num)?
                    };
                    if skipped == 0 {
                        return Err(Error::OutOfBounds(num));
                    }
                    s -= skipped as u64;
                    if s == 0 {
                        self.state = ParseState::BoxNeeded;
                    } else {
                        self.state = ParseState::SkippableBox(s);
                    }
                }
                ParseState::MetadataBox {
                    box_type,
                    mut bytes_left,
                    mut buffer,
                    is_brotli_compressed,
                } => {
                    let num = bytes_left.min(usize::MAX as u64) as usize;
                    // First consume any buffered data
                    if !self.box_buffer.is_empty() {
                        let to_read = num.min(self.box_buffer.len());
                        buffer.extend_from_slice(&self.box_buffer[..to_read]);
                        self.box_buffer.consume(to_read);
                        bytes_left -= to_read as u64;
                    } else {
                        // Read directly from input using IoSliceMut
                        let mut read_buf = vec![0u8; num.min(8192)];
                        let read = input.read(&mut [IoSliceMut::new(&mut read_buf)])?;
                        if read == 0 {
                            return Err(Error::OutOfBounds(num));
                        }
                        buffer.extend_from_slice(&read_buf[..read]);
                        bytes_left -= read as u64;
                    }
                    if bytes_left == 0 {
                        // Store completed metadata box and update aggregate size
                        let box_size = buffer.len() as u64;
                        let metadata_box = JxlMetadataBox {
                            data: buffer,
                            is_brotli_compressed,
                        };
                        match box_type {
                            MetadataBoxType::Exif => {
                                self.exif_total_size += box_size;
                                self.exif_boxes.push(metadata_box);
                            }
                            MetadataBoxType::Xml => {
                                self.xml_total_size += box_size;
                                self.xml_boxes.push(metadata_box);
                            }
                            MetadataBoxType::Jumbf => {
                                self.jumbf_total_size += box_size;
                                self.jumbf_boxes.push(metadata_box);
                            }
                        }
                        self.state = ParseState::BoxNeeded;
                    } else {
                        self.state = ParseState::MetadataBox {
                            box_type,
                            bytes_left,
                            buffer,
                            is_brotli_compressed,
                        };
                    }
                }
                ParseState::BrotliBoxHeader { bytes_left } => {
                    // We need at least 4 bytes to read the inner box type
                    self.box_buffer.refill(|b| input.read(b), None)?;
                    if self.box_buffer.len() < 4 {
                        return Err(Error::OutOfBounds(4 - self.box_buffer.len()));
                    }
                    let inner_ty: [u8; 4] = self.box_buffer[0..4].try_into().unwrap();
                    self.box_buffer.consume(4);
                    let content_len = bytes_left - 4;

                    // Check if we should capture this brob based on inner type
                    let (should_capture, box_type, size_limit, current_size) = match &inner_ty {
                        b"Exif" => (
                            self.capture_exif,
                            MetadataBoxType::Exif,
                            self.exif_size_limit,
                            self.exif_total_size,
                        ),
                        b"xml " => (
                            self.capture_xml,
                            MetadataBoxType::Xml,
                            self.xml_size_limit,
                            self.xml_total_size,
                        ),
                        b"jumb" => (
                            self.capture_jumbf,
                            MetadataBoxType::Jumbf,
                            self.jumbf_size_limit,
                            self.jumbf_total_size,
                        ),
                        _ => (false, MetadataBoxType::Exif, None, 0), // Won't be used
                    };

                    let within_limit = size_limit
                        .map(|limit| current_size.saturating_add(content_len) <= limit)
                        .unwrap_or(true);

                    if should_capture && within_limit {
                        self.state = ParseState::MetadataBox {
                            box_type,
                            bytes_left: content_len,
                            buffer: Vec::with_capacity(content_len.min(65536) as usize),
                            is_brotli_compressed: true,
                        };
                    } else {
                        self.state = ParseState::SkippableBox(content_len);
                    }
                }
                ParseState::BoxNeeded => {
                    self.box_buffer.refill(|b| input.read(b), None)?;
                    let min_len = match &self.box_buffer[..] {
                        [0, 0, 0, 1, ..] => 16,
                        _ => 8,
                    };
                    if self.box_buffer.len() <= min_len {
                        return Err(Error::OutOfBounds(min_len - self.box_buffer.len()));
                    }
                    let ty: [_; 4] = self.box_buffer[4..8].try_into().unwrap();
                    let extra_len = if &ty == b"jxlp" { 4 } else { 0 };
                    if self.box_buffer.len() <= min_len + extra_len {
                        return Err(Error::OutOfBounds(
                            min_len + extra_len - self.box_buffer.len(),
                        ));
                    }
                    let box_len = match &self.box_buffer[..] {
                        [0, 0, 0, 1, ..] => {
                            u64::from_be_bytes(self.box_buffer[8..16].try_into().unwrap())
                        }
                        _ => u32::from_be_bytes(self.box_buffer[0..4].try_into().unwrap()) as u64,
                    };
                    // Per JXL spec: jxlc box with length 0 has special meaning "extends to EOF"
                    let content_len = if box_len == 0 && (&ty == b"jxlp" || &ty == b"jxlc") {
                        u64::MAX
                    } else {
                        if box_len <= (min_len + extra_len) as u64 {
                            return Err(Error::InvalidBox);
                        }
                        box_len - min_len as u64 - extra_len as u64
                    };
                    match &ty {
                        b"jxlc" => {
                            if matches!(
                                self.box_type,
                                CodestreamBoxType::Jxlp(..) | CodestreamBoxType::LastJxlp
                            ) {
                                return Err(Error::InvalidBox);
                            }
                            self.box_type = CodestreamBoxType::Jxlc;
                            self.state = ParseState::CodestreamBox(content_len);
                        }
                        b"jxlp" => {
                            let index = u32::from_be_bytes(
                                self.box_buffer[min_len..min_len + 4].try_into().unwrap(),
                            );
                            let wanted_idx = match self.box_type {
                                CodestreamBoxType::Jxlc | CodestreamBoxType::LastJxlp => {
                                    return Err(Error::InvalidBox);
                                }
                                CodestreamBoxType::None => 0,
                                CodestreamBoxType::Jxlp(i) => i + 1,
                            };
                            let last = index & 0x80000000 != 0;
                            let idx = index & 0x7fffffff;
                            if idx != wanted_idx {
                                return Err(Error::InvalidBox);
                            }
                            self.box_type = if last {
                                CodestreamBoxType::LastJxlp
                            } else {
                                CodestreamBoxType::Jxlp(idx)
                            };
                            self.state = ParseState::CodestreamBox(content_len);
                        }
                        b"Exif" => {
                            // Capture EXIF metadata box if enabled and within aggregate limit
                            let within_limit = self
                                .exif_size_limit
                                .map(|limit| {
                                    self.exif_total_size.saturating_add(content_len) <= limit
                                })
                                .unwrap_or(true);
                            // u64::MAX is a sentinel for unbounded boxes (extends to EOF)
                            let is_bounded = content_len < u64::MAX;
                            if self.capture_exif && is_bounded && within_limit {
                                self.state = ParseState::MetadataBox {
                                    box_type: MetadataBoxType::Exif,
                                    bytes_left: content_len,
                                    buffer: Vec::with_capacity(content_len.min(65536) as usize),
                                    is_brotli_compressed: false,
                                };
                            } else {
                                self.state = ParseState::SkippableBox(content_len);
                            }
                        }
                        b"xml " => {
                            // Capture XML/XMP metadata box if enabled and within aggregate limit
                            let within_limit = self
                                .xml_size_limit
                                .map(|limit| {
                                    self.xml_total_size.saturating_add(content_len) <= limit
                                })
                                .unwrap_or(true);
                            // u64::MAX is a sentinel for unbounded boxes (extends to EOF)
                            let is_bounded = content_len < u64::MAX;
                            if self.capture_xml && is_bounded && within_limit {
                                self.state = ParseState::MetadataBox {
                                    box_type: MetadataBoxType::Xml,
                                    bytes_left: content_len,
                                    buffer: Vec::with_capacity(content_len.min(65536) as usize),
                                    is_brotli_compressed: false,
                                };
                            } else {
                                self.state = ParseState::SkippableBox(content_len);
                            }
                        }
                        b"jumb" => {
                            // Capture JUMBF metadata box if enabled and within aggregate limit
                            let within_limit = self
                                .jumbf_size_limit
                                .map(|limit| {
                                    self.jumbf_total_size.saturating_add(content_len) <= limit
                                })
                                .unwrap_or(true);
                            // u64::MAX is a sentinel for unbounded boxes (extends to EOF)
                            let is_bounded = content_len < u64::MAX;
                            if self.capture_jumbf && is_bounded && within_limit {
                                self.state = ParseState::MetadataBox {
                                    box_type: MetadataBoxType::Jumbf,
                                    bytes_left: content_len,
                                    buffer: Vec::with_capacity(content_len.min(65536) as usize),
                                    is_brotli_compressed: false,
                                };
                            } else {
                                self.state = ParseState::SkippableBox(content_len);
                            }
                        }
                        b"brob" => {
                            // Brotli-compressed box - read 4-byte inner type to decide action
                            // u64::MAX is a sentinel for unbounded boxes (extends to EOF)
                            let is_bounded = content_len < u64::MAX;
                            // brob needs at least 4 bytes for inner type
                            if is_bounded && content_len >= 4 {
                                self.state = ParseState::BrotliBoxHeader {
                                    bytes_left: content_len,
                                };
                            } else {
                                self.state = ParseState::SkippableBox(content_len);
                            }
                        }
                        _ => {
                            self.state = ParseState::SkippableBox(content_len);
                        }
                    }
                    self.box_buffer.consume(min_len + extra_len);
                }
            }
        }
    }

    pub(super) fn consume_codestream(&mut self, amount: u64) {
        if let ParseState::CodestreamBox(cb) = &mut self.state {
            *cb = cb.checked_sub(amount).unwrap();
            if *cb == 0 {
                self.state = ParseState::BoxNeeded;
            }
        } else if amount != 0 {
            unreachable!()
        }
    }

    pub(super) fn exif_boxes(&self) -> Option<&[JxlMetadataBox]> {
        self.capture_exif.then_some(&self.exif_boxes[..])
    }

    pub(super) fn xml_boxes(&self) -> Option<&[JxlMetadataBox]> {
        self.capture_xml.then_some(&self.xml_boxes[..])
    }

    pub(super) fn jumbf_boxes(&self) -> Option<&[JxlMetadataBox]> {
        self.capture_jumbf.then_some(&self.jumbf_boxes[..])
    }
}
