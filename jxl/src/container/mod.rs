// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Originally copied from jxl-bitstream of jxl-oxide.

pub mod box_header;

use box_header::*;

use crate::error::Error;

/// Container format parser.
#[derive(Default)]
pub struct ContainerParser {
    state: DetectState,
    buf: Vec<u8>,
    codestream: Vec<u8>,
    aux_boxes: Vec<(ContainerBoxType, Vec<u8>)>,
    next_jxlp_index: u32,
}

impl std::fmt::Debug for ContainerParser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContainerParser")
            .field("state", &self.state)
            .field("next_jxlp_index", &self.next_jxlp_index)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Default)]
enum DetectState {
    #[default]
    WaitingSignature,
    WaitingBoxHeader,
    WaitingJxlpIndex(ContainerBoxHeader),
    InAuxBox {
        header: ContainerBoxHeader,
        data: Vec<u8>,
        bytes_left: Option<usize>,
    },
    InCodestream {
        kind: BitstreamKind,
        bytes_left: Option<usize>,
    },
    Done(BitstreamKind),
}

/// Structure of the decoded bitstream.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum BitstreamKind {
    /// Decoder can't determine structure of the bitstream.
    Unknown,
    /// Bitstream is a direct JPEG XL codestream without box structure.
    BareCodestream,
    /// Bitstream is a JPEG XL container with box structure.
    Container,
    /// Bitstream is not a valid JPEG XL image.
    Invalid,
}

impl ContainerParser {
    const CODESTREAM_SIG: [u8; 2] = [0xff, 0x0a];
    const CONTAINER_SIG: [u8; 12] = [0, 0, 0, 0xc, b'J', b'X', b'L', b' ', 0xd, 0xa, 0x87, 0xa];

    pub fn new() -> Self {
        Self::default()
    }

    pub fn kind(&self) -> BitstreamKind {
        match self.state {
            DetectState::WaitingSignature => BitstreamKind::Unknown,
            DetectState::WaitingBoxHeader
            | DetectState::WaitingJxlpIndex(..)
            | DetectState::InAuxBox { .. } => BitstreamKind::Container,
            DetectState::InCodestream { kind, .. } | DetectState::Done(kind) => kind,
        }
    }

    pub fn feed_bytes(&mut self, input: &[u8]) -> Result<(), Error> {
        let state = &mut self.state;
        let buf = &mut self.buf;
        buf.extend_from_slice(input);

        loop {
            match state {
                DetectState::WaitingSignature => {
                    if buf.starts_with(&Self::CODESTREAM_SIG) {
                        // tracing::debug!("Codestream signature found");
                        *state = DetectState::InCodestream {
                            kind: BitstreamKind::BareCodestream,
                            bytes_left: None,
                        };
                        continue;
                    }
                    if buf.starts_with(&Self::CONTAINER_SIG) {
                        // tracing::debug!("Container signature found");
                        *state = DetectState::WaitingBoxHeader;
                        buf.drain(..Self::CONTAINER_SIG.len());
                        continue;
                    }
                    if !Self::CODESTREAM_SIG.starts_with(buf)
                        && !Self::CONTAINER_SIG.starts_with(buf)
                    {
                        // tracing::error!("Invalid signature");
                        *state = DetectState::InCodestream {
                            kind: BitstreamKind::Invalid,
                            bytes_left: None,
                        };
                        continue;
                    }
                    return Ok(());
                }
                DetectState::WaitingBoxHeader => match ContainerBoxHeader::parse(buf)? {
                    HeaderParseResult::Done { header, size } => {
                        buf.drain(..size);
                        let tbox = header.box_type();
                        if tbox == ContainerBoxType::CODESTREAM {
                            if self.next_jxlp_index == u32::MAX {
                                // tracing::error!("Duplicate jxlc box found");
                                return Err(Error::InvalidBox);
                            }
                            if self.next_jxlp_index != 0 {
                                // tracing::error!("Found jxlc box instead of jxlp box");
                                return Err(Error::InvalidBox);
                            }

                            self.next_jxlp_index = u32::MAX;
                            *state = DetectState::InCodestream {
                                kind: BitstreamKind::Container,
                                bytes_left: header.size().map(|x| x as usize),
                            };
                        } else if tbox == ContainerBoxType::PARTIAL_CODESTREAM {
                            if let Some(box_size) = header.size() {
                                if box_size < 4 {
                                    return Err(Error::InvalidBox);
                                }
                            }

                            if self.next_jxlp_index == u32::MAX {
                                // tracing::error!("jxlp box found after jxlc box");
                                return Err(Error::InvalidBox);
                            }

                            if self.next_jxlp_index >= 0x80000000 {
                                // tracing::error!(
                                //     "jxlp box #{} should be the last one, found the next one",
                                //     self.next_jxlp_index ^ 0x80000000,
                                // );
                                return Err(Error::InvalidBox);
                            }

                            *state = DetectState::WaitingJxlpIndex(header);
                        } else {
                            let bytes_left = header.size().map(|x| x as usize);
                            *state = DetectState::InAuxBox {
                                header,
                                data: Vec::new(),
                                bytes_left,
                            };
                        }
                        continue;
                    }
                    HeaderParseResult::NeedMoreData => return Ok(()),
                },
                DetectState::WaitingJxlpIndex(header) => {
                    if buf.len() < 4 {
                        return Ok(());
                    }

                    let index = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]);
                    buf.drain(..4);
                    let is_last = index & 0x80000000 != 0;
                    let index = index & 0x7fffffff;
                    // tracing::trace!(index, is_last);
                    if index != self.next_jxlp_index {
                        // tracing::error!(
                        //     "Out-of-order jxlp box found: expected {}, got {}",
                        //     self.next_jxlp_index,
                        //     index,
                        // );
                        return Err(Error::InvalidBox);
                    }

                    if is_last {
                        self.next_jxlp_index = index | 0x80000000;
                    } else {
                        self.next_jxlp_index += 1;
                    }

                    *state = DetectState::InCodestream {
                        kind: BitstreamKind::Container,
                        bytes_left: header.size().map(|x| x as usize - 4),
                    };
                }
                DetectState::InCodestream {
                    bytes_left: None, ..
                } => {
                    self.codestream.extend_from_slice(buf);
                    buf.clear();
                    return Ok(());
                }
                DetectState::InCodestream {
                    bytes_left: Some(bytes_left),
                    ..
                } => {
                    if *bytes_left <= buf.len() {
                        self.codestream.extend(buf.drain(..*bytes_left));
                        *state = DetectState::WaitingBoxHeader;
                    } else {
                        *bytes_left -= buf.len();
                        self.codestream.extend_from_slice(buf);
                        buf.clear();
                        return Ok(());
                    }
                }
                DetectState::InAuxBox {
                    data,
                    bytes_left: None,
                    ..
                } => {
                    data.extend_from_slice(buf);
                    buf.clear();
                    return Ok(());
                }
                DetectState::InAuxBox {
                    header,
                    data,
                    bytes_left: Some(bytes_left),
                } => {
                    if *bytes_left <= buf.len() {
                        data.extend(buf.drain(..*bytes_left));
                        self.aux_boxes
                            .push((header.box_type(), std::mem::take(data)));
                        *state = DetectState::WaitingBoxHeader;
                    } else {
                        *bytes_left -= buf.len();
                        data.extend_from_slice(buf);
                        buf.clear();
                        return Ok(());
                    }
                }
                DetectState::Done(_) => return Ok(()),
            }
        }
    }

    pub fn take_bytes(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.codestream)
    }

    pub fn finish(&mut self) {
        if let DetectState::InAuxBox { header, data, .. } = &mut self.state {
            self.aux_boxes
                .push((header.box_type(), std::mem::take(data)));
        }
        self.state = DetectState::Done(self.kind());
    }
}
