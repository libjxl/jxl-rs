// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Originally written for jxl-oxide.

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

struct ConcatSlice<'first, 'second> {
    slices: (&'first [u8], &'second [u8]),
    ptr: usize,
}

impl<'first, 'second> ConcatSlice<'first, 'second> {
    fn new(slice0: &'first [u8], slice1: &'second [u8]) -> Self {
        Self {
            slices: (slice0, slice1),
            ptr: 0,
        }
    }

    fn len(&self) -> usize {
        self.slices.0.len() + self.slices.1.len()
    }

    fn remaining_slices(&self) -> (&'first [u8], &'second [u8]) {
        let (slice0, slice1) = self.slices;
        let total_len = self.len();
        let ptr = self.ptr;
        if ptr >= total_len {
            (&[], &[])
        } else if let Some(second_slice_ptr) = ptr.checked_sub(slice0.len()) {
            (&[], &slice1[second_slice_ptr..])
        } else {
            (&slice0[ptr..], slice1)
        }
    }

    fn advance(&mut self, bytes: usize) {
        self.ptr += bytes;
    }

    fn peek<'out>(&self, out_buf: &'out mut [u8]) -> &'out mut [u8] {
        let (slice0, slice1) = self.remaining_slices();
        let total_len = slice0.len() + slice1.len();

        let out_bytes = out_buf.len().min(total_len);
        let out_buf = &mut out_buf[..out_bytes];

        if out_bytes <= slice0.len() {
            out_buf.copy_from_slice(&slice0[..out_bytes]);
        } else {
            let (out_first, out_second) = out_buf.split_at_mut(slice0.len());
            out_first.copy_from_slice(slice0);
            out_second.copy_from_slice(&slice1[..out_second.len()]);
        }

        out_buf
    }

    fn fill_vec(&mut self, max_bytes: Option<usize>, v: &mut Vec<u8>) -> Result<usize, Error> {
        let (slice0, slice1) = self.remaining_slices();
        let total_len = slice0.len() + slice1.len();

        let out_bytes = max_bytes.unwrap_or(usize::MAX).min(total_len);
        v.try_reserve(out_bytes)?;

        if out_bytes <= slice0.len() {
            v.extend_from_slice(&slice0[..out_bytes]);
        } else {
            let second_slice_len = out_bytes - slice0.len();
            v.extend_from_slice(slice0);
            v.extend_from_slice(&slice1[..second_slice_len]);
        }

        self.advance(out_bytes);
        Ok(out_bytes)
    }
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
        let mut reader = ConcatSlice::new(&self.buf, input);

        loop {
            match state {
                DetectState::WaitingSignature => {
                    let mut signature_buf = [0u8; 12];
                    let buf = reader.peek(&mut signature_buf);
                    if buf.starts_with(&Self::CODESTREAM_SIG) {
                        // tracing::debug!("Codestream signature found");
                        *state = DetectState::InCodestream {
                            kind: BitstreamKind::BareCodestream,
                            bytes_left: None,
                        };
                    } else if buf.starts_with(&Self::CONTAINER_SIG) {
                        // tracing::debug!("Container signature found");
                        *state = DetectState::WaitingBoxHeader;
                        reader.advance(Self::CONTAINER_SIG.len());
                    } else if !Self::CODESTREAM_SIG.starts_with(buf)
                        && !Self::CONTAINER_SIG.starts_with(buf)
                    {
                        // tracing::error!("Invalid signature");
                        *state = DetectState::InCodestream {
                            kind: BitstreamKind::Invalid,
                            bytes_left: None,
                        };
                    } else {
                        break;
                    }
                }
                DetectState::WaitingBoxHeader => match ContainerBoxHeader::parse(&reader)? {
                    HeaderParseResult::Done { header, size } => {
                        reader.advance(size);
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
                    }
                    HeaderParseResult::NeedMoreData => break,
                },
                DetectState::WaitingJxlpIndex(header) => {
                    let mut buf = [0u8; 4];
                    reader.peek(&mut buf);
                    if buf.len() < 4 {
                        break;
                    }

                    let index = u32::from_be_bytes(buf);
                    reader.advance(4);
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
                    reader.fill_vec(None, &mut self.codestream)?;
                    break;
                }
                DetectState::InCodestream {
                    bytes_left: Some(bytes_left),
                    ..
                } => {
                    let bytes_written = reader.fill_vec(Some(*bytes_left), &mut self.codestream)?;
                    *bytes_left -= bytes_written;
                    if *bytes_left == 0 {
                        *state = DetectState::WaitingBoxHeader;
                    } else {
                        break;
                    }
                }
                DetectState::InAuxBox {
                    data,
                    bytes_left: None,
                    ..
                } => {
                    reader.fill_vec(None, data)?;
                    break;
                }
                DetectState::InAuxBox {
                    header,
                    data,
                    bytes_left: Some(bytes_left),
                } => {
                    let bytes_written = reader.fill_vec(Some(*bytes_left), data)?;
                    *bytes_left -= bytes_written;
                    if *bytes_left == 0 {
                        self.aux_boxes
                            .push((header.box_type(), std::mem::take(data)));
                        *state = DetectState::WaitingBoxHeader;
                    } else {
                        break;
                    }
                }
                DetectState::Done(_) => break,
            }
        }

        let (buf_slice, input_slice) = reader.remaining_slices();
        if buf_slice.is_empty() {
            self.buf.clear();
        } else {
            let remaining_buf_from = self.buf.len() - buf_slice.len();
            self.buf.drain(..remaining_buf_from);
        }
        self.buf.try_reserve(input_slice.len())?;
        self.buf.extend_from_slice(input_slice);
        Ok(())
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
