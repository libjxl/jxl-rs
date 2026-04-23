// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::io::IoSliceMut;

use crate::container::frame_index::FrameIndexBox;
use crate::error::{Error, Result};

use crate::api::{
    JxlBitstreamInput, JxlSignatureType, check_signature_internal, inner::process::SmallBuffer,
};

#[derive(Clone, Debug)]
pub enum ParseState {
    SignatureNeeded,
    BoxNeeded,
    CodestreamBox(u64),
    SkippableBox(u64),
    /// Buffering a jxli box: (remaining bytes, accumulated content).
    BufferingFrameIndex(u64, Vec<u8>),
    /// Read first 8 bytes of `ftyp` for brand+version, then skip `skip_rest` payload bytes.
    FtypHead {
        head: [u8; 8],
        got: u8,
        skip_rest: u64,
    },
    BufferingOooJxlp {
        remaining: u64,
        buf: Vec<u8>,
        idx: u32,
        last: bool,
    },
    /// After the last codestream box, no more container bytes: no further codestream in file.
    Exhausted,
}

#[derive(Debug)]
enum CodestreamBoxType {
    None,
    Jxlc,
    Jxlp(u32),
    LastJxlp,
}

pub(super) struct BoxParser {
    pub(super) box_buffer: SmallBuffer,
    pub(super) state: ParseState,
    box_type: CodestreamBoxType,
    /// Parsed frame index box, if present in the file.
    pub(super) frame_index: Option<FrameIndexBox>,
    /// Total file bytes consumed from the underlying input.
    pub(super) total_file_consumed: u64,
    skip_jxlp_checks: bool,
    /// From `ftyp`: `0` = `jxlp` boxes must be in order; `1` = out-of-order `jxlp` allowed.
    jxl_file_format_version: u32,
    jxlp_ooo_buffer: Vec<(u32, Vec<u8>, bool)>,
    ftyp_seen: bool,
}

impl BoxParser {
    pub(super) fn new() -> Self {
        BoxParser {
            box_buffer: SmallBuffer::new(128),
            state: ParseState::SignatureNeeded,
            box_type: CodestreamBoxType::None,
            frame_index: None,
            total_file_consumed: 0,
            skip_jxlp_checks: false,
            jxl_file_format_version: 0,
            jxlp_ooo_buffer: Vec::new(),
            ftyp_seen: false,
        }
    }

    fn next_expected_jxlp_index(&self) -> Option<u32> {
        match self.box_type {
            CodestreamBoxType::None => Some(0),
            CodestreamBoxType::Jxlp(i) => Some(i + 1),
            CodestreamBoxType::LastJxlp | CodestreamBoxType::Jxlc => None,
        }
    }

    /// If the next expected `jxlp` index is already buffered (OOO), prepend it as the next codestream.
    fn try_inject_next_buffered_jxlp(&mut self) {
        if self.skip_jxlp_checks || self.jxl_file_format_version < 1 {
            return;
        }
        let Some(next) = self.next_expected_jxlp_index() else {
            return;
        };
        if let Some(pos) = self
            .jxlp_ooo_buffer
            .iter()
            .position(|(idx, _, _)| *idx == next)
        {
            let (_, payload, is_last) = self.jxlp_ooo_buffer.swap_remove(pos);
            let len = payload.len() as u64;
            self.box_buffer.inject_bytes_front(payload);
            self.box_type = if is_last {
                CodestreamBoxType::LastJxlp
            } else {
                CodestreamBoxType::Jxlp(next)
            };
            self.state = ParseState::CodestreamBox(len);
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
                ParseState::Exhausted => {
                    return Ok(0);
                }
                ParseState::SignatureNeeded => {
                    let read = self.box_buffer.refill(|b| input.read(b), None)?;
                    self.total_file_consumed += read as u64;
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
                        let skipped = input.skip(num)?;
                        self.total_file_consumed += skipped as u64;
                        skipped
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
                ParseState::BufferingFrameIndex(mut remaining, mut buf) => {
                    let num = remaining.min(usize::MAX as u64) as usize;
                    if !self.box_buffer.is_empty() {
                        let take = num.min(self.box_buffer.len());
                        buf.extend_from_slice(&self.box_buffer[..take]);
                        self.box_buffer.consume(take);
                        remaining -= take as u64;
                    } else {
                        let old_len = buf.len();
                        buf.resize(old_len + num, 0);
                        let read = input.read(&mut [IoSliceMut::new(&mut buf[old_len..])])?;
                        self.total_file_consumed += read as u64;
                        if read == 0 {
                            return Err(Error::OutOfBounds(num));
                        }
                        buf.truncate(old_len + read);
                        remaining -= read as u64;
                    }
                    if remaining == 0 {
                        // Parse the buffered frame index box.
                        self.frame_index = Some(FrameIndexBox::parse(&buf)?);
                        self.state = ParseState::BoxNeeded;
                    } else {
                        self.state = ParseState::BufferingFrameIndex(remaining, buf);
                    }
                }
                ParseState::FtypHead {
                    mut head,
                    mut got,
                    skip_rest,
                } => {
                    while got < 8 {
                        if !self.box_buffer.is_empty() {
                            head[got as usize] = self.box_buffer[0];
                            self.box_buffer.consume(1);
                            got += 1;
                        } else {
                            let mut b = [0u8; 1];
                            let n = input.read(&mut [IoSliceMut::new(&mut b)])?;
                            self.total_file_consumed += n as u64;
                            if n == 0 {
                                // Persist partial `ftyp` brand read so 1-byte `chunk_input` can
                                // resume across `process()` calls (same pattern as `BoxNeeded`).
                                self.state = ParseState::FtypHead {
                                    head,
                                    got,
                                    skip_rest,
                                };
                                return Err(Error::OutOfBounds(1));
                            }
                            head[got as usize] = b[0];
                            got += 1;
                        }
                    }
                    if &head[0..4] != b"jxl " {
                        return Err(Error::InvalidBox);
                    }
                    let ver = u32::from_be_bytes(head[4..8].try_into().unwrap());
                    if ver > 1 {
                        return Err(Error::InvalidBox);
                    }
                    self.jxl_file_format_version = ver;
                    self.ftyp_seen = true;
                    self.state = if skip_rest == 0 {
                        ParseState::BoxNeeded
                    } else {
                        ParseState::SkippableBox(skip_rest)
                    };
                }
                ParseState::BufferingOooJxlp {
                    mut remaining,
                    mut buf,
                    idx,
                    last,
                } => {
                    let num = remaining.min(usize::MAX as u64) as usize;
                    if !self.box_buffer.is_empty() {
                        let take = num.min(self.box_buffer.len());
                        buf.extend_from_slice(&self.box_buffer[..take]);
                        self.box_buffer.consume(take);
                        remaining -= take as u64;
                    } else {
                        let old_len = buf.len();
                        buf.resize(old_len + num, 0);
                        let read = input.read(&mut [IoSliceMut::new(&mut buf[old_len..])])?;
                        self.total_file_consumed += read as u64;
                        if read == 0 {
                            return Err(Error::OutOfBounds(num));
                        }
                        buf.truncate(old_len + read);
                        remaining -= read as u64;
                    }
                    if remaining == 0 {
                        self.jxlp_ooo_buffer.push((idx, buf, last));
                        self.state = ParseState::BoxNeeded;
                    } else {
                        self.state = ParseState::BufferingOooJxlp {
                            remaining,
                            buf,
                            idx,
                            last,
                        };
                    }
                }
                ParseState::BoxNeeded => {
                    let read = self.box_buffer.refill(|b| input.read(b), None)?;
                    self.total_file_consumed += read as u64;
                    if self.box_buffer.is_empty()
                        && read == 0
                        && self.ftyp_seen
                        && self.jxlp_ooo_buffer.is_empty()
                        && !self.skip_jxlp_checks
                        && matches!(
                            self.box_type,
                            CodestreamBoxType::Jxlc | CodestreamBoxType::LastJxlp
                        )
                    {
                        self.state = ParseState::Exhausted;
                        return Ok(0);
                    }
                    let min_len = match &self.box_buffer[..] {
                        [0, 0, 0, 1, ..] => 16,
                        _ => 8,
                    };
                    // Need the full fixed header (`min_len`); `== min_len` is enough (e.g. `ftyp`
                    // with `extra_len` 0). Requiring `>` min_len broke 1-byte streaming: 8 bytes
                    // in buffer is exactly the small-box size+type, not "one short".
                    if self.box_buffer.len() < min_len {
                        return Err(Error::OutOfBounds(min_len - self.box_buffer.len()));
                    }
                    let ty: [_; 4] = self.box_buffer[4..8].try_into().unwrap();
                    let extra_len = if &ty == b"jxlp" { 4 } else { 0 };
                    if self.box_buffer.len() < min_len + extra_len {
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
                    let content_len = if box_len == 0 {
                        u64::MAX
                    } else {
                        if box_len < (min_len + extra_len) as u64 {
                            return Err(Error::InvalidBox);
                        }
                        box_len - min_len as u64 - extra_len as u64
                    };
                    if !self.ftyp_seen && &ty != b"ftyp" {
                        return Err(Error::InvalidBox);
                    }
                    if self.ftyp_seen && &ty == b"ftyp" {
                        return Err(Error::InvalidBox);
                    }
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
                            let last = index & 0x80000000 != 0;
                            let idx = index & 0x7fffffff;
                            if !self.skip_jxlp_checks {
                                let wanted_idx = match self.box_type {
                                    CodestreamBoxType::Jxlc | CodestreamBoxType::LastJxlp => {
                                        return Err(Error::InvalidBox);
                                    }
                                    CodestreamBoxType::None => 0,
                                    CodestreamBoxType::Jxlp(i) => i + 1,
                                };
                                if idx < wanted_idx {
                                    return Err(Error::InvalidBox);
                                }
                                if idx > wanted_idx {
                                    if self.jxl_file_format_version < 1 {
                                        return Err(Error::InvalidBox);
                                    }
                                    if self.jxlp_ooo_buffer.iter().any(|(i, _, _)| *i == idx) {
                                        return Err(Error::InvalidBox);
                                    }
                                    if content_len == u64::MAX {
                                        return Err(Error::InvalidBox);
                                    }
                                    self.state = ParseState::BufferingOooJxlp {
                                        remaining: content_len,
                                        buf: Vec::new(),
                                        idx,
                                        last,
                                    };
                                    self.box_buffer.consume(min_len + extra_len);
                                    continue;
                                }
                            }
                            self.box_type = if last {
                                CodestreamBoxType::LastJxlp
                            } else {
                                CodestreamBoxType::Jxlp(idx)
                            };
                            self.state = ParseState::CodestreamBox(content_len);
                        }
                        b"ftyp" => {
                            if content_len < 8 {
                                return Err(Error::InvalidBox);
                            }
                            self.state = ParseState::FtypHead {
                                head: [0; 8],
                                got: 0,
                                skip_rest: content_len - 8,
                            };
                        }
                        b"jxli" => {
                            if content_len == u64::MAX || content_len > 16 * 1024 * 1024 {
                                self.state = ParseState::SkippableBox(content_len);
                            } else {
                                self.state = ParseState::BufferingFrameIndex(
                                    content_len,
                                    Vec::with_capacity(content_len as usize),
                                );
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

    /// Accounts file bytes consumed directly by codestream parser reads/skips.
    pub(super) fn mark_file_consumed(&mut self, amount: usize) {
        self.total_file_consumed += amount as u64;
    }

    /// Resets the box parser for seeking to a specific codestream position.
    ///
    /// Sets the parser to `CodestreamBox(remaining)` state with cleared
    /// buffers.  The caller must provide raw input starting from the file
    /// position that corresponds to the target codestream offset.
    ///
    /// `remaining` is the number of codestream bytes left in the current
    /// box from the target file position.  For bare-codestream files this
    /// is `u64::MAX`.
    pub(super) fn reset_for_codestream_seek(&mut self, remaining: u64) {
        self.box_buffer = SmallBuffer::new(128);
        self.state = ParseState::CodestreamBox(remaining);
        self.skip_jxlp_checks = true;
        self.jxlp_ooo_buffer.clear();
        // Keep frame_index unchanged.
    }

    pub(super) fn consume_codestream(&mut self, amount: u64) {
        if let ParseState::CodestreamBox(cb) = &mut self.state {
            *cb = cb.checked_sub(amount).unwrap();
            if *cb == 0 {
                self.state = ParseState::BoxNeeded;
                self.try_inject_next_buffered_jxlp();
            }
        } else if amount != 0 {
            unreachable!()
        }
    }
}
