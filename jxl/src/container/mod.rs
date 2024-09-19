// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Originally written for jxl-oxide.

pub mod box_header;
pub mod parse;

use box_header::*;
use parse::*;
pub use parse::ParseEvent;

/// Container format parser.
#[derive(Debug, Default)]
pub struct ContainerParser {
    state: DetectState,
    jxlp_index_state: JxlpIndexState,
    previous_consumed_bytes: usize,
}

#[derive(Debug, Default)]
enum DetectState {
    #[default]
    WaitingSignature,
    WaitingBoxHeader,
    WaitingJxlpIndex(ContainerBoxHeader),
    InAuxBox {
        #[allow(unused)]
        header: ContainerBoxHeader,
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

#[derive(Debug, Copy, Clone, Eq, PartialEq, Default)]
enum JxlpIndexState {
    #[default]
    Initial,
    SingleJxlc,
    Jxlp(u32),
    JxlpFinished,
}

impl ContainerParser {
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

    /// Parses input buffer and generates parser events.
    ///
    /// The parser might not fully consume the buffer. Use [`previous_consumed_bytes`] to get how
    /// many bytes are consumed. Bytes not consumed by the parser should be processed again.
    ///
    /// [`previous_consumed_bytes`]: ContainerDetectingReader::previous_consumed_bytes
    pub fn process_bytes<'inner, 'buf>(
        &'inner mut self,
        input: &'buf [u8],
    ) -> ParseEvents<'inner, 'buf> {
        ParseEvents::new(self, input)
    }

    /// Get how much bytes are consumed by the previous call of [`process_bytes`].
    ///
    /// Bytes not consumed by the parser should be fed into the parser again.
    ///
    /// [`process_bytes`]: ContainerDetectingReader::process_bytes
    pub fn previous_consumed_bytes(&self) -> usize {
        self.previous_consumed_bytes
    }

    pub fn finish(&mut self) {
        // FIXME: validate state
        self.state = DetectState::Done(self.kind());
    }
}

#[cfg(test)]
impl ContainerParser {
    pub(crate) fn collect_codestream(input: &[u8]) -> crate::error::Result<Vec<u8>> {
        let mut parser = Self::new();
        let mut codestream = Vec::new();
        for event in parser.process_bytes(input) {
            match event? {
                ParseEvent::BitstreamKind(_) => {}
                ParseEvent::Codestream(buf) => {
                    codestream.extend_from_slice(buf);
                }
            }
        }
        Ok(codestream)
    }
}
