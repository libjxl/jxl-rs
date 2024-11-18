// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    bit_reader::BitReader,
    error::Result,
    headers::{
        encodings::UnconditionalCoder,
        frame_header::{FrameHeader, Toc, TocNonserialized},
        FileHeader,
    },
};

pub struct Frame {
    header: FrameHeader,
    toc: Toc,
}

impl Frame {
    pub fn new(br: &mut BitReader, file_header: &FileHeader) -> Result<Self> {
        let frame_header =
            FrameHeader::read_unconditional(&(), br, &file_header.frame_header_nonserialized())
                .unwrap();
        let num_toc_entries = frame_header.num_toc_entries();
        let toc = Toc::read_unconditional(
            &(),
            br,
            &TocNonserialized {
                num_entries: num_toc_entries,
            },
        )
        .unwrap();
        br.jump_to_byte_boundary()?;
        Ok(Self {
            header: frame_header,
            toc,
        })
    }

    pub fn header(&self) -> &FrameHeader {
        &self.header
    }

    pub fn total_bytes_in_toc(&self) -> usize {
        self.toc.entries.iter().map(|x| *x as usize).sum()
    }

    pub fn is_last(&self) -> bool {
        self.header.is_last
    }

    /// Given a bit reader pointing at the end of the TOC, returns a vector of `BitReader`s, each
    /// of which reads a specific section.
    pub fn sections<'a>(&self, br: &'a mut BitReader) -> Result<Vec<BitReader<'a>>> {
        if self.toc.permuted {
            self.toc
                .permutation
                .iter()
                .map(|x| self.toc.entries[*x as usize] as usize)
                .scan(br, |br, count| Some(br.split_at(count)))
                .collect()
        } else {
            self.toc
                .entries
                .iter()
                .scan(br, |br, count| Some(br.split_at(*count as usize)))
                .collect()
        }
    }
}
