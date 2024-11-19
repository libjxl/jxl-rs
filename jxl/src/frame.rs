// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    bit_reader::BitReader,
    error::Result,
    headers::{
        encodings::UnconditionalCoder,
        frame_header::{Encoding, FrameHeader, Toc, TocNonserialized},
        FileHeader,
    },
    util::tracing_wrappers::*,
};
use quantizer::LfQuantFactors;

mod quantizer;

#[derive(Debug, PartialEq, Eq)]
pub enum Section {
    LfGlobal,
    Lf(usize),
    HfGlobal,
    Hf(usize, usize), // group, pass
}

pub struct LfGlobalState {
    // TODO(veluca93): patches
    // TODO(veluca93): splines
    // TODO(veluca93): noise
    #[allow(dead_code)]
    lf_quant: LfQuantFactors,
    // TODO(veluca93), VarDCT: HF quant matrices
    // TODO(veluca93), VarDCT: block context map
    // TODO(veluca93), VarDCT: LF color correlation
    // TODO(veluca93): Modular data
}

pub struct Frame {
    header: FrameHeader,
    toc: Toc,
    #[allow(dead_code)]
    file_header: FileHeader,
    lf_global: Option<LfGlobalState>,
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
                num_entries: num_toc_entries as u32,
            },
        )
        .unwrap();
        br.jump_to_byte_boundary()?;
        Ok(Self {
            header: frame_header,
            file_header: file_header.clone(),
            toc,
            lf_global: None,
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

    #[instrument(level = "debug", skip(self), ret)]
    pub fn get_section_idx(&self, section: Section) -> usize {
        if self.header.num_toc_entries() == 1 {
            0
        } else {
            match section {
                Section::LfGlobal => 0,
                Section::Lf(a) => 1 + a,
                Section::HfGlobal => self.header.num_dc_groups() + 1,
                Section::Hf(group, pass) => {
                    2 + self.header.num_dc_groups() + self.header.num_groups() * pass + group
                }
            }
        }
    }

    #[instrument(skip_all)]
    pub fn decode_lf_global(&mut self, br: &mut BitReader) -> Result<()> {
        assert!(self.lf_global.is_none());

        if self.header.has_patches() {
            info!("decoding patches");
            todo!("patches not implemented");
        }

        if self.header.has_splines() {
            info!("decoding splines");
            todo!("splines not implemented");
        }

        if self.header.has_noise() {
            info!("decoding noise");
            todo!("noise not implemented");
        }

        let lf_quant = LfQuantFactors::new(br)?;
        debug!(?lf_quant);

        if self.header.encoding == Encoding::VarDCT {
            info!("decoding VarDCT info");
            todo!("VarDCT not implemented");
        }

        self.lf_global = Some(LfGlobalState { lf_quant });

        Ok(())
    }
}
