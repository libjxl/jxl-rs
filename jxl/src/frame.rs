// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    bit_reader::BitReader,
    error::Result,
    features::spline::Splines,
    headers::{
        color_encoding::ColorSpace,
        encodings::UnconditionalCoder,
        extra_channels::ExtraChannelInfo,
        frame_header::{Encoding, FrameHeader, Toc, TocNonserialized},
        FileHeader,
    },
    util::tracing_wrappers::*,
};
use modular::{FullModularImage, Tree};
use quantizer::LfQuantFactors;

pub mod modular;
mod quantizer;

#[derive(Debug, PartialEq, Eq)]
pub enum Section {
    LfGlobal,
    Lf(usize),
    HfGlobal,
    Hf(usize, usize), // group, pass
}

#[allow(dead_code)]
pub struct LfGlobalState {
    // TODO(veluca93): patches
    // TODO(veluca93): splines
    splines: Option<Splines>,
    // TODO(veluca93): noise
    lf_quant: LfQuantFactors,
    // TODO(veluca93), VarDCT: HF quant matrices
    // TODO(veluca93), VarDCT: block context map
    // TODO(veluca93), VarDCT: LF color correlation
    tree: Option<Tree>,
    modular_global: FullModularImage,
}

pub struct Frame {
    header: FrameHeader,
    toc: Toc,
    modular_color_channels: usize,
    extra_channel_info: Vec<ExtraChannelInfo>,
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
        let modular_color_channels = if frame_header.encoding == Encoding::VarDCT {
            0
        } else if file_header.image_metadata.color_encoding.color_space == ColorSpace::Gray {
            1
        } else {
            3
        };
        Ok(Self {
            header: frame_header,
            modular_color_channels,
            extra_channel_info: file_header.image_metadata.extra_channel_info.clone(),
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
                Section::HfGlobal => self.header.num_lf_groups() + 1,
                Section::Hf(group, pass) => {
                    2 + self.header.num_lf_groups() + self.header.num_groups() * pass + group
                }
            }
        }
    }

    #[instrument(skip_all)]
    pub fn decode_lf_global(&mut self, br: &mut BitReader) -> Result<()> {
        assert!(self.lf_global.is_none());
        trace!(pos = br.total_bits_read());

        if self.header.has_patches() {
            info!("decoding patches");
            todo!("patches not implemented");
        }
        let mut splines = if self.header.has_splines() {
            Some(Splines::default())
        } else {
            None
        };
        if let Some(ref mut splines) = splines {
            splines.decode(br, self.header.width * self.header.height)?;
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

        let tree = if br.read(1)? == 1 {
            let size_limit = (1024
                + self.header.width as usize
                    * self.header.height as usize
                    * (self.modular_color_channels + self.extra_channel_info.len())
                    / 16)
                .min(1 << 22);
            Some(Tree::read(br, size_limit)?)
        } else {
            None
        };

        let modular_global = FullModularImage::read(
            &self.header,
            self.modular_color_channels,
            &self.extra_channel_info,
            &tree,
            br,
        )?;

        self.lf_global = Some(LfGlobalState {
            splines,
            lf_quant,
            tree,
            modular_global,
        });

        Ok(())
    }
}
