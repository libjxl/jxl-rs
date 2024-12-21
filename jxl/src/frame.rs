// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    bit_reader::BitReader,
    error::Result,
    features::{noise::Noise, patches::PatchesDictionary, spline::Splines},
    headers::{
        color_encoding::ColorSpace,
        encodings::UnconditionalCoder,
        extra_channels::ExtraChannelInfo,
        frame_header::{Encoding, FrameHeader, Toc, TocNonserialized},
        FileHeader,
    },
    image::Image,
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
    patches: Option<PatchesDictionary>,
    splines: Option<Splines>,
    noise: Option<Noise>,
    lf_quant: LfQuantFactors,
    // TODO(veluca93), VarDCT: HF quant matrices
    // TODO(veluca93), VarDCT: block context map
    // TODO(veluca93), VarDCT: LF color correlation
    tree: Option<Tree>,
    modular_global: FullModularImage,
}

#[derive(Debug)]
pub struct ReferenceFrame {
    pub frame: Vec<Image<f32>>,
    pub saved_before_color_transform: bool,
}

impl ReferenceFrame {
    // TODO(firsching): make this #[cfg(test)]
    fn blank(
        width: usize,
        height: usize,
        num_channels: usize,
        saved_before_color_transform: bool,
    ) -> Result<Self> {
        let frame = (0..num_channels)
            .map(|_| Image::new_constant((width, height), 0.0))
            .collect::<Result<_>>()?;
        Ok(Self {
            frame,
            saved_before_color_transform,
        })
    }
}

#[derive(Debug)]
pub struct DecoderState {
    file_header: FileHeader,
    reference_frames: [Option<ReferenceFrame>; Self::MAX_STORED_FRAMES],
}

impl DecoderState {
    pub const MAX_STORED_FRAMES: usize = 4;

    pub fn new(file_header: FileHeader) -> Self {
        Self {
            file_header,
            reference_frames: [None, None, None, None],
        }
    }

    pub fn extra_channel_info(&self) -> &Vec<ExtraChannelInfo> {
        &self.file_header.image_metadata.extra_channel_info
    }

    pub fn reference_frame(&self, i: usize) -> Option<&ReferenceFrame> {
        assert!(i < Self::MAX_STORED_FRAMES);
        self.reference_frames[i].as_ref()
    }
}

pub struct Frame {
    header: FrameHeader,
    toc: Toc,
    modular_color_channels: usize,
    lf_global: Option<LfGlobalState>,
    decoder_state: DecoderState,
}

impl Frame {
    pub fn new(br: &mut BitReader, decoder_state: DecoderState) -> Result<Self> {
        let frame_header = FrameHeader::read_unconditional(
            &(),
            br,
            &decoder_state.file_header.frame_header_nonserialized(),
        )
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
        } else if decoder_state
            .file_header
            .image_metadata
            .color_encoding
            .color_space
            == ColorSpace::Gray
        {
            1
        } else {
            3
        };
        Ok(Self {
            header: frame_header,
            modular_color_channels,
            toc,
            lf_global: None,
            decoder_state,
        })
    }

    pub fn header(&self) -> &FrameHeader {
        &self.header
    }

    pub fn total_bytes_in_toc(&self) -> usize {
        self.toc.entries.iter().map(|x| *x as usize).sum()
    }

    /// Given a bit reader pointing at the end of the TOC, returns a vector of `BitReader`s, each
    /// of which reads a specific sectionalpha_channel.
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

        let patches = if self.header.has_patches() {
            info!("decoding patches");
            Some(PatchesDictionary::read(
                br,
                self.header.width as usize,
                self.header.height as usize,
                &self.decoder_state,
            )?)
        } else {
            None
        };

        let splines = if self.header.has_splines() {
            info!("decoding splines");
            Some(Splines::read(br, self.header.width * self.header.height)?)
        } else {
            None
        };

        let noise = if self.header.has_noise() {
            info!("decoding noise");
            Some(Noise::read(br)?)
        } else {
            None
        };

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
                    * (self.modular_color_channels
                        + self.decoder_state.extra_channel_info().len())
                    / 16)
                .min(1 << 22);
            Some(Tree::read(br, size_limit)?)
        } else {
            None
        };

        let modular_global = FullModularImage::read(
            &self.header,
            self.modular_color_channels,
            self.decoder_state.extra_channel_info(),
            &tree,
            br,
        )?;

        self.lf_global = Some(LfGlobalState {
            patches,
            splines,
            noise,
            lf_quant,
            tree,
            modular_global,
        });

        Ok(())
    }

    pub fn finalize(mut self) -> Result<Option<DecoderState>> {
        if self.header.can_be_referenced {
            // TODO(firsching): actually decode the image here, instead of setting it
            // to a blank image here, once we can decode images.
            self.decoder_state.reference_frames[self.header.save_as_reference as usize] =
                Some(ReferenceFrame::blank(
                    self.header.width as usize,
                    self.header.height as usize,
                    // Set num_channels to "1" here unconditionally for now, since we will use this
                    // channel for the size checks
                    1,
                    self.header.save_before_ct,
                )?);
        }
        Ok(if self.header.is_last {
            None
        } else {
            Some(self.decoder_state)
        })
    }
}

#[cfg(test)]
mod test {
    use std::{panic, path::Path};

    use jxl_macros::for_each_test_file;

    use crate::{
        bit_reader::BitReader,
        container::ContainerParser,
        error::Error,
        features::spline::Point,
        headers::{FileHeader, JxlHeader},
        util::test::assert_almost_eq,
    };

    use super::{DecoderState, Frame, Section};

    fn read_frames(
        image: &[u8],
        mut callback: impl FnMut(Frame) -> Result<Option<DecoderState>, Error>,
    ) -> Result<(), Error> {
        let codestream = ContainerParser::collect_codestream(image).unwrap();
        let mut br = BitReader::new(&codestream);
        let file_header = FileHeader::read(&mut br).unwrap();
        let mut decoder_state = DecoderState::new(file_header);
        loop {
            let mut frame = Frame::new(&mut br, decoder_state)?;
            let mut sections = frame.sections(&mut br)?;
            frame.decode_lf_global(&mut sections[frame.get_section_idx(Section::LfGlobal)])?;

            // Call the callback with the frame
            if let Some(state) = callback(frame)? {
                decoder_state = state;
            } else {
                break;
            }
        }
        Ok(())
    }

    fn read_frames_from_path(path: &Path) -> Result<(), Error> {
        let data = std::fs::read(path).unwrap();
        let result = panic::catch_unwind(|| read_frames(data.as_slice(), |frame| frame.finalize()));

        match result {
            Ok(Ok(_frame)) => {}
            Ok(Err(e)) => {
                return Err(e);
            }
            Err(e) => {
                // A panic occurred
                if let Some(msg) = e.downcast_ref::<&str>() {
                    if msg.contains("VarDCT not implemented") {
                        println!("Skipping {}: VarDCT not implemented", path.display());
                    } else {
                        panic::resume_unwind(e);
                    }
                } else {
                    panic::resume_unwind(e);
                }
            }
        }

        Ok(())
    }

    for_each_test_file!(read_frames_from_path);

    #[test]
    fn splines() -> Result<(), Error> {
        let mut frames = Vec::new();
        read_frames(include_bytes!("../resources/test/splines.jxl"), |frame| {
            frames.push(frame);
            Ok(None)
        })?;
        assert_eq!(frames.len(), 1);
        let frame = &frames[0];
        let lf_global = frame.lf_global.as_ref().unwrap();
        let splines = lf_global.splines.as_ref().unwrap();
        assert_eq!(splines.quantization_adjustment, 0);
        let expected_starting_points = [Point { x: 9.0, y: 54.0 }].to_vec();
        assert_eq!(splines.starting_points, expected_starting_points);
        assert_eq!(splines.splines.len(), 1);
        let spline = splines.splines[0].clone();

        let expected_control_points = [
            (109, 105),
            (-130, -261),
            (-66, 193),
            (227, -52),
            (-170, 290),
        ]
        .to_vec();
        assert_eq!(spline.control_points.clone(), expected_control_points);

        const EXPECTED_COLOR_DCT: [[i32; 32]; 3] = [
            {
                let mut row = [0; 32];
                row[0] = 168;
                row[1] = 119;
                row
            },
            {
                let mut row = [0; 32];
                row[0] = 9;
                row[2] = 7;
                row
            },
            {
                let mut row = [0; 32];
                row[0] = -10;
                row[1] = 7;
                row
            },
        ];
        assert_eq!(spline.color_dct, EXPECTED_COLOR_DCT);

        const EXPECTED_SIGMA_DCT: [i32; 32] = {
            let mut dct = [0; 32];
            dct[0] = 4;
            dct[7] = 2;
            dct
        };
        assert_eq!(spline.sigma_dct, EXPECTED_SIGMA_DCT);
        Ok(())
    }

    #[test]
    fn noise() -> Result<(), Error> {
        let mut frames = Vec::new();
        read_frames(include_bytes!("../resources/test/8x8_noise.jxl"), |frame| {
            frames.push(frame);
            Ok(None)
        })?;

        assert_eq!(frames.len(), 1);
        let frame = &frames[0];
        let lf_global = frame.lf_global.as_ref().unwrap();
        let noise = lf_global.noise.as_ref().unwrap();
        let want_noise = [
            0.000000, 0.000977, 0.002930, 0.003906, 0.005859, 0.006836, 0.008789, 0.010742,
        ];
        for (index, noise_param) in want_noise.iter().enumerate() {
            assert_almost_eq!(noise.lut[index], *noise_param, 1e-6);
        }
        Ok(())
    }

    #[test]
    fn patches() -> Result<(), Error> {
        let mut frames = Vec::new();
        read_frames(
            include_bytes!("../resources/test/grayscale_patches_modular.jxl"),
            |frame| {
                frames.push(frame);
                Ok(None)
            },
        )?;
        // TODO(firsching) add test for patches
        Ok(())
    }
}
