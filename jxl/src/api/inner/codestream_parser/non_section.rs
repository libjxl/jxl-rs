// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::io::IoSliceMut;

use crate::{
    api::{
        Endianness, JxlBasicInfo, JxlColorEncoding, JxlColorProfile, JxlColorType, JxlDataFormat,
        JxlDecoderOptions, JxlPixelFormat, JxlPrimaries, JxlTransferFunction, JxlWhitePoint,
        inner::codestream_parser::SectionState,
    },
    bit_reader::BitReader,
    error::{Error, Result},
    frame::{DecoderState, Frame, Section},
    headers::{
        FileHeader, JxlHeader,
        color_encoding::{ColorSpace, RenderingIntent},
        encodings::UnconditionalCoder,
        frame_header::{FrameHeader, Toc, TocNonserialized},
    },
    icc::IncrementalIccReader,
};

use super::{CodestreamParser, SectionBuffer};

impl CodestreamParser {
    pub(super) fn process_non_section(&mut self, decode_options: &JxlDecoderOptions) -> Result<()> {
        if self.decoder_state.is_none() && self.file_header.is_none() {
            // We don't have a file header yet. Try parsing that.
            // TODO(veluca): make this incremental, as a file header might be multiple megabytes.
            let mut br = BitReader::new(&self.non_section_buf);
            br.skip_bits(self.non_section_bit_offset as usize)?;
            let file_header = FileHeader::read(&mut br)?;
            self.basic_info = Some(JxlBasicInfo {
                size: (
                    file_header.size.xsize() as usize,
                    file_header.size.ysize() as usize,
                ),
                bit_depth: file_header.image_metadata.bit_depth,
                orientation: file_header.image_metadata.orientation,
                extra_channels: file_header.image_metadata.extra_channel_info.clone(),
            });
            self.file_header = Some(file_header);
            let bits = br.total_bits_read();
            self.non_section_buf.consume(bits / 8);
            self.non_section_bit_offset = (bits % 8) as u8;
        }

        if self.decoder_state.is_none() && self.embedded_color_profile.is_none() {
            let file_header = self.file_header.as_ref().unwrap();
            // Parse (or extract from file header) the ICC profile.
            let mut br = BitReader::new(&self.non_section_buf);
            br.skip_bits(self.non_section_bit_offset as usize)?;
            let embedded_color_profile = if file_header.image_metadata.color_encoding.want_icc {
                if self.icc_parser.is_none() {
                    self.icc_parser = Some(IncrementalIccReader::new(&mut br)?);
                }
                let icc_parser = self.icc_parser.as_mut().unwrap();
                let mut bits = br.total_bits_read();
                for _ in 0..icc_parser.remaining() {
                    match icc_parser.read_one(&mut br) {
                        Ok(()) => bits = br.total_bits_read(),
                        Err(Error::OutOfBounds(c)) => {
                            self.non_section_buf.consume(bits / 8);
                            self.non_section_bit_offset = (bits % 8) as u8;
                            // Estimate >= one bit per remaining character to read.
                            return Err(Error::OutOfBounds(c + icc_parser.remaining() / 8));
                        }
                        Err(e) => return Err(e),
                    }
                }
                self.non_section_buf.consume(bits / 8);
                self.non_section_bit_offset = (bits % 8) as u8;
                JxlColorProfile::Icc(self.icc_parser.take().unwrap().finalize()?)
            } else {
                JxlColorProfile::Simple(JxlColorEncoding::from_internal(
                    &file_header.image_metadata.color_encoding,
                )?)
            };
            let output_color_profile = if file_header.image_metadata.xyb_encoded {
                if decode_options.xyb_output_linear {
                    JxlColorProfile::Simple(
                        if file_header.image_metadata.color_encoding.color_space == ColorSpace::Gray
                        {
                            JxlColorEncoding::GrayscaleColorSpace {
                                white_point: JxlWhitePoint::D65,
                                transfer_function: JxlTransferFunction::Linear,
                                rendering_intent: RenderingIntent::Relative,
                            }
                        } else {
                            JxlColorEncoding::RgbColorSpace {
                                white_point: JxlWhitePoint::D65,
                                primaries: JxlPrimaries::SRGB,
                                transfer_function: JxlTransferFunction::Linear,
                                rendering_intent: RenderingIntent::Relative,
                            }
                        },
                    )
                } else {
                    JxlColorProfile::Simple(JxlColorEncoding::srgb(
                        file_header.image_metadata.color_encoding.color_space == ColorSpace::Gray,
                    ))
                }
            } else {
                embedded_color_profile.clone()
            };
            self.embedded_color_profile = Some(embedded_color_profile);
            self.output_color_profile = Some(output_color_profile);
            self.pixel_format = Some(JxlPixelFormat {
                color_type: if file_header.image_metadata.color_encoding.color_space
                    == ColorSpace::Gray
                {
                    JxlColorType::Grayscale
                } else {
                    JxlColorType::Rgb
                },
                color_data_format: Some(JxlDataFormat::F32 {
                    endianness: Endianness::native(),
                }),
                extra_channel_format: vec![
                    Some(JxlDataFormat::F32 {
                        endianness: Endianness::native()
                    });
                    file_header.image_metadata.extra_channel_info.len()
                ],
            });

            let mut br = BitReader::new(&self.non_section_buf);
            br.skip_bits(self.non_section_bit_offset as usize)?;
            br.jump_to_byte_boundary()?;
            self.non_section_buf.consume(br.total_bits_read() / 8);

            // We now have image information.
            // TODO(veluca): generate BasicInfo.
            let mut decoder_state = DecoderState::new(self.file_header.take().unwrap());
            decoder_state.xyb_output_linear = decode_options.xyb_output_linear;
            decoder_state.render_spotcolors = decode_options.render_spot_colors;
            self.decoder_state = Some(decoder_state);
            // Reset bit offset to 0 since we've consumed everything up to a byte boundary
            self.non_section_bit_offset = 0;
            return Ok(());
        }

        let decoder_state = self.decoder_state.as_mut().unwrap();

        if self.frame_header.is_none() {
            // We don't have a frame header yet. Try parsing that.
            // TODO(veluca): do we need to make this incremental?
            let mut br = BitReader::new(&self.non_section_buf);
            br.skip_bits(self.non_section_bit_offset as usize)?;
            let mut frame_header = FrameHeader::read_unconditional(
                &(),
                &mut br,
                &decoder_state.file_header.frame_header_nonserialized(),
            )?;
            frame_header.postprocess(&decoder_state.file_header.frame_header_nonserialized());
            self.frame_header = Some(frame_header);
            let bits = br.total_bits_read();
            self.non_section_buf.consume(bits / 8);
            self.non_section_bit_offset = (bits % 8) as u8;
        }

        let mut br = BitReader::new(&self.non_section_buf);
        br.skip_bits(self.non_section_bit_offset as usize)?;
        let num_toc_entries = self.frame_header.as_ref().unwrap().num_toc_entries();
        let toc = Toc::read_unconditional(
            &(),
            &mut br,
            &TocNonserialized {
                num_entries: num_toc_entries as u32,
            },
        )?;
        br.jump_to_byte_boundary()?;
        let frame = Frame::from_header_and_toc(
            self.frame_header.take().unwrap(),
            toc,
            self.decoder_state.take().unwrap(),
        )?;
        let bits = br.total_bits_read();
        self.non_section_buf.consume(bits / 8);
        self.non_section_bit_offset = (bits % 8) as u8;

        let mut sections: Vec<_> = frame
            .toc()
            .entries
            .iter()
            .map(|x| SectionBuffer {
                len: *x as usize,
                data: vec![],
                section: Section::LfGlobal, // will be fixed later
            })
            .collect();

        let order = if frame.toc().permuted {
            frame.toc().permutation.0.clone()
        } else {
            (0..sections.len() as u32).collect()
        };

        if sections.len() > 1 {
            let base_sections = [Section::LfGlobal, Section::HfGlobal];
            let lf_sections = (0..frame.header().num_lf_groups()).map(|x| Section::Lf { group: x });
            let hf_sections = (0..frame.header().passes.num_passes).flat_map(|p| {
                (0..frame.header().num_groups()).map(move |g| Section::Hf {
                    group: g,
                    pass: p as usize,
                })
            });

            for section in base_sections
                .into_iter()
                .chain(lf_sections)
                .chain(hf_sections)
            {
                sections[order[frame.get_section_idx(section)] as usize].section = section;
            }
        }

        self.sections = sections.into_iter().collect();
        self.ready_section_data = 0;

        // Move data from the pre-section buffer into the sections.
        for buf in self.sections.iter_mut() {
            if self.non_section_buf.is_empty() {
                break;
            }
            buf.data = vec![0; buf.len];
            self.ready_section_data += self
                .non_section_buf
                .take(&mut [IoSliceMut::new(&mut buf.data)]);
        }

        self.section_state =
            SectionState::new(frame.header().num_lf_groups(), frame.header().num_groups());
        assert!(self.available_sections.is_empty());

        self.frame = Some(frame);

        Ok(())
    }
}
