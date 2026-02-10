// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#[cfg(feature = "jpeg-reconstruction")]
use crate::api::inner::box_parser::BoxParser;
use crate::{
    api::JxlDecoderOptions, api::JxlOutputBuffer, bit_reader::BitReader, error::Result,
    frame::Section,
};

use super::CodestreamParser;

pub(super) struct SectionState {
    lf_global_done: bool,
    remaining_lf: usize,
    hf_global_done: bool,
    completed_passes: Vec<u8>,
}

impl SectionState {
    pub(super) fn new(num_lf_groups: usize, num_groups: usize) -> Self {
        Self {
            lf_global_done: false,
            remaining_lf: num_lf_groups,
            hf_global_done: false,
            completed_passes: vec![0; num_groups],
        }
    }

    /// Returns the number of passes that are fully completed across all groups.
    /// A pass is fully completed when all groups have decoded that pass.
    pub(super) fn num_completed_passes(&self) -> usize {
        self.completed_passes.iter().copied().min().unwrap_or(0) as usize
    }
}

impl CodestreamParser {
    pub(super) fn process_sections(
        &mut self,
        decode_options: &JxlDecoderOptions,
        output_buffers: &mut Option<&mut [JxlOutputBuffer<'_>]>,
        do_flush: bool,
        #[cfg(feature = "jpeg-reconstruction")] box_parser: &mut BoxParser,
    ) -> Result<Option<usize>> {
        let frame = self.frame.as_mut().unwrap();

        // Dequeue ready sections.
        while self
            .sections
            .front()
            .is_some_and(|s| s.len <= self.ready_section_data)
        {
            let s = self.sections.pop_front().unwrap();
            self.ready_section_data -= s.len;

            match s.section {
                Section::LfGlobal => {
                    self.lf_global_section = Some(s);
                }
                Section::HfGlobal => {
                    self.hf_global_section = Some(s);
                }
                Section::Lf { .. } => {
                    self.lf_sections.push(s);
                }
                Section::Hf { group, pass } => {
                    self.hf_sections[group][pass] = Some(s);
                    self.candidate_hf_sections.insert(group);
                }
            }
        }

        let mut processed_section = false;
        let pixel_format = self.pixel_format.as_ref().unwrap();
        'process: {
            let frame_header = frame.header();
            if frame_header.num_groups() == 1 && frame_header.passes.num_passes == 1 {
                // Single-group special case.
                let Some(sec) = self.lf_global_section.take() else {
                    break 'process;
                };
                assert!(self.sections.is_empty());
                let mut br = BitReader::new(&sec.data);
                frame.decode_lf_global(&mut br)?;
                frame.decode_lf_group(0, &mut br)?;
                frame.decode_hf_global(&mut br)?;
                frame.prepare_render_pipeline(
                    self.pixel_format.as_ref().unwrap(),
                    decode_options.cms.as_deref(),
                    self.embedded_color_profile
                        .as_ref()
                        .expect("embedded_color_profile should be set before pipeline preparation"),
                    self.output_color_profile
                        .as_ref()
                        .expect("output_color_profile should be set before pipeline preparation"),
                )?;
                frame.finalize_lf()?;
                frame.decode_and_render_hf_groups(
                    output_buffers,
                    pixel_format,
                    vec![(0, vec![(0, br)])],
                    do_flush,
                )?;
                processed_section = true;
            } else {
                if let Some(lf_global) = self.lf_global_section.take() {
                    frame.decode_lf_global(&mut BitReader::new(&lf_global.data))?;
                    self.section_state.lf_global_done = true;
                    processed_section = true;
                }

                if !self.section_state.lf_global_done {
                    break 'process;
                }

                for lf_section in self.lf_sections.drain(..) {
                    let Section::Lf { group } = lf_section.section else {
                        unreachable!()
                    };
                    frame.decode_lf_group(group, &mut BitReader::new(&lf_section.data))?;
                    processed_section = true;
                    self.section_state.remaining_lf -= 1;
                }

                if self.section_state.remaining_lf != 0 {
                    break 'process;
                }

                if let Some(hf_global) = self.hf_global_section.take() {
                    frame.decode_hf_global(&mut BitReader::new(&hf_global.data))?;
                    frame.prepare_render_pipeline(
                        self.pixel_format.as_ref().unwrap(),
                        decode_options.cms.as_deref(),
                        self.embedded_color_profile.as_ref().expect(
                            "embedded_color_profile should be set before pipeline preparation",
                        ),
                        self.output_color_profile.as_ref().expect(
                            "output_color_profile should be set before pipeline preparation",
                        ),
                    )?;
                    frame.finalize_lf()?;
                    self.section_state.hf_global_done = true;
                    processed_section = true;
                }

                if !self.section_state.hf_global_done {
                    break 'process;
                }

                let mut group_readers = vec![];
                let mut processed_groups = vec![];

                let mut check_group = |g: usize| {
                    let mut sections = vec![];
                    for (pass, grp) in self.hf_sections[g]
                        .iter()
                        .enumerate()
                        .skip(self.section_state.completed_passes[g] as usize)
                    {
                        let Some(s) = &grp else {
                            break;
                        };
                        self.section_state.completed_passes[g] += 1;
                        sections.push((pass, BitReader::new(&s.data)));
                    }
                    if !sections.is_empty() {
                        group_readers.push((g, sections));
                        processed_groups.push(g);
                    }
                };

                if self.candidate_hf_sections.len() * 4 < self.hf_sections.len() {
                    for g in self.candidate_hf_sections.drain() {
                        check_group(g)
                    }
                    // Processing sections in order is more efficient because it lets us flush
                    // the pipeline faster.
                    group_readers.sort_by_key(|x| x.0);
                } else {
                    for g in 0..self.hf_sections.len() {
                        if self.candidate_hf_sections.contains(&g) {
                            check_group(g);
                        }
                    }
                    self.candidate_hf_sections.clear();
                }

                frame.decode_and_render_hf_groups(
                    output_buffers,
                    pixel_format,
                    group_readers,
                    do_flush,
                )?;

                for g in processed_groups.into_iter() {
                    for i in 0..self.section_state.completed_passes[g] {
                        self.hf_sections[g][i as usize] = None;
                    }
                    processed_section = true;
                }
            }
        }

        if !processed_section {
            let data_for_next_section =
                self.sections.front().unwrap().len - self.ready_section_data;
            return Ok(Some(data_for_next_section));
        }

        // Frame is not yet complete.
        if !self.sections.is_empty() {
            return Ok(None);
        }

        #[cfg(test)]
        {
            self.frame_callback.as_mut().map_or(Ok(()), |cb| {
                cb(self.frame.as_ref().unwrap(), self.decoded_frames)
            })?;
            self.decoded_frames += 1;
        }

        // Check if this might be a preview frame (skipped frame with preview enabled)
        let has_preview = self
            .basic_info
            .as_ref()
            .is_some_and(|info| info.preview_size.is_some());
        let might_be_preview = self.process_without_output && has_preview;

        // Extract JPEG coefficients before finalizing the frame
        #[cfg(feature = "jpeg-reconstruction")]
        if let Some(frame) = self.frame.as_mut()
            && let Some(coeffs) = frame.take_jpeg_coefficients()
        {
            let do_ycbcr = frame.header().do_ycbcr;
            // Merge coefficients into the jpeg_reconstruction data
            if let Some(ref mut jpeg_data) = box_parser.jpeg_reconstruction {
                jpeg_data.dct_coefficients = Some(coeffs);
                if let Some((qtable, qtable_den)) = frame.jpeg_raw_quant_table() {
                    jpeg_data.update_quant_tables_from_raw(qtable, qtable_den, do_ycbcr)?;
                }
                {
                    let header = frame.header();
                    let is_gray = jpeg_data.is_gray || jpeg_data.components.len() == 1;
                    let component_map = if is_gray {
                        [1usize, 1, 1]
                    } else {
                        [1usize, 0, 2]
                    };
                    let mut max_hshift = 0usize;
                    let mut max_vshift = 0usize;
                    let chans = if is_gray {
                        &[1usize][..]
                    } else {
                        &[0usize, 1, 2][..]
                    };
                    for &c in chans {
                        max_hshift = max_hshift.max(header.hshift(c));
                        max_vshift = max_vshift.max(header.vshift(c));
                    }
                    for (jpeg_idx, &vardct_chan) in component_map
                        .iter()
                        .enumerate()
                        .take(jpeg_data.components.len())
                    {
                        let hshift = header.hshift(vardct_chan);
                        let vshift = header.vshift(vardct_chan);
                        jpeg_data.components[jpeg_idx].h_samp_factor =
                            1u8 << (max_hshift.saturating_sub(hshift) as u8);
                        jpeg_data.components[jpeg_idx].v_samp_factor =
                            1u8 << (max_vshift.saturating_sub(vshift) as u8);
                    }
                }
                if let Some(profile) = self.embedded_color_profile.as_ref()
                    && let Some(icc) = profile.try_as_icc()
                {
                    jpeg_data.fill_icc_app_markers(icc.as_ref())?;
                }
            }
        }

        let decoder_state = self.frame.take().unwrap().finalize()?;
        if let Some(state) = decoder_state {
            self.decoder_state = Some(state);
        } else if might_be_preview {
            // Preview frame has is_last=true but the main frame follows.
            // Recreate decoder state from saved file header for the main frame.
            if let Some(fh) = self.saved_file_header.take() {
                let mut new_state = crate::frame::DecoderState::new(fh);
                new_state.render_spotcolors = decode_options.render_spot_colors;
                self.decoder_state = Some(new_state);
            }
        } else {
            self.has_more_frames = false;
        }
        Ok(None)
    }
}
