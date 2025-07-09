// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    api::{JxlColorType, JxlOutputBuffer},
    bit_reader::BitReader,
    error::Result,
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
}

// No guarantees on the order of calls to f, or the order of retained elements in vec.
fn retain_by_value<T>(vec: &mut Vec<T>, mut f: impl FnMut(T) -> Option<T>) {
    for pos in (0..vec.len()).rev() {
        let element_to_test = vec.swap_remove(pos);
        if let Some(v) = f(element_to_test) {
            vec.push(v);
        }
    }
}

impl CodestreamParser {
    pub(super) fn process_sections(
        &mut self,
        output_buffers: &mut Option<&mut [JxlOutputBuffer<'_>]>,
    ) -> Result<()> {
        // Dequeue ready sections.
        while self
            .sections
            .front()
            .is_some_and(|s| s.len <= self.ready_section_data)
        {
            let s = self.sections.pop_front().unwrap();
            self.ready_section_data -= s.len;
            self.available_sections.push(s);
        }
        if self.available_sections.is_empty() {
            return Ok(());
        }
        let frame = self.frame.as_mut().unwrap();
        let frame_header = frame.header();
        if frame_header.num_groups() == 1 && frame_header.passes.num_passes == 1 {
            // Single-group special case.
            assert_eq!(self.available_sections.len(), 1);
            assert!(self.sections.is_empty());
            let mut br = BitReader::new(&self.available_sections[0].data);
            frame.decode_lf_global(&mut br)?;
            frame.decode_lf_group(0, &mut br)?;
            frame.decode_hf_global(&mut br)?;
            frame.prepare_render_pipeline()?;
            frame.finalize_lf()?;
            frame.decode_hf_group(0, 0, &mut br)?;
            self.available_sections.clear();
        } else {
            let mut lf_global_section = None;
            let mut lf_sections = vec![];
            let mut hf_global_section = None;
            let mut sorted_sections_for_each_group = Vec::with_capacity(frame_header.num_groups());
            for _ in 0..frame_header.num_groups() {
                sorted_sections_for_each_group.push(vec![]);
            }

            loop {
                let initial_sz = self.available_sections.len();
                retain_by_value(&mut self.available_sections, |sec| match sec.section {
                    Section::LfGlobal => {
                        lf_global_section = Some(sec);
                        self.section_state.lf_global_done = true;
                        None
                    }
                    Section::Lf { .. } => {
                        if !self.section_state.lf_global_done {
                            Some(sec)
                        } else {
                            lf_sections.push(sec);
                            self.section_state.remaining_lf -= 1;
                            None
                        }
                    }
                    Section::HfGlobal => {
                        if self.section_state.remaining_lf != 0 {
                            Some(sec)
                        } else {
                            hf_global_section = Some(sec);
                            self.section_state.hf_global_done = true;
                            None
                        }
                    }
                    Section::Hf { group, pass } => {
                        if !self.section_state.hf_global_done
                            && self.section_state.completed_passes[group] != pass as u8
                        {
                            Some(sec)
                        } else {
                            sorted_sections_for_each_group[group].push(sec);
                            self.section_state.completed_passes[group] += 1;
                            None
                        }
                    }
                });
                if self.available_sections.len() == initial_sz {
                    break;
                }
            }

            if let Some(lf_global) = lf_global_section {
                frame.decode_lf_global(&mut BitReader::new(&lf_global.data))?;
            }

            for lf_section in lf_sections {
                let Section::Lf { group } = lf_section.section else {
                    unreachable!()
                };
                frame.decode_lf_group(group, &mut BitReader::new(&lf_section.data))?;
            }

            if let Some(hf_global) = hf_global_section {
                frame.decode_hf_global(&mut BitReader::new(&hf_global.data))?;
                frame.prepare_render_pipeline()?;
                frame.finalize_lf()?;
            }

            for g in sorted_sections_for_each_group {
                // TODO(veluca): render all the available passes at once.
                for sec in g {
                    let Section::Hf { group, pass } = sec.section else {
                        unreachable!()
                    };
                    frame.decode_hf_group(group, pass, &mut BitReader::new(&sec.data))?;
                }
            }
        }
        // Frame is not yet complete.
        if !self.sections.is_empty() {
            return Ok(());
        }
        assert!(self.available_sections.is_empty());
        let result = self.frame.take().unwrap().finalize()?;
        if let Some(state) = result.decoder_state {
            self.decoder_state = Some(state);
        } else {
            self.has_more_frames = false;
        }
        // TODO(veluca): this code should be integrated in the render pipeline.
        if let Some(channels) = result.channels
            && let Some(bufs) = output_buffers
        {
            if self.pixel_format.as_ref().unwrap().color_type == JxlColorType::Grayscale {
                for (buf, chan) in bufs.iter_mut().zip(channels.iter()) {
                    buf.write_from_f32(chan);
                }
            } else {
                bufs[0].write_from_rgb_f32(&channels[0], &channels[1], &channels[2]);
                for (buf, chan) in bufs.iter_mut().skip(1).zip(channels.iter().skip(3)) {
                    buf.write_from_f32(chan);
                }
            }
        }
        Ok(())
    }
}
