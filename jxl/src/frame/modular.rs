// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::{cell::RefCell, fmt::Debug};

use crate::{
    bit_reader::BitReader,
    error::Result,
    frame::HfMetadata,
    headers::{
        extra_channels::ExtraChannelInfo, frame_header::FrameHeader, modular::GroupHeader,
        JxlHeader,
    },
    image::{Image, Rect},
    util::{tracing_wrappers::*, CeilLog2},
};

mod borrowed_buffers;
mod decode;
mod predict;
mod transforms;
mod tree;

use borrowed_buffers::MutablyBorrowedModularBuffers;
use decode::decode_modular_subbitstream;
pub use decode::ModularStreamId;
pub use predict::Predictor;
use transforms::{make_grids, TransformStepChunk};
pub use tree::Tree;

#[derive(Clone, PartialEq, Eq)]
struct ChannelInfo {
    size: (usize, usize),
    shift: Option<(usize, usize)>, // None for meta-channels
}

impl Debug for ChannelInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}x{}", self.size.0, self.size.1)?;
        if let Some(shift) = self.shift {
            write!(f, "(shift {},{})", shift.0, shift.1)
        } else {
            write!(f, "(meta)")
        }
    }
}

impl ChannelInfo {
    fn is_meta(&self) -> bool {
        self.shift.is_none()
    }

    fn is_meta_or_small(&self, group_dim: usize) -> bool {
        self.is_meta() || (self.size.0 <= group_dim && self.size.1 <= group_dim)
    }

    fn is_shift_in_range(&self, min: usize, max: usize) -> bool {
        assert!(min <= max);
        self.shift.is_some_and(|(a, b)| {
            let shift = a.min(b);
            min <= shift && shift <= max
        })
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
enum ModularGridKind {
    // Single big channel.
    None,
    // 2048x2048 image-pixels (if modular_group_shift == 1).
    Lf,
    // 256x256 image-pixels (if modular_group_shift == 1).
    Hf,
}

impl ModularGridKind {
    fn grid_shape(&self, frame_header: &FrameHeader) -> (usize, usize) {
        match self {
            ModularGridKind::None => (1, 1),
            ModularGridKind::Lf => frame_header.size_lf_groups(),
            ModularGridKind::Hf => frame_header.size_groups(),
        }
    }
}

// Note: this type uses interior mutability to get mutable references to multiple buffers at once.
// In principle, this is not needed, but the overhead should be minimal so using `unsafe` here is
// probably not worth it.
#[derive(Debug)]
struct ModularBuffer {
    data: RefCell<Option<Image<i32>>>,
    // Holds additional information such as the weighted predictor's error channel's last row for
    // the transform chunk that produced this buffer.
    auxiliary_data: RefCell<Option<Image<i32>>>,
    // Number of times this buffer will be used, *including* when it is used for output.
    remaining_uses: usize,
    used_by_transforms: Vec<usize>,
    size: (usize, usize),
}

impl ModularBuffer {
    // Gives out a copy of the buffer + auxiliary buffer, marking the buffer as used.
    // If this was the last usage of the buffer, does not actually copy the buffer.
    fn get_buffer(&mut self) -> Result<(Image<i32>, Option<Image<i32>>)> {
        self.remaining_uses = self.remaining_uses.checked_sub(1).unwrap();
        if self.remaining_uses == 0 {
            Ok((
                self.data.borrow_mut().take().unwrap(),
                self.auxiliary_data.borrow_mut().take(),
            ))
        } else {
            Ok((
                self.data
                    .borrow()
                    .as_ref()
                    .map(|x| x.as_rect().to_image())
                    .transpose()?
                    .unwrap(),
                self.auxiliary_data
                    .borrow()
                    .as_ref()
                    .map(|x| x.as_rect().to_image())
                    .transpose()?,
            ))
        }
    }

    fn mark_used(&mut self) {
        self.remaining_uses = self.remaining_uses.checked_sub(1).unwrap();
        if self.remaining_uses == 0 {
            *self.data.borrow_mut() = None;
            *self.auxiliary_data.borrow_mut() = None;
        }
    }
}

#[derive(Debug)]
struct ModularBufferInfo {
    info: ChannelInfo,
    // Only accurate for output and coded channels.
    channel_id: usize,
    is_output: bool,
    is_coded: bool,
    #[allow(dead_code)]
    description: String,
    grid_kind: ModularGridKind,
    grid_shape: (usize, usize),
    buffer_grid: Vec<ModularBuffer>,
}

/// A modular image is a sequence of channels to which one or more transforms might have been
/// applied. We represent a modular image as a list of buffers, some of which are coded in the
/// bitstream; other buffers are obtained as the output of one of the transformation steps.
/// Some buffers are marked as `output`: those are the buffers corresponding to the pre-transform
/// image channels.
/// The buffers are internally divided in grids, matching the sizes of the groups they are coded
/// in (with appropriate shifts), or the size of the data produced by applying the appropriate
/// transforms to each of the groups in the input of the transforms.
#[allow(dead_code)]
#[derive(Debug)]
pub struct FullModularImage {
    buffer_info: Vec<ModularBufferInfo>,
    transform_steps: Vec<TransformStepChunk>,
    // List of buffer indices of the channels of the modular image encoded in each kind of section.
    // In order, LfGlobal, LfGroup, HfGroup(pass 0), ..., HfGroup(last pass).
    section_buffer_indices: Vec<Vec<usize>>,
}

impl FullModularImage {
    #[instrument(level = "debug", skip_all, ret)]
    pub fn read(
        frame_header: &FrameHeader,
        modular_color_channels: usize,
        extra_channel_info: &[ExtraChannelInfo],
        global_tree: &Option<Tree>,
        br: &mut BitReader,
    ) -> Result<Self> {
        let mut channels = vec![];
        for c in 0..modular_color_channels {
            let shift = (frame_header.hshift(c), frame_header.vshift(c));
            let size = (frame_header.width as usize, frame_header.height as usize);
            channels.push(ChannelInfo {
                size: (size.0.div_ceil(1 << shift.0), size.1.div_ceil(1 << shift.1)),
                shift: Some(shift),
            });
        }

        for info in extra_channel_info {
            let shift = info
                .dim_shift()
                .checked_sub(frame_header.upsampling.ceil_log2())
                .expect("ec_upsampling >= upsampling should be checked in frame header")
                as usize;
            let size = frame_header.size_upsampled();
            let size = (size.0 >> info.dim_shift(), size.1 >> info.dim_shift());
            channels.push(ChannelInfo {
                size,
                shift: Some((shift, shift)),
            });
        }

        if channels.is_empty() {
            return Ok(Self {
                buffer_info: vec![],
                transform_steps: vec![],
                section_buffer_indices: vec![vec![]; 2 + frame_header.passes.num_passes as usize],
            });
        }

        trace!("reading modular header");
        let header = GroupHeader::read(br)?;

        let (mut buffer_info, transform_steps) =
            transforms::meta_apply_transforms(&channels, &header.transforms)?;

        // Assign each (channel, group) pair present in the bitstream to the section in which it will be decoded.
        let mut section_buffer_indices: Vec<Vec<usize>> = vec![];

        let mut sorted_buffers: Vec<_> = buffer_info
            .iter()
            .enumerate()
            .filter_map(|(i, b)| {
                if b.is_coded {
                    Some((b.channel_id, i))
                } else {
                    None
                }
            })
            .collect();

        sorted_buffers.sort_by_key(|x| x.0);

        section_buffer_indices.push(
            sorted_buffers
                .iter()
                .take_while(|x| {
                    buffer_info[x.1]
                        .info
                        .is_meta_or_small(frame_header.group_dim())
                })
                .map(|x| x.1)
                .collect(),
        );

        section_buffer_indices.push(
            sorted_buffers
                .iter()
                .skip_while(|x| {
                    buffer_info[x.1]
                        .info
                        .is_meta_or_small(frame_header.group_dim())
                })
                .filter(|x| buffer_info[x.1].info.is_shift_in_range(3, usize::MAX))
                .map(|x| x.1)
                .collect(),
        );

        for pass in 0..frame_header.passes.num_passes as usize {
            let (min_shift, max_shift) = frame_header.passes.downsampling_bracket(pass);
            section_buffer_indices.push(
                sorted_buffers
                    .iter()
                    .filter(|x| {
                        !buffer_info[x.1]
                            .info
                            .is_meta_or_small(frame_header.group_dim())
                    })
                    .filter(|x| {
                        buffer_info[x.1]
                            .info
                            .is_shift_in_range(min_shift, max_shift)
                    })
                    .map(|x| x.1)
                    .collect(),
            );
        }

        // Ensure that the channel list in each group is sorted by actual channel ID.
        for list in section_buffer_indices.iter_mut() {
            list.sort_by_key(|x| buffer_info[*x].channel_id);
        }

        trace!(?section_buffer_indices);

        let transform_steps = make_grids(
            frame_header,
            transform_steps,
            &section_buffer_indices,
            &mut buffer_info,
        );

        {
            let mut buffers =
                MutablyBorrowedModularBuffers::new(&buffer_info, &section_buffer_indices[0], 0)?;

            decode_modular_subbitstream(
                &mut buffers.bufs,
                &buffers.channel_ids,
                ModularStreamId::GlobalData.get_id(frame_header),
                Some(header),
                global_tree,
                br,
            )?;
        }

        Ok(FullModularImage {
            buffer_info,
            transform_steps,
            section_buffer_indices,
        })
    }

    #[instrument(
        level = "debug",
        skip(self, frame_header, global_tree, on_output, br),
        ret
    )]
    pub fn read_stream(
        &mut self,
        stream: ModularStreamId,
        frame_header: &FrameHeader,
        global_tree: &Option<Tree>,
        on_output: impl Fn(usize, (usize, usize), &Image<i32>) -> Result<()>,
        br: &mut BitReader,
    ) -> Result<()> {
        if self.buffer_info.is_empty() {
            info!("No modular channels to decode");
            return Ok(());
        }
        let (section_id, grid) = match stream {
            ModularStreamId::ModularLF(group) => (1, group),
            ModularStreamId::ModularHF { pass, group } => (2 + pass, group),
            _ => {
                unreachable!("read_stream should only be used for streams that are part of the main Modular image");
            }
        };

        {
            let mut buffers = MutablyBorrowedModularBuffers::new(
                &self.buffer_info,
                &self.section_buffer_indices[section_id],
                grid,
            )?;

            decode_modular_subbitstream(
                &mut buffers.bufs,
                &buffers.channel_ids,
                stream.get_id(frame_header),
                None,
                global_tree,
                br,
            )?;
        }

        let maybe_output = |bi: &mut ModularBufferInfo, grid: usize| -> Result<()> {
            if bi.is_output {
                let (gw, _) = bi.grid_shape;
                let g = (grid % gw, grid / gw);
                on_output(
                    bi.channel_id,
                    g,
                    bi.buffer_grid[grid].data.borrow().as_ref().unwrap(),
                )?;
                bi.buffer_grid[grid].mark_used();
            }
            Ok(())
        };

        let mut new_ready_transform_chunks = vec![];
        for buf in self.section_buffer_indices[section_id].iter() {
            maybe_output(&mut self.buffer_info[*buf], grid)?;
            new_ready_transform_chunks.extend(
                self.buffer_info[*buf].buffer_grid[grid]
                    .used_by_transforms
                    .iter()
                    .copied(),
            );
        }

        while let Some(tfm) = new_ready_transform_chunks.pop() {
            for (new_buf, new_grid) in self.transform_steps[tfm].dep_ready(&mut self.buffer_info)? {
                maybe_output(&mut self.buffer_info[new_buf], new_grid)?;
                new_ready_transform_chunks.extend_from_slice(
                    &self.buffer_info[new_buf].buffer_grid[new_grid].used_by_transforms,
                );
            }
        }

        Ok(())
    }
}

pub fn decode_vardct_lf(
    group: usize,
    frame_header: &FrameHeader,
    global_tree: &Option<Tree>,
    _lf_image: &mut Image<f32>,
    br: &mut BitReader,
) -> Result<()> {
    let _extra_precision = br.read(2)?;
    assert!(frame_header.is444());
    debug!(?_extra_precision);
    let stream_id = ModularStreamId::VarDCTLF(group).get_id(frame_header);
    debug!(?stream_id);
    let r = frame_header.lf_group_rect(group);
    debug!(?r);
    let mut buffers = [
        Image::new(r.size)?,
        Image::new(r.size)?,
        Image::new(r.size)?,
    ];
    let mut buf_refs: Vec<_> = buffers.iter_mut().collect();
    decode_modular_subbitstream(&mut buf_refs, &[0, 1, 2], stream_id, None, global_tree, br)?;
    // TODO(szabadka): Generate the f32 pixels of the LF image.
    Ok(())
}

pub fn decode_hf_metadata(
    group: usize,
    frame_header: &FrameHeader,
    global_tree: &Option<Tree>,
    _hf_meta: &mut HfMetadata,
    br: &mut BitReader,
) -> Result<()> {
    let stream_id = ModularStreamId::LFMeta(group).get_id(frame_header);
    debug!(?stream_id);
    let r = frame_header.lf_group_rect(group);
    debug!(?r);
    let upper_bound = r.size.0 * r.size.1;
    let count_num_bits = upper_bound.ceil_log2();
    let count = br.read(count_num_bits)? + 1;
    debug!(?count);
    assert!(frame_header.is444());
    let cr = Rect {
        origin: (r.origin.0 >> 3, r.origin.1 >> 3),
        size: (r.size.0.div_ceil(8), r.size.1.div_ceil(8)),
    };
    let mut buffers = [
        Image::new(cr.size)?,
        Image::new(cr.size)?,
        Image::new((count as usize, 2))?,
        Image::new(r.size)?,
    ];
    let mut buf_refs: Vec<_> = buffers.iter_mut().collect();
    decode_modular_subbitstream(
        &mut buf_refs,
        &[0, 1, 2, 3],
        stream_id,
        None,
        global_tree,
        br,
    )?;
    // TODO(szabadka): Fill in HfMetadata from the modular buffers.
    Ok(())
}
