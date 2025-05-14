// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::{cell::Ref, fmt::Debug};

use num_traits::FromPrimitive;

use crate::{
    error::{Error, Result},
    frame::modular::{
        borrowed_buffers::with_buffers, ChannelInfo, ModularBufferInfo, ModularChannel,
        ModularGridKind, Predictor,
    },
    headers::{self, modular::TransformId, modular::WeightedHeader},
    util::tracing_wrappers::*,
};

use super::{RctOp, RctPermutation};

#[derive(Debug, Clone)]
pub enum TransformStep {
    Rct {
        buf_in: [usize; 3],
        buf_out: [usize; 3],
        op: RctOp,
        perm: RctPermutation,
    },
    Palette {
        buf_in: usize,
        buf_pal: usize,
        buf_out: Vec<usize>,
        num_colors: usize,
        num_deltas: usize,
        predictor: Predictor,
        wp_header: WeightedHeader,
    },
    HSqueeze {
        buf_in: [usize; 2],
        buf_out: usize,
    },
    VSqueeze {
        buf_in: [usize; 2],
        buf_out: usize,
    },
}

#[derive(Debug)]
pub struct TransformStepChunk {
    pub(super) step: TransformStep,
    // Grid position this transform should produce.
    // Note that this is a lie for Palette with AverageAll or Weighted, as the transform with
    // position (0, y) will produce the entire row of blocks (*, y) (and there will be no
    // transforms with position (x, y) with x > 0).
    pub(super) grid_pos: (usize, usize),
    // Number of inputs that are not yet available.
    pub(super) incomplete_deps: usize,
}

impl TransformStepChunk {
    // Marks that one dependency of this transform is ready, and potentially runs the transform,
    // returning the new buffers that are now ready.
    pub fn dep_ready(&mut self, buffers: &mut [ModularBufferInfo]) -> Result<Vec<(usize, usize)>> {
        self.incomplete_deps = self.incomplete_deps.checked_sub(1).unwrap();
        if self.incomplete_deps > 0 {
            return Ok(vec![]);
        }
        let buf_out: &[usize] = match &self.step {
            TransformStep::Rct { buf_out, .. } => buf_out,
            TransformStep::Palette { buf_out, .. } => buf_out,
            TransformStep::HSqueeze { buf_out, .. } | TransformStep::VSqueeze { buf_out, .. } => {
                &[*buf_out]
            }
        };

        let grid = buffers[buf_out[0]].get_grid_idx(self.grid_pos);
        for bo in buf_out {
            assert_eq!(buffers[buf_out[0]].grid_kind, buffers[*bo].grid_kind);
            assert_eq!(buffers[buf_out[0]].info.size, buffers[*bo].info.size);
        }

        match &self.step {
            TransformStep::Rct {
                buf_in,
                buf_out,
                op,
                perm,
            } => {
                for i in 0..3 {
                    assert_eq!(buffers[buf_out[0]].grid_kind, buffers[buf_in[i]].grid_kind);
                    assert_eq!(buffers[buf_out[0]].info.size, buffers[buf_in[i]].info.size);
                    // Optimistically move the buffers to the output if possible.
                    // If not, creates buffers in the output that are a copy of the input buffers.
                    // This should be rare.
                    *buffers[buf_out[i]].buffer_grid[grid].data.borrow_mut() =
                        Some(buffers[buf_in[i]].buffer_grid[grid].get_buffer()?);
                }
                with_buffers(buffers, buf_out, grid, |mut bufs| {
                    super::rct::do_rct_step(&mut bufs, *op, *perm);
                    Ok(())
                })?;
            }
            TransformStep::Palette {
                buf_in,
                buf_pal,
                buf_out,
                num_colors,
                num_deltas,
                predictor,
                wp_header,
            } if *predictor != Predictor::Weighted && *predictor != Predictor::AverageAll => {
                assert_eq!(buffers[buf_out[0]].grid_kind, buffers[*buf_in].grid_kind);
                assert_eq!(buffers[buf_out[0]].info.size, buffers[*buf_in].info.size);

                {
                    let img_in = Ref::map(buffers[*buf_in].buffer_grid[grid].data.borrow(), |x| {
                        x.as_ref().unwrap()
                    });
                    let img_pal = Ref::map(buffers[*buf_pal].buffer_grid[0].data.borrow(), |x| {
                        x.as_ref().unwrap()
                    });
                    with_buffers(buffers, buf_out, grid, |mut bufs| {
                        super::palette::do_palette_step_general(
                            &img_in,
                            &img_pal,
                            &mut bufs,
                            *num_colors,
                            *num_deltas,
                            *predictor,
                            wp_header,
                        );
                        Ok(())
                    })?;
                }
                buffers[*buf_in].buffer_grid[grid].mark_used();
                buffers[*buf_pal].buffer_grid[0].mark_used();
            }
            _ => {
                todo!()
            }
        };

        Ok(buf_out.iter().map(|x| (*x, grid)).collect())
    }
}

#[instrument(level = "trace", err)]
fn check_equal_channels(
    channels: &[(usize, ChannelInfo)],
    first_chan: usize,
    num: usize,
) -> Result<()> {
    if first_chan + num > channels.len() {
        return Err(Error::InvalidChannelRange(
            first_chan,
            first_chan + num,
            channels.len(),
        ));
    }
    for inc in 1..num {
        if channels[first_chan].1 != channels[first_chan + inc].1 {
            return Err(Error::MixingDifferentChannels);
        }
    }
    Ok(())
}

fn meta_apply_single_transform(
    transform: &headers::modular::Transform,
    header: &headers::modular::GroupHeader,
    channels: &mut Vec<(usize, ChannelInfo)>,
    transform_steps: &mut Vec<TransformStep>,
    mut add_transform_buffer: impl FnMut(ChannelInfo, String) -> usize,
) -> Result<()> {
    match transform.id {
        TransformId::Rct => {
            let begin_channel = transform.begin_channel as usize;
            let op = RctOp::from_u32(transform.rct_type % 7).unwrap();
            let perm = RctPermutation::from_u32(transform.rct_type / 7)
                .expect("header decoding should ensure rct_type < 42");
            check_equal_channels(channels, begin_channel, 3)?;
            let mut buf_in = [0; 3];
            let buf_out = [
                channels[begin_channel].0,
                channels[begin_channel + 1].0,
                channels[begin_channel + 2].0,
            ];
            for i in 0..3 {
                let c = &mut channels[begin_channel + i];
                c.0 = add_transform_buffer(
                    c.1,
                    format!(
                        "RCT (op {:?} perm {:?}) starting at channel {}, input {}",
                        op, perm, begin_channel, i
                    ),
                );
                buf_in[i] = c.0;
            }
            transform_steps.push(TransformStep::Rct {
                buf_out,
                buf_in,
                op,
                perm,
            });
            trace!("applied RCT: {channels:?}");
        }
        TransformId::Squeeze => {
            let steps = if transform.squeezes.is_empty() {
                super::squeeze::default_squeeze(channels)
            } else {
                transform.squeezes.clone()
            };
            for step in steps {
                super::squeeze::check_squeeze_params(channels, &step)?;
                let in_place = step.in_place;
                let horizontal = step.horizontal;
                let begin_channel = step.begin_channel as usize;
                let num_channels = step.num_channels as usize;
                let end_channel = begin_channel + num_channels;
                let new_chan_offset = if in_place {
                    end_channel
                } else {
                    channels.len()
                };
                for ic in 0..num_channels {
                    let chan = &channels[begin_channel + ic].1;
                    let new_shift = if let Some(shift) = chan.shift {
                        if shift.0 > 30 || shift.1 > 30 {
                            return Err(Error::TooManySqueezes);
                        }
                        if horizontal {
                            Some((shift.0 + 1, shift.1))
                        } else {
                            Some((shift.0, shift.1 + 1))
                        }
                    } else {
                        None
                    };
                    let w = chan.size.0;
                    let h = chan.size.1;
                    let (new_size_0, new_size_1) = if horizontal {
                        ((w.div_ceil(2), h), (w - w.div_ceil(2), h))
                    } else {
                        ((w, h.div_ceil(2)), (w, h - h.div_ceil(2)))
                    };
                    let new_0 = ChannelInfo {
                        shift: new_shift,
                        size: new_size_0,
                        bit_depth: chan.bit_depth,
                    };
                    let buf_0 = add_transform_buffer(
                        new_0,
                        format!("Squeezed channel, original channel {}", begin_channel + ic),
                    );
                    let new_1 = ChannelInfo {
                        shift: new_shift,
                        size: new_size_1,
                        bit_depth: chan.bit_depth,
                    };
                    let buf_1 = add_transform_buffer(
                        new_1,
                        format!("Squeeze residual, original channel {}", begin_channel + ic),
                    );
                    if horizontal {
                        transform_steps.push(TransformStep::HSqueeze {
                            buf_in: [buf_0, buf_1],
                            buf_out: channels[begin_channel + ic].0,
                        });
                    } else {
                        transform_steps.push(TransformStep::VSqueeze {
                            buf_in: [buf_0, buf_1],
                            buf_out: channels[begin_channel + ic].0,
                        });
                    }
                    channels[begin_channel + ic] = (buf_0, new_0);
                    channels.insert(new_chan_offset + ic, (buf_1, new_1));
                    trace!("applied squeeze: {channels:?}");
                }
            }
        }
        TransformId::Palette => {
            let begin_channel = transform.begin_channel as usize;
            let num_channels = transform.num_channels as usize;
            let num_colors = transform.num_colors as usize;
            let num_deltas = transform.num_deltas as usize;
            let pred = Predictor::from_u32(transform.predictor_id)
                .expect("header decoding should ensure a valid predictor");
            check_equal_channels(channels, begin_channel, num_channels)?;
            // We already checked the bit_depth for all channels from `begin_channel` is
            // equal in the line above.
            let bit_depth = channels[begin_channel].1.bit_depth;
            let pchan_info = ChannelInfo {
                shift: None,
                size: (num_colors + num_deltas, num_channels),
                bit_depth,
            };
            let pchan = add_transform_buffer(
                pchan_info,
                format!(
                    "Palette for palette transform starting at channel {} with {} channels",
                    begin_channel, num_channels
                ),
            );
            let inchan = add_transform_buffer(
                channels[begin_channel].1,
                format!(
                    "Pixel data for palette transform starting at channel {} with {} channels",
                    begin_channel, num_channels
                ),
            );
            transform_steps.push(TransformStep::Palette {
                buf_in: inchan,
                buf_pal: pchan,
                buf_out: channels[begin_channel..(begin_channel + num_channels)]
                    .iter()
                    .map(|x| x.0)
                    .collect(),
                num_colors,
                num_deltas,
                predictor: pred,
                wp_header: header.wp_header.clone(),
            });
            channels.drain(begin_channel + 1..begin_channel + num_channels);
            channels[begin_channel].0 = inchan;
            channels.insert(0, (pchan, pchan_info));
            trace!("applied palette: {channels:?}");
        }
        TransformId::Invalid => {
            unreachable!("header decoding for invalid transforms should fail");
        }
    }
    Ok(())
}

#[instrument(level = "trace", ret)]
pub fn meta_apply_transforms(
    channels: &[ChannelInfo],
    header: &headers::modular::GroupHeader,
) -> Result<(Vec<ModularBufferInfo>, Vec<TransformStep>)> {
    let mut buffer_info = vec![];
    let mut transform_steps = vec![];
    // (buffer id, channel info)
    let mut channels: Vec<_> = channels.iter().cloned().enumerate().collect();

    // First, add all the pre-transform channels to the buffer list.
    for chan in channels.iter() {
        buffer_info.push(ModularBufferInfo {
            info: chan.1,
            channel_id: chan.0,
            is_output: true,
            is_coded: false,
            description: format!(
                "Input channel {}, size {}x{}",
                chan.0, chan.1.size.0, chan.1.size.1
            ),
            // To be filled by make_grids.
            grid_kind: ModularGridKind::None,
            grid_shape: (0, 0),
            buffer_grid: vec![],
        });
    }

    let mut add_transform_buffer = |info, description| {
        buffer_info.push(ModularBufferInfo {
            info,
            channel_id: 0,
            is_output: false,
            is_coded: false,
            description,
            // To be filled by make_grids.
            grid_kind: ModularGridKind::None,
            grid_shape: (0, 0),
            buffer_grid: vec![],
        });
        buffer_info.len() - 1
    };

    // Apply transforms to the channel list.
    for transform in &header.transforms {
        meta_apply_single_transform(
            transform,
            header,
            &mut channels,
            &mut transform_steps,
            &mut add_transform_buffer,
        )?;
    }

    // All the channels left over at the end of applying transforms are the channels that are
    // actually coded.
    for (chid, chan) in channels.iter().enumerate() {
        buffer_info[chan.0].is_coded = true;
        buffer_info[chan.0].channel_id = chid;
    }

    debug!(?transform_steps);

    Ok((buffer_info, transform_steps))
}

#[derive(Debug)]
pub enum LocalTransformBuffer<'a> {
    // This channel has been consumed by some transform.
    Empty,
    // This channel has not been written to yet.
    Placeholder(ChannelInfo),
    // Temporary, locally-allocated channel.
    Owned(ModularChannel),
    // Channel belonging to the global image.
    Borrowed(&'a mut ModularChannel),
}

impl LocalTransformBuffer<'_> {
    fn channel_info(&self) -> ChannelInfo {
        match self {
            LocalTransformBuffer::Empty => unreachable!("an empty buffer has no channel info"),
            LocalTransformBuffer::Owned(m) => m.channel_info(),
            LocalTransformBuffer::Placeholder(c) => *c,
            LocalTransformBuffer::Borrowed(m) => m.channel_info(),
        }
    }

    fn borrow_mut(&mut self) -> &mut ModularChannel {
        match self {
            LocalTransformBuffer::Owned(m) => m,
            LocalTransformBuffer::Borrowed(m) => m,
            LocalTransformBuffer::Empty => unreachable!("tried to borrow an empty channel"),
            LocalTransformBuffer::Placeholder(_) => {
                unreachable!("tried to borrow a placeholder channel")
            }
        }
    }

    fn take(&mut self) -> Self {
        assert!(!matches!(self, LocalTransformBuffer::Empty));
        let mut r = LocalTransformBuffer::Empty;
        std::mem::swap(self, &mut r);
        r
    }

    fn allocate_if_needed(&mut self) -> Result<()> {
        if let LocalTransformBuffer::Placeholder(c) = self {
            *self = LocalTransformBuffer::Owned(ModularChannel::new_with_shift(
                c.size,
                c.shift,
                c.bit_depth,
            )?);
        }
        Ok(())
    }
}

#[instrument(level = "trace", ret)]
pub fn meta_apply_local_transforms<'a, 'b>(
    channels_in: Vec<&'a mut ModularChannel>,
    buffer_storage: &'b mut Vec<LocalTransformBuffer<'a>>,
    header: &headers::modular::GroupHeader,
) -> Result<(Vec<&'b mut ModularChannel>, Vec<TransformStep>)> {
    let mut transform_steps = vec![];

    // (buffer id, channel info)
    let mut channels: Vec<_> = channels_in
        .iter()
        .map(|x| x.channel_info())
        .enumerate()
        .collect();

    debug!(?channels, "initial channels");

    // First, add all the pre-transform channels to the buffer list.
    buffer_storage.extend(channels_in.into_iter().map(LocalTransformBuffer::Borrowed));

    #[allow(unused_variables)]
    let mut add_transform_buffer = |info, description| {
        trace!(description, ?info, "adding channel buffer");
        buffer_storage.push(LocalTransformBuffer::Placeholder(info));
        buffer_storage.len() - 1
    };

    // Apply transforms to the channel list.
    for transform in &header.transforms {
        meta_apply_single_transform(
            transform,
            header,
            &mut channels,
            &mut transform_steps,
            &mut add_transform_buffer,
        )?;
    }

    debug!(?channels, ?buffer_storage, "channels after transforms");
    debug!(?transform_steps);

    // Ensure that the buffer indices in `channels` appear in increasing order, by reordering them
    // if necessary.
    if !channels.iter().map(|x| x.0).is_sorted() {
        let mut buf_new_position: Vec<_> = channels.iter().map(|x| x.0).collect();
        buf_new_position.sort();
        let buf_tmp: Vec<_> = channels
            .iter()
            .map(|x| {
                let mut b = LocalTransformBuffer::Empty;
                std::mem::swap(&mut b, &mut buffer_storage[x.0]);
                b
            })
            .collect();

        let mut buf_remap: Vec<_> = (0..buffer_storage.len()).collect();

        for (new_pos, (ch_info, buf)) in buf_new_position
            .iter()
            .cloned()
            .zip(channels.iter_mut().zip(buf_tmp.into_iter()))
        {
            assert!(matches!(
                buffer_storage[new_pos],
                LocalTransformBuffer::Empty
            ));
            buf_remap[ch_info.0] = new_pos;
            buffer_storage[new_pos] = buf;
            ch_info.0 = new_pos;
        }

        for step in transform_steps.iter_mut() {
            use std::iter::once;
            match step {
                TransformStep::Rct {
                    buf_in, buf_out, ..
                } => {
                    for b in buf_in.iter_mut().chain(buf_out.iter_mut()) {
                        *b = buf_remap[*b];
                    }
                }
                TransformStep::Palette {
                    buf_in,
                    buf_pal,
                    buf_out,
                    ..
                } => {
                    for b in once(buf_in).chain(once(buf_pal)).chain(buf_out.iter_mut()) {
                        *b = buf_remap[*b];
                    }
                }
                TransformStep::HSqueeze { buf_in, buf_out }
                | TransformStep::VSqueeze { buf_in, buf_out } => {
                    for b in once(buf_out).chain(buf_in.iter_mut()) {
                        *b = buf_remap[*b];
                    }
                }
            }
        }
    }

    debug!(?channels, ?buffer_storage, "sorted channels");

    debug!(?transform_steps);

    // Since RCT steps will try to transfer buffers from the source channels to the destination
    // channels, make sure we do the reverse transformation here (to have the caller-provided
    // buffers be used for writing temporary data).
    for ts in transform_steps.iter() {
        if let TransformStep::Rct {
            buf_in, buf_out, ..
        } = ts
        {
            for c in 0..3 {
                assert_eq!(
                    buffer_storage[buf_in[c]].channel_info(),
                    buffer_storage[buf_out[c]].channel_info()
                );
                assert!(matches!(
                    buffer_storage[buf_in[c]],
                    LocalTransformBuffer::Placeholder(_)
                ));
                buffer_storage.swap(buf_in[c], buf_out[c]);
            }
        }
    }

    debug!(?channels, ?buffer_storage, "RCT-adjusted channels");

    // Allocate all the coded channels if they aren't yet.
    for (buf, _) in channels.iter() {
        buffer_storage[*buf].allocate_if_needed()?;
    }

    debug!(?channels, ?buffer_storage, "allocated buffers");

    // Extract references to to-be-decoded buffers.
    let mut coded_buffers = Vec::with_capacity(channels.len());
    let mut buffer_tail = &mut buffer_storage[..];
    let mut last_buffer = None;
    for (buf, _) in channels {
        let offset = if let Some(lb) = last_buffer {
            buf.checked_sub(lb).unwrap()
        } else {
            buf + 1
        };
        let cur_buf;
        (cur_buf, buffer_tail) = buffer_tail.split_at_mut(offset);
        coded_buffers.push(cur_buf.last_mut().unwrap().borrow_mut());
        last_buffer = Some(buf);
    }

    Ok((coded_buffers, transform_steps))
}

impl TransformStep {
    // Marks that one dependency of this transform is ready, and potentially runs the transform,
    // returning the new buffers that are now ready.
    pub fn local_apply(&self, buffers: &mut [LocalTransformBuffer]) -> Result<()> {
        match self {
            TransformStep::Rct {
                buf_in,
                buf_out,
                op,
                perm,
            } => {
                for i in 0..3 {
                    assert_eq!(
                        buffers[buf_in[i]].channel_info(),
                        buffers[buf_out[i]].channel_info()
                    );
                }
                let [mut a, mut b, mut c] = [
                    buffers[buf_in[0]].take(),
                    buffers[buf_in[1]].take(),
                    buffers[buf_in[2]].take(),
                ];
                {
                    let mut bufs = [a.borrow_mut(), b.borrow_mut(), c.borrow_mut()];
                    super::rct::do_rct_step(&mut bufs, *op, *perm);
                }
                buffers[buf_out[0]] = a;
                buffers[buf_out[1]] = b;
                buffers[buf_out[2]] = c;
            }
            TransformStep::Palette {
                buf_in,
                buf_pal,
                buf_out,
                num_colors,
                num_deltas,
                predictor,
                wp_header,
            } if *predictor != Predictor::Weighted && *predictor != Predictor::AverageAll => {
                for b in buf_out.iter() {
                    assert_eq!(
                        buffers[*b].channel_info().size,
                        buffers[*buf_in].channel_info().size
                    );
                    buffers[*b].allocate_if_needed()?;
                }
                let mut img_in = buffers[*buf_in].take();
                let mut img_pal = buffers[*buf_pal].take();
                let mut out_bufs: Vec<_> = buf_out.iter().map(|x| buffers[*x].take()).collect();
                {
                    let mut bufs: Vec<_> = out_bufs.iter_mut().map(|x| x.borrow_mut()).collect();
                    super::palette::do_palette_step_general(
                        img_in.borrow_mut(),
                        img_pal.borrow_mut(),
                        &mut bufs,
                        *num_colors,
                        *num_deltas,
                        *predictor,
                        wp_header,
                    );
                }
                for (pos, buf) in buf_out.iter().zip(out_bufs.into_iter()) {
                    buffers[*pos] = buf;
                }
            }
            _ => {
                todo!()
            }
        };

        Ok(())
    }
}
