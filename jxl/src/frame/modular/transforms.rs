// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::cell::RefCell;

use apply::TransformStep;
pub use apply::TransformStepChunk;
use num_derive::FromPrimitive;
use num_traits::FromPrimitive;

use crate::error::Error;
use crate::frame::modular::ModularBuffer;
use crate::headers::frame_header::FrameHeader;
use crate::util::tracing_wrappers::*;
use crate::{
    error::Result,
    headers::{self, modular::TransformId},
};

use super::{ChannelInfo, ModularBufferInfo, ModularGridKind, Predictor};

pub(super) mod apply;
mod palette;
mod rct;
mod squeeze;

#[derive(Debug, FromPrimitive, PartialEq, Clone, Copy)]
pub enum RctPermutation {
    Rgb = 0,
    Gbr = 1,
    Brg = 2,
    Rbg = 3,
    Grb = 4,
    Bgr = 5,
}

#[derive(Debug, FromPrimitive, PartialEq, Clone, Copy)]
pub enum RctOp {
    Noop = 0,
    AddFirstToThird = 1,
    AddFirstToSecond = 2,
    AddFirstToSecondAndThird = 3,
    AddAvgToSecond = 4,
    AddFirstToThirdAndAvgToSecond = 5,
    YCoCg = 6,
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

#[instrument(level = "trace", ret)]
pub fn meta_apply_transforms(
    channels: &[ChannelInfo],
    transforms: &[headers::modular::Transform],
) -> Result<(Vec<ModularBufferInfo>, Vec<TransformStep>)> {
    let mut buffer_info = vec![];
    let mut transform_steps = vec![];
    // (buffer id, channel info)
    let mut channels: Vec<_> = channels.iter().cloned().enumerate().collect();

    // First, add all the pre-transform channels to the buffer list.
    for chan in channels.iter() {
        buffer_info.push(ModularBufferInfo {
            info: chan.1.clone(),
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

    let add_transform_buffer = |buffer_info: &mut Vec<ModularBufferInfo>, info, description| {
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
    for transform in transforms {
        match transform.id {
            TransformId::Rct => {
                let begin_channel = transform.begin_channel as usize;
                let op = RctOp::from_u32(transform.rct_type % 7).unwrap();
                let perm = RctPermutation::from_u32(transform.rct_type / 7)
                    .expect("header decoding should ensure rct_type < 42");
                check_equal_channels(&channels, begin_channel, 3)?;
                let mut buf_in = [0; 3];
                let buf_out = [
                    channels[begin_channel].0,
                    channels[begin_channel + 1].0,
                    channels[begin_channel + 2].0,
                ];
                for i in 0..3 {
                    let c = &mut channels[begin_channel + i];
                    c.0 = add_transform_buffer(
                        &mut buffer_info,
                        c.1.clone(),
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
                    squeeze::default_squeeze(&channels)
                } else {
                    transform.squeezes.clone()
                };
                for step in steps {
                    squeeze::check_squeeze_params(&channels, &step)?;
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
                        };
                        let buf_0 = add_transform_buffer(
                            &mut buffer_info,
                            new_0.clone(),
                            format!("Squeezed channel, original channel {}", begin_channel + ic),
                        );
                        let new_1 = ChannelInfo {
                            shift: new_shift,
                            size: new_size_1,
                        };
                        let buf_1 = add_transform_buffer(
                            &mut buffer_info,
                            new_1.clone(),
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
                check_equal_channels(&channels, begin_channel, num_channels)?;
                let pchan_info = ChannelInfo {
                    shift: None,
                    size: (num_colors + num_deltas, num_channels),
                };
                let pchan = add_transform_buffer(
                    &mut buffer_info,
                    pchan_info.clone(),
                    format!(
                        "Palette for palette transform starting at channel {} with {} channels",
                        begin_channel, num_channels
                    ),
                );
                let inchan = add_transform_buffer(
                    &mut buffer_info,
                    channels[begin_channel].1.clone(),
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
    }

    // All the channels left over at the end of applying transforms are the channels that are
    // actually coded.
    for (chid, chan) in channels.iter().enumerate() {
        buffer_info[chan.0].is_coded = true;
        buffer_info[chan.0].channel_id = chid;
    }

    Ok((buffer_info, transform_steps))
}

#[instrument(level = "trace", skip_all, ret)]
pub fn make_grids(
    frame_header: &FrameHeader,
    transform_steps: Vec<TransformStep>,
    section_buffer_indices: &[Vec<usize>],
    buffer_info: &mut Vec<ModularBufferInfo>,
) -> Vec<TransformStepChunk> {
    // Initialize grid sizes, starting from coded channels.
    for i in section_buffer_indices[1].iter() {
        buffer_info[*i].grid_kind = ModularGridKind::Lf;
    }
    for buffer_indices in section_buffer_indices.iter().skip(2) {
        for i in buffer_indices.iter() {
            buffer_info[*i].grid_kind = ModularGridKind::Hf;
        }
    }

    trace!(?buffer_info, "post set grid kind for coded channels");

    // Transforms can be un-applied in the opposite order they appear with in the array,
    // so we can use that information to propagate grid kinds.

    for step in transform_steps.iter().rev() {
        match step {
            TransformStep::Rct {
                buf_in, buf_out, ..
            } => {
                let grid_in = buffer_info[buf_in[0]].grid_kind;
                for i in 0..3 {
                    assert_eq!(grid_in, buffer_info[buf_in[i]].grid_kind);
                }
                for i in 0..3 {
                    buffer_info[buf_out[i]].grid_kind = grid_in;
                }
            }
            TransformStep::Palette {
                buf_in, buf_out, ..
            } => {
                for buf in buf_out.iter() {
                    buffer_info[*buf].grid_kind = buffer_info[*buf_in].grid_kind;
                }
            }
            TransformStep::HSqueeze { buf_in, buf_out }
            | TransformStep::VSqueeze { buf_in, buf_out } => {
                buffer_info[*buf_out].grid_kind = buffer_info[buf_in[0]]
                    .grid_kind
                    .max(buffer_info[buf_in[1]].grid_kind);
            }
        }
    }

    // Set grid shapes.
    for buf in buffer_info.iter_mut() {
        buf.grid_shape = buf.grid_kind.grid_shape(frame_header);
    }

    trace!(?buffer_info, "post propagate grid kind");

    let get_grid_indices = |shape: (usize, usize)| {
        (0..shape.1).flat_map(move |y| (0..shape.0).map(move |x| (x as isize, y as isize)))
    };

    // Create grids.
    for g in buffer_info.iter_mut() {
        let is_output = g.is_output;
        g.buffer_grid = get_grid_indices(g.grid_shape)
            .map(|(x, y)| {
                let chan_size = g.info.size;
                let size = match g.grid_kind {
                    ModularGridKind::None => chan_size,
                    ModularGridKind::Lf | ModularGridKind::Hf => {
                        let group_dim = if g.grid_kind == ModularGridKind::Lf {
                            frame_header.lf_group_dim()
                        } else {
                            frame_header.group_dim()
                        };
                        let dx = group_dim >> g.info.shift.unwrap().0;
                        let bx = x as usize * dx;
                        let dy = group_dim >> g.info.shift.unwrap().1;
                        let by = y as usize * dy;
                        ((chan_size.0 - bx).min(dx), (chan_size.1 - by).min(dy))
                    }
                };
                ModularBuffer {
                    data: RefCell::new(None),
                    remaining_uses: if is_output { 1 } else { 0 },
                    used_by_transforms: vec![],
                    size,
                }
            })
            .collect();
    }

    trace!(?buffer_info, "with grids");

    let add_transform_step =
        |transform: &TransformStep,
         grid_pos: (isize, isize),
         grid_transform_steps: &mut Vec<TransformStepChunk>| {
            let ts = grid_transform_steps.len();
            grid_transform_steps.push(TransformStepChunk {
                step: transform.clone(),
                grid_pos: (grid_pos.0 as usize, grid_pos.1 as usize),
                incomplete_deps: 0,
            });
            ts
        };

    let add_grid_use = |ts: usize,
                        input_grid_idx: usize,
                        output_grid_kind: ModularGridKind,
                        input_grid_pos_out_kind: (isize, isize),
                        grid_transform_steps: &mut Vec<TransformStepChunk>,
                        buffer_info: &mut Vec<ModularBufferInfo>| {
        let input_grid_pos = match (output_grid_kind, buffer_info[input_grid_idx].grid_kind) {
            (_, ModularGridKind::None) => (0, 0),
            (ModularGridKind::Lf, ModularGridKind::Lf)
            | (ModularGridKind::Hf, ModularGridKind::Hf) => input_grid_pos_out_kind,
            (ModularGridKind::Hf, ModularGridKind::Lf) => {
                (input_grid_pos_out_kind.0 / 8, input_grid_pos_out_kind.1 / 8)
            }
            _ => unreachable!("invalid combination of output grid kind and buffer grid kind"),
        };
        let input_grid_size = buffer_info[input_grid_idx]
            .grid_kind
            .grid_shape(frame_header);
        let input_grid_size = (input_grid_size.0 as isize, input_grid_size.1 as isize);
        if input_grid_pos.0 < 0
            || input_grid_pos.0 >= input_grid_size.0
            || input_grid_pos.1 < 0
            || input_grid_pos.1 >= input_grid_size.1
        {
            // Skip adding uses of non-existent grid positions.
            return;
        }
        let grid_pos = (input_grid_size.0 * input_grid_pos.1 + input_grid_pos.0) as usize;
        buffer_info[input_grid_idx].buffer_grid[grid_pos].remaining_uses += 1;
        buffer_info[input_grid_idx].buffer_grid[grid_pos]
            .used_by_transforms
            .push(ts);
        grid_transform_steps[ts].incomplete_deps += 1;
    };

    // Add grid-ed transforms.
    let mut grid_transform_steps = vec![];

    for transform in transform_steps {
        match &transform {
            TransformStep::Rct {
                buf_in, buf_out, ..
            } => {
                // Easy case: we just depend on the 3 input buffers in the same location.
                let out_kind = buffer_info[buf_out[0]].grid_kind;
                let out_shape = buffer_info[buf_out[0]].grid_shape;
                for (x, y) in get_grid_indices(out_shape) {
                    let ts = add_transform_step(&transform, (x, y), &mut grid_transform_steps);
                    for bin in buf_in {
                        add_grid_use(
                            ts,
                            *bin,
                            out_kind,
                            (x, y),
                            &mut grid_transform_steps,
                            buffer_info,
                        );
                    }
                }
            }
            TransformStep::Palette {
                buf_in,
                buf_pal,
                buf_out,
                predictor: Predictor::AverageAll | Predictor::Weighted,
                ..
            } => {
                // Delta palette with AverageAll or Weighted. Those are special, because we can
                // only make progress one full image row at a time (since we need decoded values
                // from the previous row or two rows).
                let out_kind = buffer_info[buf_out[0]].grid_kind;
                let out_shape = buffer_info[buf_out[0]].grid_shape;
                let mut ts = 0;
                for (x, y) in get_grid_indices(out_shape) {
                    if x == 0 {
                        ts = add_transform_step(&transform, (x, y), &mut grid_transform_steps);
                        add_grid_use(
                            ts,
                            *buf_pal,
                            out_kind,
                            (x, y),
                            &mut grid_transform_steps,
                            buffer_info,
                        );
                    }
                    add_grid_use(
                        ts,
                        *buf_in,
                        out_kind,
                        (x, y),
                        &mut grid_transform_steps,
                        buffer_info,
                    );
                    for out in buf_out.iter() {
                        add_grid_use(
                            ts,
                            *out,
                            out_kind,
                            (x, y - 1),
                            &mut grid_transform_steps,
                            buffer_info,
                        );
                    }
                }
            }
            TransformStep::Palette {
                buf_in,
                buf_pal,
                buf_out,
                predictor,
                ..
            } => {
                // Maybe-delta palette: we depend on the palette and the input buffer in the same
                // location. We may also depend on other grid positions in the output buffer,
                // according to the used predictor.
                let out_kind = buffer_info[buf_out[0]].grid_kind;
                let out_shape = buffer_info[buf_out[0]].grid_shape;
                for (x, y) in get_grid_indices(out_shape) {
                    let ts = add_transform_step(&transform, (x, y), &mut grid_transform_steps);
                    add_grid_use(
                        ts,
                        *buf_pal,
                        out_kind,
                        (x, y),
                        &mut grid_transform_steps,
                        buffer_info,
                    );
                    add_grid_use(
                        ts,
                        *buf_in,
                        out_kind,
                        (x, y),
                        &mut grid_transform_steps,
                        buffer_info,
                    );
                    let offsets = match predictor {
                        Predictor::Zero => [].as_slice(),
                        Predictor::West | Predictor::WestWest => &[(-1, 0)],
                        Predictor::North => &[(0, -1)],
                        Predictor::Select
                        | Predictor::Gradient
                        | Predictor::NorthWest
                        | Predictor::AverageWestAndNorth
                        | Predictor::AverageWestAndNorthWest
                        | Predictor::AverageNorthAndNorthWest => &[(0, -1), (-1, 0), (-1, -1)],
                        Predictor::NorthEast | Predictor::AverageNorthAndNorthEast => {
                            &[(0, -1), (1, 0), (1, -1)]
                        }
                        Predictor::AverageAll | Predictor::Weighted => unreachable!(),
                    };
                    for (dx, dy) in offsets {
                        for out in buf_out.iter() {
                            add_grid_use(
                                ts,
                                *out,
                                out_kind,
                                (x + dx, y + dy),
                                &mut grid_transform_steps,
                                buffer_info,
                            );
                        }
                    }
                }
            }
            TransformStep::HSqueeze { buf_in, buf_out } => {
                let out_kind = buffer_info[*buf_out].grid_kind;
                let out_shape = buffer_info[*buf_out].grid_shape;
                for (x, y) in get_grid_indices(out_shape) {
                    let ts = add_transform_step(&transform, (x, y), &mut grid_transform_steps);
                    // Average and residuals from the same position
                    for bin in buf_in {
                        add_grid_use(
                            ts,
                            *bin,
                            out_kind,
                            (x, y),
                            &mut grid_transform_steps,
                            buffer_info,
                        );
                    }
                    // Next average
                    add_grid_use(
                        ts,
                        buf_in[0],
                        out_kind,
                        (x + 1, y),
                        &mut grid_transform_steps,
                        buffer_info,
                    );
                    // Previous decoded
                    add_grid_use(
                        ts,
                        *buf_out,
                        out_kind,
                        (x - 1, y),
                        &mut grid_transform_steps,
                        buffer_info,
                    );
                }
            }
            TransformStep::VSqueeze { buf_in, buf_out } => {
                let out_kind = buffer_info[*buf_out].grid_kind;
                let out_shape = buffer_info[*buf_out].grid_shape;
                for (x, y) in get_grid_indices(out_shape) {
                    let ts = add_transform_step(&transform, (x, y), &mut grid_transform_steps);
                    // Average and residuals from the same position
                    for bin in buf_in {
                        add_grid_use(
                            ts,
                            *bin,
                            out_kind,
                            (x, y),
                            &mut grid_transform_steps,
                            buffer_info,
                        );
                    }
                    // Next average
                    add_grid_use(
                        ts,
                        buf_in[0],
                        out_kind,
                        (x, y + 1),
                        &mut grid_transform_steps,
                        buffer_info,
                    );
                    // Previous decoded
                    add_grid_use(
                        ts,
                        *buf_out,
                        out_kind,
                        (x, y - 1),
                        &mut grid_transform_steps,
                        buffer_info,
                    );
                }
            }
        }
    }

    trace!(?grid_transform_steps, ?buffer_info);

    grid_transform_steps
}
