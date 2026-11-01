// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::ops::Range;

use crate::error::Result;
use crate::image::Rect;
use crate::render::LowMemoryRenderPipeline;
use crate::render::buffer_splitter::BufferSplitter;
use crate::render::internal::{ChannelInfo, Stage};
use crate::util::ShiftRightCeil;
use crate::util::tracing_wrappers::*;

// Finds a small set of rectangles that cover all the "true" values in `ready_mask`,
// and calls `f` on each such rectangle.
fn foreach_ready_rect(
    ready_mask: [bool; 9],
    mut f: impl FnMut(Range<u8>, Range<u8>) -> Result<()>,
) -> Result<()> {
    // x range in middle row
    let xrange = (1 - ready_mask[3] as u8)..(2 + ready_mask[5] as u8);
    let can_extend_top = xrange.clone().all(|x| ready_mask[x as usize]);
    let can_extend_bottom = xrange.clone().all(|x| ready_mask[6 + x as usize]);
    let yrange = (1 - can_extend_top as u8)..(2 + can_extend_bottom as u8);
    f(xrange.clone(), yrange)?;

    if !can_extend_top {
        if ready_mask[1] {
            let xrange = (1 - ready_mask[0] as u8)..(2 + ready_mask[2] as u8);
            f(xrange, 0..1)?;
        } else {
            if ready_mask[0] {
                f(0..1, 0..1)?;
            }
            if ready_mask[2] {
                f(2..3, 0..1)?;
            }
        }
    } else {
        if ready_mask[0] && !xrange.contains(&0) {
            f(0..1, 0..1)?;
        }
        if ready_mask[2] && !xrange.contains(&2) {
            f(2..3, 0..1)?;
        }
    }

    if !can_extend_bottom {
        if ready_mask[7] {
            let xrange = (1 - ready_mask[6] as u8)..(2 + ready_mask[8] as u8);
            f(xrange, 2..3)?;
        } else {
            if ready_mask[6] {
                f(0..1, 2..3)?;
            }
            if ready_mask[8] {
                f(2..3, 2..3)?;
            }
        }
    } else {
        if ready_mask[6] && !xrange.contains(&0) {
            f(0..1, 2..3)?;
        }
        if ready_mask[8] && !xrange.contains(&2) {
            f(2..3, 2..3)?;
        }
    }

    Ok(())
}

fn boundary_points(
    origin: usize,
    end: usize,
    is_first: bool,
    is_last: bool,
    border: usize,
    input_size: usize,
    alignment_shift: usize,
) -> [usize; 4] {
    let alignment = 1usize << alignment_shift;
    debug_assert!(border.is_multiple_of(alignment));
    debug_assert!(origin.is_multiple_of(alignment));

    let p0 = if is_first {
        origin
    } else {
        origin.saturating_sub(border)
    };
    let p1 = (origin + border).min(end);
    // `end` might not be aligned.
    let p2 = end
        .saturating_sub(border)
        .next_multiple_of(alignment)
        .min(end)
        .max(p1);
    let p3 = if is_last {
        end
    } else {
        (end + border).min(input_size)
    };
    [p0, p1, p2, p3]
}

#[allow(clippy::too_many_arguments)]
fn ready_image_area(
    group_rect: Rect,
    group_position: (usize, usize),
    group_count: (usize, usize),
    input_size: (usize, usize),
    border_size: (usize, usize),
    downsample_align: (usize, usize),
    xrange: Range<u8>,
    yrange: Range<u8>,
) -> Option<Rect> {
    let (gx, gy) = group_position;
    let xs = boundary_points(
        group_rect.origin.0,
        group_rect.end().0,
        gx == 0,
        gx + 1 == group_count.0,
        border_size.0,
        input_size.0,
        downsample_align.0,
    );
    let ys = boundary_points(
        group_rect.origin.1,
        group_rect.end().1,
        gy == 0,
        gy + 1 == group_count.1,
        border_size.1,
        input_size.1,
        downsample_align.1,
    );

    let x0 = xs[xrange.start as usize];
    let x1 = xs[xrange.end as usize];
    let y0 = ys[yrange.start as usize];
    let y1 = ys[yrange.end as usize];

    (x1 > x0 && y1 > y0).then(|| Rect {
        origin: (x0, y0),
        size: (x1 - x0, y1 - y0),
    })
}

impl LowMemoryRenderPipeline {
    pub(super) fn render_with_new_group(
        &self,
        g: usize,
        buffer_splitter: &BufferSplitter,
    ) -> Result<()> {
        let buf = self.input_buffers.get(g);
        let (gx, gy) = self.shared.group_position(g);
        debug!("new data ready for group {gx},{gy}");

        // Prepare output buffers for the group.
        let (origin, size) = if let Some(e) = self.shared.extend_stage_index {
            let (dx, dy) = self.downsampling_for_stage[e];
            let Stage::Extend(e) = &self.shared.stages[e] else {
                unreachable!("extend stage is not an extend stage");
            };
            // BufferSplitter accepts virtual coordinates, while Extend works
            // in the resolution at its position in the pipeline.
            (
                (e.frame_origin.0 << dx, e.frame_origin.1 << dy),
                (e.image_size.0 << dx, e.image_size.1 << dy),
            )
        } else {
            ((0, 0), self.shared.input_size)
        };
        let gsz = 1 << self.shared.log_group_size;
        let group_rect = Rect {
            size: (gsz, gsz),
            origin: (gsz * gx, gsz * gy),
        }
        .clip(self.shared.input_size);

        if self.border_size != (0, 0) {
            for c in 0..self.shared.num_channels() {
                if !self.shared.channel_is_used[c] {
                    continue;
                }
                let (bx, by) = self.border_size;
                let ChannelInfo {
                    ty,
                    downsample: (dx, dy),
                } = self.shared.channel_info[0][c];
                let bx = bx >> dx;
                let by = by >> dy;
                if bx == 0 && by == 0 {
                    continue;
                }
                let data = buf.data[c].try_read().unwrap();
                let input = data.as_ref().unwrap();

                // Crop input buffers to the actual group size.
                let (gw, gh) = self.shared.group_size(g);
                let (sx, sy) = input.byte_size();
                let sx = sx.min(gw.shrc(dx) * ty.size());
                let sy = sy.min(gh.shrc(dy));

                if by != 0 {
                    let mut topbottom =
                        if let Some(b) = buf.topbottom[c].try_write().unwrap().take() {
                            b
                        } else {
                            let height = 4 * by;
                            let width = (1 << self.shared.log_group_size) * ty.size();
                            self.shared
                                .buffer_recycler
                                .get_raw_buffer((width, height), false)?
                        };
                    for y in 0..(2 * by).min(sy) {
                        topbottom.row_mut(y)[..sx].copy_from_slice(&input.row(y)[..sx]);
                        topbottom.row_mut(4 * by - 1 - y)[..sx]
                            .copy_from_slice(&input.row(sy - y - 1)[..sx]);
                    }
                    *buf.topbottom[c].try_write().unwrap() = Some(topbottom);
                }
                if bx != 0 {
                    let mut leftright =
                        if let Some(b) = buf.leftright[c].try_write().unwrap().take() {
                            b
                        } else {
                            let height = 1 << self.shared.log_group_size;
                            let width = 4 * bx * ty.size();
                            self.shared
                                .buffer_recycler
                                .get_raw_buffer((width, height), false)?
                        };
                    let cs = (bx * 2 * ty.size()).min(sx);
                    for y in 0..sy {
                        let row_out = leftright.row_mut(y);
                        let row_in = input.row(y);
                        row_out[..cs].copy_from_slice(&row_in[..cs]);
                        row_out[4 * bx * ty.size() - cs..].copy_from_slice(&row_in[sx - cs..sx]);
                    }
                    *buf.leftright[c].try_write().unwrap() = Some(leftright);
                }
            }
        }

        let ready_mask = self.input_buffers.mark_ready(g);

        let mut data = self.per_thread_data.get();
        data.ensure_populated(self)?;

        foreach_ready_rect(ready_mask, |xrange, yrange| {
            let Some(image_area) = ready_image_area(
                group_rect,
                (gx, gy),
                self.shared.group_count,
                self.shared.input_size,
                self.border_size,
                self.downsample_align,
                xrange,
                yrange,
            ) else {
                return Ok(());
            };

            let mut local_buffers = buffer_splitter.get_local_buffers(
                &self.save_buffer_info,
                image_area,
                false,
                self.shared.input_size,
                size,
                origin,
            );

            self.render_group(&mut data, (gx, gy), image_area, &mut local_buffers)?;
            Ok(())
        })?;

        self.input_buffers.mark_done(g, &self.shared, |image| {
            self.shared.buffer_recycler.recycle_raw_buffer(image);
        });

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_foreach_ready_rect() {
        for i in 0..512 {
            let mut ready_mask = [false; 9];
            for j in 0..9 {
                if (i >> j) & 1 == 1 {
                    ready_mask[j] = true;
                }
            }
            if !ready_mask[4] {
                continue;
            }

            let mut covered = [false; 9];
            foreach_ready_rect(ready_mask, |xr, yr| {
                for y in yr {
                    for x in xr.clone() {
                        let idx = (y as usize) * 3 + (x as usize);
                        assert!(
                            ready_mask[idx],
                            "Covered not ready index {} in mask {:?} (x={}, y={})",
                            idx, ready_mask, x, y
                        );
                        assert!(
                            !covered[idx],
                            "Double coverage of index {} in mask {:?}",
                            idx, ready_mask
                        );
                        covered[idx] = true;
                    }
                }
                Ok(())
            })
            .unwrap();

            for j in 0..9 {
                if ready_mask[j] {
                    assert!(
                        covered[j],
                        "Failed to cover index {} in mask {:?}",
                        j, ready_mask
                    );
                }
            }
        }
    }

    #[test]
    fn test_ready_image_area_clips_tiny_edge_group() {
        let area = ready_image_area(
            Rect {
                origin: (512, 512),
                size: (8, 8),
            },
            (1, 1),
            (2, 2),
            (520, 520),
            (18, 18),
            (0, 0),
            0..1,
            0..1,
        );
        assert_eq!(
            area,
            Some(Rect {
                origin: (494, 494),
                size: (26, 26),
            })
        );
    }

    // The last group of a 513-pixel-wide image is 1 pixel wide, narrower than
    // the border, so its interior band starts past the right edge of the image.
    #[test]
    fn test_ready_image_area_interior_band_past_image_edge() {
        let area = ready_image_area(
            Rect {
                origin: (512, 0),
                size: (1, 256),
            },
            (2, 0),
            (3, 2),
            (513, 321),
            (3, 3),
            (0, 0),
            1..3,
            0..2,
        );
        assert_eq!(area, None);
    }
}
