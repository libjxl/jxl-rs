// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::api::JxlParallelRunner;
use crate::error::Result;
use crate::headers::frame_header::FrameHeader;
use crate::image::{Image, Rect};
use crate::render::buffer_splitter::OutputChannelSplitter;
use num_traits::abs;

#[allow(clippy::excessive_precision)]
const W_SIDE: f32 = 0.20345139757231578;
#[allow(clippy::excessive_precision)]
const W_CORNER: f32 = 0.0334829185968739;
const W_CENTER: f32 = 1.0 - 4.0 * (W_SIDE + W_CORNER);

fn chroma_upsample_lf(
    frame_header: &FrameHeader,
    lf_image: &mut Image<f32>,
    upsample_h: bool,
    upsample_v: bool,
) -> Result<()> {
    if !upsample_h && !upsample_v {
        return Ok(());
    }

    let (xsize, ysize) = lf_image.size();
    let group_dim = frame_header.group_dim();
    let num_lf_groups = frame_header.num_lf_groups();
    let mut upsampled = Image::<f32>::new((xsize, ysize))?;

    if upsample_h {
        for g in 0..num_lf_groups {
            let mut r = frame_header.lf_group_rect(g);
            if upsample_v {
                r.size.1 = r.size.1.div_ceil(2);
            }
            let r_right = if r.origin.0 + group_dim < xsize {
                Some(Rect {
                    origin: (r.origin.0 + group_dim, r.origin.1),
                    size: (1, r.size.1),
                })
            } else {
                None
            };
            let r_left = if r.origin.0 > (group_dim >> 1) {
                Some(Rect {
                    origin: (r.origin.0 - (group_dim >> 1) - 1, r.origin.1),
                    size: (1, r.size.1),
                })
            } else {
                None
            };
            let sw = r.size.0.div_ceil(2);

            let view = lf_image.get_rect(r);
            let view_r = r_right.map(|r| lf_image.get_rect(r));
            let view_l = r_left.map(|r| lf_image.get_rect(r));
            let mut out = upsampled.get_rect_mut(r);

            for y in 0..r.size.1 {
                let row = view.row(y);
                let row_out = out.row(y);
                let rightmost_sample = view_r.map(|v| v.row(y)[0]).unwrap_or(row[sw - 1]);
                let leftmost_sample = view_l.map(|v| v.row(y)[0]).unwrap_or(row[0]);

                for sx in 0..sw {
                    let x = sx << 1;
                    let a = if sx == 0 {
                        leftmost_sample
                    } else {
                        row[sx - 1]
                    };
                    let b = row[sx];
                    let c = if sx == sw - 1 {
                        rightmost_sample
                    } else {
                        row[sx + 1]
                    };

                    row_out[x] = 0.25 * a + 0.75 * b;
                    if let Some(out) = row_out.get_mut(x + 1) {
                        *out = 0.75 * b + 0.25 * c;
                    }
                }
            }
        }
        std::mem::swap(lf_image, &mut upsampled);
    }

    if upsample_v {
        for g in 0..num_lf_groups {
            let r = frame_header.lf_group_rect(g);
            let r_bottom = if r.origin.1 + group_dim < ysize {
                Some(Rect {
                    origin: (r.origin.0, r.origin.1 + group_dim),
                    size: (r.size.0, 1),
                })
            } else {
                None
            };
            let r_top = if r.origin.1 > (group_dim >> 1) {
                Some(Rect {
                    origin: (r.origin.0, r.origin.1 - (group_dim >> 1) - 1),
                    size: (r.size.0, 1),
                })
            } else {
                None
            };
            let sh = r.size.1.div_ceil(2);

            let view = lf_image.get_rect(r);
            let view_b = r_bottom.map(|r| lf_image.get_rect(r));
            let view_t = r_top.map(|r| lf_image.get_rect(r));
            let mut out = upsampled.get_rect_mut(r);

            for sy in 0..sh {
                let y = sy << 1;
                let row_b = view.row(sy);
                let row_a = if sy == 0 {
                    view_t.map(|v| v.row(0)).unwrap_or(row_b)
                } else {
                    view.row(sy - 1)
                };
                let row_c = if sy == sh - 1 {
                    view_b.map(|v| v.row(0)).unwrap_or(row_b)
                } else {
                    view.row(sy + 1)
                };

                let row_out = out.row(y);
                for x in 0..r.size.0 {
                    row_out[x] = 0.25 * row_a[x] + 0.75 * row_b[x];
                }
                if out.size().1 > y + 1 {
                    let row_out_b = out.row(y + 1);
                    for x in 0..r.size.0 {
                        row_out_b[x] = 0.75 * row_b[x] + 0.25 * row_c[x];
                    }
                }
            }
        }
        std::mem::swap(lf_image, &mut upsampled);
    }

    Ok(())
}

fn compute_pixel_channel(
    dc_factor: f32,
    gap: f32,
    x: usize,
    row_top: &[f32],
    row: &[f32],
    row_bottom: &[f32],
) -> (f32, f32, f32) {
    let tl = row_top[x - 1];
    let tc = row_top[x];
    let tr = row_top[x + 1];
    let ml = row[x - 1];
    let mc = row[x];
    let mr = row[x + 1];
    let bl = row_bottom[x - 1];
    let bc = row_bottom[x];
    let br = row_bottom[x + 1];
    let corner = tl + tr + bl + br;
    let side = ml + mr + tc + bc;
    let sm = corner * W_CORNER + side * W_SIDE + mc * W_CENTER;
    (mc, sm, gap.max(abs((mc - sm) / dc_factor)))
}

// TODO(veluca): consider SIMDfying this.
pub fn adaptive_lf_smoothing(
    lf_factors: [f32; 3],
    frame_header: &FrameHeader,
    lf_image: &mut [Image<f32>; 3],
    parallel_runner: &mut dyn JxlParallelRunner,
) -> Result<()> {
    let (xsize, ysize) = lf_image[0].size();
    if ysize <= 2 || xsize <= 2 {
        return Ok(());
    }

    let shifts: [_; 3] =
        std::array::from_fn(|i| (frame_header.hshift(i) as u8, frame_header.vshift(i) as u8));

    for ch in 0..3 {
        chroma_upsample_lf(
            frame_header,
            &mut lf_image[ch],
            shifts[ch].0 != 0,
            shifts[ch].1 != 0,
        )?;
    }

    let mut smoothed0 = Image::<f32>::new((xsize, ysize))?;
    let mut smoothed1 = Image::<f32>::new((xsize, ysize))?;
    let mut smoothed2 = Image::<f32>::new((xsize, ysize))?;

    let splitter0 = OutputChannelSplitter::from_image(&mut smoothed0);
    let splitter1 = OutputChannelSplitter::from_image(&mut smoothed1);
    let splitter2 = OutputChannelSplitter::from_image(&mut smoothed2);

    let num_lf_groups = frame_header.num_lf_groups();

    parallel_runner.run(num_lf_groups, &|g| {
        let r = frame_header.lf_group_rect(g);
        let mut out_ref_0 = splitter0.borrow_typed_rect::<f32>(r);
        let mut out_ref_1 = splitter1.borrow_typed_rect::<f32>(r);
        let mut out_ref_2 = splitter2.borrow_typed_rect::<f32>(r);

        for ly in 0..r.size.1 {
            let gy = r.origin.1 + ly;
            let sly = shifts.map(|(_, vshift)| ly >> vshift);
            let mut row_0 =
                (sly[0] << shifts[0].1 == ly).then(|| out_ref_0.typed_row_mut::<f32>(sly[0]));
            let mut row_1 =
                (sly[1] << shifts[1].1 == ly).then(|| out_ref_1.typed_row_mut::<f32>(sly[1]));
            let mut row_2 =
                (sly[2] << shifts[2].1 == ly).then(|| out_ref_2.typed_row_mut::<f32>(sly[2]));

            for lx in 0..r.size.0 {
                let gx = r.origin.0 + lx;
                let slx = shifts.map(|(hshift, _)| lx >> hshift);

                if gy == 0 || gy == ysize - 1 || gx == 0 || gx == xsize - 1 {
                    if let Some(row_0) = &mut row_0 {
                        row_0[slx[0]] = lf_image[0].row(gy)[gx];
                    }
                    if let Some(row_1) = &mut row_1 {
                        row_1[slx[1]] = lf_image[1].row(gy)[gx];
                    }
                    if let Some(row_2) = &mut row_2 {
                        row_2[slx[2]] = lf_image[2].row(gy)[gx];
                    }
                    continue;
                }

                let mut row_0 = row_0
                    .as_mut()
                    .and_then(|r| (slx[0] << shifts[0].0 == lx).then_some(&mut **r));
                let mut row_1 = row_1
                    .as_mut()
                    .and_then(|r| (slx[1] << shifts[1].0 == lx).then_some(&mut **r));
                let mut row_2 = row_2
                    .as_mut()
                    .and_then(|r| (slx[2] << shifts[2].0 == lx).then_some(&mut **r));

                let gap = 0.5;
                let (mc_x, sm_x, gap) = compute_pixel_channel(
                    lf_factors[0],
                    gap,
                    gx,
                    lf_image[0].row(gy - 1),
                    lf_image[0].row(gy),
                    lf_image[0].row(gy + 1),
                );
                let (mc_y, sm_y, gap) = compute_pixel_channel(
                    lf_factors[1],
                    gap,
                    gx,
                    lf_image[1].row(gy - 1),
                    lf_image[1].row(gy),
                    lf_image[1].row(gy + 1),
                );
                let (mc_b, sm_b, gap) = compute_pixel_channel(
                    lf_factors[2],
                    gap,
                    gx,
                    lf_image[2].row(gy - 1),
                    lf_image[2].row(gy),
                    lf_image[2].row(gy + 1),
                );
                let factor = (3.0 - 4.0 * gap).max(0.0);
                if let Some(row_0) = &mut row_0 {
                    row_0[slx[0]] = (sm_x - mc_x) * factor + mc_x;
                }
                if let Some(row_1) = &mut row_1 {
                    row_1[slx[1]] = (sm_y - mc_y) * factor + mc_y;
                }
                if let Some(row_2) = &mut row_2 {
                    row_2[slx[2]] = (sm_b - mc_b) * factor + mc_b;
                }
            }
        }
        Ok(())
    })?;

    drop(splitter0);
    drop(splitter1);
    drop(splitter2);
    *lf_image = [smoothed0, smoothed1, smoothed2];
    Ok(())
}
