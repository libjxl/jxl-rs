// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::error::Result;
use crate::headers::frame_header::FrameHeader;
use crate::image::Image;
use crate::util::AtomicRefCell;
use num_traits::abs;

#[allow(clippy::excessive_precision)]
const W_SIDE: f32 = 0.20345139757231578;
#[allow(clippy::excessive_precision)]
const W_CORNER: f32 = 0.0334829185968739;
const W_CENTER: f32 = 1.0 - 4.0 * (W_SIDE + W_CORNER);

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

fn compute_pixel_channel_from_neighborhood(
    dc_factor: f32,
    gap: f32,
    row_top: [f32; 3],
    row_mid: [f32; 3],
    row_bot: [f32; 3],
) -> (f32, f32, f32) {
    let tl = row_top[0];
    let tc = row_top[1];
    let tr = row_top[2];
    let ml = row_mid[0];
    let mc = row_mid[1];
    let mr = row_mid[2];
    let bl = row_bot[0];
    let bc = row_bot[1];
    let br = row_bot[2];
    let corner = tl + tr + bl + br;
    let side = ml + mr + tc + bc;
    let sm = corner * W_CORNER + side * W_SIDE + mc * W_CENTER;
    (mc, sm, gap.max(abs((mc - sm) / dc_factor)))
}

type LfImage = [Image<f32>; 3];

pub fn adaptive_lf_smoothing(
    lf_factors: [f32; 3],
    header: &FrameHeader,
    lf_image: &[AtomicRefCell<Option<LfImage>>],
) -> Result<()> {
    let size_blocks = header.size_blocks();
    let xsize = size_blocks.0;
    let ysize = size_blocks.1;
    if ysize <= 2 || xsize <= 2 {
        return Ok(());
    }

    let group_dim = header.group_dim();
    let num_lf_x = header.size_lf_groups().0;
    let num_lf_y = header.size_lf_groups().1;
    let num_lf_groups = header.num_lf_groups();

    let mut smoothed_chunks: Vec<Option<[Image<f32>; 3]>> = (0..num_lf_groups)
        .map(|g| {
            let cell = lf_image[g].borrow();
            if let Some(img) = cell.as_ref() {
                let sz = img[0].size();
                Ok(Some([
                    Image::<f32>::new(sz)?,
                    Image::<f32>::new(sz)?,
                    Image::<f32>::new(sz)?,
                ]))
            } else {
                Ok(None)
            }
        })
        .collect::<Result<Vec<_>>>()?;

    let get_pixel = |c: usize, x: usize, y: usize| -> f32 {
        let lgx = x / group_dim;
        let lgy = y / group_dim;
        let local_x = x % group_dim;
        let local_y = y % group_dim;
        let lg_idx = lgy * num_lf_x + lgx;
        let guard = lf_image[lg_idx].borrow();
        guard.as_ref().unwrap()[c].row(local_y)[local_x]
    };

    for lgy in 0..num_lf_y {
        for lgx in 0..num_lf_x {
            let lg_idx = lgy * num_lf_x + lgx;
            let guard = lf_image[lg_idx].borrow();
            let img = match guard.as_ref() {
                Some(i) => i,
                None => continue,
            };
            let (w, h) = img[0].size();
            let smoothed = smoothed_chunks[lg_idx].as_mut().unwrap();

            for ly in 0..h {
                let gy = lgy * group_dim + ly;
                for lx in 0..w {
                    let gx = lgx * group_dim + lx;

                    if gy == 0 || gy == ysize - 1 || gx == 0 || gx == xsize - 1 {
                        for c in 0..3 {
                            smoothed[c].row_mut(ly)[lx] = img[c].row(ly)[lx];
                        }
                        continue;
                    }

                    if ly > 0 && ly < h - 1 && lx > 0 && lx < w - 1 {
                        // Fast path: all 9 stencil pixels are inside this chunk!
                        let gap = 0.5;
                        let (mc_x, sm_x, gap) = compute_pixel_channel(
                            lf_factors[0],
                            gap,
                            lx,
                            img[0].row(ly - 1),
                            img[0].row(ly),
                            img[0].row(ly + 1),
                        );
                        let (mc_y, sm_y, gap) = compute_pixel_channel(
                            lf_factors[1],
                            gap,
                            lx,
                            img[1].row(ly - 1),
                            img[1].row(ly),
                            img[1].row(ly + 1),
                        );
                        let (mc_b, sm_b, gap) = compute_pixel_channel(
                            lf_factors[2],
                            gap,
                            lx,
                            img[2].row(ly - 1),
                            img[2].row(ly),
                            img[2].row(ly + 1),
                        );
                        let factor = (3.0 - 4.0 * gap).max(0.0);
                        smoothed[0].row_mut(ly)[lx] = (sm_x - mc_x) * factor + mc_x;
                        smoothed[1].row_mut(ly)[lx] = (sm_y - mc_y) * factor + mc_y;
                        smoothed[2].row_mut(ly)[lx] = (sm_b - mc_b) * factor + mc_b;
                    } else {
                        // Slow path for pixels on LF group chunk boundaries
                        let gap = 0.5;

                        let row_top = [
                            get_pixel(0, gx - 1, gy - 1),
                            get_pixel(0, gx, gy - 1),
                            get_pixel(0, gx + 1, gy - 1),
                        ];
                        let row_mid = [
                            get_pixel(0, gx - 1, gy),
                            get_pixel(0, gx, gy),
                            get_pixel(0, gx + 1, gy),
                        ];
                        let row_bot = [
                            get_pixel(0, gx - 1, gy + 1),
                            get_pixel(0, gx, gy + 1),
                            get_pixel(0, gx + 1, gy + 1),
                        ];
                        let (mc_x, sm_x, gap) = compute_pixel_channel_from_neighborhood(
                            lf_factors[0],
                            gap,
                            row_top,
                            row_mid,
                            row_bot,
                        );

                        let row_top = [
                            get_pixel(1, gx - 1, gy - 1),
                            get_pixel(1, gx, gy - 1),
                            get_pixel(1, gx + 1, gy - 1),
                        ];
                        let row_mid = [
                            get_pixel(1, gx - 1, gy),
                            get_pixel(1, gx, gy),
                            get_pixel(1, gx + 1, gy),
                        ];
                        let row_bot = [
                            get_pixel(1, gx - 1, gy + 1),
                            get_pixel(1, gx, gy + 1),
                            get_pixel(1, gx + 1, gy + 1),
                        ];
                        let (mc_y, sm_y, gap) = compute_pixel_channel_from_neighborhood(
                            lf_factors[1],
                            gap,
                            row_top,
                            row_mid,
                            row_bot,
                        );

                        let row_top = [
                            get_pixel(2, gx - 1, gy - 1),
                            get_pixel(2, gx, gy - 1),
                            get_pixel(2, gx + 1, gy - 1),
                        ];
                        let row_mid = [
                            get_pixel(2, gx - 1, gy),
                            get_pixel(2, gx, gy),
                            get_pixel(2, gx + 1, gy),
                        ];
                        let row_bot = [
                            get_pixel(2, gx - 1, gy + 1),
                            get_pixel(2, gx, gy + 1),
                            get_pixel(2, gx + 1, gy + 1),
                        ];
                        let (mc_b, sm_b, gap) = compute_pixel_channel_from_neighborhood(
                            lf_factors[2],
                            gap,
                            row_top,
                            row_mid,
                            row_bot,
                        );

                        let factor = (3.0 - 4.0 * gap).max(0.0);

                        smoothed[0].row_mut(ly)[lx] = (sm_x - mc_x) * factor + mc_x;
                        smoothed[1].row_mut(ly)[lx] = (sm_y - mc_y) * factor + mc_y;
                        smoothed[2].row_mut(ly)[lx] = (sm_b - mc_b) * factor + mc_b;
                    }
                }
            }
        }
    }

    for (g, smoothed_opt) in smoothed_chunks.into_iter().enumerate() {
        if let Some(smoothed) = smoothed_opt {
            *lf_image[g].borrow_mut().as_mut().unwrap() = smoothed;
        }
    }

    Ok(())
}
