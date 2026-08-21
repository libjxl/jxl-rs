// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use num_traits::abs;

use crate::api::JxlParallelRunner;
use crate::error::Result;
use crate::headers::frame_header::FrameHeader;
use crate::image::Image;
use crate::render::buffer_splitter::OutputChannelSplitter;

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
) -> (f32, f32) {
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
    (sm, gap.max(abs((mc - sm) / dc_factor)))
}

// TODO(veluca): consider SIMDfying this.
pub fn adaptive_lf_smoothing(
    lf_factors: [f32; 3],
    frame_header: &FrameHeader,
    lf_image: &mut [Image<f32>; 3],
    parallel_runner: &mut dyn JxlParallelRunner,
) -> Result<()> {
    let (xsize, ysize) = lf_image[0].size();
    let h_upsample = frame_header.maxhs;
    let v_upsample = frame_header.maxvs;
    let bw = 1usize << h_upsample;
    let bh = 1usize << v_upsample;
    let raw_hs: [_; 3] = std::array::from_fn(|ch| frame_header.raw_hshift(ch));
    let raw_vs: [_; 3] = std::array::from_fn(|ch| frame_header.raw_vshift(ch));

    let sw = xsize >> h_upsample;
    let sh = ysize >> v_upsample;
    assert!(
        (sw << h_upsample) == xsize && (sh << v_upsample) == ysize,
        "LF image size is not multiple of MCU size \
        (xsize={xsize}, ysize={ysize}, bw={bw}, bh={bh})",
    );

    if sh <= 2 || sw <= 2 {
        return Ok(());
    }

    let smoothed0 = Image::<f32>::new((xsize, ysize))?;
    let smoothed1 = Image::<f32>::new((xsize, ysize))?;
    let smoothed2 = Image::<f32>::new((xsize, ysize))?;
    let mut smoothed_images = [smoothed0, smoothed1, smoothed2];

    let splitters = smoothed_images
        .each_mut()
        .map(OutputChannelSplitter::from_image);

    let num_lf_groups = frame_header.num_lf_groups();

    parallel_runner.run(num_lf_groups, &|g| {
        let r = frame_header.lf_group_rect(g);
        let mut out_refs = splitters.each_ref().map(|s| s.borrow_typed_rect::<f32>(r));

        for sly in 0..(r.size.1 >> v_upsample) {
            let lys: [_; 3] = std::array::from_fn(|ch| sly << raw_vs[ch]);
            let gys = lys.map(|ly| r.origin.1 + ly);
            let sgy = (r.origin.1 >> v_upsample) + sly;

            for slx in 0..(r.size.0 >> h_upsample) {
                let lxs: [_; 3] = std::array::from_fn(|ch| slx << raw_hs[ch]);
                let gxs = lxs.map(|lx| r.origin.0 + lx);
                let sgx = (r.origin.0 >> h_upsample) + slx;

                if sgy == 0 || sgy == sh - 1 || sgx == 0 || sgx == sw - 1 {
                    for ch in 0..3 {
                        let ly = lys[ch];
                        let lx = lxs[ch];
                        let gy = gys[ch];
                        let gx = gxs[ch];
                        let lf_image = &lf_image[ch];
                        let out_ref = &mut out_refs[ch];

                        for dy in 0..bh {
                            let in_row = lf_image.row(gy + dy);
                            let row = out_ref.typed_row_mut::<f32>(ly + dy);
                            row[lx..][..bw].copy_from_slice(&in_row[gx..][..bw]);
                        }
                    }
                    continue;
                }

                let mut gap_acc = 0.5;
                for ch in 0..3 {
                    let ly = lys[ch];
                    let lx = lxs[ch];
                    let gy = gys[ch];
                    let gx = gxs[ch];
                    let lf_factor = lf_factors[ch];
                    let lf_image = &lf_image[ch];

                    for dy in 0..bh {
                        let in_row_u = lf_image.row(gy + dy - 1);
                        let in_row_c = lf_image.row(gy + dy);
                        let in_row_b = lf_image.row(gy + dy + 1);
                        let row = out_refs[ch].typed_row_mut::<f32>(ly + dy);
                        for dx in 0..bw {
                            let (sm, gap) = compute_pixel_channel(
                                lf_factor,
                                gap_acc,
                                gx + dx,
                                in_row_u,
                                in_row_c,
                                in_row_b,
                            );
                            row[lx + dx] = sm;
                            gap_acc = gap;
                        }
                    }
                }
                let factor = (3.0 - 4.0 * gap_acc).max(0.0);

                for ch in 0..3 {
                    let ly = lys[ch];
                    let lx = lxs[ch];
                    let gy = gys[ch];
                    let gx = gxs[ch];
                    let lf_image = &lf_image[ch];
                    let out_ref = &mut out_refs[ch];

                    for dy in 0..bh {
                        let in_row = lf_image.row(gy + dy);
                        let row = out_ref.typed_row_mut::<f32>(ly + dy);
                        for dx in 0..bw {
                            let mc = in_row[gx + dx];
                            let sm = row[lx + dx];
                            row[lx + dx] = (sm - mc) * factor + mc;
                        }
                    }
                }
            }
        }
        Ok(())
    })?;

    drop(splitters);
    *lf_image = smoothed_images;
    Ok(())
}
