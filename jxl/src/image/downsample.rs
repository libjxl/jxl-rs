// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use super::{Image, ImageDataType, Rect};
use crate::util::ShiftRightCeil;

/// Nearest-grid alignment, with ties towards positive infinity. Use the same
/// rule for frame origins and patch positions, including negative frame crops.
pub(crate) fn align_origin(origin: isize, shift: usize) -> isize {
    if shift == 0 {
        origin
    } else {
        (origin + (1isize << (shift - 1))) >> shift
    }
}

/// Alpha weights apply only to straight samples. Associated samples and alpha
/// itself must use an ordinary average to preserve their representation.
pub(crate) fn box_average<T: ImageDataType, Dest>(
    source: &Image<T>,
    rect: Rect,
    shift: usize,
    alpha: Option<&Image<T>>,
    destination: &mut Dest,
    get_destination_row: impl Fn(&mut Dest, usize) -> &mut [T],
) {
    let valid_size = rect.size;
    let output_size = (valid_size.0.shrc(shift), valid_size.1.shrc(shift));
    assert!(source.size().0 >= rect.end().0 && source.size().1 >= rect.end().1);
    if let Some(alpha) = alpha {
        assert!(alpha.size().0 >= rect.end().0 && alpha.size().1 >= rect.end().1);
    }

    let source = source.get_rect(rect);
    let alpha = alpha.map(|image| image.get_rect(rect));

    for y in 0..output_size.1 {
        let output = get_destination_row(destination, y);
        let y0 = y << shift;
        let y1 = ((y + 1) << shift).min(valid_size.1);
        for (x, value) in output[..output_size.0].iter_mut().enumerate() {
            let x0 = x << shift;
            let x1 = ((x + 1) << shift).min(source.size().0);

            let mut sum = 0.0;
            let mut weight = 0.0;
            for sy in y0..y1 {
                let src_row = &source.row(sy)[x0..x1];
                if let Some(alpha) = alpha {
                    let alpha_row = &alpha.row(sy)[x0..x1];
                    for (sample, alpha) in src_row.iter().zip(alpha_row) {
                        let a = alpha.to_f64();
                        sum += sample.to_f64() * a;
                        weight += a;
                    }
                } else {
                    for sample in src_row {
                        sum += sample.to_f64();
                        weight += 1.0;
                    }
                }
            }

            *value = T::from_f64(if weight > 0.0 { sum / weight } else { 0.0 });
        }
    }
}
