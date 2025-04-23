// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

pub struct TwoDArray(pub Vec<Vec<f32>>);

impl TwoDArray {
    pub fn copy(source: &[&[f32]]) -> TwoDArray {
        TwoDArray(source.iter().map(|s| s.to_vec()).collect::<Vec<Vec<f32>>>())
    }
    pub fn copy_mut(source: &mut [&mut [f32]]) -> TwoDArray {
        TwoDArray(
            source
                .iter_mut()
                .map(|s| s.to_vec())
                .collect::<Vec<Vec<f32>>>(),
        )
    }
    pub fn blank(rows: usize, cols: usize) -> TwoDArray {
        TwoDArray(vec![vec![0.0; cols]; rows])
    }
    pub fn as_refs(&self) -> Vec<&[f32]> {
        self.0
            .iter()
            .map(|inner_vec| inner_vec.as_slice())
            .collect()
    }
    pub fn as_mut_refs(&mut self) -> Vec<&mut [f32]> {
        self.0
            .iter_mut()
            .map(|inner_vec| inner_vec.as_mut_slice())
            .collect()
    }
}
