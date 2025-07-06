// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::color::tf;
use crate::headers::color_encoding::CustomTransferFunction;
use crate::render::{RenderPipelineInPlaceStage, RenderPipelineStage};

/// Convert encoded non-linear color samples to display-referred linear color samples.
#[derive(Debug)]
pub struct ToLinearStage {
    first_channel: usize,
    tf: TransferFunction,
}

impl ToLinearStage {
    pub fn new(first_channel: usize, tf: TransferFunction) -> Self {
        Self { first_channel, tf }
    }

    #[allow(unused, reason = "tirr-c: remove once we use this!")]
    pub fn sdr(first_channel: usize, tf: CustomTransferFunction) -> Self {
        let tf = TransferFunction::try_from(tf).expect("transfer function is not an SDR one");
        Self::new(first_channel, tf)
    }

    #[allow(unused, reason = "tirr-c: remove once we use this!")]
    pub fn pq(first_channel: usize, intensity_target: f32) -> Self {
        let tf = TransferFunction::Pq { intensity_target };
        Self::new(first_channel, tf)
    }

    #[allow(unused, reason = "tirr-c: remove once we use this!")]
    pub fn hlg(first_channel: usize, intensity_target: f32, luminance_rgb: [f32; 3]) -> Self {
        let tf = TransferFunction::Hlg {
            intensity_target,
            luminance_rgb,
        };
        Self::new(first_channel, tf)
    }
}

impl std::fmt::Display for ToLinearStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let channel = self.first_channel;
        write!(
            f,
            "Convert transfer function {:?} to display-referred linear TF for channel [{},{},{}]",
            self.tf,
            channel,
            channel + 1,
            channel + 2
        )
    }
}

impl RenderPipelineStage for ToLinearStage {
    type Type = RenderPipelineInPlaceStage<f32>;

    fn uses_channel(&self, c: usize) -> bool {
        (self.first_channel..self.first_channel + 3).contains(&c)
    }

    fn process_row_chunk(
        &self,
        _position: (usize, usize),
        xsize: usize,
        row: &mut [&mut [f32]],
        _state: Option<&mut dyn std::any::Any>,
    ) {
        let [row_r, row_g, row_b] = row else {
            panic!(
                "incorrect number of channels; expected 3, found {}",
                row.len()
            );
        };

        match self.tf {
            TransferFunction::Bt709 => {
                for row in row {
                    tf::bt709_to_linear(&mut row[..xsize]);
                }
            }
            TransferFunction::Srgb => {
                for row in row {
                    tf::srgb_to_linear(&mut row[..xsize]);
                }
            }
            TransferFunction::Pq { intensity_target } => {
                for row in row {
                    tf::pq_to_linear(intensity_target, &mut row[..xsize]);
                }
            }
            TransferFunction::Hlg {
                intensity_target,
                luminance_rgb,
            } => {
                tf::hlg_to_scene(&mut row_r[..xsize]);
                tf::hlg_to_scene(&mut row_g[..xsize]);
                tf::hlg_to_scene(&mut row_b[..xsize]);

                let rows = [
                    &mut row_r[..xsize],
                    &mut row_g[..xsize],
                    &mut row_b[..xsize],
                ];
                tf::hlg_scene_to_display(intensity_target, luminance_rgb, rows);
            }
            TransferFunction::Gamma(g) => {
                for row in row {
                    for v in &mut row[..xsize] {
                        *v = crate::util::fast_powf(*v, g);
                    }
                }
            }
        }
    }
}

#[derive(Debug)]
pub enum TransferFunction {
    Bt709,
    Srgb,
    Pq {
        /// Original Intensity Target
        intensity_target: f32,
    },
    Hlg {
        /// Original Intensity Target
        intensity_target: f32,
        luminance_rgb: [f32; 3],
    },
    /// Gamma in range `(0, 1]`
    Gamma(f32),
}

impl TryFrom<CustomTransferFunction> for TransferFunction {
    type Error = ();

    fn try_from(ctf: CustomTransferFunction) -> Result<Self, ()> {
        use crate::headers::color_encoding::TransferFunction;

        if ctf.have_gamma {
            Ok(Self::Gamma(ctf.gamma()))
        } else {
            match ctf.transfer_function {
                TransferFunction::BT709 => Ok(Self::Bt709),
                TransferFunction::Unknown => Err(()),
                TransferFunction::Linear => Err(()),
                TransferFunction::SRGB => Ok(Self::Srgb),
                TransferFunction::PQ => Err(()),
                TransferFunction::DCI => Ok(Self::Gamma(2.6f32.recip())),
                TransferFunction::HLG => Err(()),
            }
        }
    }
}

#[cfg(test)]
mod test {
    use test_log::test;

    use super::*;
    use crate::error::Result;
    use crate::image::Image;
    use crate::render::test::make_and_run_simple_pipeline;
    use crate::util::test::assert_all_almost_eq;

    const LUMINANCE_BT2020: [f32; 3] = [0.2627, 0.678, 0.0593];

    #[test]
    fn consistency_hlg() -> Result<()> {
        crate::render::test::test_stage_consistency::<_, f32, f32>(
            ToLinearStage::hlg(0, 1000f32, LUMINANCE_BT2020),
            (500, 500),
            3,
        )
    }

    #[test]
    fn consistency_pq() -> Result<()> {
        crate::render::test::test_stage_consistency::<_, f32, f32>(
            ToLinearStage::pq(0, 10000f32),
            (500, 500),
            3,
        )
    }

    #[test]
    fn consistency_srgb() -> Result<()> {
        crate::render::test::test_stage_consistency::<_, f32, f32>(
            ToLinearStage::new(0, TransferFunction::Srgb),
            (500, 500),
            3,
        )
    }

    #[test]
    fn consistency_bt709() -> Result<()> {
        crate::render::test::test_stage_consistency::<_, f32, f32>(
            ToLinearStage::new(0, TransferFunction::Bt709),
            (500, 500),
            3,
        )
    }

    #[test]
    fn consistency_gamma22() -> Result<()> {
        crate::render::test::test_stage_consistency::<_, f32, f32>(
            ToLinearStage::new(0, TransferFunction::Gamma(0.4545455)),
            (500, 500),
            3,
        )
    }

    #[test]
    fn sdr_white_hlg() -> Result<()> {
        let intensity_target = 1000f32;
        // Reversed version of FromLinear test
        let input_r = Image::new_constant((1, 1), 0.75)?;
        let input_g = Image::new_constant((1, 1), 0.75)?;
        let input_b = Image::new_constant((1, 1), 0.75)?;

        // 75% HLG
        let stage = ToLinearStage::hlg(0, intensity_target, LUMINANCE_BT2020);
        let output = make_and_run_simple_pipeline::<_, f32, f32>(
            stage,
            &[input_r, input_g, input_b],
            (1, 1),
            0,
            256,
        )?
        .1;

        assert_all_almost_eq!(output[0].as_rect().row(0), &[0.203], 1e-3);
        assert_all_almost_eq!(output[1].as_rect().row(0), &[0.203], 1e-3);
        assert_all_almost_eq!(output[2].as_rect().row(0), &[0.203], 1e-3);

        Ok(())
    }

    #[test]
    fn sdr_white_pq() -> Result<()> {
        let intensity_target = 1000f32;
        // Reversed version of FromLinear test
        let input_r = Image::new_constant((1, 1), 0.5807)?;
        let input_g = Image::new_constant((1, 1), 0.5807)?;
        let input_b = Image::new_constant((1, 1), 0.5807)?;

        // 58% PQ
        let stage = ToLinearStage::pq(0, intensity_target);
        let output = make_and_run_simple_pipeline::<_, f32, f32>(
            stage,
            &[input_r, input_g, input_b],
            (1, 1),
            0,
            256,
        )?
        .1;

        assert_all_almost_eq!(output[0].as_rect().row(0), &[0.203], 1e-3);
        assert_all_almost_eq!(output[1].as_rect().row(0), &[0.203], 1e-3);
        assert_all_almost_eq!(output[2].as_rect().row(0), &[0.203], 1e-3);

        Ok(())
    }
}
