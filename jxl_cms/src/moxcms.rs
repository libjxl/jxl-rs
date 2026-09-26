// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Color Management System implementation using moxcms.

use std::sync::Arc;

use jxl::api::JxlColorProfile;
use moxcms::{
    ColorProfile, DataColorSpace, Layout, RenderingIntent, TransformF32Executor, TransformOptions,
};

use crate::{Error, JxlCms, JxlCmsTransformer, Result};

/// CMS implementation using moxcms.
pub struct MoxCms;

impl JxlCms for MoxCms {
    fn initialize_transforms(
        &self,
        n: usize,
        _max_pixels_per_transform: usize,
        input: JxlColorProfile,
        output: JxlColorProfile,
        _intensity_target: f32,
    ) -> Result<(usize, Vec<Box<dyn JxlCmsTransformer + Send>>)> {
        // Convert profiles to ICC
        let input_icc = input.try_as_icc().ok_or(Error::InputIccError)?;
        let output_icc = output.try_as_icc().ok_or(Error::OutputIccError)?;

        // Determine channel counts from parsed ICC profiles
        let input = ColorProfile::new_from_slice(&input_icc)
            .map_err(|e| Error::CmsInputParseError(e.to_string()))?;
        let output = ColorProfile::new_from_slice(&output_icc)
            .map_err(|e| Error::CmsOutputParseError(e.to_string()))?;
        let input_layout = layout_from_color_space(input.color_space)?;
        let output_layout = layout_from_color_space(input.color_space)?;
        let input_channels = input_layout.channels();
        let output_channels = output_layout.channels();

        let options = TransformOptions {
            rendering_intent: RenderingIntent::RelativeColorimetric,
            ..Default::default()
        };
        let transform = input
            .create_transform_f32(input_layout, &output, output_layout, options)
            .map_err(|e| Error::CmsTransformError(e.to_string()))?;
        let transforms = (0..n)
            .map(|_| {
                Box::new(MoxCmsTransformer {
                    transform: Arc::clone(&transform),
                    input_channels,
                    output_channels,
                    cmyk_buffer: (input.color_space == DataColorSpace::Cmyk).then(Vec::new),
                }) as Box<dyn JxlCmsTransformer + Send>
            })
            .collect();
        Ok((output_channels, transforms))
    }
}

fn layout_from_color_space(color_space: DataColorSpace) -> Result<Layout> {
    Ok(match color_space {
        DataColorSpace::Gray => Layout::Gray,
        DataColorSpace::Cmyk => Layout::Rgba,
        DataColorSpace::Xyz
        | DataColorSpace::Lab
        | DataColorSpace::Luv
        | DataColorSpace::YCbr
        | DataColorSpace::Yxy
        | DataColorSpace::Rgb
        | DataColorSpace::Hsv
        | DataColorSpace::Hls
        | DataColorSpace::Cmy
        | DataColorSpace::Color3 => Layout::Rgb,
        color_space => {
            return Err(Error::CmsTransformError(format!(
                "Cannot handle ICC color space {color_space:?}"
            )));
        }
    })
}

struct MoxCmsTransformer {
    transform: Arc<TransformF32Executor>,
    input_channels: usize,
    output_channels: usize,
    cmyk_buffer: Option<Vec<f32>>,
}

impl JxlCmsTransformer for MoxCmsTransformer {
    fn do_transform(&mut self, input: &[f32], output: &mut [f32]) -> Result<()> {
        let num_pixels = input.len() / self.input_channels;

        // Verify output buffer size
        let expected_output_len = num_pixels * self.output_channels;
        if output.len() < expected_output_len {
            return Err(Error::OutputBufferTooSmall(
                expected_output_len,
                output.len(),
            ));
        }

        let input = if let Some(buf) = &mut self.cmyk_buffer {
            buf.resize(input.len(), 0.0);
            for (dst, &src) in buf.iter_mut().zip(input) {
                *dst = 1.0 - src;
            }
            buf
        } else {
            input
        };
        self.transform
            .transform(input, &mut output[..expected_output_len])
            .map_err(|e| Error::CmsTransformError(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use jxl::api::{JxlColorEncoding, JxlPrimaries, JxlTransferFunction, JxlWhitePoint};
    use jxl::headers::color_encoding::RenderingIntent;

    use super::*;

    fn srgb_profile() -> JxlColorProfile {
        JxlColorProfile::Simple(JxlColorEncoding::RgbColorSpace {
            white_point: JxlWhitePoint::D65,
            primaries: JxlPrimaries::SRGB,
            transfer_function: JxlTransferFunction::SRGB,
            rendering_intent: RenderingIntent::Relative,
        })
    }

    fn linear_srgb_profile() -> JxlColorProfile {
        JxlColorProfile::Simple(JxlColorEncoding::RgbColorSpace {
            white_point: JxlWhitePoint::D65,
            primaries: JxlPrimaries::SRGB,
            transfer_function: JxlTransferFunction::Linear,
            rendering_intent: RenderingIntent::Relative,
        })
    }

    #[test]
    fn test_create_transform() {
        let cms = MoxCms;
        let result =
            cms.initialize_transforms(1, 1024, srgb_profile(), linear_srgb_profile(), 255.0);
        assert!(result.is_ok());
        let (output_channels, transforms) = result.unwrap();
        assert_eq!(output_channels, 3);
        assert_eq!(transforms.len(), 1);
    }

    #[test]
    fn test_transform_identity() {
        let cms = MoxCms;
        let (_, mut transforms) = cms
            .initialize_transforms(1, 1024, srgb_profile(), srgb_profile(), 255.0)
            .unwrap();

        let input = [0.5f32; 3]; // Gray
        let mut output = [0.0f32; 3];

        transforms[0].do_transform(&input, &mut output).unwrap();

        // Should be approximately the same (identity transform)
        for i in 0..3 {
            assert!(
                (input[i] - output[i]).abs() < 0.001,
                "Mismatch at {i}: {} vs {}",
                input[i],
                output[i]
            );
        }
    }

    #[test]
    fn test_transform_srgb_to_linear() {
        let cms = MoxCms;
        let (_, mut transforms) = cms
            .initialize_transforms(1, 1024, srgb_profile(), linear_srgb_profile(), 255.0)
            .unwrap();

        // sRGB mid-gray (0.5) should map to approximately 0.214 in linear
        let input = [0.5f32; 3];
        let mut output = [0.0f32; 3];

        transforms[0].do_transform(&input, &mut output).unwrap();

        // Linear value for sRGB 0.5 is approximately 0.214
        for (i, element) in output.into_iter().enumerate() {
            assert!(
                (element - 0.214).abs() < 0.01,
                "Output {i} = {element}, expected ~0.214",
            );
        }
    }
}
