// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Color Management System implementation using lcms2.

use jxl::api::{JxlCms, JxlCmsTransformer, JxlColorProfile};
use jxl::error::{Error, Result};
use lcms2::{
    AllowCache, ColorSpaceSignatureExt, Intent, PixelFormat, Profile, ThreadContext, Transform,
};

/// CMS implementation using Little CMS (lcms2).
pub struct Lcms2Cms;

impl JxlCms for Lcms2Cms {
    fn initialize_transforms(
        &self,
        n: usize,
        _max_pixels_per_transform: usize,
        input: JxlColorProfile,
        output: JxlColorProfile,
        _intensity_target: f32,
    ) -> Result<(usize, Vec<Box<dyn JxlCmsTransformer + Send>>)> {
        // Convert profiles to ICC
        let input_icc = input
            .try_as_icc()
            .ok_or_else(|| Error::CmsError("Cannot create ICC for input profile".into()))?;
        let output_icc = output
            .try_as_icc()
            .ok_or_else(|| Error::CmsError("Cannot create ICC for output profile".into()))?;

        // Parse profiles once to determine channel counts
        let temp_input_profile = Profile::new_icc(input_icc.as_slice())
            .map_err(|e| Error::CmsError(format!("lcms2 failed to parse input ICC: {e}")))?;
        let temp_output_profile = Profile::new_icc(output_icc.as_slice())
            .map_err(|e| Error::CmsError(format!("lcms2 failed to parse output ICC: {e}")))?;

        let input_channels = temp_input_profile.color_space().channels() as usize;
        let output_channels = temp_output_profile.color_space().channels() as usize;

        let input_format = channels_to_pixel_format(input_channels);
        let output_format = channels_to_pixel_format(output_channels);

        // Create transforms using ThreadContext for thread safety (implements Send).
        // Use u8 pixel type with PixelFormat describing the actual f32 data layout.
        let mut transforms: Vec<Box<dyn JxlCmsTransformer + Send>> = Vec::with_capacity(n);

        for _ in 0..n {
            let context = ThreadContext::new();

            // Create profiles with the thread context
            let input_profile = Profile::new_icc_context(&context, input_icc.as_slice())
                .map_err(|e| Error::CmsError(format!("lcms2 failed to parse input ICC: {e}")))?;
            let output_profile = Profile::new_icc_context(&context, output_icc.as_slice())
                .map_err(|e| Error::CmsError(format!("lcms2 failed to parse output ICC: {e}")))?;

            let transform: Transform<u8, u8, ThreadContext, AllowCache> = Transform::new_context(
                context,
                &input_profile,
                input_format,
                &output_profile,
                output_format,
                Intent::RelativeColorimetric,
            )
            .map_err(|e| Error::CmsError(format!("lcms2 failed to create transform: {e}")))?;

            transforms.push(Box::new(Lcms2Transformer {
                transform,
                input_channels,
                output_channels,
            }));
        }

        Ok((output_channels, transforms))
    }
}

/// Maps channel count to lcms2 PixelFormat for f32 data.
fn channels_to_pixel_format(channels: usize) -> PixelFormat {
    match channels {
        1 => PixelFormat::GRAY_FLT,
        3 => PixelFormat::RGB_FLT,
        4 => PixelFormat::CMYK_FLT,
        _ => PixelFormat::RGB_FLT, // Default to RGB
    }
}

/// Transformer implementation using lcms2 with ThreadContext for thread safety.
struct Lcms2Transformer {
    transform: Transform<u8, u8, ThreadContext, AllowCache>,
    input_channels: usize,
    output_channels: usize,
}

impl JxlCmsTransformer for Lcms2Transformer {
    fn do_transform(&mut self, input: &[f32], output: &mut [f32]) -> Result<()> {
        let num_pixels = input.len() / self.input_channels;

        // Verify output buffer size
        let expected_output_len = num_pixels * self.output_channels;
        if output.len() < expected_output_len {
            return Err(Error::CmsError(format!(
                "Output buffer too small: expected {expected_output_len}, got {}",
                output.len()
            )));
        }

        // Convert f32 slices to byte slices using bytemuck for safe casting
        let input_bytes: &[u8] = bytemuck::cast_slice(input);
        let output_bytes: &mut [u8] = bytemuck::cast_slice_mut(output);

        self.transform.transform_pixels(input_bytes, output_bytes);

        Ok(())
    }

    fn do_transform_inplace(&mut self, inout: &mut [f32]) -> Result<()> {
        // For in-place transform, input and output channel counts must match
        if self.input_channels != self.output_channels {
            return Err(Error::CmsError(
                "In-place transform requires matching channel counts".into(),
            ));
        }

        // Convert f32 slice to byte slice
        let inout_bytes: &mut [u8] = bytemuck::cast_slice_mut(inout);

        self.transform.transform_in_place(inout_bytes);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jxl::api::{JxlColorEncoding, JxlPrimaries, JxlTransferFunction, JxlWhitePoint};
    use jxl::headers::color_encoding::RenderingIntent;

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
        let cms = Lcms2Cms;
        let result =
            cms.initialize_transforms(1, 1024, srgb_profile(), linear_srgb_profile(), 255.0);
        assert!(result.is_ok());
        let (output_channels, transforms) = result.unwrap();
        assert_eq!(output_channels, 3);
        assert_eq!(transforms.len(), 1);
    }

    #[test]
    fn test_transform_identity() {
        let cms = Lcms2Cms;
        let (_, mut transforms) = cms
            .initialize_transforms(1, 1024, srgb_profile(), srgb_profile(), 255.0)
            .unwrap();

        let input = [0.5f32, 0.5, 0.5]; // Gray
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
        let cms = Lcms2Cms;
        let (_, mut transforms) = cms
            .initialize_transforms(1, 1024, srgb_profile(), linear_srgb_profile(), 255.0)
            .unwrap();

        // sRGB mid-gray (0.5) should map to approximately 0.214 in linear
        let input = [0.5f32, 0.5, 0.5];
        let mut output = [0.0f32; 3];

        transforms[0].do_transform(&input, &mut output).unwrap();

        // Linear value for sRGB 0.5 is approximately 0.214
        for (i, element) in output.iter().enumerate() {
            assert!(
                (element - 0.214).abs() < 0.01,
                "Output {i} = {}, expected ~0.214",
                element
            );
        }
    }
}
