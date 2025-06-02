// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{bit_reader::BitReader, error::Error, headers::encodings::*};
use jxl_macros::UnconditionalCoder;
use num_derive::FromPrimitive;
use std::fmt;

// Define type aliases for clarity
pub type Matrix3x3 = [[f32; 3]; 3];
pub type Vector3 = [f32; 3];

// Bradford matrices for chromatic adaptation
const K_BRADFORD: Matrix3x3 = [
    [0.8951, 0.2664, -0.1614],
    [-0.7502, 1.7135, 0.0367],
    [0.0389, -0.0685, 1.0296],
];

const K_BRADFORD_INV: Matrix3x3 = [
    [0.9869929, -0.1470543, 0.1599627],
    [0.4323053, 0.5183603, 0.0492912],
    [-0.0085287, 0.0400428, 0.9684867],
];

fn mul_3x3_vector(matrix: &Matrix3x3, vector: &Vector3) -> Vector3 {
    std::array::from_fn(|i| {
        matrix[i]
            .iter()
            .zip(vector.iter())
            .map(|(&matrix_element, &vector_element)| matrix_element * vector_element)
            .sum()
    })
}

fn mul_3x3_matrix(mat1: &Matrix3x3, mat2: &Matrix3x3) -> Matrix3x3 {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| mat1[i][k] * mat2[k][j]).sum()))
}

fn det2x2(a: f32, b: f32, c: f32, d: f32) -> f32 {
    a * d - b * c
}

fn calculate_cofactor(m: &Matrix3x3, r: usize, c: usize) -> f32 {
    // Determine indices for the 2x2 submatrix for the minor M_rc
    let r1 = (r + 1) % 3;
    let r2 = (r + 2) % 3;
    let c1 = (c + 1) % 3;
    let c2 = (c + 2) % 3;

    let minor_val = det2x2(m[r1][c1], m[r1][c2], m[r2][c1], m[r2][c2]);

    if (r + c) % 2 == 0 {
        minor_val
    } else {
        -minor_val
    }
}

/// Calculates the inverse of a 3x3 matrix.
fn inv_3x3_matrix(m: &Matrix3x3) -> Result<Matrix3x3, Error> {
    let cofactor_matrix: [[f32; 3]; 3] = std::array::from_fn(|r_idx| {
        std::array::from_fn(|c_idx| calculate_cofactor(m, r_idx, c_idx))
    });

    let det: f32 = m[0]
        .iter()
        .zip(cofactor_matrix[0].iter())
        .map(|(&m_element, &cof_element)| m_element * cof_element)
        .sum::<f32>();

    // Check for numerical singularity.
    const EPSILON: f32 = 1e-12;
    if det.abs() < EPSILON {
        return Err(Error::MatrixInversionFailed(det.abs()));
    }

    let inv_det = 1.0 / det;

    let adjugate_matrix: [[f32; 3]; 3] =
        std::array::from_fn(|r_idx| std::array::from_fn(|c_idx| cofactor_matrix[c_idx][r_idx]));

    // Inverse matrix = (1/det) * Adjugate matrix.
    Ok(std::array::from_fn(|r_idx| {
        std::array::from_fn(|c_idx| adjugate_matrix[r_idx][c_idx] * inv_det)
    }))
}

#[allow(clippy::too_many_arguments)]
pub fn primaries_to_xyz(
    rx: f32,
    ry: f32,
    gx: f32,
    gy: f32,
    bx: f32,
    by: f32,
    wx: f32,
    wy: f32,
) -> Result<Matrix3x3, Error> {
    // Validate white point coordinates
    if !((0.0..=1.0).contains(&wx) && (wy > 0.0 && wy <= 1.0)) {
        return Err(Error::IccInvalidWhitePoint(
            wx,
            wy,
            "White point coordinates out of range ([0,1] for x, (0,1] for y)".to_string(),
        ));
    }
    // Comment from libjxl:
    // TODO(lode): also require rx, ry, gx, gy, bx, to be in range 0-1? ICC
    // profiles in theory forbid negative XYZ values, but in practice the ACES P0
    // color space uses a negative y for the blue primary.

    // Construct the primaries matrix P. Its columns are the XYZ coordinates
    // of the R, G, B primaries (derived from their chromaticities x, y, z=1-x-y).
    // P = [[xr, xg, xb],
    //      [yr, yg, yb],
    //      [zr, zg, zb]]
    let rz = 1.0 - rx - ry;
    let gz = 1.0 - gx - gy;
    let bz = 1.0 - bx - by;
    let p_matrix = [[rx, gx, bx], [ry, gy, by], [rz, gz, bz]];

    let p_inv_matrix = inv_3x3_matrix(&p_matrix)?;

    // Convert reference white point (wx, wy) to XYZ form with Y=1
    // This is WhitePoint_XYZ_wp = [Wx/Wy, 1, (1-Wx-Wy)/Wy]
    let x_over_y_wp = wx / wy;
    let z_over_y_wp = (1.0 - wx - wy) / wy;

    if !x_over_y_wp.is_finite() || !z_over_y_wp.is_finite() {
        return Err(Error::IccInvalidWhitePoint(
            wx,
            wy,
            "Calculated X/Y or Z/Y for white point is not finite.".to_string(),
        ));
    }
    let white_point_xyz_vec: Vector3 = [x_over_y_wp, 1.0, z_over_y_wp];

    // Calculate scaling factors S = [Sr, Sg, Sb] such that P * S = WhitePoint_XYZ_wp
    // So, S = P_inv * WhitePoint_XYZ_wp
    let s_vec = mul_3x3_vector(&p_inv_matrix, &white_point_xyz_vec);

    // Construct diagonal matrix S_diag from s_vec
    let s_diag_matrix = [
        [s_vec[0], 0.0, 0.0],
        [0.0, s_vec[1], 0.0],
        [0.0, 0.0, s_vec[2]],
    ];

    // The final RGB-to-XYZ matrix is P * S_diag
    let result_matrix = mul_3x3_matrix(&p_matrix, &s_diag_matrix);

    Ok(result_matrix)
}

fn adapt_to_xyz_d50(wx: f32, wy: f32) -> Result<Matrix3x3, Error> {
    if !((0.0..=1.0).contains(&wx) && (wy > 0.0 && wy <= 1.0)) {
        return Err(Error::IccInvalidWhitePoint(
            wx,
            wy,
            "White point coordinates out of range ([0,1] for x, (0,1] for y)".to_string(),
        ));
    }

    // Convert white point (wx, wy) to XYZ with Y=1
    let x_over_y = wx / wy;
    let z_over_y = (1.0 - wx - wy) / wy;

    // Check for finiteness, as 1.0 / tiny float can overflow.
    if !x_over_y.is_finite() || !z_over_y.is_finite() {
        return Err(Error::IccInvalidWhitePoint(
            wx,
            wy,
            "Calculated X/Y or Z/Y for white point is not finite.".to_string(),
        ));
    }
    let w: Vector3 = [x_over_y, 1.0, z_over_y];

    // D50 white point in XYZ (Y=1 form)
    // These are X_D50/Y_D50, 1.0, Z_D50/Y_D50
    let w50: Vector3 = [0.96422, 1.0, 0.82521];

    // Transform to LMS color space
    let lms_source = mul_3x3_vector(&K_BRADFORD, &w);
    let lms_d50 = mul_3x3_vector(&K_BRADFORD, &w50);

    // Check for invalid LMS values which would lead to division by zero
    if lms_source.contains(&0.0) {
        return Err(Error::IccInvalidWhitePoint(
            wx,
            wy,
            "LMS components for source white point are zero, leading to division by zero."
                .to_string(),
        ));
    }

    // Create diagonal scaling matrix in LMS space
    let mut a_diag_matrix: Matrix3x3 = [[0.0; 3]; 3];
    for i in 0..3 {
        a_diag_matrix[i][i] = lms_d50[i] / lms_source[i];
        if !a_diag_matrix[i][i].is_finite() {
            return Err(Error::IccInvalidWhitePoint(
                wx,
                wy,
                format!("Diagonal adaptation matrix component {} is not finite.", i),
            ));
        }
    }

    // Combine transformations
    let b_matrix = mul_3x3_matrix(&a_diag_matrix, &K_BRADFORD);
    let final_adaptation_matrix = mul_3x3_matrix(&K_BRADFORD_INV, &b_matrix);

    Ok(final_adaptation_matrix)
}

#[allow(clippy::too_many_arguments)]
pub fn primaries_to_xyz_d50(
    rx: f32,
    ry: f32,
    gx: f32,
    gy: f32,
    bx: f32,
    by: f32,
    wx: f32,
    wy: f32,
) -> Result<Matrix3x3, Error> {
    // Get the matrix to convert RGB to XYZ, adapted to its native white point (wx, wy).
    let rgb_to_xyz_native_wp_matrix = primaries_to_xyz(rx, ry, gx, gy, bx, by, wx, wy)?;

    // Get the chromatic adaptation matrix from the native white point (wx, wy) to D50.
    let adaptation_to_d50_matrix = adapt_to_xyz_d50(wx, wy)?;
    // This matrix converts XYZ values relative to white point (wx, wy)
    // to XYZ values relative to D50.

    // Combine the matrices: M_RGBtoD50XYZ = M_AdaptToD50 * M_RGBtoNativeXYZ
    // Applying M_RGBtoNativeXYZ first gives XYZ relative to native white point.
    // Then applying M_AdaptToD50 converts these XYZ values to be relative to D50.
    let result_matrix = mul_3x3_matrix(&adaptation_to_d50_matrix, &rgb_to_xyz_native_wp_matrix);

    Ok(result_matrix)
}

#[allow(clippy::too_many_arguments)]
pub fn create_icc_rgb_matrix(
    rx: f32,
    ry: f32,
    gx: f32,
    gy: f32,
    bx: f32,
    by: f32,
    wx: f32,
    wy: f32,
) -> Result<Matrix3x3, Error> {
    // TODO: think about if we need/want to change precision to f64 for some calculations here
    primaries_to_xyz_d50(rx, ry, gx, gy, bx, by, wx, wy)
}

pub fn create_icc_chad_matrix(wx: f32, wy: f32) -> Result<Matrix3x3, Error> {
    // The libjxl checks `wy == 0.0` check here, but this is redundant since
    // `adapt_to_xyz_d50` handles it robustly.
    // TODO(firsching): call `adapt_to_xyz_d50` directly when libjxl calls `create_icc_chad_matrix`
    adapt_to_xyz_d50(wx, wy)
}

#[allow(clippy::upper_case_acronyms)]
#[derive(UnconditionalCoder, Copy, Clone, PartialEq, Debug, FromPrimitive)]
pub enum ColorSpace {
    RGB,
    Gray,
    XYB,
    Unknown,
}

impl fmt::Display for ColorSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                ColorSpace::RGB => "RGB",
                ColorSpace::Gray => "Gra",
                ColorSpace::XYB => "XYB",
                ColorSpace::Unknown => "CS?",
            }
        )
    }
}

#[allow(clippy::upper_case_acronyms)]
#[derive(UnconditionalCoder, Copy, Clone, PartialEq, Debug, FromPrimitive)]
pub enum WhitePoint {
    D65 = 1,
    Custom = 2,
    E = 10,
    DCI = 11,
}

impl fmt::Display for WhitePoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                WhitePoint::D65 => "D65",
                WhitePoint::Custom => "Cst",
                WhitePoint::E => "EER",
                WhitePoint::DCI => "DCI",
            }
        )
    }
}

impl WhitePoint {
    pub fn to_xy_coords(&self, custom_xy_data: Option<&CustomXY>) -> Result<(f32, f32), Error> {
        match self {
            WhitePoint::Custom => custom_xy_data
                .map(|data| data.as_f32_coords())
                .ok_or(Error::MissingCustomWhitePointData),
            WhitePoint::D65 => Ok((0.3127, 0.3290)),
            WhitePoint::DCI => Ok((0.314, 0.351)),
            WhitePoint::E => Ok((1.0 / 3.0, 1.0 / 3.0)),
        }
    }
}

#[allow(clippy::upper_case_acronyms)]
#[derive(UnconditionalCoder, Copy, Clone, PartialEq, Debug, FromPrimitive)]
pub enum Primaries {
    SRGB = 1,
    Custom = 2,
    BT2100 = 9,
    P3 = 11,
}

impl fmt::Display for Primaries {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Primaries::SRGB => "SRG",
                Primaries::Custom => "Cst", // Base string for Custom
                Primaries::BT2100 => "202",
                Primaries::P3 => "DCI",
            }
        )
    }
}

impl Primaries {
    pub fn to_xy_coords(
        &self,
        custom_xy_data: Option<&[CustomXY; 3]>,
    ) -> Result<[(f32, f32); 3], Error> {
        match self {
            Primaries::Custom => {
                let data = custom_xy_data.ok_or(Error::MissingCustomPrimariesData)?;
                Ok([
                    data[0].as_f32_coords(),
                    data[1].as_f32_coords(),
                    data[2].as_f32_coords(),
                ])
            }
            Primaries::SRGB => Ok([
                // libjxl uses values very close to that, like 0.639998686; check with Sami what the point of that is
                (0.640, 0.330), // R
                (0.300, 0.600), // G
                (0.150, 0.060), // B
            ]),
            Primaries::BT2100 => Ok([
                (0.708, 0.292), // R
                (0.170, 0.797), // G
                (0.131, 0.046), // B
            ]),
            Primaries::P3 => Ok([
                (0.680, 0.320), // R
                (0.265, 0.690), // G
                (0.150, 0.060), // B
            ]),
        }
    }
}

#[allow(clippy::upper_case_acronyms)]
#[derive(UnconditionalCoder, Copy, Clone, PartialEq, Debug, FromPrimitive)]
pub enum TransferFunction {
    BT709 = 1,
    Unknown = 2,
    Linear = 8,
    SRGB = 13,
    PQ = 16,
    DCI = 17,
    HLG = 18,
}

impl fmt::Display for TransferFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                TransferFunction::BT709 => "709",
                TransferFunction::Unknown => "TF?",
                TransferFunction::Linear => "Lin",
                TransferFunction::SRGB => "SRG",
                TransferFunction::PQ => "PeQ",
                TransferFunction::DCI => "DCI",
                TransferFunction::HLG => "HLG",
            }
        )
    }
}

#[derive(UnconditionalCoder, Copy, Clone, PartialEq, Debug, FromPrimitive)]
pub enum RenderingIntent {
    Perceptual = 0,
    Relative,
    Saturation,
    Absolute,
}

impl fmt::Display for RenderingIntent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                RenderingIntent::Perceptual => "Per",
                RenderingIntent::Relative => "Rel",
                RenderingIntent::Saturation => "Sat",
                RenderingIntent::Absolute => "Abs",
            }
        )
    }
}

#[derive(UnconditionalCoder, Debug, Clone)]
pub struct CustomXY {
    #[default(0)]
    #[coder(u2S(Bits(19), Bits(19) + 524288, Bits(20) + 1048576, Bits(21) + 2097152))]
    pub x: i32,
    #[default(0)]
    #[coder(u2S(Bits(19), Bits(19) + 524288, Bits(20) + 1048576, Bits(21) + 2097152))]
    pub y: i32,
}

impl CustomXY {
    /// Converts the stored scaled integer coordinates to f32 (x, y) values.
    pub fn as_f32_coords(&self) -> (f32, f32) {
        (self.x as f32 / 1_000_000.0, self.y as f32 / 1_000_000.0)
    }
}

pub struct CustomTransferFunctionNonserialized {
    color_space: ColorSpace,
}

#[derive(UnconditionalCoder, Debug, Clone)]
#[nonserialized(CustomTransferFunctionNonserialized)]
#[validate]
pub struct CustomTransferFunction {
    #[condition(nonserialized.color_space != ColorSpace::XYB)]
    #[default(false)]
    pub have_gamma: bool,
    #[condition(have_gamma)]
    #[default(3333333)] // XYB gamma
    #[coder(Bits(24))]
    pub gamma: u32,
    #[condition(!have_gamma && nonserialized.color_space != ColorSpace::XYB)]
    #[default(TransferFunction::SRGB)]
    pub transfer_function: TransferFunction,
}

impl CustomTransferFunction {
    #[cfg(test)]
    pub fn empty() -> CustomTransferFunction {
        CustomTransferFunction {
            have_gamma: false,
            gamma: 0,
            transfer_function: TransferFunction::Unknown,
        }
    }
    pub fn gamma(&self) -> f32 {
        assert!(self.have_gamma);
        self.gamma as f32 * 0.0000001
    }

    pub fn check(&self, _: &CustomTransferFunctionNonserialized) -> Result<(), Error> {
        if self.have_gamma {
            let gamma = self.gamma();
            if gamma > 1.0 || gamma * 8192.0 < 1.0 {
                Err(Error::InvalidGamma(gamma))
            } else {
                Ok(())
            }
        } else {
            Ok(())
        }
    }
}

/// Writes a u32 value in big-endian format to the slice at the given position.
pub fn write_u32_be(slice: &mut [u8], pos: usize, value: u32) -> Result<(), Error> {
    if pos.checked_add(4).is_none_or(|end| end > slice.len()) {
        return Err(Error::IccWriteOutOfBounds);
    }
    slice[pos..pos + 4].copy_from_slice(&value.to_be_bytes());
    Ok(())
}

/// Writes a u16 value in big-endian format to the slice at the given position.
pub fn write_u16_be(slice: &mut [u8], pos: usize, value: u16) -> Result<(), Error> {
    if pos.checked_add(2).is_none_or(|end| end > slice.len()) {
        return Err(Error::IccWriteOutOfBounds);
    }
    slice[pos..pos + 2].copy_from_slice(&value.to_be_bytes());
    Ok(())
}

/// Writes a u8 value to the slice at the given position.
pub fn write_u8(slice: &mut [u8], pos: usize, value: u8) -> Result<(), Error> {
    if pos.checked_add(1).is_none_or(|end| end > slice.len()) {
        return Err(Error::IccWriteOutOfBounds);
    }
    slice[pos] = value;
    Ok(())
}

/// Writes a 4-character ASCII tag string to the slice at the given position.
pub fn write_icc_tag(slice: &mut [u8], pos: usize, tag_str: &str) -> Result<(), Error> {
    if tag_str.len() != 4 || !tag_str.is_ascii() {
        return Err(Error::IccInvalidTagString(tag_str.to_string()));
    }
    if pos.checked_add(4).is_none_or(|end| end > slice.len()) {
        return Err(Error::IccWriteOutOfBounds);
    }
    slice[pos..pos + 4].copy_from_slice(tag_str.as_bytes());
    Ok(())
}

/// Creates an ICC 'mluc' tag with a single "enUS" record.
///
/// The input `text` must be ASCII, as it will be encoded as UTF-16BE by prepending
/// a null byte to each ASCII character.
pub fn create_icc_mluc_tag(tags: &mut Vec<u8>, text: &str) -> Result<(), Error> {
    // libjxl comments that "The input text must be ASCII".
    // We enforce this.
    if !text.is_ascii() {
        return Err(Error::IccMlucTextNotAscii(text.to_string()));
    }
    // Tag signature 'mluc' (4 bytes)
    tags.extend_from_slice(b"mluc");
    // Reserved, must be 0 (4 bytes)
    tags.extend_from_slice(&0u32.to_be_bytes());
    // Number of records (u32, 4 bytes) - Hardcoded to 1.
    tags.extend_from_slice(&1u32.to_be_bytes());
    // Record size (u32, 4 bytes) - Each record descriptor is 12 bytes.
    // (Language Code [2] + Country Code [2] + String Length [4] + String Offset [4])
    tags.extend_from_slice(&12u32.to_be_bytes());
    // Language Code (2 bytes) - "en" for English
    tags.extend_from_slice(b"en");
    // Country Code (2 bytes) - "US" for United States
    tags.extend_from_slice(b"US");
    // Length of the string (u32, 4 bytes)
    // For ASCII text encoded as UTF-16BE, each char becomes 2 bytes.
    let string_actual_byte_length = text.len() * 2;
    tags.extend_from_slice(&(string_actual_byte_length as u32).to_be_bytes());
    // Offset of the string (u32, 4 bytes)
    // The string data for this record starts at offset 28.
    tags.extend_from_slice(&28u32.to_be_bytes());
    // The actual string data, encoded as UTF-16BE.
    // For ASCII char 'X', UTF-16BE is 0x00 0x58.
    for ascii_char_code in text.as_bytes() {
        tags.push(0u8);
        tags.push(*ascii_char_code);
    }

    Ok(())
}

pub struct TagInfo {
    signature: [u8; 4],
    // Offset of this tag's data relative to the START of the `tags_data` block
    offset_in_tags_blob: u32,
    // Unpadded size of this tag's actual data content.
    size_unpadded: u32,
}

fn pad_to_4_byte_boundary(data: &mut Vec<u8>) {
    data.resize(data.len().next_multiple_of(4), 0u8);
}

/// Converts an f32 to s15Fixed16 format and appends it as big-endian bytes.
/// s15Fixed16 is a signed 32-bit number with 1 sign bit, 15 integer bits,
/// and 16 fractional bits.
fn append_s15_fixed_16(tags_data: &mut Vec<u8>, value: f32) -> Result<(), Error> {
    // In libjxl, the following specific range check is used: (-32767.995f <= value) && (value <= 32767.995f)
    // This is slightly tighter than the theoretical max positive s15.16 value.
    // We replicate this for consistency.
    if !(value.is_finite() && (-32767.995..=32767.995).contains(&value)) {
        return Err(Error::IccValueOutOfRangeS15Fixed16(value));
    }

    // Multiply by 2^16 and round to nearest integer
    let scaled_value = (value * 65536.0).round();
    // Cast to i32 for correct two's complement representation
    let int_value = scaled_value as i32;
    tags_data.extend_from_slice(&int_value.to_be_bytes());
    Ok(())
}

/// Creates the data for an ICC 'XYZ ' tag and appends it to `tags_data`.
/// The 'XYZ ' tag contains three s15Fixed16Number values.
fn create_icc_xyz_tag(tags_data: &mut Vec<u8>, xyz_color: &[f32; 3]) -> Result<TagInfo, Error> {
    // Tag signature 'XYZ ' (4 bytes, note the trailing space)
    let signature = b"XYZ ";
    let start_offset = tags_data.len() as u32;
    tags_data.extend_from_slice(signature);

    // Reserved, must be 0 (4 bytes)
    tags_data.extend_from_slice(&0u32.to_be_bytes());

    // XYZ data (3 * s15Fixed16Number = 3 * 4 bytes)
    for &val in xyz_color {
        append_s15_fixed_16(tags_data, val)?;
    }

    Ok(TagInfo {
        signature: *signature,
        offset_in_tags_blob: start_offset,
        size_unpadded: (tags_data.len() as u32) - start_offset,
    })
}

pub fn create_icc_chad_tag(
    tags_data: &mut Vec<u8>,
    chad_matrix: &Matrix3x3,
) -> Result<TagInfo, Error> {
    // The tag type signature "sf32" (4 bytes).
    let signature = b"sf32";
    let start_offset = tags_data.len() as u32;
    tags_data.extend_from_slice(signature);

    // A reserved field (4 bytes), which must be set to 0.
    tags_data.extend_from_slice(&0u32.to_be_bytes());

    // The 9 matrix elements as s15Fixed16Number values.
    // m[0][0], m[0][1], m[0][2], m[1][0], ..., m[2][2]
    for row_array in chad_matrix.iter() {
        for &value in row_array.iter() {
            append_s15_fixed_16(tags_data, value)?;
        }
    }
    Ok(TagInfo {
        signature: *signature,
        offset_in_tags_blob: start_offset,
        size_unpadded: (tags_data.len() as u32) - start_offset,
    })
}

/// Converts CIE xy white point coordinates to CIE XYZ values (Y is normalized to 1.0).
fn cie_xyz_from_white_cie_xy(wx: f32, wy: f32) -> Result<[f32; 3], Error> {
    // Check for wy being too close to zero to prevent division by zero or extreme values.
    if wy.abs() < 1e-12 {
        return Err(Error::IccInvalidWhitePointY(wy));
    }
    let factor = 1.0 / wy;
    let x_val = wx * factor;
    let y_val = 1.0f32;
    let z_val = (1.0 - wx - wy) * factor;
    Ok([x_val, y_val, z_val])
}

#[derive(UnconditionalCoder, Debug, Clone)]
#[validate]
pub struct ColorEncoding {
    #[all_default]
    // TODO(firsching): remove once we use this!
    #[allow(dead_code)]
    all_default: bool,
    #[default(false)]
    pub want_icc: bool,
    #[default(ColorSpace::RGB)]
    pub color_space: ColorSpace,
    #[condition(!want_icc && color_space != ColorSpace::XYB)]
    #[default(WhitePoint::D65)]
    pub white_point: WhitePoint,
    // TODO(veluca): can this be merged in the enum?!
    #[condition(white_point == WhitePoint::Custom)]
    #[default(CustomXY::default(&field_nonserialized))]
    pub white: CustomXY,
    #[condition(!want_icc && color_space != ColorSpace::XYB && color_space != ColorSpace::Gray)]
    #[default(Primaries::SRGB)]
    pub primaries: Primaries,
    #[condition(primaries == Primaries::Custom)]
    #[default([CustomXY::default(&field_nonserialized), CustomXY::default(&field_nonserialized), CustomXY::default(&field_nonserialized)])]
    pub custom_primaries: [CustomXY; 3],
    #[condition(!want_icc)]
    #[default(CustomTransferFunction::default(&field_nonserialized))]
    #[nonserialized(color_space: color_space)]
    pub tf: CustomTransferFunction,
    #[condition(!want_icc)]
    #[default(RenderingIntent::Relative)]
    pub rendering_intent: RenderingIntent,
}

impl ColorEncoding {
    pub fn check(&self, _: &Empty) -> Result<(), Error> {
        if !self.want_icc
            && (self.color_space == ColorSpace::Unknown
                || self.tf.transfer_function == TransferFunction::Unknown)
        {
            Err(Error::InvalidColorEncoding)
        } else {
            Ok(())
        }
    }

    pub fn get_resolved_white_point_xy(&self) -> Result<(f32, f32), Error> {
        let custom_data_for_wp = if self.white_point == WhitePoint::Custom {
            Some(&self.white)
        } else {
            None
        };
        self.white_point.to_xy_coords(custom_data_for_wp)
    }

    pub fn get_resolved_primaries_xy(&self) -> Result<[(f32, f32); 3], Error> {
        let custom_data = if self.primaries == Primaries::Custom {
            Some(&self.custom_primaries)
        } else {
            None
        };
        self.primaries.to_xy_coords(custom_data)
    }

    fn create_icc_cicp_tag_data(&self, tags_data: &mut Vec<u8>) -> Result<Option<TagInfo>, Error> {
        if self.color_space != ColorSpace::RGB {
            return Ok(None);
        }

        // Determine the CICP value for primaries.
        let primaries_val: u8 = if self.primaries == Primaries::P3 {
            if self.white_point == WhitePoint::D65 {
                12 // P3 D65
            } else if self.white_point == WhitePoint::DCI {
                11 // P3 DCI
            } else {
                return Ok(None);
            }
        } else if self.primaries != Primaries::Custom && self.white_point == WhitePoint::D65 {
            // These JXL enum values match the ones for CICP with a D65 white point.
            self.primaries as u8
        } else {
            return Ok(None);
        };

        // Custom gamma or unknown transfer functions cannot be represented.
        if self.tf.have_gamma || self.tf.transfer_function == TransferFunction::Unknown {
            return Ok(None);
        }
        let tf_val = self.tf.transfer_function as u8;

        let signature = b"cicp";
        let start_offset = tags_data.len() as u32;
        tags_data.extend_from_slice(signature);
        let data_len = tags_data.len();
        write_u32_be(tags_data, data_len, 0)?;
        tags_data.push(primaries_val);
        tags_data.push(tf_val);
        // Matrix Coefficients (RGB is non-constant luminance)
        tags_data.push(0);
        // Video Full Range Flag
        tags_data.push(1);

        Ok(Some(TagInfo {
            signature: *signature,
            offset_in_tags_blob: start_offset,
            size_unpadded: 12,
        }))
    }

    fn can_tone_map_for_icc(&self) -> bool {
        // Placeholder
        // TODO(firsching): implement this function
        false
    }

    pub fn get_color_encoding_description(&self) -> String {
        // Helper for formatting custom XY float values.
        // Your CustomXY stores i32, which are float * 1_000_000.
        let format_xy_float = |val: i32| -> String { format!("{:.7}", val as f64 / 1_000_000.0) };
        // Helper for formatting gamma float value.
        let format_gamma_float = |val: f32| -> String { format!("{:.7}", val) };

        // Handle special known color spaces first
        if self.color_space == ColorSpace::RGB && self.white_point == WhitePoint::D65 {
            if self.rendering_intent == RenderingIntent::Perceptual
                && !self.tf.have_gamma
                && self.tf.transfer_function == TransferFunction::SRGB
            {
                if self.primaries == Primaries::SRGB {
                    return "sRGB".to_string();
                }
                if self.primaries == Primaries::P3 {
                    return "DisplayP3".to_string();
                }
            }
            if self.rendering_intent == RenderingIntent::Relative
                && self.primaries == Primaries::BT2100
            {
                if !self.tf.have_gamma && self.tf.transfer_function == TransferFunction::PQ {
                    return "Rec2100PQ".to_string();
                }
                if !self.tf.have_gamma && self.tf.transfer_function == TransferFunction::HLG {
                    return "Rec2100HLG".to_string();
                }
            }
        }

        // Build the string part by part for other case
        let mut d = String::with_capacity(64);

        // Append ColorSpace string
        d.push_str(&self.color_space.to_string());

        let explicit_wp_tf = self.color_space != ColorSpace::XYB;

        if explicit_wp_tf {
            d.push('_');
            if self.white_point == WhitePoint::Custom {
                // For Custom, we append the specific xy values
                d.push_str(&format_xy_float(self.white.x));
                d.push(';');
                d.push_str(&format_xy_float(self.white.y));
            } else {
                d.push_str(&self.white_point.to_string());
            }
        }

        if self.color_space != ColorSpace::Gray && self.color_space != ColorSpace::XYB {
            d.push('_');
            if self.primaries == Primaries::Custom {
                // For Custom, append specific r,g,b xy values
                // Red primaries
                d.push_str(&format_xy_float(self.custom_primaries[0].x));
                d.push(';');
                d.push_str(&format_xy_float(self.custom_primaries[0].y));
                d.push(';');
                // Green primaries
                d.push_str(&format_xy_float(self.custom_primaries[1].x));
                d.push(';');
                d.push_str(&format_xy_float(self.custom_primaries[1].y));
                d.push(';');
                // Blue primaries
                d.push_str(&format_xy_float(self.custom_primaries[2].x));
                d.push(';');
                d.push_str(&format_xy_float(self.custom_primaries[2].y));
            } else {
                d.push_str(&self.primaries.to_string());
            }
        }

        d.push('_');
        d.push_str(&self.rendering_intent.to_string());

        if explicit_wp_tf {
            d.push('_');
            if self.tf.have_gamma {
                d.push('g');
                d.push_str(&format_gamma_float(self.tf.gamma()));
            } else {
                d.push_str(&self.tf.transfer_function.to_string());
            }
        }
        d
    }

    pub fn create_icc_header(&self) -> Result<Vec<u8>, Error> {
        let mut header_data = vec![0u8; 128];

        // Profile size - To be filled in at the end of profile creation.
        write_u32_be(&mut header_data, 0, 0)?;
        const CMM_TAG: &str = "jxl ";
        // CMM Type
        write_icc_tag(&mut header_data, 4, CMM_TAG)?;

        // Profile version - ICC v4.4 (0x04400000)
        // Conformance tests have v4.3, libjxl produces v4.4
        write_u32_be(&mut header_data, 8, 0x04400000u32)?;

        let profile_class_str = match self.color_space {
            ColorSpace::XYB => "scnr",
            _ => "mntr",
        };
        write_icc_tag(&mut header_data, 12, profile_class_str)?;

        // Data color space
        let data_color_space_str = match self.color_space {
            ColorSpace::Gray => "GRAY",
            _ => "RGB ",
        };
        write_icc_tag(&mut header_data, 16, data_color_space_str)?;

        // PCS - Profile Connection Space
        // Corresponds to: if (kEnable3DToneMapping && CanToneMap(c))
        // Assuming kEnable3DToneMapping is true for this port for now.
        const K_ENABLE_3D_ICC_TONEMAPPING: bool = true;
        if K_ENABLE_3D_ICC_TONEMAPPING && self.can_tone_map_for_icc() {
            write_icc_tag(&mut header_data, 20, "Lab ")?;
        } else {
            write_icc_tag(&mut header_data, 20, "XYZ ")?;
        }

        // Date and Time - Placeholder values from libjxl
        write_u16_be(&mut header_data, 24, 2019)?; // Year
        write_u16_be(&mut header_data, 26, 12)?; // Month
        write_u16_be(&mut header_data, 28, 1)?; // Day
        write_u16_be(&mut header_data, 30, 0)?; // Hours
        write_u16_be(&mut header_data, 32, 0)?; // Minutes
        write_u16_be(&mut header_data, 34, 0)?; // Seconds

        write_icc_tag(&mut header_data, 36, "acsp")?;
        write_icc_tag(&mut header_data, 40, "APPL")?;

        // Profile flags
        write_u32_be(&mut header_data, 44, 0)?;
        // Device manufacturer
        write_u32_be(&mut header_data, 48, 0)?;
        // Device model
        write_u32_be(&mut header_data, 52, 0)?;
        // Device attributes
        write_u32_be(&mut header_data, 56, 0)?;
        write_u32_be(&mut header_data, 60, 0)?;

        // Rendering Intent
        write_u32_be(&mut header_data, 64, self.rendering_intent as u32)?;

        // Whitepoint is fixed to D50 for ICC.
        write_u32_be(&mut header_data, 68, 0x0000F6D6)?;
        write_u32_be(&mut header_data, 72, 0x00010000)?;
        write_u32_be(&mut header_data, 76, 0x0000D32D)?;

        // Profile Creator
        write_icc_tag(&mut header_data, 80, CMM_TAG)?;

        // Profile ID (MD5 checksum) (offset 84) - 16 bytes.
        // This is calculated at the end of profile creation and written here.

        // Reserved (offset 100-127) - already zeroed here.

        Ok(header_data)
    }

    pub fn maybe_create_profile(&self) -> Result<Option<Vec<u8>>, Error> {
        // TODO can reuse `check` above? or at least simplify logic/dedup somehow?
        if self.color_space == ColorSpace::Unknown
            || self.tf.transfer_function == TransferFunction::Unknown
        {
            return Ok(None);
        }
        if !matches!(
            self.color_space,
            ColorSpace::RGB | ColorSpace::Gray | ColorSpace::XYB
        ) {
            return Err(Error::InvalidColorSpace);
        }

        if self.color_space == ColorSpace::XYB
            && self.rendering_intent != RenderingIntent::Perceptual
        {
            return Err(Error::InvalidRenderingIntent);
        }
        let header = self.create_icc_header()?;
        let mut tags_data: Vec<u8> = Vec::new();
        let mut collected_tags: Vec<TagInfo> = Vec::new();

        // Create 'desc' (ProfileDescription) tag
        let description_string = self.get_color_encoding_description();

        let desc_tag_start_offset = tags_data.len() as u32; // 0 at this point ...
        create_icc_mluc_tag(&mut tags_data, &description_string)?;
        let desc_tag_unpadded_size = (tags_data.len() as u32) - desc_tag_start_offset;
        pad_to_4_byte_boundary(&mut tags_data);
        collected_tags.push(TagInfo {
            signature: *b"desc",
            offset_in_tags_blob: desc_tag_start_offset,
            size_unpadded: desc_tag_unpadded_size,
        });

        // Create 'cprt' (Copyright) tag
        let copyright_string = "CC0";
        let cprt_tag_start_offset = tags_data.len() as u32;
        create_icc_mluc_tag(&mut tags_data, copyright_string)?;
        let cprt_tag_unpadded_size = (tags_data.len() as u32) - cprt_tag_start_offset;
        pad_to_4_byte_boundary(&mut tags_data);
        collected_tags.push(TagInfo {
            signature: *b"cprt",
            offset_in_tags_blob: cprt_tag_start_offset,
            size_unpadded: cprt_tag_unpadded_size,
        });

        // It is the other way around in libjxl for some reason? TODO: check with Sami!
        match self.color_space {
            ColorSpace::Gray => {
                const D50: [f32; 3] = [0.964203f32, 1.0, 0.824905];
                collected_tags.push(create_icc_xyz_tag(&mut tags_data, &D50)?);
            }
            _ => {
                let (wx, wy) = self.get_resolved_white_point_xy()?;
                collected_tags.push(create_icc_xyz_tag(
                    &mut tags_data,
                    &cie_xyz_from_white_cie_xy(wx, wy)?,
                )?);
            }
        }
        pad_to_4_byte_boundary(&mut tags_data);
        if self.color_space != ColorSpace::Gray {
            let (wx, wy) = self.get_resolved_white_point_xy()?;
            let chad_matrix = create_icc_chad_matrix(wx, wy)?;
            collected_tags.push(create_icc_chad_tag(&mut tags_data, &chad_matrix)?);
            pad_to_4_byte_boundary(&mut tags_data);
        }

        if self.color_space == ColorSpace::RGB {
            if let Some(tag_info) = self.create_icc_cicp_tag_data(&mut tags_data)? {
                collected_tags.push(tag_info);
            }
        }

        // TODO: add the more tags

        // Construct the Tag Table bytes
        let mut tag_table_bytes: Vec<u8> = Vec::new();
        // First, the number of tags (u32)
        tag_table_bytes.extend_from_slice(&(collected_tags.len() as u32).to_be_bytes());

        let header_size = header.len() as u32;
        // Each entry in the tag table on disk is 12 bytes: signature (4), offset (4), size (4)
        let tag_table_on_disk_size = 4 + (collected_tags.len() as u32 * 12);

        for tag_info in &collected_tags {
            tag_table_bytes.extend_from_slice(&tag_info.signature);
            // The offset in the tag table is absolute from the start of the ICC profile file
            let final_profile_offset_for_tag =
                header_size + tag_table_on_disk_size + tag_info.offset_in_tags_blob;
            tag_table_bytes.extend_from_slice(&final_profile_offset_for_tag.to_be_bytes());
            // In https://www.color.org/specification/ICC.1-2022-05.pdf, section 7.3.5 reads:
            //
            // "The value of the tag data element size shall be the number of actual data
            // bytes and shall not include any padding at the end of the tag data element."
            //
            // The reference from conformance tests and libjxl use the padded size here instead.
            tag_table_bytes.extend_from_slice(&tag_info.size_unpadded.to_be_bytes());
        }

        // Assemble the final ICC profile parts: header + tag_table + tags_data
        let mut final_icc_profile_data: Vec<u8> =
            Vec::with_capacity(header.len() + tag_table_bytes.len() + tags_data.len());
        final_icc_profile_data.extend_from_slice(&header);
        final_icc_profile_data.extend_from_slice(&tag_table_bytes);
        final_icc_profile_data.extend_from_slice(&tags_data);

        // Update the profile size in the header (at offset 0)
        let total_profile_size = final_icc_profile_data.len() as u32;
        write_u32_be(&mut final_icc_profile_data, 0, total_profile_size)?;
        // TODO: MD5 hashing?
        Ok(Some(final_icc_profile_data))
    }
}
