// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::error::{Error, Result};
use enum_iterator::{cardinality, Sequence};

#[allow(clippy::upper_case_acronyms)]
#[derive(Copy, Clone, Debug, PartialEq, Sequence)]
pub enum HfTransformType {
    // Regular block size DCT
    DCT = 0,
    // Encode pixels without transforming
    IDENTITY = 1, // a.k.a HORNUSS
    // Use 2-by-2 DCT
    DCT2X2 = 2,
    // Use 4-by-4 DCT
    DCT4X4 = 3,
    // Use 16-by-16 DCT
    DCT16X16 = 4,
    // Use 32-by-32 DCT
    DCT32X32 = 5,
    // Use 16-by-8 DCT
    DCT16X8 = 6,
    // Use 8-by-16 DCT
    DCT8X16 = 7,
    // Use 32-by-8 DCT
    DCT32X8 = 8,
    // Use 8-by-32 DCT
    DCT8X32 = 9,
    // Use 32-by-16 DCT
    DCT32X16 = 10,
    // Use 16-by-32 DCT
    DCT16X32 = 11,
    // 4x8 and 8x4 DCT
    DCT4X8 = 12,
    DCT8X4 = 13,
    // Corner-DCT.
    AFV0 = 14,
    AFV1 = 15,
    AFV2 = 16,
    AFV3 = 17,
    // Larger DCTs
    DCT64X64 = 18,
    DCT64X32 = 19,
    DCT32X64 = 20,
    // No transforms smaller than 64x64 are allowed below.
    DCT128X128 = 21,
    DCT128X64 = 22,
    DCT64X128 = 23,
    DCT256X256 = 24,
    DCT256X128 = 25,
    DCT128X256 = 26,
}

pub const INVALID_TRANSFORM: u8 = cardinality::<HfTransformType>() as u8;

pub fn get_transform_type(raw_type: i32) -> Result<HfTransformType, Error> {
    let lut: [HfTransformType; cardinality::<HfTransformType>()] = [
        HfTransformType::DCT,
        HfTransformType::IDENTITY,
        HfTransformType::DCT2X2,
        HfTransformType::DCT4X4,
        HfTransformType::DCT16X16,
        HfTransformType::DCT32X32,
        HfTransformType::DCT16X8,
        HfTransformType::DCT8X16,
        HfTransformType::DCT32X8,
        HfTransformType::DCT8X32,
        HfTransformType::DCT32X16,
        HfTransformType::DCT16X32,
        HfTransformType::DCT4X8,
        HfTransformType::DCT8X4,
        HfTransformType::AFV0,
        HfTransformType::AFV1,
        HfTransformType::AFV2,
        HfTransformType::AFV3,
        HfTransformType::DCT64X64,
        HfTransformType::DCT64X32,
        HfTransformType::DCT32X64,
        HfTransformType::DCT128X128,
        HfTransformType::DCT128X64,
        HfTransformType::DCT64X128,
        HfTransformType::DCT256X256,
        HfTransformType::DCT256X128,
        HfTransformType::DCT128X256,
    ];
    if raw_type < 0 || raw_type >= INVALID_TRANSFORM.into() {
        Err(Error::InvalidVarDCTTransform(raw_type))
    } else {
        Ok(lut[raw_type as usize])
    }
}

pub fn covered_blocks_x(transform: HfTransformType) -> u32 {
    let lut: [u32; cardinality::<HfTransformType>()] = [
        1, 1, 1, 1, 2, 4, 1, 2, 1, 4, 2, 4, 1, 1, 1, 1, 1, 1, 8, 4, 8, 16, 8, 16, 32, 16, 32,
    ];
    lut[transform as usize]
}

pub fn covered_blocks_y(transform: HfTransformType) -> u32 {
    let lut: [u32; cardinality::<HfTransformType>()] = [
        1, 1, 1, 1, 2, 4, 2, 1, 4, 1, 4, 2, 1, 1, 1, 1, 1, 1, 8, 8, 4, 16, 16, 8, 32, 32, 16,
    ];
    lut[transform as usize]
}

pub fn block_shape_id(transform: HfTransformType) -> u32 {
    let lut: [u32; cardinality::<HfTransformType>()] = [
        0, 1, 1, 1, 2, 3, 4, 4, 5, 5, 6, 6, 1, 1, 1, 1, 1, 1, 7, 8, 8, 9, 10, 10, 11, 12, 12,
    ];
    lut[transform as usize]
}
/// Enum representing the encoding mode for dequantization matrices,
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DequantEncodingMode {
    Identity, // a.k.a Hornuss
    Dct2,
    Dct4,
    Dct4x8,
    Afv,
    Dct,
    Raw,
}

/// Structure holding the default parameters for dequantization matrices
#[derive(Debug, Clone, Copy)]
pub struct DequantizationDefaults {
    pub mode: DequantEncodingMode,
    /// Optional dct_params matrix. Structure: Some(&[row1, row2, ...]), where row is &[val1, val2, ...]
    pub dct_params: Option<&'static [&'static [f64]]>,
    /// Optional params matrix.
    pub params: Option<&'static [&'static [f64]]>,
}

/// Structure holding dequantization metadata for a given HfTransformType.
/// This includes the parameter index, matrix size, and the default parameters.
#[derive(Debug, Clone, Copy)]
pub struct HfTransformDequantMeta {
    /// Parameter index for dequantization matrices
    pub parameter_index: u8,
    /// Dequantization matrix size (rows, columns).
    pub matrix_size: (u16, u16),
    /// The static struct containing default dequantization parameters.
    pub default_params: &'static DequantizationDefaults,
}

// --- Macros to save some writing
macro_rules! r {
    ($($val:expr),* $(,)?) => { &[$($val),*] as &'static [f64] };
}
macro_rules! define_params {
    ($($row:expr),* $(,)?) => { Some(&[$($row),*]) };
}


const DCT4X8_PARAMS_CONST_VAL: &[&[f64]] = &[
    r![2198.050556016380522, -0.96269623020744692, -0.76194253026666783, -0.6551140670773547],
    r![764.3655248643528689, -0.92630200888366945, -0.9675229603596517, -0.27845290869168118],
    r![527.107573587542228, -1.4594385811273854, -1.450082094097871593, -1.5843722511996204],
];

const DCT4X4_PARAMS_CONST_VAL: &[&[f64]] = &[
    r![2200.0, 0.0, 0.0, 0.0],
    r![392.0, 0.0, 0.0, 0.0],
    r![112.0, -0.25, -0.25, -0.5],
];


static DEFAULT_PARAMS_DCT8X8: DequantizationDefaults = DequantizationDefaults {
    mode: DequantEncodingMode::Dct,
    dct_params: define_params!(
        r![3150.0, 0.0, -0.4, -0.4, -0.4, -2.0],
        r![560.0, 0.0, -0.3, -0.3, -0.3, -0.3],
        r![512.0, -2.0, -1.0, 0.0, -1.0, -2.0]
    ),
    params: None,
};

static DEFAULT_PARAMS_HORNUSS: DequantizationDefaults = DequantizationDefaults {
    mode: DequantEncodingMode::Identity,
    dct_params: None,
    params: define_params!(
        r![280.0, 3160.0, 3160.0],
        r![60.0, 864.0, 864.0],
        r![18.0, 200.0, 200.0]
    ),
};

static DEFAULT_PARAMS_DCT2X2: DequantizationDefaults = DequantizationDefaults {
    mode: DequantEncodingMode::Dct2,
    dct_params: None,
    params: define_params!(
        r![3840.0, 2560.0, 1280.0, 640.0, 480.0, 300.0],
        r![960.0, 640.0, 320.0, 180.0, 140.0, 120.0],
        r![640.0, 320.0, 128.0, 64.0, 32.0, 16.0]
    ),
};

static DEFAULT_PARAMS_DCT4X4: DequantizationDefaults = DequantizationDefaults {
    mode: DequantEncodingMode::Dct4,
    dct_params: Some(&DCT4X4_PARAMS_CONST_VAL),
    params: define_params!(
        r![1.0, 1.0],
        r![1.0, 1.0],
        r![1.0, 1.0]
    ),
};

static DEFAULT_PARAMS_DCT16X16: DequantizationDefaults = DequantizationDefaults {
    mode: DequantEncodingMode::Dct,
    dct_params: define_params!(
        r![8996.8725711814115328, -1.3000777393353804, -0.49424529824571225, -0.439093774457103443, -0.6350101832695744, -0.90177264050827612, -1.6162099239887414],
        r![3191.48366296844234752, -0.67424582104194355, -0.80745813428471001, -0.44925837484843441, -0.35865440981033403, -0.31322389111877305, -0.37615025315725483],
        r![1157.50408145487200256, -2.0531423165804414, -1.4, -0.50687130033378396, -0.42708730624733904, -1.4856834539296244, -4.9209142884401604]
    ),
    params: None,
};

static DEFAULT_PARAMS_DCT32X32: DequantizationDefaults = DequantizationDefaults {
    mode: DequantEncodingMode::Dct,
    dct_params: define_params!(
        r![15718.40830982518931456, -1.025, -0.98, -0.9012, -0.4, -0.48819395464, -0.421064, -0.27],
        r![7305.7636810695983104, -0.8041958212306401, -0.7633036457487539, -0.55660379990111464, -0.49785304658857626, -0.43699592683512467, -0.40180866526242109, -0.27321683125358037],
        r![3803.53173721215041536, -3.060733579805728, -2.0413270132490346, -2.0235650159727417, -0.5495389509954993, -0.4, -0.4, -0.3]
    ),
    params: None,
};

static DEFAULT_PARAMS_DCT16X8_8X16: DequantizationDefaults = DequantizationDefaults {
    mode: DequantEncodingMode::Dct,
    dct_params: define_params!(
        r![7240.7734393502, -0.7, -0.7, -0.2, -0.2, -0.2, -0.5],
        r![1448.15468787004, -0.5, -0.5, -0.5, -0.2, -0.2, -0.2],
        r![506.854140754517, -1.4, -0.2, -0.5, -0.5, -1.5, -3.6]
    ),
    params: None,
};

static DEFAULT_PARAMS_DCT32X8_8X32: DequantizationDefaults = DequantizationDefaults {
    mode: DequantEncodingMode::Dct,
    dct_params: define_params!(
        r![16283.2494710648897, -1.7812845336559429, -1.6309059012653515, -1.0382179034313539, -0.85, -0.7, -0.9, -1.2360638576849587],
        r![5089.15750884921511936, -0.320049391452786891, -0.35362849922161446, -0.30340000000000003, -0.61, -0.5, -0.5, -0.6],
        r![3397.77603275308720128, -0.321327362693153371, -0.34507619223117997, -0.70340000000000003, -0.9, -1.0, -1.0, -1.1754605576265209]
    ),
    params: None,
};

static DEFAULT_PARAMS_DCT16X32_32X16: DequantizationDefaults = DequantizationDefaults {
    mode: DequantEncodingMode::Dct,
    dct_params: define_params!(
        r![13844.97076442300573, -0.97113799999999995, -0.658, -0.42026, -0.22712, -0.2206, -0.226, -0.6],
        r![4798.964084220744293, -0.61125308982767057, -0.83770786552491361, -0.79014862079498627, -0.2692727459704829, -0.38272769465388551, -0.22924222653091453, -0.20719098826199578],
        r![1807.236946760964614, -1.2, -1.2, -0.7, -0.7, -0.7, -0.4, -0.5]
    ),
    params: None,
};

static DEFAULT_PARAMS_DCT4X8_8X4: DequantizationDefaults = DequantizationDefaults {
    mode: DequantEncodingMode::Dct4x8,
    dct_params: Some(&DCT4X8_PARAMS_CONST_VAL),
    params: define_params!( r![1.0], r![1.0], r![1.0] ),
};

static DEFAULT_PARAMS_AFV: DequantizationDefaults = DequantizationDefaults {
    mode: DequantEncodingMode::Afv,
    dct_params: Some(&DCT4X8_PARAMS_CONST_VAL),
    params: define_params!(
        r![3072.0, 3072.0, 256.0, 256.0, 256.0, 414.0, 0.0, 0.0, 0.0],
        r![1024.0, 1024.0, 50.0, 50.0, 50.0, 58.0, 0.0, 0.0, 0.0],
        r![384.0, 384.0, 12.0, 12.0, 12.0, 22.0, -0.25, -0.25, -0.25]
    ),
};

static DEFAULT_PARAMS_DCT64X64: DequantizationDefaults = DequantizationDefaults {
    mode: DequantEncodingMode::Dct,
    dct_params: define_params!(
        r![23966.1665298448605, -1.025, -0.78, -0.65012, -0.19041574084286472, -0.20819395464, -0.421064, -0.32733845535848671],
        r![8380.19148390090414, -0.3041958212306401, -0.3633036457487539, -0.35660379990111464, -0.3443074455424403, -0.33699592683512467, -0.30180866526242109, -0.27321683125358037],
        r![4493.02378009847706, -1.2, -1.2, -0.8, -0.7, -0.7, -0.4, -0.5]
    ),
    params: None,
};

static DEFAULT_PARAMS_DCT32X64_64X32: DequantizationDefaults = DequantizationDefaults {
    mode: DequantEncodingMode::Dct,
    dct_params: define_params!(
        r![15358.89804933239925, -1.025, -0.78, -0.65012, -0.19041574084286472, -0.20819395464, -0.421064, -0.32733845535848671],
        r![5597.360516150652990, -0.3041958212306401, -0.3633036457487539, -0.35660379990111464, -0.3443074455424403, -0.33699592683512467, -0.30180866526242109, -0.27321683125358037],
        r![2919.961618960011210, -1.2, -1.2, -0.8, -0.7, -0.7, -0.4, -0.5]
    ),
    params: None,
};

static DEFAULT_PARAMS_DCT128X128: DequantizationDefaults = DequantizationDefaults {
    mode: DequantEncodingMode::Dct,
    dct_params: define_params!(
        r![47932.3330596897210, -1.025, -0.78, -0.65012, -0.19041574084286472, -0.20819395464, -0.421064, -0.32733845535848671],
        r![16760.38296780180828, -0.3041958212306401, -0.3633036457487539, -0.35660379990111464, -0.3443074455424403, -0.33699592683512467, -0.30180866526242109, -0.27321683125358037],
        r![8986.04756019695412, -1.2, -1.2, -0.8, -0.7, -0.7, -0.4, -0.5]
    ),
    params: None,
};

static DEFAULT_PARAMS_DCT64X128_128X64: DequantizationDefaults = DequantizationDefaults {
    mode: DequantEncodingMode::Dct,
    dct_params: define_params!(
        r![30717.796098664792, -1.025, -0.78, -0.65012, -0.19041574084286472, -0.20819395464, -0.421064, -0.32733845535848671],
        r![11194.72103230130598, -0.3041958212306401, -0.3633036457487539, -0.35660379990111464, -0.3443074455424403, -0.33699592683512467, -0.30180866526242109, -0.27321683125358037],
        r![5839.92323792002242, -1.2, -1.2, -0.8, -0.7, -0.7, -0.4, -0.5]
    ),
    params: None,
};

static DEFAULT_PARAMS_DCT256X256: DequantizationDefaults = DequantizationDefaults {
    mode: DequantEncodingMode::Dct,
    dct_params: define_params!(
        r![95864.6661193794420, -1.025, -0.78, -0.65012, -0.19041574084286472, -0.20819395464, -0.421064, -0.32733845535848671],
        r![33520.76593560361656, -0.3041958212306401, -0.3633036457487539, -0.35660379990111464, -0.3443074455424403, -0.33699592683512467, -0.30180866526242109, -0.27321683125358037],
        r![17972.09512039390824, -1.2, -1.2, -0.8, -0.7, -0.7, -0.4, -0.5]
    ),
    params: None,
};

static DEFAULT_PARAMS_DCT128X256_256X128: DequantizationDefaults = DequantizationDefaults {
    mode: DequantEncodingMode::Dct,
    dct_params: define_params!(
        r![61435.5921973295970, -1.025, -0.78, -0.65012, -0.19041574084286472, -0.20819395464, -0.421064, -0.32733845535848671],
        r![24209.44206460261196, -0.3041958212306401, -0.3633036457487539, -0.35660379990111464, -0.3443074455424403, -0.33699592683512467, -0.30180866526242109, -0.27321683125358037],
        r![12979.84647584004484, -1.2, -1.2, -0.8, -0.7, -0.7, -0.4, -0.5]
    ),
    params: None,
};


pub const HF_TRANSFORM_DEQUANT_META_LUT: [HfTransformDequantMeta; cardinality::<HfTransformType>()] = [
    HfTransformDequantMeta { parameter_index: 0, matrix_size: (8,8), default_params: &DEFAULT_PARAMS_DCT8X8 },
    HfTransformDequantMeta { parameter_index: 1, matrix_size: (8,8), default_params: &DEFAULT_PARAMS_HORNUSS },
    HfTransformDequantMeta { parameter_index: 2, matrix_size: (8,8), default_params: &DEFAULT_PARAMS_DCT2X2 },
    HfTransformDequantMeta { parameter_index: 3, matrix_size: (8,8), default_params: &DEFAULT_PARAMS_DCT4X4 },
    HfTransformDequantMeta { parameter_index: 4, matrix_size: (16,16), default_params: &DEFAULT_PARAMS_DCT16X16 },
    HfTransformDequantMeta { parameter_index: 5, matrix_size: (32,32), default_params: &DEFAULT_PARAMS_DCT32X32 },
    HfTransformDequantMeta { parameter_index: 6, matrix_size: (8,16), default_params: &DEFAULT_PARAMS_DCT16X8_8X16 },
    HfTransformDequantMeta { parameter_index: 6, matrix_size: (8,16), default_params: &DEFAULT_PARAMS_DCT16X8_8X16 },
    HfTransformDequantMeta { parameter_index: 7, matrix_size: (8,32), default_params: &DEFAULT_PARAMS_DCT32X8_8X32 },
    HfTransformDequantMeta { parameter_index: 7, matrix_size: (8,32), default_params: &DEFAULT_PARAMS_DCT32X8_8X32 },
    HfTransformDequantMeta { parameter_index: 8, matrix_size: (16,32), default_params: &DEFAULT_PARAMS_DCT16X32_32X16 },
    HfTransformDequantMeta { parameter_index: 8, matrix_size: (16,32), default_params: &DEFAULT_PARAMS_DCT16X32_32X16 },
    HfTransformDequantMeta { parameter_index: 9, matrix_size: (8,8), default_params: &DEFAULT_PARAMS_DCT4X8_8X4 },
    HfTransformDequantMeta { parameter_index: 9, matrix_size: (8,8), default_params: &DEFAULT_PARAMS_DCT4X8_8X4 },
    HfTransformDequantMeta { parameter_index: 10, matrix_size: (8,8), default_params: &DEFAULT_PARAMS_AFV },
    HfTransformDequantMeta { parameter_index: 10, matrix_size: (8,8), default_params: &DEFAULT_PARAMS_AFV },
    HfTransformDequantMeta { parameter_index: 10, matrix_size: (8,8), default_params: &DEFAULT_PARAMS_AFV },
    HfTransformDequantMeta { parameter_index: 10, matrix_size: (8,8), default_params: &DEFAULT_PARAMS_AFV },
    HfTransformDequantMeta { parameter_index: 11, matrix_size: (64,64), default_params: &DEFAULT_PARAMS_DCT64X64 },
    HfTransformDequantMeta { parameter_index: 12, matrix_size: (32,64), default_params: &DEFAULT_PARAMS_DCT32X64_64X32 },
    HfTransformDequantMeta { parameter_index: 12, matrix_size: (32,64), default_params: &DEFAULT_PARAMS_DCT32X64_64X32 },
    HfTransformDequantMeta { parameter_index: 13, matrix_size: (128,128), default_params: &DEFAULT_PARAMS_DCT128X128 },
    HfTransformDequantMeta { parameter_index: 14, matrix_size: (64,128), default_params: &DEFAULT_PARAMS_DCT64X128_128X64 },
    HfTransformDequantMeta { parameter_index: 14, matrix_size: (64,128), default_params: &DEFAULT_PARAMS_DCT64X128_128X64 },
    HfTransformDequantMeta { parameter_index: 15, matrix_size: (256,256), default_params: &DEFAULT_PARAMS_DCT256X256 },
    HfTransformDequantMeta { parameter_index: 16, matrix_size: (128,256), default_params: &DEFAULT_PARAMS_DCT128X256_256X128 },
    HfTransformDequantMeta { parameter_index: 16, matrix_size: (128,256), default_params: &DEFAULT_PARAMS_DCT128X256_256X128 },
];