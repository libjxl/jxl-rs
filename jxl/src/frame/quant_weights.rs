// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(dead_code)]

use std::sync::OnceLock;

use enum_iterator::{cardinality, Sequence};

use crate::{
    bit_reader::BitReader,
    error::Result,
    frame::transform_map::{self, HfTransformType},
    BLOCK_SIZE,
};

pub const INV_LF_QUANT: [f32; 3] = [4096.0, 512.0, 256.0];

pub const LF_QUANT: [f32; 3] = [
    1.0 / INV_LF_QUANT[0],
    1.0 / INV_LF_QUANT[1],
    1.0 / INV_LF_QUANT[2],
];

const MAX_QUANT_TABLE_SIZE: usize = transform_map::MAX_COEFF_AREA;
const NUM_PREDEFINED_TABLES: usize = 1;
const CEIL_LOG2_NUM_PREDEFINED_TABLES: usize = 0;
const LOG2_NUM_QUANT_MODES: usize = 3;
type FixedDctQuantWeightParams<const NUM_DISTANCE_BANDS: usize> = [[f32; NUM_DISTANCE_BANDS]; 3];
struct DctQuantWeightParams {
    params: [[f32; Self::MAX_DISTANCE_BANDS]; 3],
    num_bands: usize,
}
impl DctQuantWeightParams {
    const LOG2_MAX_DISTANCE_BANDS: usize = 4;
    const MAX_DISTANCE_BANDS: usize = 1 + (1 << Self::LOG2_MAX_DISTANCE_BANDS);

    pub fn from_array<const N: usize>(values: &[[f32; N]; 3]) -> Self {
        let mut result = Self {
            params: [[0.0; Self::MAX_DISTANCE_BANDS]; 3],
            num_bands: N,
        };
        for (params, values) in result.params.iter_mut().zip(values) {
            params.copy_from_slice(values);
        }
        result
    }
}

enum QuantEncoding {
    Library {
        predefined: u8,
    },
    Identity {
        xyb_weights: [[f32; 3]; 3],
    },
    Dct2 {
        xyb_weights: [[f32; 6]; 3],
    },
    Dct4 {
        params: FixedDctQuantWeightParams<4>,
        xyb_mul: [[f32; 2]; 3],
    },
    Dct4x8 {
        params: FixedDctQuantWeightParams<4>,
        xyb_mul: [f32; 3],
    },
    Dct {
        params: DctQuantWeightParams,
    },
    Afv {
        params4x8: FixedDctQuantWeightParams<4>,
        params4x4: FixedDctQuantWeightParams<4>,
        weights: [[f32; 9]; 3],
    },
    Raw {
        qtable: Vec<i32>,
        qtable_den: f32,
    },
}

impl QuantEncoding {
    pub fn raw_from_qtable(qtable: Vec<i32>, shift: i32) -> Self {
        Self::Raw {
            qtable,
            qtable_den: (1 << shift) as f32 * (1.0 / (8.0 * 255.0)),
        }
    }
}

#[derive(Sequence)]
enum QuantTable {
    Dct,
    Identity,
    Dct2x2,
    Dct4x4,
    Dct16x16,
    Dct32x32,
    // Dct16x8
    Dct8x16,
    // Dct32x8
    Dct8x32,
    // Dct32x16
    Dct16x32,
    Dct4x8,
    // Dct8x4
    Afv0,
    // Afv1
    // Afv2
    // Afv3
    Dct64x64,
    // Dct64x32,
    Dct32x64,
    Dct128x128,
    // Dct128x64,
    Dct64x128,
    Dct256x256,
    // Dct256x128,
    Dct128x256,
}

const NUM_QUANT_TABLES: usize = cardinality::<QuantTable>();

impl QuantTable {
    fn for_strategy(strategy: HfTransformType) -> QuantTable {
        match strategy {
            HfTransformType::DCT => QuantTable::Dct,
            HfTransformType::IDENTITY => QuantTable::Identity,
            HfTransformType::DCT2X2 => QuantTable::Dct2x2,
            HfTransformType::DCT4X4 => QuantTable::Dct4x4,
            HfTransformType::DCT16X16 => QuantTable::Dct16x16,
            HfTransformType::DCT32X32 => QuantTable::Dct32x32,
            HfTransformType::DCT16X8 | HfTransformType::DCT8X16 => QuantTable::Dct8x16,
            HfTransformType::DCT32X8 | HfTransformType::DCT8X32 => QuantTable::Dct8x32,
            HfTransformType::DCT32X16 | HfTransformType::DCT16X32 => QuantTable::Dct16x32,
            HfTransformType::DCT4X8 | HfTransformType::DCT8X4 => QuantTable::Dct4x8,
            HfTransformType::AFV0
            | HfTransformType::AFV1
            | HfTransformType::AFV2
            | HfTransformType::AFV3 => QuantTable::Afv0,
            HfTransformType::DCT64X64 => QuantTable::Dct64x64,
            HfTransformType::DCT64X32 | HfTransformType::DCT32X64 => QuantTable::Dct32x64,
            HfTransformType::DCT128X128 => QuantTable::Dct128x128,
            HfTransformType::DCT128X64 | HfTransformType::DCT64X128 => QuantTable::Dct64x128,
            HfTransformType::DCT256X256 => QuantTable::Dct256x256,
            HfTransformType::DCT256X128 | HfTransformType::DCT128X256 => QuantTable::Dct128x256,
        }
    }
}

struct DequantMatrices {
    computed_mask: u32,
    table: [f32; Self::TOTAL_TABLE_SIZE],
    inv_table: [f32; Self::TOTAL_TABLE_SIZE],
    table_offsets: [usize; HfTransformType::CARDINALITY * 3],
    encodings: Vec<QuantEncoding>,
}

#[allow(clippy::excessive_precision)]
impl DequantMatrices {
    fn dct() -> QuantEncoding {
        QuantEncoding::Dct {
            params: DctQuantWeightParams::from_array(&[
                [3150.0, 0.0, -0.4, -0.4, -0.4, -2.0],
                [560.0, 0.0, -0.3, -0.3, -0.3, -0.3],
                [512.0, -2.0, -1.0, 0.0, -1.0, -2.0],
            ]),
        }
    }
    fn id() -> QuantEncoding {
        QuantEncoding::Identity {
            xyb_weights: [
                [280.0, 3160.0, 3160.0],
                [60.0, 864.0, 864.0],
                [18.0, 200.0, 200.0],
            ],
        }
    }
    fn dct2x2() -> QuantEncoding {
        QuantEncoding::Dct2 {
            xyb_weights: [
                [3840.0, 2560.0, 1280.0, 640.0, 480.0, 300.0],
                [960.0, 640.0, 320.0, 180.0, 140.0, 120.0],
                [640.0, 320.0, 128.0, 64.0, 32.0, 16.0],
            ],
        }
    }
    fn dct4x4() -> QuantEncoding {
        QuantEncoding::Dct4 {
            params: [
                [2200.0, 0.0, 0.0, 0.0],
                [392.0, 0.0, 0.0, 0.0],
                [112.0, -0.25, -0.25, -0.5],
            ],
            xyb_mul: [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
        }
    }
    fn dct16x16() -> QuantEncoding {
        QuantEncoding::Dct {
            params: DctQuantWeightParams::from_array(&[
                [
                    8996.8725711814115328,
                    -1.3000777393353804,
                    -0.49424529824571225,
                    -0.439093774457103443,
                    -0.6350101832695744,
                    -0.90177264050827612,
                    -1.6162099239887414,
                ],
                [
                    3191.48366296844234752,
                    -0.67424582104194355,
                    -0.80745813428471001,
                    -0.44925837484843441,
                    -0.35865440981033403,
                    -0.31322389111877305,
                    -0.37615025315725483,
                ],
                [
                    1157.50408145487200256,
                    -2.0531423165804414,
                    -1.4,
                    -0.50687130033378396,
                    -0.42708730624733904,
                    -1.4856834539296244,
                    -4.9209142884401604,
                ],
            ]),
        }
    }
    fn dct32x32() -> QuantEncoding {
        QuantEncoding::Dct {
            params: DctQuantWeightParams::from_array(&[
                [
                    15718.40830982518931456,
                    -1.025,
                    -0.98,
                    -0.9012,
                    -0.4,
                    -0.48819395464,
                    -0.421064,
                    -0.27,
                ],
                [
                    7305.7636810695983104,
                    -0.8041958212306401,
                    -0.7633036457487539,
                    -0.55660379990111464,
                    -0.49785304658857626,
                    -0.43699592683512467,
                    -0.40180866526242109,
                    -0.27321683125358037,
                ],
                [
                    3803.53173721215041536,
                    -3.060733579805728,
                    -2.0413270132490346,
                    -2.0235650159727417,
                    -0.5495389509954993,
                    -0.4,
                    -0.4,
                    -0.3,
                ],
            ]),
        }
    }

    // dct16x8
    fn dct8x16() -> QuantEncoding {
        QuantEncoding::Dct {
            params: DctQuantWeightParams::from_array(&[
                [7240.7734393502, -0.7, -0.7, -0.2, -0.2, -0.2, -0.5],
                [1448.15468787004, -0.5, -0.5, -0.5, -0.2, -0.2, -0.2],
                [506.854140754517, -1.4, -0.2, -0.5, -0.5, -1.5, -3.6],
            ]),
        }
    }

    // dct32x8
    fn dct8x32() -> QuantEncoding {
        QuantEncoding::Dct {
            params: DctQuantWeightParams::from_array(&[
                [
                    16283.2494710648897,
                    -1.7812845336559429,
                    -1.6309059012653515,
                    -1.0382179034313539,
                    -0.85,
                    -0.7,
                    -0.9,
                    -1.2360638576849587,
                ],
                [
                    5089.15750884921511936,
                    -0.320049391452786891,
                    -0.35362849922161446,
                    -0.30340000000000003,
                    -0.61,
                    -0.5,
                    -0.5,
                    -0.6,
                ],
                [
                    3397.77603275308720128,
                    -0.321327362693153371,
                    -0.34507619223117997,
                    -0.70340000000000003,
                    -0.9,
                    -1.0,
                    -1.0,
                    -1.1754605576265209,
                ],
            ]),
        }
    }

    // dct32x16
    fn dct16x32() -> QuantEncoding {
        QuantEncoding::Dct {
            params: DctQuantWeightParams::from_array(&[
                [
                    13844.97076442300573,
                    -0.97113799999999995,
                    -0.658,
                    -0.42026,
                    -0.22712,
                    -0.2206,
                    -0.226,
                    -0.6,
                ],
                [
                    4798.964084220744293,
                    -0.61125308982767057,
                    -0.83770786552491361,
                    -0.79014862079498627,
                    -0.2692727459704829,
                    -0.38272769465388551,
                    -0.22924222653091453,
                    -0.20719098826199578,
                ],
                [
                    1807.236946760964614,
                    -1.2,
                    -1.2,
                    -0.7,
                    -0.7,
                    -0.7,
                    -0.4,
                    -0.5,
                ],
            ]),
        }
    }

    // dct8x4
    fn dct4x8() -> QuantEncoding {
        QuantEncoding::Dct4x8 {
            params: [
                [
                    2198.050556016380522,
                    -0.96269623020744692,
                    -0.76194253026666783,
                    -0.6551140670773547,
                ],
                [
                    764.3655248643528689,
                    -0.92630200888366945,
                    -0.9675229603596517,
                    -0.27845290869168118,
                ],
                [
                    527.107573587542228,
                    -1.4594385811273854,
                    -1.450082094097871593,
                    -1.5843722511996204,
                ],
            ],
            xyb_mul: [1.0, 1.0, 1.0],
        }
    }
    // AFV
    fn afv0() -> QuantEncoding {
        let QuantEncoding::Dct4x8 {
            params: params4x8, ..
        } = Self::dct4x8()
        else {
            unreachable!();
        };
        let QuantEncoding::Dct4 {
            params: params4x4, ..
        } = Self::dct4x4()
        else {
            unreachable!()
        };
        QuantEncoding::Afv {
            params4x8,
            params4x4,
            weights: [
                [
                    3072.0, 3072.0, // 4x4/4x8 DC tendency.
                    256.0, 256.0, 256.0, // AFV corner.
                    414.0, 0.0, 0.0, 0.0, // AFV high freqs.
                ],
                [
                    1024.0, 1024.0, // 4x4/4x8 DC tendency.
                    50.0, 50.0, 50.0, // AFV corner.
                    58.0, 0.0, 0.0, 0.0, // AFV high freqs.
                ],
                [
                    384.0, 384.0, // 4x4/4x8 DC tendency.
                    12.0, 12.0, 12.0, // AFV corner.
                    22.0, -0.25, -0.25, -0.25, // AFV high freqs.
                ],
            ],
        }
    }

    fn dct64x64() -> QuantEncoding {
        QuantEncoding::Dct {
            params: DctQuantWeightParams::from_array(&[
                [
                    0.9 * 26629.073922049845,
                    -1.025,
                    -0.78,
                    -0.65012,
                    -0.19041574084286472,
                    -0.20819395464,
                    -0.421064,
                    -0.32733845535848671,
                ],
                [
                    0.9 * 9311.3238710010046,
                    -0.3041958212306401,
                    -0.3633036457487539,
                    -0.35660379990111464,
                    -0.3443074455424403,
                    -0.33699592683512467,
                    -0.30180866526242109,
                    -0.27321683125358037,
                ],
                [
                    0.9 * 4992.2486445538634,
                    -1.2,
                    -1.2,
                    -0.8,
                    -0.7,
                    -0.7,
                    -0.4,
                    -0.5,
                ],
            ]),
        }
    }

    // dct64x32
    fn dct32x64() -> QuantEncoding {
        QuantEncoding::Dct {
            params: DctQuantWeightParams::from_array(&[
                [
                    0.65 * 23629.073922049845,
                    -1.025,
                    -0.78,
                    -0.65012,
                    -0.19041574084286472,
                    -0.20819395464,
                    -0.421064,
                    -0.32733845535848671,
                ],
                [
                    0.65 * 8611.3238710010046,
                    -0.3041958212306401,
                    -0.3633036457487539,
                    -0.35660379990111464,
                    -0.3443074455424403,
                    -0.33699592683512467,
                    -0.30180866526242109,
                    -0.27321683125358037,
                ],
                [
                    0.65 * 4492.2486445538634,
                    -1.2,
                    -1.2,
                    -0.8,
                    -0.7,
                    -0.7,
                    -0.4,
                    -0.5,
                ],
            ]),
        }
    }
    fn dct128x128() -> QuantEncoding {
        QuantEncoding::Dct {
            params: DctQuantWeightParams::from_array(&[
                [
                    1.8 * 26629.073922049845,
                    -1.025,
                    -0.78,
                    -0.65012,
                    -0.19041574084286472,
                    -0.20819395464,
                    -0.421064,
                    -0.32733845535848671,
                ],
                [
                    1.8 * 9311.3238710010046,
                    -0.3041958212306401,
                    -0.3633036457487539,
                    -0.35660379990111464,
                    -0.3443074455424403,
                    -0.33699592683512467,
                    -0.30180866526242109,
                    -0.27321683125358037,
                ],
                [
                    1.8 * 4992.2486445538634,
                    -1.2,
                    -1.2,
                    -0.8,
                    -0.7,
                    -0.7,
                    -0.4,
                    -0.5,
                ],
            ]),
        }
    }

    // dct128x64
    fn dct64x128() -> QuantEncoding {
        QuantEncoding::Dct {
            params: DctQuantWeightParams::from_array(&[
                [
                    1.3 * 23629.073922049845,
                    -1.025,
                    -0.78,
                    -0.65012,
                    -0.19041574084286472,
                    -0.20819395464,
                    -0.421064,
                    -0.32733845535848671,
                ],
                [
                    1.3 * 8611.3238710010046,
                    -0.3041958212306401,
                    -0.3633036457487539,
                    -0.35660379990111464,
                    -0.3443074455424403,
                    -0.33699592683512467,
                    -0.30180866526242109,
                    -0.27321683125358037,
                ],
                [
                    1.3 * 4492.2486445538634,
                    -1.2,
                    -1.2,
                    -0.8,
                    -0.7,
                    -0.7,
                    -0.4,
                    -0.5,
                ],
            ]),
        }
    }
    fn dct256x256() -> QuantEncoding {
        QuantEncoding::Dct {
            params: DctQuantWeightParams::from_array(&[
                [
                    3.6 * 26629.073922049845,
                    -1.025,
                    -0.78,
                    -0.65012,
                    -0.19041574084286472,
                    -0.20819395464,
                    -0.421064,
                    -0.32733845535848671,
                ],
                [
                    3.6 * 9311.3238710010046,
                    -0.3041958212306401,
                    -0.3633036457487539,
                    -0.35660379990111464,
                    -0.3443074455424403,
                    -0.33699592683512467,
                    -0.30180866526242109,
                    -0.27321683125358037,
                ],
                [
                    3.6 * 4992.2486445538634,
                    -1.2,
                    -1.2,
                    -0.8,
                    -0.7,
                    -0.7,
                    -0.4,
                    -0.5,
                ],
            ]),
        }
    }

    // dct256x128
    fn dct128x256() -> QuantEncoding {
        QuantEncoding::Dct {
            params: DctQuantWeightParams::from_array(&[
                [
                    2.6 * 23629.073922049845,
                    -1.025,
                    -0.78,
                    -0.65012,
                    -0.19041574084286472,
                    -0.20819395464,
                    -0.421064,
                    -0.32733845535848671,
                ],
                [
                    2.6 * 8611.3238710010046,
                    -0.3041958212306401,
                    -0.3633036457487539,
                    -0.35660379990111464,
                    -0.3443074455424403,
                    -0.33699592683512467,
                    -0.30180866526242109,
                    -0.27321683125358037,
                ],
                [
                    2.6 * 4492.2486445538634,
                    -1.2,
                    -1.2,
                    -0.8,
                    -0.7,
                    -0.7,
                    -0.4,
                    -0.5,
                ],
            ]),
        }
    }
    pub fn library() -> &'static [QuantEncoding; NUM_PREDEFINED_TABLES * NUM_QUANT_TABLES] {
        static QUANTS: OnceLock<[QuantEncoding; NUM_PREDEFINED_TABLES * NUM_QUANT_TABLES]> =
            OnceLock::new();
        QUANTS.get_or_init(|| {
            [
                DequantMatrices::dct(),
                DequantMatrices::id(),
                DequantMatrices::dct2x2(),
                DequantMatrices::dct4x4(),
                DequantMatrices::dct16x16(),
                DequantMatrices::dct32x32(),
                DequantMatrices::dct8x16(),
                DequantMatrices::dct8x32(),
                DequantMatrices::dct16x32(),
                DequantMatrices::dct4x8(),
                DequantMatrices::afv0(),
                DequantMatrices::dct64x64(),
                DequantMatrices::dct32x64(),
                // Same default for large transforms (128+) as for 64x* transforms.
                DequantMatrices::dct128x128(),
                DequantMatrices::dct64x128(),
                DequantMatrices::dct256x256(),
                DequantMatrices::dct128x256(),
            ]
        })
    }

    fn matrix(&self, quant_kind: HfTransformType, c: usize) -> &[f32] {
        assert_eq!((1 << quant_kind as u32) & self.computed_mask, 1);
        &self.table[self.table_offsets[quant_kind as usize * 3 + c]..]
    }

    fn inv_matrix(&self, quant_kind: HfTransformType, c: usize) -> &[f32] {
        assert_eq!((1 << quant_kind as u32) & self.computed_mask, 1);
        &self.inv_table[self.table_offsets[quant_kind as usize * 3 + c]..]
    }

    pub fn decode(_br: &mut BitReader) -> Result<Self> {
        todo!();
    }

    pub const REQUIRED_SIZE_X: [usize; NUM_QUANT_TABLES] =
        [1, 1, 1, 1, 2, 4, 1, 1, 2, 1, 1, 8, 4, 16, 8, 32, 16];

    pub const REQUIRED_SIZE_Y: [usize; NUM_QUANT_TABLES] =
        [1, 1, 1, 1, 2, 4, 2, 4, 4, 1, 1, 8, 8, 16, 16, 32, 32];

    pub const SUM_REQUIRED_X_Y: usize = 2056;

    pub const TOTAL_TABLE_SIZE: usize = Self::SUM_REQUIRED_X_Y * BLOCK_SIZE * 3;

    pub fn ensure_computed(&mut self, _acs_mask: u32) {
        todo!();
    }
}

#[test]
fn check_required_x_y() {
    assert_eq!(
        DequantMatrices::SUM_REQUIRED_X_Y,
        DequantMatrices::REQUIRED_SIZE_X
            .iter()
            .zip(DequantMatrices::REQUIRED_SIZE_Y)
            .map(|(&x, y)| x * y)
            .sum()
    );
}
