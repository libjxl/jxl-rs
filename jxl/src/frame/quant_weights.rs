// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(dead_code)]

use std::{f32::consts::SQRT_2, sync::OnceLock};

use enum_iterator::Sequence;
use half::f16;

use crate::{
    bit_reader::BitReader,
    error::{
        Error::{
            HfQuantFactorTooSmall, InvalidDistanceBand, InvalidQuantEncoding,
            InvalidQuantEncodingMode, InvalidQuantizationTableWeight, InvalidRawQuantTable,
        },
        Result,
    },
    frame::{
        modular::{decode::decode_modular_subbitstream, ModularChannel, ModularStreamId},
        transform_map::{self, HfTransformType},
        LfGlobalState,
    },
    headers::{bit_depth::BitDepth, frame_header::FrameHeader},
    BLOCK_DIM, BLOCK_SIZE,
};

pub const INV_LF_QUANT: [f32; 3] = [4096.0, 512.0, 256.0];

pub const LF_QUANT: [f32; 3] = [
    1.0 / INV_LF_QUANT[0],
    1.0 / INV_LF_QUANT[1],
    1.0 / INV_LF_QUANT[2],
];

const ALMOST_ZERO: f32 = 1e-8;

const MAX_QUANT_TABLE_SIZE: usize = transform_map::MAX_COEFF_AREA;
const LOG2_NUM_QUANT_MODES: usize = 3;

#[derive(Debug)]
pub struct DctQuantWeightParams {
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
            params[..values.len()].copy_from_slice(values);
        }
        result
    }

    pub fn decode(br: &mut BitReader) -> Result<Self> {
        let num_bands = br.read(Self::LOG2_MAX_DISTANCE_BANDS)? as usize + 1;
        let mut params = [[0.0; Self::MAX_DISTANCE_BANDS]; 3];
        for row in params.iter_mut() {
            for item in row[..num_bands].iter_mut() {
                *item = f32::from(f16::from_bits(br.read(16)? as u16));
            }
            if row[0] < ALMOST_ZERO {
                return Err(HfQuantFactorTooSmall(row[0]));
            }
            row[0] *= 64.0;
        }
        Ok(DctQuantWeightParams { params, num_bands })
    }
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub enum QuantEncoding {
    Library,
    // a.k.a. "Hornuss"
    Identity {
        xyb_weights: [[f32; 3]; 3],
    },
    Dct2 {
        xyb_weights: [[f32; 6]; 3],
    },
    Dct4 {
        params: DctQuantWeightParams,
        xyb_mul: [[f32; 2]; 3],
    },
    Dct4x8 {
        params: DctQuantWeightParams,
        xyb_mul: [f32; 3],
    },
    Afv {
        params4x8: DctQuantWeightParams,
        params4x4: DctQuantWeightParams,
        weights: [[f32; 9]; 3],
    },
    Dct {
        params: DctQuantWeightParams,
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

    pub fn decode(
        mut required_size_x: usize,
        mut required_size_y: usize,
        index: usize,
        header: &FrameHeader,
        lf_global: &LfGlobalState,
        br: &mut BitReader,
    ) -> Result<Self> {
        let required_size = required_size_x * required_size_y;
        required_size_x *= BLOCK_DIM;
        required_size_y *= BLOCK_DIM;
        let mode = br.read(LOG2_NUM_QUANT_MODES)? as u8;
        match mode {
            0 => Ok(Self::Library),
            1 => {
                if required_size != 1 {
                    return Err(InvalidQuantEncoding {
                        mode,
                        required_size,
                    });
                }
                let mut xyb_weights = [[0.0; 3]; 3];
                for row in xyb_weights.iter_mut() {
                    for item in row.iter_mut() {
                        *item = f32::from(f16::from_bits(br.read(16)? as u16));
                        if item.abs() < ALMOST_ZERO {
                            return Err(HfQuantFactorTooSmall(*item));
                        }
                        *item *= 64.0;
                    }
                }
                Ok(Self::Identity { xyb_weights })
            }
            2 => {
                if required_size != 1 {
                    return Err(InvalidQuantEncoding {
                        mode,
                        required_size,
                    });
                }
                let mut xyb_weights = [[0.0; 6]; 3];
                for row in xyb_weights.iter_mut() {
                    for item in row.iter_mut() {
                        *item = f32::from(f16::from_bits(br.read(16)? as u16));
                        if item.abs() < ALMOST_ZERO {
                            return Err(HfQuantFactorTooSmall(*item));
                        }
                        *item *= 64.0;
                    }
                }
                Ok(Self::Dct2 { xyb_weights })
            }
            3 => {
                if required_size != 1 {
                    return Err(InvalidQuantEncoding {
                        mode,
                        required_size,
                    });
                }
                let mut xyb_mul = [[0.0; 2]; 3];
                for row in xyb_mul.iter_mut() {
                    for item in row.iter_mut() {
                        *item = f32::from(f16::from_bits(br.read(16)? as u16));
                        if item.abs() < ALMOST_ZERO {
                            return Err(HfQuantFactorTooSmall(*item));
                        }
                    }
                }
                let params = DctQuantWeightParams::decode(br)?;
                Ok(Self::Dct4 { params, xyb_mul })
            }
            4 => {
                if required_size != 1 {
                    return Err(InvalidQuantEncoding {
                        mode,
                        required_size,
                    });
                }
                let mut xyb_mul = [0.0; 3];
                for item in xyb_mul.iter_mut() {
                    *item = f32::from(f16::from_bits(br.read(16)? as u16));
                    if item.abs() < ALMOST_ZERO {
                        return Err(HfQuantFactorTooSmall(*item));
                    }
                }
                let params = DctQuantWeightParams::decode(br)?;
                Ok(Self::Dct4x8 { params, xyb_mul })
            }
            5 => {
                if required_size != 1 {
                    return Err(InvalidQuantEncoding {
                        mode,
                        required_size,
                    });
                }
                let mut weights = [[0.0; 9]; 3];
                for row in weights.iter_mut() {
                    for item in row.iter_mut() {
                        *item = f32::from(f16::from_bits(br.read(16)? as u16));
                    }
                    for item in row[0..6].iter_mut() {
                        *item *= 64.0;
                    }
                }
                let params4x8 = DctQuantWeightParams::decode(br)?;
                let params4x4 = DctQuantWeightParams::decode(br)?;
                Ok(Self::Afv {
                    params4x8,
                    params4x4,
                    weights,
                })
            }
            6 => {
                let params = DctQuantWeightParams::decode(br)?;
                Ok(Self::Dct { params })
            }
            7 => {
                let qtable_den = f32::from(f16::from_bits(br.read(16)? as u16));
                if qtable_den < ALMOST_ZERO {
                    // qtable[] values are already checked for <= 0 so the denominator may not be negative.
                    return Err(InvalidRawQuantTable);
                }
                let bit_depth = BitDepth::integer_samples(8);
                let mut image = [
                    ModularChannel::new((required_size_x, required_size_y), bit_depth)?,
                    ModularChannel::new((required_size_x, required_size_y), bit_depth)?,
                    ModularChannel::new((required_size_x, required_size_y), bit_depth)?,
                ];
                let stream_id = ModularStreamId::QuantTable(index).get_id(header);
                decode_modular_subbitstream(
                    image.iter_mut().collect(),
                    stream_id,
                    None,
                    &lf_global.tree,
                    br,
                )?;
                let mut qtable = Vec::with_capacity(required_size_x * required_size_y * 3);
                for channel in image.iter_mut() {
                    for entry in channel.data.as_rect().iter() {
                        qtable.push(entry);
                        if entry <= 0 {
                            return Err(InvalidRawQuantTable);
                        }
                    }
                }
                Ok(Self::Raw { qtable, qtable_den })
            }
            _ => Err(InvalidQuantEncoding {
                mode,
                required_size,
            }),
        }
    }
}

#[derive(Sequence, Clone, Copy, Debug)]
enum QuantTable {
    // Update QuantTable::VALUES when changing this!
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

impl QuantTable {
    pub const CARDINALITY: usize = Self::VALUES.len();
    pub const VALUES: [QuantTable; 17] = [
        QuantTable::Dct,
        QuantTable::Identity,
        QuantTable::Dct2x2,
        QuantTable::Dct4x4,
        QuantTable::Dct16x16,
        QuantTable::Dct32x32,
        // QuantTable::Dct16x8
        QuantTable::Dct8x16,
        // QuantTable::Dct32x8
        QuantTable::Dct8x32,
        // QuantTable::Dct32x16
        QuantTable::Dct16x32,
        QuantTable::Dct4x8,
        // QuantTable::Dct8x4
        QuantTable::Afv0,
        // QuantTable::Afv1
        // QuantTable::Afv2
        // QuantTable::Afv3
        QuantTable::Dct64x64,
        // QuantTable::Dct64x32,
        QuantTable::Dct32x64,
        QuantTable::Dct128x128,
        // QuantTable::Dct128x64,
        QuantTable::Dct64x128,
        QuantTable::Dct256x256,
        // QuantTable::Dct256x128,
        QuantTable::Dct128x256,
    ];
    pub fn from_usize(idx: usize) -> Result<QuantTable> {
        match QuantTable::VALUES.get(idx) {
            Some(table) => Ok(*table),
            None => Err(InvalidQuantEncodingMode),
        }
    }
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

pub struct DequantMatrices {
    computed_mask: u32,
    table: Vec<f32>,
    inv_table: Vec<f32>,
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
            params: DctQuantWeightParams::from_array(&[
                [2200.0, 0.0, 0.0, 0.0],
                [392.0, 0.0, 0.0, 0.0],
                [112.0, -0.25, -0.25, -0.5],
            ]),
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
            params: DctQuantWeightParams::from_array(&[
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
            ]),
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

    pub fn library() -> &'static [QuantEncoding; QuantTable::CARDINALITY] {
        static QUANTS: OnceLock<[QuantEncoding; QuantTable::CARDINALITY]> = OnceLock::new();
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

    pub fn decode(
        header: &FrameHeader,
        lf_global: &LfGlobalState,
        br: &mut BitReader,
    ) -> Result<Self> {
        let all_default = br.read(1)? == 1;
        let mut encodings = Vec::with_capacity(QuantTable::CARDINALITY);
        if all_default {
            for _ in 0..QuantTable::CARDINALITY {
                encodings.push(QuantEncoding::Library)
            }
        } else {
            for (i, (&required_size_x, required_size_y)) in Self::REQUIRED_SIZE_X
                .iter()
                .zip(Self::REQUIRED_SIZE_Y)
                .enumerate()
            {
                encodings.push(QuantEncoding::decode(
                    required_size_x,
                    required_size_y,
                    i,
                    header,
                    lf_global,
                    br,
                )?);
            }
        }
        Ok(Self {
            computed_mask: 0,
            table: vec![0.0; Self::TOTAL_TABLE_SIZE],
            inv_table: vec![0.0; Self::TOTAL_TABLE_SIZE],
            table_offsets: [0; HfTransformType::CARDINALITY * 3],
            encodings,
        })
    }

    pub const REQUIRED_SIZE_X: [usize; QuantTable::CARDINALITY] =
        [1, 1, 1, 1, 2, 4, 1, 1, 2, 1, 1, 8, 4, 16, 8, 32, 16];

    pub const REQUIRED_SIZE_Y: [usize; QuantTable::CARDINALITY] =
        [1, 1, 1, 1, 2, 4, 2, 4, 4, 1, 1, 8, 8, 16, 16, 32, 32];

    pub const SUM_REQUIRED_X_Y: usize = 2056;

    pub const TOTAL_TABLE_SIZE: usize = Self::SUM_REQUIRED_X_Y * BLOCK_SIZE * 3;

    pub fn ensure_computed(&mut self, acs_mask: u32) -> Result<()> {
        let mut offsets = [0usize; QuantTable::CARDINALITY * 3];
        let mut pos = 0usize;
        for i in 0..QuantTable::CARDINALITY {
            let num = DequantMatrices::REQUIRED_SIZE_X[i]
                * DequantMatrices::REQUIRED_SIZE_Y[i]
                * BLOCK_SIZE;
            for c in 0..3 {
                offsets[3 * i + c] = pos + c * num;
            }
            pos += 3 * num;
        }
        for i in 0..HfTransformType::CARDINALITY {
            for c in 0..3 {
                self.table_offsets[i * 3 + c] = offsets
                    [QuantTable::for_strategy(HfTransformType::from_usize(i)?) as usize * 3 + c];
            }
        }
        let mut kind_mask = 0u32;
        for i in 0..HfTransformType::CARDINALITY {
            if acs_mask & (1u32 << i) != 0 {
                kind_mask |= 1u32 << QuantTable::for_strategy(HfTransformType::VALUES[i]) as u32;
            }
        }
        let mut computed_kind_mask = 0u32;
        for i in 0..HfTransformType::CARDINALITY {
            if self.computed_mask & (1u32 << i) != 0 {
                computed_kind_mask |=
                    1u32 << QuantTable::for_strategy(HfTransformType::VALUES[i]) as u32;
            }
        }
        for table in 0..QuantTable::CARDINALITY {
            if (1u32 << table) & computed_kind_mask != 0 {
                continue;
            }
            if (1u32 << table) & !kind_mask != 0 {
                continue;
            }
            match self.encodings[table] {
                QuantEncoding::Library => {
                    self.compute_quant_table(true, table, offsets[table * 3])?
                }
                _ => self.compute_quant_table(false, table, offsets[table * 3])?,
            };
        }
        self.computed_mask |= acs_mask;
        Ok(())
    }
    fn compute_quant_table(
        &mut self,
        library: bool,
        table_num: usize,
        offset: usize,
    ) -> Result<usize> {
        let encoding = if library {
            &DequantMatrices::library()[table_num]
        } else {
            &self.encodings[table_num]
        };
        let quant_table_idx = QuantTable::from_usize(table_num)? as usize;
        let wrows = 8 * DequantMatrices::REQUIRED_SIZE_X[quant_table_idx];
        let wcols = 8 * DequantMatrices::REQUIRED_SIZE_Y[quant_table_idx];
        let num = wrows * wcols;
        let mut weights = vec![0f32; 3 * num];
        match encoding {
            QuantEncoding::Library => {
                // Library and copy quant encoding should get replaced by the actual
                // parameters by the caller.
                return Err(InvalidQuantEncodingMode);
            }
            QuantEncoding::Identity { xyb_weights } => {
                for c in 0..3 {
                    for i in 0..64 {
                        weights[64 * c + i] = xyb_weights[c][0];
                    }
                    weights[64 * c + 1] = xyb_weights[c][1];
                    weights[64 * c + 8] = xyb_weights[c][1];
                    weights[64 * c + 9] = xyb_weights[c][2];
                }
            }
            QuantEncoding::Dct2 { xyb_weights } => {
                for (c, xyb_weight) in xyb_weights.iter().enumerate() {
                    let start = c * 64;
                    weights[start] = 0xBAD as f32;
                    weights[start + 1] = xyb_weight[0];
                    weights[start + 8] = xyb_weight[0];
                    weights[start + 9] = xyb_weight[1];
                    for y in 0..2 {
                        for x in 0..2 {
                            weights[start + y * 8 + x + 2] = xyb_weight[2];
                            weights[start + (y + 2) * 8 + x] = xyb_weight[2];
                        }
                    }
                    for y in 0..2 {
                        for x in 0..2 {
                            weights[start + (y + 2) * 8 + x + 2] = xyb_weight[3];
                        }
                    }
                    for y in 0..4 {
                        for x in 0..4 {
                            weights[start + y * 8 + x + 4] = xyb_weight[4];
                            weights[start + (y + 4) * 8 + x] = xyb_weight[4];
                        }
                    }
                    for y in 0..4 {
                        for x in 0..4 {
                            weights[start + (y + 4) * 8 + x + 4] = xyb_weight[5];
                        }
                    }
                }
            }
            QuantEncoding::Dct4 { params, xyb_mul } => {
                let mut weights4x4 = [0f32; 3 * 4 * 4];
                get_quant_weights(4, 4, params, &mut weights4x4)?;
                for c in 0..3 {
                    for y in 0..BLOCK_DIM {
                        for x in 0..BLOCK_DIM {
                            weights[c * num + y * BLOCK_DIM + x] =
                                weights4x4[c * 16 + (y / 2) * 4 + (x / 2)];
                        }
                    }
                    weights[c * num + 1] /= xyb_mul[c][0];
                    weights[c * num + BLOCK_DIM] /= xyb_mul[c][0];
                    weights[c * num + BLOCK_DIM + 1] /= xyb_mul[c][1];
                }
            }
            QuantEncoding::Dct4x8 { params, xyb_mul } => {
                let mut weights4x8 = [0f32; 3 * 4 * 8];
                get_quant_weights(4, 8, params, &mut weights4x8)?;
                for c in 0..3 {
                    for y in 0..BLOCK_DIM {
                        for x in 0..BLOCK_DIM {
                            weights[c * num + y * BLOCK_DIM + x] =
                                weights4x8[c * 32 + (y / 2) * 8 + x];
                        }
                    }
                    weights[c * num + BLOCK_DIM] /= xyb_mul[c];
                }
            }
            QuantEncoding::Dct { params } => {
                get_quant_weights(wrows, wcols, params, &mut weights)?;
            }
            QuantEncoding::Raw { qtable, qtable_den } => {
                if qtable.len() != 3 * num {
                    return Err(InvalidRawQuantTable);
                }
                for i in 0..3 * num {
                    weights[i] = 1f32 / (qtable_den * qtable[i] as f32);
                }
            }
            QuantEncoding::Afv {
                params4x8,
                params4x4,
                weights: afv_weights,
            } => {
                const FREQS: [f32; 16] = [
                    0xBAD as f32,
                    0xBAD as f32,
                    0.8517778890324296,
                    5.37778436506804,
                    0xBAD as f32,
                    0xBAD as f32,
                    4.734747904497923,
                    5.449245381693219,
                    1.6598270267479331,
                    4f32,
                    7.275749096817861,
                    10.423227632456525,
                    2.662932286148962,
                    7.630657783650829,
                    8.962388608184032,
                    12.97166202570235,
                ];
                let mut weights4x8 = [0f32; 3 * 4 * 8];
                get_quant_weights(4, 8, params4x8, &mut weights4x8)?;
                let mut weights4x4 = [0f32; 3 * 4 * 4];
                get_quant_weights(4, 4, params4x4, &mut weights4x4)?;
                const LO: f32 = 0.8517778890324296;
                const HI: f32 = 12.97166202570235f32 - LO + 1e-6f32;
                for c in 0..3 {
                    let mut bands = [0f32; 4];
                    bands[0] = afv_weights[c][5];
                    if bands[0] < ALMOST_ZERO {
                        return Err(InvalidDistanceBand(0, bands[0]));
                    }
                    for i in 1..4 {
                        bands[i] = bands[i - 1] * mult(afv_weights[c][i + 5]);
                        if bands[i] < ALMOST_ZERO {
                            return Err(InvalidDistanceBand(i, bands[i]));
                        }
                    }

                    {
                        let start = c * 64;
                        weights[start] = 1f32;
                        let mut set = |x, y, val| {
                            weights[start + y * 8 + x] = val;
                        };
                        set(0, 1, afv_weights[c][0]);
                        set(1, 0, afv_weights[c][1]);
                        set(0, 2, afv_weights[c][2]);
                        set(2, 0, afv_weights[c][3]);
                        set(2, 2, afv_weights[c][4]);

                        for y in 0..4 {
                            for x in 0..4 {
                                if x < 2 && y < 2 {
                                    continue;
                                }
                                let val = interpolate(FREQS[y * 4 + x] - LO, HI, &bands);
                                set(2 * x, 2 * y, val);
                            }
                        }
                    }

                    for y in 0..BLOCK_DIM / 2 {
                        for x in 0..BLOCK_DIM {
                            if x == 0 && y == 0 {
                                continue;
                            }
                            weights[c * num + (2 * y + 1) * BLOCK_DIM + x] =
                                weights4x8[c * 32 + y * 8 + x];
                        }
                    }

                    for y in 0..BLOCK_DIM / 2 {
                        for x in 0..BLOCK_DIM / 2 {
                            if x == 0 && y == 0 {
                                continue;
                            }
                            weights[c * num + (2 * y) * BLOCK_DIM + 2 * x + 1] =
                                weights4x4[c * 16 + y * 4 + x];
                        }
                    }
                }
            }
        }
        for (i, weight) in weights.iter().enumerate() {
            if !(ALMOST_ZERO..=1.0 / ALMOST_ZERO).contains(weight) {
                println!("weight index {} is {}", i, *weight);
                return Err(InvalidQuantizationTableWeight(*weight));
            }
            self.table[offset + i] = 1f32 / weight;
            self.inv_table[offset + i] = *weight;
        }
        let (xs, ys) = coefficient_layout(
            DequantMatrices::REQUIRED_SIZE_X[quant_table_idx],
            DequantMatrices::REQUIRED_SIZE_Y[quant_table_idx],
        );

        for c in 0..3 {
            for y in 0..ys {
                for x in 0..xs {
                    self.inv_table[offset + c * ys * xs * BLOCK_SIZE + y * BLOCK_DIM * xs + x] =
                        0f32;
                }
            }
        }

        Ok(0)
    }
}

fn coefficient_layout(rows: usize, cols: usize) -> (usize, usize) {
    (
        if rows < cols { rows } else { cols },
        if rows < cols { cols } else { rows },
    )
}

fn get_quant_weights(
    rows: usize,
    cols: usize,
    distance_bands: &DctQuantWeightParams,
    out: &mut [f32],
) -> Result<()> {
    for c in 0..3 {
        let mut bands = [0f32; DctQuantWeightParams::MAX_DISTANCE_BANDS];
        bands[0] = distance_bands.params[c][0];
        if bands[0] < ALMOST_ZERO {
            return Err(InvalidDistanceBand(0, bands[0]));
        }
        for i in 1..distance_bands.num_bands {
            bands[i] = bands[i - 1] * mult(distance_bands.params[c][i]);
            if bands[i] < ALMOST_ZERO {
                return Err(InvalidDistanceBand(i, bands[i]));
            }
        }
        let scale = (distance_bands.num_bands - 1) as f32 / (SQRT_2 + 1e-6);
        let rcpcol = scale / (cols - 1) as f32;
        let rcprow = scale / (rows - 1) as f32;
        for y in 0..rows {
            let dy = y as f32 * rcprow;
            let dy2 = dy * dy;
            for x in 0..cols {
                let dx = x as f32 * rcpcol;
                let scaled_distance = (dx * dx + dy2).sqrt();
                let weight = if distance_bands.num_bands == 1 {
                    bands[0]
                } else {
                    interpolate_vec(scaled_distance, &bands)
                };
                out[c * cols * rows + y * cols + x] = weight;
            }
        }
    }
    Ok(())
}

fn interpolate_vec(scaled_pos: f32, array: &[f32]) -> f32 {
    let idxf32 = scaled_pos.floor();
    let frac = scaled_pos - idxf32;
    let idx = idxf32 as usize;
    let a = array[idx];
    let b = array[1..][idx];
    (b / a).powf(frac) * a
}

fn interpolate(pos: f32, max: f32, array: &[f32]) -> f32 {
    let scaled_pos = pos * (array.len() - 1) as f32 / max;
    let idx = scaled_pos as usize;
    let a = array[idx];
    let b = array[idx + 1];
    a * (b / a).powf(scaled_pos - idx as f32)
}

fn mult(v: f32) -> f32 {
    if v > 0f32 {
        1f32 + v
    } else {
        1f32 / (1f32 - v)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::error::Result;
    use crate::frame::quant_weights::DequantMatrices;
    use crate::util::test::{assert_all_almost_eq, assert_almost_eq};

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

    #[test]
    fn check_dequant_matrix_correctness() -> Result<()> {
        let mut matrices = DequantMatrices {
            computed_mask: 0,
            table: vec![0.0; DequantMatrices::TOTAL_TABLE_SIZE],
            inv_table: vec![0.0; DequantMatrices::TOTAL_TABLE_SIZE],
            table_offsets: [0; HfTransformType::CARDINALITY * 3],
            encodings: (0..QuantTable::CARDINALITY)
                .map(|_| QuantEncoding::Library)
                .collect(),
        };
        matrices.ensure_computed(!0)?;

        // Golden data produced by libjxl.
        let target_offsets: [usize; 81] = [
            0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 1024, 1280, 1536, 2560,
            3584, 4608, 4736, 4864, 4608, 4736, 4864, 4992, 5248, 5504, 4992, 5248, 5504, 5760,
            6272, 6784, 5760, 6272, 6784, 7296, 7360, 7424, 7296, 7360, 7424, 7488, 7552, 7616,
            7488, 7552, 7616, 7488, 7552, 7616, 7488, 7552, 7616, 7680, 11776, 15872, 19968, 22016,
            24064, 19968, 22016, 24064, 26112, 42496, 58880, 75264, 83456, 91648, 75264, 83456,
            91648, 99840, 165376, 230912, 296448, 329216, 361984, 296448, 329216, 361984,
        ];
        assert_all_almost_eq!(matrices.table_offsets, target_offsets, 0);

        // Golden data produced by libjxl.
        let target_table = [
            0.000317, 0.000629, 0.000457, 0.000367, 0.000378, 0.000709, 0.000593, 0.000566,
            0.000629, 0.001192, 0.000943, 0.001786, 0.003042, 0.002372, 0.001998, 0.002044,
            0.003341, 0.002907, 0.002804, 0.003042, 0.004229, 0.003998, 0.001953, 0.011969,
            0.011719, 0.007886, 0.008374, 0.015337, 0.011719, 0.011719, 0.011969, 0.032080,
            0.025368, 0.003571, 0.003571, 0.003571, 0.003571, 0.003571, 0.003571, 0.003571,
            0.003571, 0.003571, 0.003571, 0.003571, 0.016667, 0.016667, 0.016667, 0.016667,
            0.016667, 0.016667, 0.016667, 0.016667, 0.016667, 0.016667, 0.016667, 0.055556,
            0.055556, 0.055556, 0.055556, 0.055556, 0.055556, 0.055556, 0.055556, 0.055556,
            0.055556, 0.055556, 0.000335, 0.002083, 0.002083, 0.001563, 0.000781, 0.002083,
            0.003333, 0.002083, 0.002083, 0.003333, 0.003333, 0.000335, 0.007143, 0.007143,
            0.005556, 0.003125, 0.007143, 0.008333, 0.007143, 0.007143, 0.008333, 0.008333,
            0.000335, 0.031250, 0.031250, 0.015625, 0.007812, 0.031250, 0.062500, 0.031250,
            0.031250, 0.062500, 0.062500, 0.000455, 0.000455, 0.000455, 0.000455, 0.000455,
            0.000455, 0.000455, 0.000455, 0.000455, 0.000455, 0.000455, 0.002551, 0.002551,
            0.002551, 0.002551, 0.002551, 0.002551, 0.002551, 0.002551, 0.002551, 0.002551,
            0.002551, 0.008929, 0.014654, 0.012241, 0.011161, 0.010455, 0.015352, 0.013951,
            0.012706, 0.014654, 0.020926, 0.017433, 0.000111, 0.000469, 0.000258, 0.000640,
            0.000388, 0.001007, 0.000566, 0.001880, 0.000946, 0.000886, 0.001880, 0.000313,
            0.001168, 0.000531, 0.001511, 0.000962, 0.001959, 0.001399, 0.002531, 0.001908,
            0.001850, 0.002531, 0.000864, 0.007969, 0.002684, 0.010653, 0.006434, 0.015981,
            0.009743, 0.040354, 0.014631, 0.013468, 0.040354, 0.000064, 0.000135, 0.000279,
            0.000521, 0.000760, 0.001145, 0.000502, 0.000647, 0.000911, 0.001286, 0.001685,
            0.000137, 0.000257, 0.000464, 0.000739, 0.001126, 0.001645, 0.000706, 0.000959,
            0.001327, 0.001839, 0.002404, 0.000263, 0.001155, 0.003800, 0.010779, 0.016740,
            0.024003, 0.010258, 0.014299, 0.019509, 0.026824, 0.035546, 0.000138, 0.000515,
            0.000425, 0.000333, 0.000362, 0.000559, 0.000507, 0.000500, 0.000538, 0.000686,
            0.000666, 0.000691, 0.002504, 0.001785, 0.001353, 0.001443, 0.002721, 0.002469,
            0.002432, 0.002617, 0.003340, 0.003241, 0.001973, 0.010000, 0.006529, 0.005339,
            0.005497, 0.012033, 0.009689, 0.009374, 0.011033, 0.031220, 0.026814, 0.000138,
            0.000447, 0.000379, 0.000569, 0.000555, 0.000885, 0.001434, 0.003015, 0.002469,
            0.002623, 0.003417, 0.000691, 0.001995, 0.001495, 0.002768, 0.002701, 0.003754,
            0.005481, 0.018655, 0.009689, 0.011082, 0.037175, 0.001973, 0.007296, 0.005584,
            0.012499, 0.011829, 0.081609, 0.001256, 0.000640, 0.000410, 0.000492, 0.002408,
            0.000061, 0.000983, 0.000567, 0.002720, 0.002399, 0.012238, 0.000453, 0.000377,
            0.001004, 0.000937, 0.002516, 0.000196, 0.000483, 0.000383, 0.001058, 0.000961,
            0.002652, 0.000872, 0.000603, 0.002873, 0.002551, 0.013503, 0.000294, 0.000959,
            0.000623, 0.003142, 0.002666, 0.014731, 0.000323, 0.000163, 0.000416, 0.000296,
            0.000536, 0.000061, 0.000164, 0.000414, 0.000837, 0.001528, 0.002577, 0.000163,
            0.000352, 0.000714, 0.001311, 0.002240, 0.000196, 0.000256, 0.000342, 0.000443,
            0.000680, 0.001015, 0.000256, 0.000325, 0.000417, 0.000604, 0.000912, 0.000294,
            0.000384, 0.000510, 0.000833, 0.001520, 0.002927, 0.000384, 0.000485, 0.000740,
            0.001296, 0.002438, 0.000072, 0.000138, 0.000226, 0.000321, 0.000397, 0.000482,
            0.000118, 0.000196, 0.000289, 0.000374, 0.000453, 0.000208, 0.000329, 0.000586,
            0.001026, 0.001347, 0.001811, 0.000294, 0.000493, 0.000863, 0.001253, 0.001639,
            0.000553, 0.001178, 0.002507, 0.004258, 0.007080, 0.011771, 0.000975, 0.002001,
            0.003635, 0.006026, 0.010000, 0.000072, 0.000570, 0.001464, 0.003774, 0.004518,
            0.000343, 0.000137, 0.000288, 0.000131, 0.000264, 0.000389, 0.000208, 0.002204,
            0.008305, 0.000840, 0.000229, 0.000347, 0.000161, 0.000300, 0.000180, 0.000284,
            0.000438, 0.000553, 0.016224, 0.017241, 0.000082, 0.000243, 0.000355, 0.000199,
            0.000317, 0.000245, 0.000310, 0.000501, 0.000455, 0.001007, 0.000141, 0.000229,
            0.000285, 0.000342, 0.000095, 0.000151, 0.000239, 0.000289, 0.000345, 0.001308,
            0.002909, 0.000141, 0.000230, 0.000286, 0.000342, 0.000099, 0.000155, 0.000244,
            0.000290, 0.000347, 0.001897, 0.005643, 0.000141, 0.000231, 0.000286, 0.000343,
            0.000103, 0.000159, 0.000248, 0.000292, 0.000349, 0.000455, 0.000263, 0.000278,
            0.000382, 0.000300, 0.000782, 0.002154, 0.002697, 0.000452, 0.000292, 0.001052,
            0.001308, 0.000265, 0.000282, 0.000383, 0.000306, 0.000792, 0.002197, 0.002809,
            0.000456, 0.000293, 0.001079, 0.001897, 0.000267, 0.000286, 0.000384, 0.000312,
            0.000802, 0.002243, 0.002926, 0.000460, 0.000295, 0.001107, 1.000000, 0.000299,
            0.000268, 0.000265, 0.000290, 0.000643, 0.000386, 0.000322, 0.000318, 0.000372,
            0.000813, 1.000000, 0.000300, 0.000270, 0.000268, 0.000294, 0.000653, 0.000388,
            0.000326, 0.000324, 0.000381, 0.000824, 1.000000, 0.000301, 0.000272, 0.000271,
            0.000298, 0.000663, 0.000389, 0.000330, 0.000330, 0.000389, 0.000835, 1.000000,
            0.000322, 0.000477, 0.000030, 0.000106, 0.000175, 0.000171, 0.000289, 0.000158,
            0.000782, 0.002583, 1.000000, 0.000328, 0.000482, 0.000120, 0.000163, 0.000291,
            0.000093, 0.000182, 0.000874, 0.002065, 0.004617, 1.000000, 0.000335, 0.000488,
            0.000031, 0.000108, 0.000176, 0.000172, 0.000291, 0.000161, 0.000798, 0.002631,
            1.000000, 0.000290, 0.000318, 0.002291, 0.000464, 0.001137, 0.000028, 0.000136,
            0.000102, 0.000201, 0.000172, 1.000000, 0.000294, 0.000324, 0.002343, 0.000469,
            0.001168, 0.000114, 0.000058, 0.000159, 0.000141, 0.000280, 1.000000, 0.000298,
            0.000330, 0.002397, 0.000475, 0.001200, 0.000029, 0.000136, 0.000104, 0.000202,
            0.000173,
        ];
        let mut target_table_index = 0;
        for i in 0..QuantTable::CARDINALITY {
            let size = DequantMatrices::REQUIRED_SIZE_X[i]
                * DequantMatrices::REQUIRED_SIZE_Y[i]
                * BLOCK_SIZE;
            for c in 0..3 {
                let start = matrices.table_offsets[3 * i + c];
                for j in (start..start + size).step_by(size / 10) {
                    assert_almost_eq!(matrices.table[j], target_table[target_table_index], 1e-5);
                    target_table_index += 1;
                }
            }
        }
        Ok(())
    }
}
