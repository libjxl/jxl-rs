// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use super::LUT_TABLE_SIZE;
use crate::bit_reader::BitReader;
use crate::entropy_coding::decode::{Histograms, SymbolReader};
use crate::frame::modular::decode::References;
use crate::frame::modular::decode::channel::ModularChannelDecoder;
use crate::frame::modular::decode::common::{make_pixel, precompute_references};
use crate::frame::modular::decode::specialized_trees::{
    DynamicReader, GRADIENT_PROPERTY, LUT_MAX_SPLITVAL, LUT_MIN_SPLITVAL, MaybeWeightedPredictor,
};
use crate::frame::modular::flat_tree::TreeLeaf;
use crate::frame::modular::predict::PredictionData;
use crate::frame::modular::tree::{PredictionResult, TreeNode};
use crate::frame::modular::{ModularChannel, ModularStorage};
use crate::util::box_array;

pub(super) const MAX_NUM_PROPERTIES: usize = 8;

#[derive(Clone, Copy, Default)]
struct StackEntry {
    ranges: [(u16, u16); MAX_NUM_PROPERTIES],
    node: u32,
}

pub struct BitsetTreeScratch {
    pub(super) property_luts: Box<[[u8; LUT_TABLE_SIZE]; MAX_NUM_PROPERTIES]>,
    leaf_masks: Box<[[u64; MAX_NUM_PROPERTIES * 256]; 4]>,
    leaves: Box<[TreeLeaf; 257]>,
    stack: Box<[StackEntry; 257]>,
}

impl Default for BitsetTreeScratch {
    fn default() -> Self {
        BitsetTreeScratch {
            property_luts: box_array([0u8; LUT_TABLE_SIZE]),
            leaves: box_array(TreeLeaf::default()),
            leaf_masks: box_array([0u64; MAX_NUM_PROPERTIES * 256]),
            stack: box_array(StackEntry::default()),
        }
    }
}

pub struct BitsetTreeInner<'a> {
    property_luts: &'a mut [[u8; LUT_TABLE_SIZE]; MAX_NUM_PROPERTIES],
    leaf_masks: &'a mut [[u64; MAX_NUM_PROPERTIES * 256]; 4],
    leaves: &'a mut [TreeLeaf; 257],
    properties: [u8; MAX_NUM_PROPERTIES],
    num_properties: u8,
    num_leaf_buckets: u8,
    references: References<'a>,
    property_buffer: &'a mut [i32; 256],
    storage: ModularStorage,
}

impl<'a> BitsetTreeInner<'a> {
    pub fn new(
        mut property_mask: u64,
        pruned_tree: &[TreeNode],
        scratch: &'a mut BitsetTreeScratch,
        references: References<'a>,
        property_buffer: &'a mut [i32; 256],
        storage: ModularStorage,
    ) -> Self {
        let mut properties = [0; MAX_NUM_PROPERTIES];
        let mut prop_to_idx = [0u8; 64];
        let mut num_properties = 0;
        while property_mask != 0 {
            let prop = property_mask.trailing_zeros() as u8;
            properties[num_properties as usize] = prop;
            prop_to_idx[prop as usize] = num_properties;
            num_properties += 1;
            property_mask &= property_mask - 1;
        }

        let BitsetTreeScratch {
            property_luts,
            leaf_masks,
            leaves,
            stack,
        } = scratch;

        let num_props = num_properties as usize;
        if num_props > 1 {
            for prop_lut in &mut property_luts[..num_props] {
                prop_lut.fill(0);
            }
            for node in pruned_tree {
                if let TreeNode::Split { property, val, .. } = *node {
                    let p_idx = prop_to_idx[property as usize] as usize;
                    let split_pos = (val + 1 - LUT_MIN_SPLITVAL) as usize;
                    property_luts[p_idx][split_pos] = 1;
                }
            }
            let mut next_mask_idx = 0usize;
            for p in 0..num_props {
                let mut cur_mask_idx = next_mask_idx;
                for mask_bucket in leaf_masks.iter_mut() {
                    mask_bucket[cur_mask_idx] = 0;
                }
                for v in 0..LUT_TABLE_SIZE {
                    if property_luts[p][v] != 0 {
                        cur_mask_idx += 1;
                        for mask_bucket in leaf_masks.iter_mut() {
                            mask_bucket[cur_mask_idx] = 0;
                        }
                    }
                    property_luts[p][v] = (cur_mask_idx - next_mask_idx) as u8;
                }
                next_mask_idx += 256;
            }
        }

        stack[0] = StackEntry {
            ranges: [(0, LUT_TABLE_SIZE as u16); MAX_NUM_PROPERTIES],
            node: 0,
        };
        let mut stack_len = 1usize;

        let mut num_leaves = 0usize;
        while stack_len > 0 {
            stack_len -= 1;
            let StackEntry { ranges, node } = stack[stack_len];
            match pruned_tree[node as usize] {
                TreeNode::Split {
                    property,
                    val,
                    left,
                    right,
                } => {
                    let p_idx = prop_to_idx[property as usize] as usize;
                    let first_left = (val + 1 - LUT_MIN_SPLITVAL) as u16;
                    let (start, end) = ranges[p_idx];
                    let mut left_ranges = ranges;
                    left_ranges[p_idx] = (first_left, end);
                    stack[stack_len] = StackEntry {
                        ranges: left_ranges,
                        node: left,
                    };
                    stack_len += 1;
                    let mut right_ranges = ranges;
                    right_ranges[p_idx] = (start, first_left);
                    stack[stack_len] = StackEntry {
                        ranges: right_ranges,
                        node: right,
                    };
                    stack_len += 1;
                }
                TreeNode::Leaf {
                    predictor,
                    offset,
                    multiplier,
                    id,
                } => {
                    let leaf_idx = num_leaves;
                    num_leaves += 1;
                    leaves[leaf_idx] = TreeLeaf {
                        predictor,
                        multiplier,
                        context: id,
                        offset,
                    };
                    if num_props == 1 {
                        let (start, end) = ranges[0];
                        property_luts[0][start as usize..end as usize].fill(leaf_idx as u8);
                    } else {
                        let bucket = leaf_idx / 64;
                        let bit = 1u64 << (leaf_idx % 64);
                        for p in 0..num_props {
                            let (start, end) = ranges[p];
                            let start_interval =
                                p * 256 + property_luts[p][start as usize] as usize;
                            let end_interval =
                                p * 256 + property_luts[p][end as usize - 1] as usize;
                            for interval in start_interval..=end_interval {
                                leaf_masks[bucket][interval] |= bit;
                            }
                        }
                    }
                }
            }
        }

        let num_leaf_buckets = num_leaves.div_ceil(64) as u8;

        BitsetTreeInner {
            property_luts: &mut *property_luts,
            leaf_masks: &mut *leaf_masks,
            leaves: &mut *leaves,
            properties,
            num_properties,
            num_leaf_buckets,
            references,
            property_buffer,
            storage,
        }
    }

    #[inline(always)]
    #[allow(clippy::needless_range_loop)]
    fn predict_impl<const PROPERTIES: usize, const LEAF_BUCKETS: usize>(
        &self,
        prediction_data: PredictionData,
        wp_pred: i64,
    ) -> PredictionResult {
        let p_to_idx = {
            #[inline(always)]
            |p: i32| (p.clamp(LUT_MIN_SPLITVAL, LUT_MAX_SPLITVAL) - LUT_MIN_SPLITVAL) as usize
        };
        if PROPERTIES == 1 {
            let p = self.property_buffer[self.properties[0] as usize];
            let index = p_to_idx(p);
            let leaf = self.property_luts[0][index];
            // No need to look at masks for a single property.
            return self.leaves[leaf as usize].predict(prediction_data, wp_pred);
        }

        let mut masks = [!0u64; 4];
        let mut ps = [0; PROPERTIES];
        for i in 0..PROPERTIES {
            ps[i] = self.property_buffer[self.properties[i] as usize];
        }
        let mut indices = [0; PROPERTIES];
        for i in 0..PROPERTIES {
            let idx = p_to_idx(ps[i]);
            indices[i] = i * 256 + self.property_luts[i][idx] as usize;
        }
        for i in 0..PROPERTIES {
            for l in 0..LEAF_BUCKETS {
                masks[l] &= self.leaf_masks[l][indices[i]];
            }
        }
        let leaf = if masks[0] != 0 || LEAF_BUCKETS == 1 {
            masks[0].trailing_zeros()
        } else if masks[1] != 0 || LEAF_BUCKETS == 2 {
            64 + masks[1].trailing_zeros()
        } else if masks[2] != 0 || LEAF_BUCKETS == 3 {
            128 + masks[2].trailing_zeros()
        } else {
            192 + masks[3].trailing_zeros()
        };
        self.leaves[leaf as usize].predict(prediction_data, wp_pred)
    }

    #[inline(always)]
    fn predict(&self, prediction_data: PredictionData, wp_pred: i64) -> PredictionResult {
        match (self.num_properties, self.num_leaf_buckets) {
            (1, _) => self.predict_impl::<1, 0>(prediction_data, wp_pred),
            (2, 1) => self.predict_impl::<2, 1>(prediction_data, wp_pred),
            (2, _) => self.predict_impl::<2, 4>(prediction_data, wp_pred),
            (3, 1) => self.predict_impl::<3, 1>(prediction_data, wp_pred),
            (3, _) => self.predict_impl::<3, 4>(prediction_data, wp_pred),
            (4, 1) => self.predict_impl::<4, 1>(prediction_data, wp_pred),
            (4, _) => self.predict_impl::<4, 4>(prediction_data, wp_pred),
            (5, 1) => self.predict_impl::<5, 1>(prediction_data, wp_pred),
            (5, _) => self.predict_impl::<5, 4>(prediction_data, wp_pred),
            (6, 1) => self.predict_impl::<6, 1>(prediction_data, wp_pred),
            (6, _) => self.predict_impl::<6, 4>(prediction_data, wp_pred),
            (7, 1) => self.predict_impl::<7, 1>(prediction_data, wp_pred),
            (7, _) => self.predict_impl::<7, 4>(prediction_data, wp_pred),
            (_, 1) => self.predict_impl::<8, 1>(prediction_data, wp_pred),
            (_, _) => self.predict_impl::<8, 4>(prediction_data, wp_pred),
        }
    }
}

pub struct BitsetTree<'a, WP> {
    inner: BitsetTreeInner<'a>,
    reader: DynamicReader,
    wp_state: WP,
}

impl<'a, WP: MaybeWeightedPredictor> BitsetTree<'a, WP> {
    pub fn new(inner: BitsetTreeInner<'a>, reader: DynamicReader, wp_state: WP) -> Self {
        Self {
            inner,
            reader,
            wp_state,
        }
    }
}

impl<WP: MaybeWeightedPredictor> ModularChannelDecoder for BitsetTree<'_, WP> {
    fn init_row(&mut self, buffers: &mut [&mut ModularChannel], chan: usize, y: usize) {
        precompute_references(
            buffers,
            chan,
            y,
            &mut self.inner.references,
            self.inner.storage,
        );
        self.inner.property_buffer[GRADIENT_PROPERTY as usize] = 0;
    }

    #[inline(always)]
    fn decode_one(
        &mut self,
        prediction_data: PredictionData,
        pos: (usize, usize),
        reader: &mut SymbolReader,
        br: &mut BitReader,
        histograms: &Histograms,
    ) -> i32 {
        let wp_pred = self.wp_state.compute_properties(
            prediction_data,
            pos,
            &self.inner.references,
            self.inner.property_buffer,
        );
        let prediction_result: PredictionResult = self.inner.predict(prediction_data, wp_pred);
        let dec = self
            .reader
            .read(reader, histograms, br, prediction_result.context as usize);
        let val = make_pixel(dec, prediction_result.multiplier, prediction_result.guess);
        self.wp_state.update_errors(val, pos);
        val
    }
}
