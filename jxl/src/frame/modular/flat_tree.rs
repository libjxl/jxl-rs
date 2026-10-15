// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::collections::VecDeque;
use std::fmt::Debug;

use super::Predictor;
use super::predict::{PredictionData, WeightedPredictorState, clamped_gradient};
use crate::error::Result;
use crate::frame::modular::Tree;
use crate::frame::modular::tree::{
    NUM_NONREF_PROPERTIES, PredictionResult, TreeNode, compute_properties,
};
use crate::image::Image;
use crate::util::NewWithCapacity;

/// Bitmask for local properties `{9, 10, 11, 12, 15}`.
const LOCAL_AND_WP_PROP_MASK: u16 = (1 << 9) | (1 << 10) | (1 << 11) | (1 << 12) | (1 << 15);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum LeafPredMode {
    OnlyWeighted,
    OnlyGradient,
    OnlyZero,
    GradOrWeighted,
    General,
}

impl LeafPredMode {
    #[inline(always)]
    fn eval_leaf<const LOCAL_ONLY: bool>(
        self,
        leaf_token: u16,
        prediction_data: PredictionData,
        wp_pred: i64,
    ) -> (i64, usize) {
        let cluster = (leaf_token & 0xff) as usize;
        let pred_id = (leaf_token >> 8) as u8;
        let PredictionData {
            left, top, topleft, ..
        } = prediction_data;

        let guess = match self {
            Self::OnlyWeighted => wp_pred,
            Self::OnlyGradient => clamped_gradient(left as i64, top as i64, topleft as i64),
            Self::OnlyZero => 0,
            Self::GradOrWeighted => {
                if pred_id == Predictor::Weighted as u8 {
                    wp_pred
                } else {
                    clamped_gradient(left as i64, top as i64, topleft as i64)
                }
            }
            Self::General => {
                if pred_id == Predictor::Weighted as u8 {
                    wp_pred
                } else if pred_id == Predictor::Gradient as u8 {
                    clamped_gradient(left as i64, top as i64, topleft as i64)
                } else if pred_id == Predictor::Zero as u8 {
                    0
                } else {
                    Self::predict_secondary::<LOCAL_ONLY>(pred_id, prediction_data)
                }
            }
        };
        (guess, cluster)
    }

    #[inline(always)]
    fn predict_secondary<const LOCAL_ONLY: bool>(
        pred_id: u8,
        PredictionData {
            left,
            top,
            toptop,
            topleft,
            topright,
            leftleft,
            toprightright,
        }: PredictionData,
    ) -> i64 {
        match pred_id {
            1 => left as i64,
            2 => top as i64,
            3 => (top as i64 + left as i64) / 2,
            4 => {
                let l = left as i64;
                let t = top as i64;
                let tl = topleft as i64;
                let p = l + t - tl;
                if (p - l).abs() < (p - t).abs() { l } else { t }
            }
            7 => topright as i64,
            8 => topleft as i64,
            9 => leftleft as i64,
            10 => (left as i64 + topleft as i64) / 2,
            11 => (top as i64 + topleft as i64) / 2,
            12 => (top as i64 + topright as i64) / 2,
            _ => {
                if LOCAL_ONLY {
                    0
                } else {
                    (6 * top as i64 - 2 * toptop as i64
                        + 7 * left as i64
                        + leftleft as i64
                        + toprightright as i64
                        + 3 * topright as i64
                        + 8)
                        / 16
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct TreeMeta {
    pub all_simple_leaves: bool,
    pub only_local_and_wp: bool,
    pub pred_mode: LeafPredMode,
    pub min_splitval: i32,
    pub max_splitval: i32,
    pub used_props: [u8; 6],
    pub num_props: usize,
}

impl TreeMeta {
    pub(super) fn analyze(nodes: &[TreeNode], max_property_count: usize) -> Self {
        let mut all_simple_leaves = true;
        let mut uses_average_all = false;
        let mut prop_mask: u16 = 0;
        let mut pred_mask: u16 = 0;
        let mut min_splitval = i32::MAX;
        let mut max_splitval = i32::MIN;
        let mut used_props = [0u8; 6];
        let mut num_props = 0usize;
        let mut seen_props = [false; 256];

        for node in nodes {
            match *node {
                TreeNode::Split { property, val, .. } => {
                    let p = property as usize;
                    if p < 16 {
                        prop_mask |= 1u16 << p;
                    }
                    if !seen_props[p] {
                        seen_props[p] = true;
                        if num_props < 6 {
                            used_props[num_props] = property;
                        }
                        num_props += 1;
                    }
                    min_splitval = min_splitval.min(val);
                    max_splitval = max_splitval.max(val);
                }
                TreeNode::Leaf {
                    predictor,
                    offset,
                    multiplier,
                    id,
                } => {
                    if offset != 0 || multiplier != 1 || id > 255 {
                        all_simple_leaves = false;
                    }
                    if predictor == Predictor::AverageAll {
                        uses_average_all = true;
                    }
                    pred_mask |= 1u16 << (predictor as u8);
                }
            }
        }

        let only_local_and_wp = !uses_average_all
            && max_property_count <= NUM_NONREF_PROPERTIES
            && (prop_mask & !LOCAL_AND_WP_PROP_MASK) == 0;

        let wm = 1u16 << (Predictor::Weighted as u8);
        let gm = 1u16 << (Predictor::Gradient as u8);
        let zm = 1u16 << (Predictor::Zero as u8);
        let pred_mode = if pred_mask == wm {
            LeafPredMode::OnlyWeighted
        } else if pred_mask == gm {
            LeafPredMode::OnlyGradient
        } else if pred_mask == zm {
            LeafPredMode::OnlyZero
        } else if (pred_mask & !(wm | gm)) == 0 {
            LeafPredMode::GradOrWeighted
        } else {
            LeafPredMode::General
        };

        Self {
            all_simple_leaves,
            only_local_and_wp,
            pred_mode,
            min_splitval,
            max_splitval,
            used_props,
            num_props,
        }
    }
}

/// Flattened tree node for optimized traversal.
/// Stores parent + info about both children to evaluate 3 nodes per iteration.
#[derive(Debug, Clone, Copy)]
pub(super) enum FlatTreeNode {
    Split {
        properties: [u8; 3],
        splitvals: [i32; 3],
        child_id: u32,
    },
    Leaf {
        predictor: Predictor,
        multiplier: u32,
        context: u32,
        offset: i32,
    },
}

#[inline(always)]
pub(super) fn predict_flat(
    flat_tree: &[FlatTreeNode],
    mdd: Option<&MddTree>,
    prediction_data: PredictionData,
    wp_state: Option<&mut WeightedPredictorState>,
    pos: (usize, usize),
    references: &Image<i32>,
    property_buffer: &mut [i32; 256],
) -> PredictionResult {
    let wp_pred = compute_properties(
        prediction_data,
        wp_state,
        pos.0,
        pos.1,
        references,
        property_buffer,
    );

    if let Some(mdd) = mdd {
        let (guess, cluster) = mdd.eval_general_props(prediction_data, wp_pred, property_buffer);
        return PredictionResult {
            guess,
            multiplier: 1,
            context: cluster as u32,
        };
    }

    let mut pos = 0;
    loop {
        // Removing this bound check doesn't seem to have a significant effect.
        let node = flat_tree[pos];
        match node {
            FlatTreeNode::Split {
                properties,
                splitvals,
                child_id,
            } => {
                // This bound check is elided by virtue of `property_buffer` having 256 elements.
                let props = properties.map(|x| property_buffer[x as usize]);
                let p0 = props[0] <= splitvals[0];
                let p1 = props[1] <= splitvals[1];
                let p2 = props[2] <= splitvals[2];
                pos = child_id as usize + if p0 { 2 | p2 as usize } else { p1 as usize };
            }
            FlatTreeNode::Leaf {
                predictor,
                multiplier,
                context,
                offset,
            } => {
                let pred = predictor.predict_one(prediction_data, wp_pred);
                return PredictionResult {
                    guess: pred + offset as i64,
                    multiplier,
                    context,
                };
            }
        };
    }
}

impl Tree {
    /// Build flat tree using BFS traversal.
    /// Each flat node stores parent + both children info to reduce branches.
    pub(super) fn build_flat_tree(nodes: &[TreeNode]) -> Result<Vec<FlatTreeNode>> {
        if nodes.is_empty() {
            return Ok(vec![]);
        }

        let mut flat_nodes = Vec::new_with_capacity(nodes.len())?;
        let mut queue: VecDeque<usize> = VecDeque::new();
        queue.push_back(0); // Start with root

        while let Some(cur_idx) = queue.pop_front() {
            match nodes[cur_idx] {
                TreeNode::Leaf {
                    predictor,
                    offset,
                    multiplier,
                    id,
                } => {
                    flat_nodes.push(FlatTreeNode::Leaf {
                        predictor,
                        offset,
                        multiplier,
                        context: id,
                    });
                }
                TreeNode::Split {
                    property,
                    val,
                    left,
                    right,
                } => {
                    // childID points to first of 4 grandchildren in output
                    let child_id = (flat_nodes.len() + queue.len() + 1) as u32;

                    let mut splitvals = [val, 0, 0];
                    let mut properties = [property, 0, 0];

                    // Process left (i=0) and right (i=1) children
                    for (i, &child_idx) in [left as usize, right as usize].iter().enumerate() {
                        match &nodes[child_idx] {
                            TreeNode::Leaf { .. } => {
                                // Child is leaf: enqueue leaf twice
                                queue.push_back(child_idx);
                                queue.push_back(child_idx);
                            }
                            TreeNode::Split {
                                property: cp,
                                val: cv,
                                left: cl,
                                right: cr,
                            } => {
                                // Child is split: store property/splitval and enqueue grandchildren
                                properties[i + 1] = *cp;
                                splitvals[i + 1] = *cv;
                                queue.push_back(*cl as usize);
                                queue.push_back(*cr as usize);
                            }
                        }
                    }

                    flat_nodes.push(FlatTreeNode::Split {
                        properties,
                        splitvals,
                        child_id,
                    });
                }
            }
        }

        Ok(flat_nodes)
    }
}

// ============================================================================
// Multi-Valued Decision Diagram (MDD) Tree Compilation & Evaluation
// ============================================================================

const MAX_Q_SPAN: usize = 2048;
const Q_MASK: usize = MAX_Q_SPAN - 1;
const MIN_Q_VAL: i32 = -(MAX_Q_SPAN as i32 / 2);
const MAX_Q_VAL: i32 = (MAX_Q_SPAN as i32 / 2) - 1;
// Pad stride by 32 u16s (64B = 1 cache line) to avoid 4KB L1D cache set aliasing across properties.
const Q_STRIDE: usize = MAX_Q_SPAN + 32;
const MAX_STAGE0_FUSED_ENTRIES: usize = 4096;
const MAX_FULL_LUT_ENTRIES: usize = 4096;
const MAX_FUSED_2D_ENTRIES: usize = 8192;
const MAX_MDD_TOTAL_ENTRIES: usize = 16384;
const TABLE_MASK: usize = MAX_MDD_TOTAL_ENTRIES - 1;
const TABLES_OFFSET: usize = 6 * Q_STRIDE;
const MDD_BUF_LEN: usize = TABLES_OFFSET + MAX_MDD_TOTAL_ENTRIES;

const DAG_HASH_BITS: usize = 12;
const DAG_HASH_SIZE: usize = 1 << DAG_HASH_BITS;
const DAG_HASH_MASK: usize = DAG_HASH_SIZE - 1;

#[derive(Clone, Copy)]
enum DagNode {
    Leaf(u16),
    Split {
        prop_rank: u8,
        split_bucket: u16,
        left: u32,
        right: u32,
    },
}

#[derive(Clone, Copy, Default)]
struct HashSlot {
    generation: u32,
    key: u64,
    val: u32,
}

#[derive(Clone, Copy, Default)]
struct MemoSlot {
    generation: u32,
    val: u32,
}

struct MddCompiler {
    generation: u32,
    dag_nodes: Vec<DagNode>,
    dag_table: Vec<HashSlot>,
    state_table: Vec<HashSlot>,
    memo: Vec<MemoSlot>,
    thresholds: [Vec<i32>; 6],
    level_tables: [Vec<u16>; 5],
    level_states: [Vec<u32>; 5],
    dirty_start: [u16; 6],
}

pub(super) struct MddScratch {
    compiler: MddCompiler,
    cached_buf: Option<Box<[u16; MDD_BUF_LEN]>>,
}

impl MddScratch {
    pub(super) const fn new() -> Self {
        Self {
            compiler: MddCompiler {
                generation: 1,
                dag_nodes: Vec::new(),
                dag_table: Vec::new(),
                state_table: Vec::new(),
                memo: Vec::new(),
                thresholds: [const { Vec::new() }; 6],
                level_tables: [const { Vec::new() }; 5],
                level_states: [const { Vec::new() }; 5],
                dirty_start: [MAX_Q_SPAN as u16; 6],
            },
            cached_buf: None,
        }
    }
}

impl MddCompiler {
    #[inline(always)]
    fn bump_gen(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.dag_table.fill(HashSlot::default());
            self.state_table.fill(HashSlot::default());
            self.memo.fill(MemoSlot::default());
            self.generation = 1;
        }
    }

    #[inline(always)]
    fn hash_u64(k: u64) -> usize {
        (k.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - DAG_HASH_BITS)) as usize
    }

    #[inline(always)]
    fn intern_dag(&mut self, key: u64, node: DagNode) -> u32 {
        let mut idx = Self::hash_u64(key);
        let generation = self.generation;
        loop {
            let slot = &mut self.dag_table[idx & DAG_HASH_MASK];
            if slot.generation != generation {
                let id = self.dag_nodes.len() as u32;
                self.dag_nodes.push(node);
                slot.generation = generation;
                slot.key = key;
                slot.val = id;
                return id;
            }
            if slot.key == key {
                return slot.val;
            }
            idx = (idx + 1) & DAG_HASH_MASK;
        }
    }

    fn build_dag(&mut self, tree_nodes: &[TreeNode], rank_of_prop: &[u8; 256]) -> Option<u32> {
        let n = tree_nodes.len();
        if n > 2048 {
            return None;
        }
        if self.dag_table.is_empty() {
            self.dag_table.resize(DAG_HASH_SIZE, HashSlot::default());
            self.state_table.resize(DAG_HASH_SIZE, HashSlot::default());
        }
        self.dag_nodes.clear();
        self.bump_gen();
        let mut ids = [0u32; 2048];
        for i in (0..n).rev() {
            ids[i] = match tree_nodes[i] {
                TreeNode::Leaf { predictor, id, .. } => {
                    let token = ((predictor as u16) << 8) | (id as u16);
                    let key = (1u64 << 63) | (token as u64);
                    self.intern_dag(key, DagNode::Leaf(token))
                }
                TreeNode::Split {
                    property,
                    val,
                    left,
                    right,
                } => {
                    let r = rank_of_prop[property as usize];
                    let t_idx = self.thresholds[r as usize].binary_search(&val).ok()?;
                    let split_bucket = (t_idx + 1) as u16;
                    let l_id = ids[left as usize];
                    let r_id = ids[right as usize];
                    if l_id == r_id {
                        l_id
                    } else {
                        let key = ((r as u64) << 56)
                            | ((split_bucket as u64) << 40)
                            | ((l_id as u64) << 20)
                            | (r_id as u64);
                        self.intern_dag(
                            key,
                            DagNode::Split {
                                prop_rank: r,
                                split_bucket,
                                left: l_id,
                                right: r_id,
                            },
                        )
                    }
                }
            };
        }
        Some(ids[0])
    }

    fn restrict_rec(&mut self, dag_id: u32, prop_rank: u8, bucket: u16) -> u32 {
        let idx = dag_id as usize;
        if self.memo[idx].generation == self.generation {
            return self.memo[idx].val;
        }
        let res = match self.dag_nodes[idx] {
            DagNode::Leaf(_) => dag_id,
            DagNode::Split {
                prop_rank: r,
                split_bucket,
                left,
                right,
            } => {
                if r == prop_rank {
                    let child = if bucket >= split_bucket { left } else { right };
                    self.restrict_rec(child, prop_rank, bucket)
                } else {
                    let new_l = self.restrict_rec(left, prop_rank, bucket);
                    let new_r = self.restrict_rec(right, prop_rank, bucket);
                    if new_l == new_r {
                        new_l
                    } else if new_l == left && new_r == right {
                        dag_id
                    } else {
                        let key = ((r as u64) << 56)
                            | ((split_bucket as u64) << 40)
                            | ((new_l as u64) << 20)
                            | (new_r as u64);
                        self.intern_dag(
                            key,
                            DagNode::Split {
                                prop_rank: r,
                                split_bucket,
                                left: new_l,
                                right: new_r,
                            },
                        )
                    }
                }
            }
        };
        if idx >= self.memo.len() {
            self.memo
                .resize(self.dag_nodes.len() + 64, MemoSlot::default());
        }
        self.memo[idx] = MemoSlot {
            generation: self.generation,
            val: res,
        };
        res
    }

    fn restrict_level(
        &mut self,
        root: u32,
        level_idx: usize,
        prop_rank: u8,
        num_buckets: usize,
    ) -> Option<()> {
        let num_states = if level_idx == 0 {
            1
        } else {
            self.level_states[level_idx - 1].len()
        };
        let table_len = num_states * num_buckets;
        if table_len > MAX_MDD_TOTAL_ENTRIES {
            return None;
        }
        self.level_tables[level_idx].clear();
        self.level_tables[level_idx].reserve(table_len);
        self.level_states[level_idx].clear();

        if num_buckets == 1 {
            if level_idx == 0 {
                self.level_states[0].push(root);
                self.level_tables[0].push(0);
            } else {
                for s_idx in 0..num_states {
                    let dag = self.level_states[level_idx - 1][s_idx];
                    self.level_states[level_idx].push(dag);
                    self.level_tables[level_idx].push(s_idx as u16);
                }
            }
            return Some(());
        }

        self.bump_gen();
        let state_gen = self.generation;

        for s_idx in 0..num_states {
            let state_dag = if level_idx == 0 {
                root
            } else {
                self.level_states[level_idx - 1][s_idx]
            };
            for b in 0..num_buckets {
                self.bump_gen();
                if self.memo.len() < self.dag_nodes.len() {
                    self.memo
                        .resize(self.dag_nodes.len() + 64, MemoSlot::default());
                }
                let next_dag = self.restrict_rec(state_dag, prop_rank, b as u16);

                let key = (next_dag as u64) + 1;
                let mut h_idx = Self::hash_u64(key);
                let next_state_idx = loop {
                    let slot = &mut self.state_table[h_idx & DAG_HASH_MASK];
                    if slot.generation != state_gen {
                        let new_idx = self.level_states[level_idx].len() as u32;
                        if new_idx >= 1024 {
                            return None;
                        }
                        self.level_states[level_idx].push(next_dag);
                        slot.generation = state_gen;
                        slot.key = key;
                        slot.val = new_idx;
                        break new_idx as u16;
                    }
                    if slot.key == key {
                        break slot.val as u16;
                    }
                    h_idx = (h_idx + 1) & DAG_HASH_MASK;
                };
                self.level_tables[level_idx].push(next_state_idx);
            }
        }
        Some(())
    }
}

fn fill_quantizer(q: &mut [u16], dirty_start: &mut u16, thresholds: &[i32], stride: u16) {
    let prev_start = *dirty_start as usize;
    if thresholds.is_empty() || stride == 0 {
        if prev_start < MAX_Q_SPAN {
            q[prev_start..MAX_Q_SPAN].fill(0);
            *dirty_start = MAX_Q_SPAN as u16;
        }
        return;
    }
    let first_nonzero = (thresholds[0] - MIN_Q_VAL + 1) as usize;
    if prev_start < first_nonzero {
        q[prev_start..first_nonzero].fill(0);
    }
    *dirty_start = first_nonzero as u16;
    let mut start_idx = first_nonzero;
    for (b_minus_1, &t) in thresholds[1..].iter().enumerate() {
        let end_idx = (t - MIN_Q_VAL + 1) as usize;
        q[start_idx..end_idx].fill(((b_minus_1 + 1) as u16) * stride);
        start_idx = end_idx;
    }
    q[start_idx..MAX_Q_SPAN].fill((thresholds.len() as u16) * stride);
}

fn write_box(
    tables: &mut [u16],
    bounds: &[(usize, usize); 6],
    strides: &[usize; 6],
    r: usize,
    active_dims: usize,
    base: usize,
    token: u16,
) {
    let (lo, hi) = bounds[r];
    if r + 1 >= active_dims {
        let st = strides[r];
        if st == 1 {
            tables[base + lo..base + hi].fill(token);
        } else {
            for b in lo..hi {
                tables[base + b * st] = token;
            }
        }
    } else {
        let st = strides[r];
        for b in lo..hi {
            write_box(
                tables,
                bounds,
                strides,
                r + 1,
                active_dims,
                base + b * st,
                token,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn fill_box_tree(
    tree_nodes: &[TreeNode],
    node_idx: usize,
    bounds: [(usize, usize); 6],
    strides: &[usize; 6],
    active_dims: usize,
    rank_of_prop: &[u8; 256],
    thresholds: &[Vec<i32>; 6],
    tables: &mut [u16],
) {
    match tree_nodes[node_idx] {
        TreeNode::Leaf { predictor, id, .. } => {
            let token = ((predictor as u16) << 8) | (id as u16);
            write_box(tables, &bounds, strides, 0, active_dims, 0, token);
        }
        TreeNode::Split {
            property,
            val,
            left,
            right,
        } => {
            let r = rank_of_prop[property as usize] as usize;
            let Ok(t_idx) = thresholds[r].binary_search(&val) else {
                return;
            };
            let split_b = t_idx + 1;
            let (lo, hi) = bounds[r];
            if split_b <= lo {
                fill_box_tree(
                    tree_nodes,
                    left as usize,
                    bounds,
                    strides,
                    active_dims,
                    rank_of_prop,
                    thresholds,
                    tables,
                );
            } else if split_b >= hi {
                fill_box_tree(
                    tree_nodes,
                    right as usize,
                    bounds,
                    strides,
                    active_dims,
                    rank_of_prop,
                    thresholds,
                    tables,
                );
            } else {
                let mut left_bounds = bounds;
                left_bounds[r] = (split_b, hi);
                fill_box_tree(
                    tree_nodes,
                    left as usize,
                    left_bounds,
                    strides,
                    active_dims,
                    rank_of_prop,
                    thresholds,
                    tables,
                );
                let mut right_bounds = bounds;
                right_bounds[r] = (lo, split_b);
                fill_box_tree(
                    tree_nodes,
                    right as usize,
                    right_bounds,
                    strides,
                    active_dims,
                    rank_of_prop,
                    thresholds,
                    tables,
                );
            }
        }
    }
}

fn fill_1d_dag(dag_id: u32, lo: usize, hi: usize, dag_nodes: &[DagNode], out: &mut [u16]) {
    match dag_nodes[dag_id as usize] {
        DagNode::Leaf(token) => {
            out[lo..hi].fill(token);
        }
        DagNode::Split {
            split_bucket,
            left,
            right,
            ..
        } => {
            let sb = split_bucket as usize;
            if sb <= lo {
                fill_1d_dag(left, lo, hi, dag_nodes, out);
            } else if sb >= hi {
                fill_1d_dag(right, lo, hi, dag_nodes, out);
            } else {
                fill_1d_dag(right, lo, sb, dag_nodes, out);
                fill_1d_dag(left, sb, hi, dag_nodes, out);
            }
        }
    }
}

fn fill_2d_dag(
    dag_id: u32,
    b0: (usize, usize),
    b1: (usize, usize),
    rank0: u8,
    stride0: usize,
    dag_nodes: &[DagNode],
    out: &mut [u16],
) {
    match dag_nodes[dag_id as usize] {
        DagNode::Leaf(token) => {
            let (lo0, hi0) = b0;
            let (lo1, hi1) = b1;
            if lo1 == 0 && hi1 == stride0 {
                out[lo0 * stride0..hi0 * stride0].fill(token);
            } else {
                for r0 in lo0..hi0 {
                    let row = r0 * stride0;
                    out[row + lo1..row + hi1].fill(token);
                }
            }
        }
        DagNode::Split {
            prop_rank,
            split_bucket,
            left,
            right,
        } => {
            let sb = split_bucket as usize;
            if prop_rank == rank0 {
                let (lo, hi) = b0;
                if sb <= lo {
                    fill_2d_dag(left, b0, b1, rank0, stride0, dag_nodes, out);
                } else if sb >= hi {
                    fill_2d_dag(right, b0, b1, rank0, stride0, dag_nodes, out);
                } else {
                    fill_2d_dag(right, (lo, sb), b1, rank0, stride0, dag_nodes, out);
                    fill_2d_dag(left, (sb, hi), b1, rank0, stride0, dag_nodes, out);
                }
            } else {
                let (lo, hi) = b1;
                if sb <= lo {
                    fill_2d_dag(left, b0, b1, rank0, stride0, dag_nodes, out);
                } else if sb >= hi {
                    fill_2d_dag(right, b0, b1, rank0, stride0, dag_nodes, out);
                } else {
                    fill_2d_dag(right, b0, (lo, sb), rank0, stride0, dag_nodes, out);
                    fill_2d_dag(left, b0, (sb, hi), rank0, stride0, dag_nodes, out);
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MddStages {
    S1,
    S2,
    S3,
    S3Swap34,
    S4,
    S5,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct MddTree<'a> {
    pub props: [u8; 6],
    pub buf: &'a [u16; MDD_BUF_LEN],
    pub stages: MddStages,
    pub pred_mode: LeafPredMode,
}

impl<'a> MddTree<'a> {
    #[inline(always)]
    fn q_idx(val: i32) -> usize {
        ((val.clamp(MIN_Q_VAL, MAX_Q_VAL) - MIN_Q_VAL) as usize) & Q_MASK
    }

    #[inline(always)]
    pub(super) fn eval_local(
        &self,
        prediction_data: PredictionData,
        wp_state: Option<&mut WeightedPredictorState>,
        pos: (usize, usize),
    ) -> (i64, usize) {
        let PredictionData {
            left,
            top,
            topleft,
            topright,
            ..
        } = prediction_data;
        let (wp_pred, wp_prop) = if let Some(wp) = wp_state {
            wp.predict_and_property(pos, &prediction_data)
        } else {
            (0, 0)
        };
        let buf = self.buf;
        let p9 = left.wrapping_add(top).wrapping_sub(topleft);
        let p10 = left.wrapping_sub(topleft);
        let p11 = topleft.wrapping_sub(top);
        let p12 = top.wrapping_sub(topright);

        let c0 = buf[Self::q_idx(p10)] as usize;
        let c1 = buf[Q_STRIDE + Self::q_idx(p11)] as usize;
        let c2 = buf[2 * Q_STRIDE + Self::q_idx(p12)] as usize;
        let c3 = buf[3 * Q_STRIDE + Self::q_idx(p9)] as usize;
        let c4 = buf[4 * Q_STRIDE + Self::q_idx(wp_prop)] as usize;

        let leaf_token = match self.stages {
            MddStages::S1 => buf[TABLES_OFFSET + ((c0 + c1 + c2 + c3 + c4) & TABLE_MASK)],
            MddStages::S2 => {
                let s1 = buf[TABLES_OFFSET + ((c0 + c1 + c2) & TABLE_MASK)] as usize;
                buf[TABLES_OFFSET + ((s1 + c3 + c4) & TABLE_MASK)]
            }
            MddStages::S3 => {
                let s1 = buf[TABLES_OFFSET + ((c0 + c1 + c2) & TABLE_MASK)] as usize;
                let s2 = buf[TABLES_OFFSET + ((s1 + c3) & TABLE_MASK)] as usize;
                buf[TABLES_OFFSET + ((s2 + c4) & TABLE_MASK)]
            }
            MddStages::S3Swap34 => {
                let s1 = buf[TABLES_OFFSET + ((c0 + c1 + c2) & TABLE_MASK)] as usize;
                let s2 = buf[TABLES_OFFSET + ((s1 + c4) & TABLE_MASK)] as usize;
                buf[TABLES_OFFSET + ((s2 + c3) & TABLE_MASK)]
            }
            MddStages::S4 | MddStages::S5 => {
                let s1 = buf[TABLES_OFFSET + ((c0 + c1) & TABLE_MASK)] as usize;
                let s2 = buf[TABLES_OFFSET + ((s1 + c2) & TABLE_MASK)] as usize;
                let s3 = buf[TABLES_OFFSET + ((s2 + c3) & TABLE_MASK)] as usize;
                buf[TABLES_OFFSET + ((s3 + c4) & TABLE_MASK)]
            }
        };
        self.pred_mode
            .eval_leaf::<true>(leaf_token, prediction_data, wp_pred)
    }

    #[inline(always)]
    pub(super) fn eval_general_props(
        &self,
        prediction_data: PredictionData,
        wp_pred: i64,
        property_buffer: &[i32; 256],
    ) -> (i64, usize) {
        let buf = self.buf;
        let c0 = buf[Self::q_idx(property_buffer[self.props[0] as usize])] as usize;
        let c1 = buf[Q_STRIDE + Self::q_idx(property_buffer[self.props[1] as usize])] as usize;
        let c2 = buf[2 * Q_STRIDE + Self::q_idx(property_buffer[self.props[2] as usize])] as usize;
        let c3 = buf[3 * Q_STRIDE + Self::q_idx(property_buffer[self.props[3] as usize])] as usize;
        let c4 = buf[4 * Q_STRIDE + Self::q_idx(property_buffer[self.props[4] as usize])] as usize;
        let c5 = buf[5 * Q_STRIDE + Self::q_idx(property_buffer[self.props[5] as usize])] as usize;

        let leaf_token = match self.stages {
            MddStages::S1 => buf[TABLES_OFFSET + ((c0 + c1 + c2 + c3 + c4 + c5) & TABLE_MASK)],
            MddStages::S2 => {
                let s1 = buf[TABLES_OFFSET + ((c0 + c1 + c2) & TABLE_MASK)] as usize;
                buf[TABLES_OFFSET + ((s1 + c3 + c4) & TABLE_MASK)]
            }
            MddStages::S3 | MddStages::S3Swap34 => {
                let s1 = buf[TABLES_OFFSET + ((c0 + c1 + c2) & TABLE_MASK)] as usize;
                let s2 = buf[TABLES_OFFSET + ((s1 + c3) & TABLE_MASK)] as usize;
                buf[TABLES_OFFSET + ((s2 + c4 + c5) & TABLE_MASK)]
            }
            MddStages::S4 => {
                let s1 = buf[TABLES_OFFSET + ((c0 + c1) & TABLE_MASK)] as usize;
                let s2 = buf[TABLES_OFFSET + ((s1 + c2) & TABLE_MASK)] as usize;
                let s3 = buf[TABLES_OFFSET + ((s2 + c3) & TABLE_MASK)] as usize;
                buf[TABLES_OFFSET + ((s3 + c4) & TABLE_MASK)]
            }
            MddStages::S5 => {
                let s1 = buf[TABLES_OFFSET + ((c0 + c1) & TABLE_MASK)] as usize;
                let s2 = buf[TABLES_OFFSET + ((s1 + c2) & TABLE_MASK)] as usize;
                let s3 = buf[TABLES_OFFSET + ((s2 + c3) & TABLE_MASK)] as usize;
                let s4 = buf[TABLES_OFFSET + ((s3 + c4) & TABLE_MASK)] as usize;
                buf[TABLES_OFFSET + ((s4 + c5) & TABLE_MASK)]
            }
        };
        self.pred_mode
            .eval_leaf::<false>(leaf_token, prediction_data, wp_pred)
    }

    pub(super) fn try_new(
        tree_nodes: &[TreeNode],
        meta: &TreeMeta,
        local_layout: bool,
        num_pixels: usize,
        mdd_scratch: &'a mut MddScratch,
    ) -> Option<Self> {
        if !meta.all_simple_leaves
            || meta.num_props == 0
            || meta.num_props > 6
            || num_pixels < 4096
            || tree_nodes.len() < 8
            || meta.min_splitval < MIN_Q_VAL
            || meta.max_splitval >= MAX_Q_VAL
        {
            return None;
        }
        let max_stage0_3d = if num_pixels <= 65536 {
            1024
        } else {
            MAX_STAGE0_FUSED_ENTRIES
        };
        let max_fused_2d = if num_pixels <= 65536 {
            1536
        } else {
            MAX_FUSED_2D_ENTRIES
        };

        let MddScratch {
            compiler: scratch,
            cached_buf,
        } = mdd_scratch;

        for t in &mut scratch.thresholds {
            t.clear();
        }

        let mut props = [0u8; 6];
        let mut rank_of_prop = [0u8; 256];
        let mut slot_of_rank: [usize; 6] = [0, 1, 2, 3, 4, 5];
        let active_dims: usize;

        if local_layout {
            props = [10, 11, 12, 9, 15, 0];
            active_dims = 5;
            for (r, &p) in props[..5].iter().enumerate() {
                rank_of_prop[p as usize] = r as u8;
            }
            for node in tree_nodes {
                if let TreeNode::Split { property, val, .. } = *node {
                    scratch.thresholds[rank_of_prop[property as usize] as usize].push(val);
                }
            }
            for thresh in &mut scratch.thresholds[..5] {
                thresh.sort_unstable();
                thresh.dedup();
                if thresh.len() >= 254 {
                    return None;
                }
            }
            let b012 = (scratch.thresholds[0].len() + 1)
                * (scratch.thresholds[1].len() + 1)
                * (scratch.thresholds[2].len() + 1);
            if b012 <= max_stage0_3d {
                let cmp_len = |l: usize| if l == 0 { usize::MAX } else { l };
                for i in 1..3 {
                    let mut j = i;
                    while j > 0
                        && cmp_len(scratch.thresholds[j].len())
                            < cmp_len(scratch.thresholds[j - 1].len())
                    {
                        scratch.thresholds.swap(j, j - 1);
                        props.swap(j, j - 1);
                        slot_of_rank.swap(j, j - 1);
                        j -= 1;
                    }
                }
                if cmp_len(scratch.thresholds[4].len()) < cmp_len(scratch.thresholds[3].len()) {
                    scratch.thresholds.swap(3, 4);
                    props.swap(3, 4);
                    slot_of_rank.swap(3, 4);
                }
                for (r, &p) in props[..5].iter().enumerate() {
                    rank_of_prop[p as usize] = r as u8;
                }
            }
        } else {
            let k = meta.num_props;
            active_dims = k;
            let mut prop_lookup = [0usize; 256];
            for (i, (&p, dst)) in meta.used_props[..k].iter().zip(&mut props[..k]).enumerate() {
                *dst = p;
                prop_lookup[p as usize] = i;
            }
            for node in tree_nodes {
                if let TreeNode::Split { property, val, .. } = *node {
                    scratch.thresholds[prop_lookup[property as usize]].push(val);
                }
            }
            for thresh in &mut scratch.thresholds[..k] {
                thresh.sort_unstable();
                thresh.dedup();
                if thresh.len() >= 254 {
                    return None;
                }
            }
            for i in 1..k {
                let mut j = i;
                while j > 0 && scratch.thresholds[j].len() < scratch.thresholds[j - 1].len() {
                    scratch.thresholds.swap(j, j - 1);
                    props.swap(j, j - 1);
                    j -= 1;
                }
            }
            for (r, &p) in props[..k].iter().enumerate() {
                rank_of_prop[p as usize] = r as u8;
            }
        }

        let mut m = [1usize; 6];
        for (dst, thresh) in m.iter_mut().zip(&scratch.thresholds).take(active_dims) {
            *dst = thresh.len() + 1;
        }

        let buf: &'a mut [u16; MDD_BUF_LEN] = cached_buf.get_or_insert_with(|| {
            scratch.dirty_start = [MAX_Q_SPAN as u16; 6];
            crate::util::box_array(0u16)
        });

        let build_result: Option<([u16; 6], MddStages)> = 'stage_select: {
            let tables = &mut buf[TABLES_OFFSET..];

            // 1. Try 1-Stage Full LUT (`S1`, 0 dependent state loads).
            let max_full = if meta.num_props <= 2 {
                MAX_MDD_TOTAL_ENTRIES
            } else {
                MAX_FULL_LUT_ENTRIES
            };
            let total_prod = m[0]
                .saturating_mul(m[1])
                .saturating_mul(m[2])
                .saturating_mul(m[3])
                .saturating_mul(m[4])
                .saturating_mul(m[5]);
            if total_prod <= max_full {
                let mut strides = [1usize; 6];
                for r in (0..active_dims.saturating_sub(1)).rev() {
                    strides[r] = strides[r + 1] * m[r + 1];
                }
                let full_bounds: [(usize, usize); 6] = std::array::from_fn(|r| (0, m[r]));
                fill_box_tree(
                    tree_nodes,
                    0,
                    full_bounds,
                    &strides,
                    active_dims,
                    &rank_of_prop,
                    &scratch.thresholds,
                    &mut tables[..total_prod],
                );
                let strides_u16: [u16; 6] = std::array::from_fn(|r| {
                    if r < active_dims {
                        strides[r] as u16
                    } else {
                        0
                    }
                });
                break 'stage_select Some((strides_u16, MddStages::S1));
            }

            let Some(root) = scratch.build_dag(tree_nodes, &rank_of_prop) else {
                break 'stage_select None;
            };
            if scratch.restrict_level(root, 0, 0, m[0]).is_none()
                || scratch.restrict_level(root, 1, 1, m[1]).is_none()
                || scratch.restrict_level(root, 2, 2, m[2]).is_none()
            {
                break 'stage_select None;
            }
            let b012_len = m[0] * m[1] * m[2];
            let states_3_len = scratch.level_states[2].len();

            let fill_stage0_3d = |scratch: &MddCompiler, tables: &mut [u16], next_stride: usize| {
                let t0 = &scratch.level_tables[0];
                let t1 = &scratch.level_tables[1];
                let t2 = &scratch.level_tables[2];
                for (b0, &s1_u16) in t0.iter().enumerate().take(m[0]) {
                    let s1 = s1_u16 as usize;
                    for b1 in 0..m[1] {
                        let s2 = t1[s1 * m[1] + b1] as usize;
                        let row_s2 = s2 * m[2];
                        let out_base = (b0 * m[1] + b1) * m[2];
                        for b2 in 0..m[2] {
                            let s3 = t2[row_s2 + b2] as usize;
                            tables[out_base + b2] = (b012_len + s3 * next_stride) as u16;
                        }
                    }
                }
            };

            // 2. Try 2-Stage `3 + 2` Fused MDD (`S2`, 1 dependent state load) when `m[5] == 1`.
            if m[5] == 1 && b012_len <= max_stage0_3d {
                let b34_len = m[3] * m[4];
                let stage1_2d_len = states_3_len * b34_len;
                if b012_len + stage1_2d_len <= max_fused_2d {
                    fill_stage0_3d(scratch, tables, b34_len);
                    for (s3_idx, &dag_id) in scratch.level_states[2].iter().enumerate() {
                        let start = b012_len + s3_idx * b34_len;
                        fill_2d_dag(
                            dag_id,
                            (0, m[3]),
                            (0, m[4]),
                            3,
                            m[4],
                            &scratch.dag_nodes,
                            &mut tables[start..start + b34_len],
                        );
                    }
                    let strides: [u16; 6] =
                        [(m[1] * m[2]) as u16, m[2] as u16, 1, m[4] as u16, 1, 0];
                    break 'stage_select Some((strides, MddStages::S2));
                }
            }

            if scratch.restrict_level(root, 3, 3, m[3]).is_none() {
                break 'stage_select None;
            }
            let states_4_len = scratch.level_states[3].len();
            let b45_len = m[4] * m[5];

            // 3. Try 3-Stage `3 + 1 + 2` (or `3 + 1 + 1` when `m[5] == 1`) (`S3`, 2 dependent state loads).
            if b012_len <= max_stage0_3d {
                let s1_len = states_3_len * m[3];
                let s2_len = states_4_len * b45_len;
                let total_len = b012_len + s1_len + s2_len;
                let limit = if m[5] == 1 {
                    MAX_MDD_TOTAL_ENTRIES
                } else {
                    max_fused_2d
                };
                if total_len <= limit {
                    fill_stage0_3d(scratch, tables, m[3]);
                    let base2 = b012_len + s1_len;
                    for (i, &s4) in scratch.level_tables[3].iter().enumerate() {
                        tables[b012_len + i] = (base2 + (s4 as usize) * b45_len) as u16;
                    }
                    for (s4_idx, &dag_id) in scratch.level_states[3].iter().enumerate() {
                        let start = base2 + s4_idx * b45_len;
                        if m[5] == 1 {
                            fill_1d_dag(
                                dag_id,
                                0,
                                m[4],
                                &scratch.dag_nodes,
                                &mut tables[start..start + m[4]],
                            );
                        } else {
                            fill_2d_dag(
                                dag_id,
                                (0, m[4]),
                                (0, m[5]),
                                4,
                                m[5],
                                &scratch.dag_nodes,
                                &mut tables[start..start + b45_len],
                            );
                        }
                    }
                    let strides: [u16; 6] =
                        [(m[1] * m[2]) as u16, m[2] as u16, 1, 1, m[5] as u16, 1];
                    let stage = if local_layout && slot_of_rank[3] == 4 {
                        MddStages::S3Swap34
                    } else {
                        MddStages::S3
                    };
                    break 'stage_select Some((strides, stage));
                }
                if local_layout && slot_of_rank != [0, 1, 2, 3, 4, 5] {
                    break 'stage_select None;
                }
            }

            // 4. Fallback `S4` (`2 + 1 + 1 + 1` when `m[5] == 1`) or `S5` (`2 + 1 + 1 + 1 + 1`).
            let b01_len = m[0] * m[1];
            if b01_len > MAX_STAGE0_FUSED_ENTRIES {
                break 'stage_select None;
            }
            let s1_len = scratch.level_states[1].len() * m[2];
            let s2_len = states_3_len * m[3];
            let s3_len = states_4_len * m[4];
            let s4_len = if m[5] == 1 {
                0
            } else {
                if scratch.restrict_level(root, 4, 4, m[4]).is_none() {
                    break 'stage_select None;
                }
                scratch.level_states[4].len() * m[5]
            };
            let total_len = b01_len + s1_len + s2_len + s3_len + s4_len;
            if total_len > MAX_MDD_TOTAL_ENTRIES {
                break 'stage_select None;
            }
            let base2 = b01_len + s1_len;
            let base3 = base2 + s2_len;
            let base4 = base3 + s3_len;
            {
                let t0 = &scratch.level_tables[0];
                let t1 = &scratch.level_tables[1];
                let t2 = &scratch.level_tables[2];
                let t3 = &scratch.level_tables[3];
                for b0 in 0..m[0] {
                    let s1 = t0[b0] as usize;
                    for b1 in 0..m[1] {
                        let s2 = t1[s1 * m[1] + b1] as usize;
                        tables[b0 * m[1] + b1] = (b01_len + s2 * m[2]) as u16;
                    }
                }
                for (i, &s3) in t2.iter().enumerate() {
                    tables[b01_len + i] = (base2 + (s3 as usize) * m[3]) as u16;
                }
                for (i, &s4) in t3.iter().enumerate() {
                    tables[base2 + i] = (base3 + (s4 as usize) * m[4]) as u16;
                }
            }
            if m[5] == 1 {
                for (s4_idx, &dag_id) in scratch.level_states[3].iter().enumerate() {
                    let start = base3 + s4_idx * m[4];
                    fill_1d_dag(
                        dag_id,
                        0,
                        m[4],
                        &scratch.dag_nodes,
                        &mut tables[start..start + m[4]],
                    );
                }
                let strides: [u16; 6] = [m[1] as u16, 1, 1, 1, 1, 0];
                break 'stage_select Some((strides, MddStages::S4));
            }

            for (i, &s5) in scratch.level_tables[4].iter().enumerate() {
                tables[base3 + i] = (base4 + (s5 as usize) * m[5]) as u16;
            }
            for (s5_idx, &dag_id) in scratch.level_states[4].iter().enumerate() {
                let start = base4 + s5_idx * m[5];
                fill_1d_dag(
                    dag_id,
                    0,
                    m[5],
                    &scratch.dag_nodes,
                    &mut tables[start..start + m[5]],
                );
            }
            let strides: [u16; 6] = [m[1] as u16, 1, 1, 1, 1, 1];
            Some((strides, MddStages::S5))
        };

        let (strides, stages) = build_result?;

        let MddCompiler {
            dirty_start,
            thresholds,
            ..
        } = scratch;
        let num_q_slots = if local_layout { 5 } else { 6 };
        for r in 0..num_q_slots {
            let slot = slot_of_rank[r];
            let q_slice = &mut buf[slot * Q_STRIDE..(slot + 1) * Q_STRIDE];
            if r < active_dims {
                fill_quantizer(q_slice, &mut dirty_start[slot], &thresholds[r], strides[r]);
            } else {
                fill_quantizer(q_slice, &mut dirty_start[slot], &[], 0);
            }
        }

        Some(Self {
            props,
            buf,
            stages,
            pred_mode: meta.pred_mode,
        })
    }
}
