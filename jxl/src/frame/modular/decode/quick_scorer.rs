// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! QuickScorer-style evaluation of MA trees.
//!
//! Instead of walking the tree, every split is turned into a bitmask over the leaves (in DFS
//! order) that are unreachable when the split goes a given way. For each used property, the
//! thresholds of all splits on that property partition the value range into buckets; each
//! bucket stores the AND of the masks of all splits on that property. The reached leaf is then
//! the lowest set bit of the AND of one bucket mask per property, which is branchless.
//!
//! The bucket of a property value is found with a lookup table, so this is only used for trees
//! with at most 8 properties and thresholds within `[QS_LUT_MIN_SPLITVAL, QS_LUT_MAX_SPLITVAL)`.

use crate::frame::modular::flat_tree::TreeLeaf;
use crate::frame::modular::predict::PredictionData;
use crate::frame::modular::tree::TreeNode;

const MAX_QS_LEAVES: usize = 256;
const QS_LUT_MIN_SPLITVAL: i32 = -1024;
const QS_LUT_MAX_SPLITVAL: i32 = 1023;
const QS_LUT_SIZE: usize = (QS_LUT_MAX_SPLITVAL - QS_LUT_MIN_SPLITVAL + 1) as usize;
/// Trees shallower than this are faster to evaluate with `FlatTree` (at most two iterations
/// of its 3-node step), so QuickScorer is not used for them.
const QS_MIN_DEPTH: u32 = 5;

static DUMMY_LUT: [u8; QS_LUT_SIZE] = [0; QS_LUT_SIZE];
static DUMMY_MASKS: [[u64; 4]; 1] = [[!0u64; 4]];

#[derive(Clone, Copy)]
struct QsSplit {
    prop: u8,
    val: i32,
    /// Leaves (in DFS order) of the subtree taken when `property > val`.
    left: (usize, usize),
    /// Leaves (in DFS order) of the subtree taken when `property <= val`.
    right: (usize, usize),
}

#[derive(Clone, Copy)]
enum DfsAction {
    Visit { node: u32, depth: u32 },
    Merge { prop: u8, val: i32 },
}

#[derive(Default)]
pub(crate) struct QuickScorerScratch {
    leaves: Vec<TreeLeaf>,
    splits: Vec<QsSplit>,
    used_props: Vec<u8>,
    thresholds: Vec<Vec<i32>>,
    masks: Vec<[u64; 4]>,
    mask_offsets: Vec<usize>,
    #[allow(clippy::vec_box)]
    quantizers: Vec<Box<[u8; QS_LUT_SIZE]>>,
    dfs_stack: Vec<DfsAction>,
    range_stack: Vec<(usize, usize)>,
}

#[inline(always)]
fn get_local_prop(p: u8, pd: &PredictionData, wp_prop: i32) -> i32 {
    match p {
        9 => pd.left.wrapping_add(pd.top).wrapping_sub(pd.topleft),
        10 => pd.left.wrapping_sub(pd.topleft),
        11 => pd.topleft.wrapping_sub(pd.top),
        12 => pd.top.wrapping_sub(pd.topright),
        15 => wp_prop,
        _ => 0,
    }
}

/// Returns the index of the first leaf that survives the AND of `N` bucket masks.
#[inline(always)]
fn first_leaf<const N: usize>(wide: bool, mask: impl Fn(usize) -> [u64; 4]) -> usize {
    if !wide {
        let mut m = !0u64;
        for i in 0..N {
            m &= mask(i)[0];
        }
        m.trailing_zeros() as usize
    } else {
        let mut w = [!0u64; 4];
        for i in 0..N {
            let m = mask(i);
            for j in 0..4 {
                w[j] &= m[j];
            }
        }
        if w[0] != 0 {
            w[0].trailing_zeros() as usize
        } else if w[1] != 0 {
            64 + w[1].trailing_zeros() as usize
        } else if w[2] != 0 {
            128 + w[2].trailing_zeros() as usize
        } else {
            192 + w[3].trailing_zeros() as usize
        }
    }
}

/// Buckets found via one lookup table per property.
#[derive(Clone, Copy)]
struct QsLut<'a, const N: usize> {
    props: [u8; N],
    quantizers: [&'a [u8; QS_LUT_SIZE]; N],
    masks: [&'a [[u64; 4]]; N],
}

impl<const N: usize> QsLut<'_, N> {
    #[inline(always)]
    fn leaf(&self, wide: bool, prop: impl Fn(u8) -> i32) -> usize {
        first_leaf::<N>(wide, |i| {
            let p = prop(self.props[i]);
            let v =
                (p.clamp(QS_LUT_MIN_SPLITVAL, QS_LUT_MAX_SPLITVAL) - QS_LUT_MIN_SPLITVAL) as usize;
            self.masks[i][self.quantizers[i][v] as usize]
        })
    }
}

#[derive(Clone, Copy)]
enum QsKind<'a> {
    Lut1(QsLut<'a, 1>),
    Lut2(QsLut<'a, 2>),
    Lut4(QsLut<'a, 4>),
    // Five properties are common in practice (e.g. all trees of flower_foveon_d0p.jxl).
    Lut5(QsLut<'a, 5>),
    Lut8(QsLut<'a, 8>),
}

#[derive(Clone, Copy)]
pub(crate) struct QuickScorer<'a> {
    kind: QsKind<'a>,
    leaves: &'a [TreeLeaf],
    /// More than 64 leaves: masks use all four words.
    wide: bool,
    /// All used properties can be computed from `PredictionData` and the WP property alone.
    is_local: bool,
}

impl<'a> QuickScorer<'a> {
    #[inline(always)]
    fn leaf(&self, prop: impl Fn(u8) -> i32) -> usize {
        let wide = self.wide;
        match &self.kind {
            QsKind::Lut1(q) => q.leaf(wide, prop),
            QsKind::Lut2(q) => q.leaf(wide, prop),
            QsKind::Lut4(q) => q.leaf(wide, prop),
            QsKind::Lut5(q) => q.leaf(wide, prop),
            QsKind::Lut8(q) => q.leaf(wide, prop),
        }
    }

    #[inline(always)]
    pub(crate) fn eval(&self, property_buffer: &[i32; 256]) -> TreeLeaf {
        self.leaves[self.leaf(|p| property_buffer[p as usize])]
    }

    /// Returns the context of the reached leaf; only valid if `is_local()`.
    #[inline(always)]
    pub(crate) fn eval_local(&self, prediction_data: &PredictionData, wp_prop: i32) -> usize {
        self.leaves[self.leaf(|p| get_local_prop(p, prediction_data, wp_prop))].context as usize
    }

    pub(crate) fn is_local(&self) -> bool {
        self.is_local
    }

    pub(crate) fn leaves(&self) -> &[TreeLeaf] {
        self.leaves
    }

    fn padded_props<const N: usize>(used_props: &[u8]) -> [u8; N] {
        let mut props = [0u8; N];
        props[..used_props.len()].copy_from_slice(used_props);
        props
    }

    fn build_lut<const N: usize>(scratch: &'a QuickScorerScratch) -> QsLut<'a, N> {
        let num_props = scratch.used_props.len();
        let mut quantizers = [&DUMMY_LUT; N];
        let mut masks: [&[[u64; 4]]; N] = [&DUMMY_MASKS[..]; N];
        for i in 0..num_props {
            quantizers[i] = &scratch.quantizers[i];
            let offset = scratch.mask_offsets[i];
            masks[i] = &scratch.masks[offset..offset + scratch.thresholds[i].len() + 1];
        }
        QsLut {
            props: Self::padded_props(&scratch.used_props),
            quantizers,
            masks,
        }
    }

    /// Tries to build a QuickScorer for a pruned tree (as produced by
    /// `run_on_specialized_tree`). Returns `None` if the tree is too shallow, too large, uses
    /// too many properties or has thresholds outside of the LUT range, in which case `FlatTree`
    /// should be used.
    pub(crate) fn try_build(
        pruned_tree: &[TreeNode],
        scratch: &'a mut QuickScorerScratch,
    ) -> Option<Self> {
        let plan = scratch.prepare(pruned_tree)?;
        let s: &'a QuickScorerScratch = scratch;
        let kind = match plan.num_props {
            1 => QsKind::Lut1(Self::build_lut(s)),
            2 => QsKind::Lut2(Self::build_lut(s)),
            3 | 4 => QsKind::Lut4(Self::build_lut(s)),
            5 => QsKind::Lut5(Self::build_lut(s)),
            _ => QsKind::Lut8(Self::build_lut(s)),
        };
        Some(Self {
            kind,
            leaves: &s.leaves,
            wide: s.leaves.len() > 64,
            is_local: s
                .used_props
                .iter()
                .all(|&p| matches!(p, 9 | 10 | 11 | 12 | 15)),
        })
    }
}

struct QsPlan {
    num_props: usize,
}

impl QuickScorerScratch {
    fn prepare(&mut self, pruned_tree: &[TreeNode]) -> Option<QsPlan> {
        let Self {
            leaves,
            splits,
            used_props,
            thresholds,
            masks,
            mask_offsets,
            quantizers,
            dfs_stack,
            range_stack,
        } = self;
        leaves.clear();
        splits.clear();
        used_props.clear();
        masks.clear();
        mask_offsets.clear();
        dfs_stack.clear();
        range_stack.clear();

        if pruned_tree.is_empty() {
            return None;
        }

        // Iterative DFS: assign leaf indices in DFS order and record, for every split, the leaf
        // ranges of its two subtrees.
        let mut max_depth = 0;
        dfs_stack.push(DfsAction::Visit { node: 0, depth: 0 });
        while let Some(action) = dfs_stack.pop() {
            match action {
                DfsAction::Visit { node, depth } => match pruned_tree[node as usize] {
                    TreeNode::Leaf {
                        predictor,
                        offset,
                        multiplier,
                        id,
                    } => {
                        max_depth = max_depth.max(depth);
                        let leaf_idx = leaves.len();
                        if leaf_idx >= MAX_QS_LEAVES {
                            return None;
                        }
                        leaves.push(TreeLeaf {
                            predictor,
                            offset,
                            multiplier,
                            context: id,
                        });
                        range_stack.push((leaf_idx, leaf_idx + 1));
                    }
                    TreeNode::Split {
                        property,
                        val,
                        left,
                        right,
                    } => {
                        dfs_stack.push(DfsAction::Merge {
                            prop: property,
                            val,
                        });
                        dfs_stack.push(DfsAction::Visit {
                            node: right,
                            depth: depth + 1,
                        });
                        dfs_stack.push(DfsAction::Visit {
                            node: left,
                            depth: depth + 1,
                        });
                    }
                },
                DfsAction::Merge { prop, val } => {
                    let right = range_stack.pop()?;
                    let left = range_stack.pop()?;
                    splits.push(QsSplit {
                        prop,
                        val,
                        left,
                        right,
                    });
                    range_stack.push((left.0, right.1));
                }
            }
        }

        if max_depth < QS_MIN_DEPTH {
            return None;
        }
        let num_leaves = leaves.len();

        // Properties are clamped to the LUT range before the lookup, so a split at
        // QS_LUT_MAX_SPLITVAL could not distinguish the values above it.
        let mut in_lut_range = true;
        for s in splits.iter() {
            in_lut_range &= (QS_LUT_MIN_SPLITVAL..QS_LUT_MAX_SPLITVAL).contains(&s.val);
            if !used_props.contains(&s.prop) {
                if used_props.len() >= 8 {
                    return None;
                }
                used_props.push(s.prop);
            }
        }
        used_props.sort_unstable();
        let num_props = used_props.len();
        let prop_index = |p: u8| used_props.iter().position(|&x| x == p).unwrap();

        if thresholds.len() < num_props {
            thresholds.resize(num_props, Vec::new());
        }
        for t in thresholds.iter_mut() {
            t.clear();
        }
        for s in splits.iter() {
            thresholds[prop_index(s.prop)].push(s.val);
        }
        let mut max_thresholds = 0;
        for t in thresholds[..num_props].iter_mut() {
            t.sort_unstable();
            t.dedup();
            max_thresholds = max_thresholds.max(t.len());
        }

        if !in_lut_range || max_thresholds >= 255 {
            return None;
        }

        // Bucket masks: property `i` has `thresholds[i].len() + 1` buckets.
        let mut offset = 0;
        for t in &thresholds[..num_props] {
            mask_offsets.push(offset);
            offset += t.len() + 1;
        }
        let mut all_leaves = [0u64; 4];
        for i in 0..num_leaves {
            all_leaves[i / 64] |= 1u64 << (i % 64);
        }
        masks.resize(offset, all_leaves);

        let range_mask = |(start, end): (usize, usize)| {
            let mut m = [0u64; 4];
            for bit in start..end {
                m[bit / 64] |= 1u64 << (bit % 64);
            }
            m
        };
        for s in splits.iter() {
            let pi = prop_index(s.prop);
            let thr_idx = thresholds[pi].binary_search(&s.val).unwrap();
            let num_buckets = thresholds[pi].len() + 1;
            let base = mask_offsets[pi];
            // Buckets up to `thr_idx` have `property <= val`, so the left subtree is
            // unreachable; the others have `property > val`.
            let left = range_mask(s.left);
            let right = range_mask(s.right);
            for (b, m) in masks[base..base + num_buckets].iter_mut().enumerate() {
                let unreachable = if b <= thr_idx { &left } else { &right };
                for j in 0..4 {
                    m[j] &= !unreachable[j];
                }
            }
        }

        while quantizers.len() < num_props {
            quantizers.push(crate::util::box_array(0u8));
        }
        for (thr, q) in thresholds.iter().zip(quantizers.iter_mut()).take(num_props) {
            // q[v - QS_LUT_MIN_SPLITVAL] = number of thresholds < v, i.e. the bucket of v.
            let mut b = 0u8;
            for (i, entry) in q.iter_mut().enumerate() {
                let val = i as i32 + QS_LUT_MIN_SPLITVAL;
                while (b as usize) < thr.len() && val > thr[b as usize] {
                    b += 1;
                }
                *entry = b;
            }
        }

        Some(QsPlan { num_props })
    }
}
