// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::fmt::Debug;

use super::{Predictor, predict::WeightedPredictorState};
use crate::{
    bit_reader::BitReader,
    entropy_coding::decode::Histograms,
    entropy_coding::decode::SymbolReader,
    error::{Error, Result},
    frame::modular::predict::PredictionData,
    image::Image,
    util::{NewWithCapacity, tracing_wrappers::*},
};

#[derive(Debug, Clone, Copy)]
pub enum TreeNode {
    Split {
        property: u8,
        val: i32,
        left: u32,
        right: u32,
    },
    Leaf {
        predictor: Predictor,
        offset: i32,
        multiplier: u32,
        id: u32,
    },
}

/// Flattened tree node for optimized traversal (matches C++ FlatDecisionNode).
/// Stores parent + info about both children to evaluate 3 nodes per iteration.
/// Leaf nodes store `Predictor` directly to avoid runtime integer-to-enum conversion.
#[derive(Debug, Clone, Copy)]
pub(super) struct FlatTreeNode {
    property0: i32,                    // Property to test, -1 if leaf
    splitval0: i32,                    // Split value (only used for split nodes)
    predictor: Predictor,              // Stored for leaf nodes (avoids runtime conversion)
    splitvals_or_multiplier: [i32; 2], // Child splitvals, or multiplier if leaf
    child_id: u32,                     // Index to first grandchild, or context if leaf
    properties_or_offset: [i16; 2],    // Child properties, or offset if leaf
}

impl FlatTreeNode {
    #[inline]
    fn leaf(predictor: Predictor, offset: i32, multiplier: u32, context: u32) -> Self {
        Self {
            property0: -1,
            splitval0: 0,
            predictor,
            splitvals_or_multiplier: [multiplier as i32, 0],
            child_id: context,
            properties_or_offset: [offset as i16, 0],
        }
    }

    #[inline]
    fn split(
        property0: i32,
        splitval0: i32,
        child_id: u32,
        properties: [i16; 2],
        splitvals: [i32; 2],
    ) -> Self {
        Self {
            property0,
            splitval0,
            predictor: Predictor::Zero, // unused for split nodes
            splitvals_or_multiplier: splitvals,
            child_id,
            properties_or_offset: properties,
        }
    }
}

pub struct Tree {
    pub nodes: Vec<TreeNode>,
    pub histograms: Histograms,
}

fn validate_tree(tree: &[TreeNode], num_properties: usize) -> Result<()> {
    const HEIGHT_LIMIT: usize = 2048;

    if tree.is_empty() {
        return Ok(());
    }

    // This mirrors libjxl's ValidateTree(), but avoids allocating
    // `num_properties * tree.len()` entries.
    //
    // We do an explicit DFS and keep the property ranges only for the current root->node path.
    // When descending into a child we update exactly one property's range (the one we split on)
    // and store the previous range in the child frame; when returning from that child we restore
    // it. This makes memory O(num_properties + height) instead of O(num_properties * tree_size).

    #[derive(Clone, Copy, Debug)]
    enum Stage {
        Enter,
        AfterLeft,
        AfterRight,
    }

    struct Frame {
        node: usize,
        depth: usize,
        stage: Stage,
        restore: Option<(usize, (i32, i32))>,
    }

    let mut property_ranges: Vec<(i32, i32)> = vec![(i32::MIN, i32::MAX); num_properties];
    let mut stack = vec![Frame {
        node: 0,
        depth: 0,
        stage: Stage::Enter,
        restore: None,
    }];

    while let Some(mut frame) = stack.pop() {
        if frame.depth > HEIGHT_LIMIT {
            return Err(Error::TreeTooTall(frame.depth, HEIGHT_LIMIT));
        }

        match (frame.stage, tree[frame.node]) {
            (Stage::Enter, TreeNode::Leaf { .. }) => {
                if let Some((p, old)) = frame.restore {
                    property_ranges[p] = old;
                }
            }
            (
                Stage::Enter,
                TreeNode::Split {
                    property,
                    val,
                    left,
                    right: _,
                },
            ) => {
                let p = property as usize;
                let (l, u) = property_ranges[p];
                if l > val || u <= val {
                    return Err(Error::TreeSplitOnEmptyRange(property, val, l, u));
                }

                frame.stage = Stage::AfterLeft;
                let depth = frame.depth;
                stack.push(frame);

                // Descend into left child: range becomes (val+1, u).
                let old = property_ranges[p];
                property_ranges[p] = (val + 1, u);
                stack.push(Frame {
                    node: left as usize,
                    depth: depth + 1,
                    stage: Stage::Enter,
                    restore: Some((p, old)),
                });
            }
            (
                Stage::AfterLeft,
                TreeNode::Split {
                    property,
                    val,
                    left: _,
                    right,
                },
            ) => {
                let p = property as usize;
                let (l, u) = property_ranges[p];
                if l > val || u <= val {
                    return Err(Error::TreeSplitOnEmptyRange(property, val, l, u));
                }

                frame.stage = Stage::AfterRight;
                let depth = frame.depth;
                stack.push(frame);

                // Descend into right child: range becomes (l, val).
                let old = property_ranges[p];
                property_ranges[p] = (l, val);
                stack.push(Frame {
                    node: right as usize,
                    depth: depth + 1,
                    stage: Stage::Enter,
                    restore: Some((p, old)),
                });
            }
            (Stage::AfterRight, TreeNode::Split { .. }) => {
                if let Some((p, old)) = frame.restore {
                    property_ranges[p] = old;
                }
            }
            _ => unreachable!("invalid tree validation state"),
        }
    }

    Ok(())
}

impl Debug for Tree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tree[{:?}]", self.nodes)
    }
}

#[derive(Debug)]
pub struct PredictionResult {
    pub guess: i64,
    pub multiplier: u32,
    pub context: u32,
}

pub const NUM_NONREF_PROPERTIES: usize = 16;
pub const PROPERTIES_PER_PREVCHAN: usize = 4;

const SPLIT_VAL_CONTEXT: usize = 0;
const PROPERTY_CONTEXT: usize = 1;
const PREDICTOR_CONTEXT: usize = 2;
const OFFSET_CONTEXT: usize = 3;
const MULTIPLIER_LOG_CONTEXT: usize = 4;
const MULTIPLIER_BITS_CONTEXT: usize = 5;
const NUM_TREE_CONTEXTS: usize = 6;

// Note: `property_buffer` is passed as input because this implementation relies on having the
// previous values available for computing the local gradient property.
// Also, the first two properties (the static properties) should be already set by the caller.
// All other properties should be 0 on the first call in a row.

/// Computes all properties needed by the tree, guided by `used_mask`.
/// Bit `i` in `used_mask` means property `i` is referenced by some split node.
/// Properties not in the mask are skipped.
/// Note: property 9 (local gradient) must always be computed if property 8 is used,
/// since property 8 depends on the *previous* value of property 9.
#[inline(always)]
fn compute_properties_common(
    prediction_data: PredictionData,
    references: &Image<i32>,
    property_buffer: &mut [i32],
    x: usize,
    y: usize,
    used_mask: u32,
) {
    let PredictionData {
        left,
        top,
        toptop,
        topleft,
        topright,
        leftleft,
        toprightright: _,
    } = prediction_data;

    // Properties 0,1 (channel, stream) are set once in init, never change.

    // Only compute properties that the tree actually splits on.
    // used_mask is constant per channel, so branches predict perfectly.
    // TESTED: unconditional (all 13 stores) was -7.3% (cache pressure).
    // TESTED: hybrid (7 unconditional + 6 gated) was also worse.
    // Full mask-based approach is best: branch cost < store savings.

    if used_mask & 0x000C != 0 {
        if used_mask & (1 << 2) != 0 {
            property_buffer[2] = y as i32;
        }
        if used_mask & (1 << 3) != 0 {
            property_buffer[3] = x as i32;
        }
    }
    if used_mask & 0x0030 != 0 {
        if used_mask & (1 << 4) != 0 {
            property_buffer[4] = top.wrapping_abs();
        }
        if used_mask & (1 << 5) != 0 {
            property_buffer[5] = left.wrapping_abs();
        }
    }
    if used_mask & 0x00C0 != 0 {
        if used_mask & (1 << 6) != 0 {
            property_buffer[6] = top;
        }
        if used_mask & (1 << 7) != 0 {
            property_buffer[7] = left;
        }
    }
    if used_mask & 0x0300 != 0 {
        property_buffer[8] = left.wrapping_sub(property_buffer[9]);
        property_buffer[9] = left.wrapping_add(top).wrapping_sub(topleft);
    }
    if used_mask & 0x7C00 != 0 {
        if used_mask & (1 << 10) != 0 {
            property_buffer[10] = left.wrapping_sub(topleft);
        }
        if used_mask & (1 << 11) != 0 {
            property_buffer[11] = topleft.wrapping_sub(top);
        }
        if used_mask & (1 << 12) != 0 {
            property_buffer[12] = top.wrapping_sub(topright);
        }
        if used_mask & (1 << 13) != 0 {
            property_buffer[13] = top.wrapping_sub(toptop);
        }
        if used_mask & (1 << 14) != 0 {
            property_buffer[14] = left.wrapping_sub(leftleft);
        }
    }

    // Reference properties - only copy if used (this involves a memcpy).
    if used_mask >> NUM_NONREF_PROPERTIES as u32 != 0 {
        let num_refs = references.size().0;
        if num_refs != 0 {
            let ref_properties = &mut property_buffer[NUM_NONREF_PROPERTIES..];
            ref_properties[..num_refs].copy_from_slice(&references.row(x)[..num_refs]);
        }
    }
}

/// Computes properties without weighted predictor. Returns 0 for wp_pred.
#[inline(always)]
fn compute_properties_no_wp(
    prediction_data: PredictionData,
    references: &Image<i32>,
    property_buffer: &mut [i32],
    x: usize,
    y: usize,
    used_mask: u32,
) {
    compute_properties_common(
        prediction_data,
        references,
        property_buffer,
        x,
        y,
        used_mask,
    );
    if used_mask & (1 << 15) != 0 {
        property_buffer[15] = 0;
    }
}

/// Computes properties with weighted predictor. Returns the WP prediction value.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn compute_properties_with_wp(
    prediction_data: PredictionData,
    xsize: usize,
    wp_state: &mut WeightedPredictorState,
    references: &Image<i32>,
    property_buffer: &mut [i32],
    x: usize,
    y: usize,
    used_mask: u32,
) -> i64 {
    compute_properties_common(
        prediction_data,
        references,
        property_buffer,
        x,
        y,
        used_mask,
    );
    // If property 15 (WP max error) is used by the tree, compute it.
    // Otherwise skip the expensive abs() comparisons.
    if used_mask & (1 << 15) != 0 {
        let (wp_pred, wp_prop) = wp_state.predict_and_property((x, y), xsize, &prediction_data);
        property_buffer[15] = wp_prop;
        wp_pred
    } else {
        wp_state.predict_no_property((x, y), xsize, &prediction_data)
    }
}

/// Interior version: no edge checks for WP predictor. x > 0 and x < xsize-1 guaranteed.
#[inline(always)]
#[allow(clippy::too_many_arguments, unsafe_code)]
fn compute_properties_with_wp_interior(
    prediction_data: PredictionData,
    xsize: usize,
    wp_state: &mut WeightedPredictorState,
    references: &Image<i32>,
    property_buffer: &mut [i32],
    x: usize,
    y: usize,
    used_mask: u32,
) -> i64 {
    compute_properties_common(
        prediction_data,
        references,
        property_buffer,
        x,
        y,
        used_mask,
    );
    if used_mask & (1 << 15) != 0 {
        let (wp_pred, wp_prop) = wp_state.predict_and_property_interior(x, xsize, &prediction_data);
        property_buffer[15] = wp_prop;
        wp_pred
    } else {
        wp_state.predict_no_property_interior(x, xsize, &prediction_data)
    }
}

/// Computes all properties for tree traversal (non-flat path).
/// Returns the weighted predictor prediction value.
#[inline]
fn compute_properties(
    prediction_data: PredictionData,
    xsize: usize,
    wp_state: Option<&mut WeightedPredictorState>,
    x: usize,
    y: usize,
    references: &Image<i32>,
    property_buffer: &mut [i32],
) -> i64 {
    // Non-flat path: compute all properties (cold path, no mask optimization)
    compute_properties_common(prediction_data, references, property_buffer, x, y, u32::MAX);

    // Weighted predictor property.
    let (wp_pred, wp_prop) = wp_state
        .map(|wp_state| wp_state.predict_and_property((x, y), xsize, &prediction_data))
        .unwrap_or((0, 0));
    property_buffer[15] = wp_prop;

    wp_pred
}

/// Prediction using standard tree traversal.
/// Used for small channels where building a flat tree isn't worth it.
#[inline]
#[instrument(level = "trace", ret)]
#[allow(clippy::too_many_arguments)]
pub(super) fn predict(
    tree: &[TreeNode],
    prediction_data: PredictionData,
    xsize: usize,
    wp_state: Option<&mut WeightedPredictorState>,
    x: usize,
    y: usize,
    references: &Image<i32>,
    property_buffer: &mut [i32],
) -> PredictionResult {
    let wp_pred = compute_properties(
        prediction_data,
        xsize,
        wp_state,
        x,
        y,
        references,
        property_buffer,
    );

    trace!(?property_buffer, "new properties");

    let mut tree_node = 0;
    while let TreeNode::Split {
        property,
        val,
        left,
        right,
    } = tree[tree_node]
    {
        if property_buffer[property as usize] > val {
            trace!(
                "left at node {tree_node} [{} > {val}]",
                property_buffer[property as usize]
            );
            tree_node = left as usize;
        } else {
            trace!(
                "right at node {tree_node} [{} <= {val}]",
                property_buffer[property as usize]
            );
            tree_node = right as usize;
        }
    }

    trace!(leaf = ?tree[tree_node]);

    let TreeNode::Leaf {
        predictor,
        offset,
        multiplier,
        id,
    } = tree[tree_node]
    else {
        unreachable!();
    };

    let pred = predictor.predict_one(prediction_data, wp_pred);

    PredictionResult {
        guess: pred + offset as i64,
        multiplier,
        context: id,
    }
}

/// Optimized prediction using flat tree (matches C++ context_predict.h:351-371).
/// Note: prefer predict_flat_no_wp / predict_flat_with_wp for better codegen.
#[inline]
#[allow(clippy::too_many_arguments, unsafe_code, dead_code)]
pub(super) fn predict_flat(
    flat_tree: &[FlatTreeNode],
    prediction_data: PredictionData,
    xsize: usize,
    wp_state: Option<&mut WeightedPredictorState>,
    x: usize,
    y: usize,
    references: &Image<i32>,
    property_buffer: &mut [i32],
) -> PredictionResult {
    let wp_pred = compute_properties(
        prediction_data,
        xsize,
        wp_state,
        x,
        y,
        references,
        property_buffer,
    );

    // Flat tree traversal -- this is the hottest loop in modular decoding.
    // Uses targeted unsafe for bounds-check elimination matching C++ performance.
    // Macro matches C++ TRAVERSE_THE_TREE pattern, unrolled 2x per loop iteration.
    let mut pos = 0;

    macro_rules! traverse {
        ($pos:expr) => {{
            // SAFETY: pos is always in bounds -- the flat tree is built via BFS from a
            // validated tree, and each split node computes pos = child_id + offset (0..3)
            // which always lands within the flat_nodes array.
            let node = unsafe { flat_tree.get_unchecked($pos) };
            if node.property0 < 0 {
                // Leaf node -- predictor is stored directly, no conversion needed
                let pred = node.predictor.predict_one(prediction_data, wp_pred);
                return PredictionResult {
                    guess: pred + node.properties_or_offset[0] as i32 as i64,
                    multiplier: node.splitvals_or_multiplier[0] as u32,
                    context: node.child_id,
                };
            }
            // Split node: C++ logic from context_predict.h:361-365
            // SAFETY: property indices are bounded by max_property_count which is validated
            // against property_buffer.len() when the tree is built and the buffer is allocated.
            let p0 = unsafe {
                *property_buffer.get_unchecked(node.property0 as usize) <= node.splitval0
            };
            // SAFETY: properties_or_offset[0] is validated to be a valid property index.
            let off0 = unsafe {
                (*property_buffer.get_unchecked(node.properties_or_offset[0] as usize)
                    <= node.splitvals_or_multiplier[0]) as u32
            };
            // SAFETY: properties_or_offset[1] is validated to be a valid property index.
            let off1 = unsafe {
                2 | (*property_buffer.get_unchecked(node.properties_or_offset[1] as usize)
                    <= node.splitvals_or_multiplier[1]) as u32
            };
            (node.child_id + if p0 { off1 } else { off0 }) as usize
        }};
    }

    loop {
        pos = traverse!(pos);
        pos = traverse!(pos);
    }
}

/// Specialized predict_flat without weighted predictor (avoids Option overhead).
#[inline(always)]
#[allow(unsafe_code)]
pub(super) fn predict_flat_no_wp(
    flat_tree: &[FlatTreeNode],
    prediction_data: PredictionData,
    x: usize,
    y: usize,
    references: &Image<i32>,
    property_buffer: &mut [i32],
    used_mask: u32,
) -> PredictionResult {
    compute_properties_no_wp(
        prediction_data,
        references,
        property_buffer,
        x,
        y,
        used_mask,
    );

    let mut pos = 0;
    macro_rules! traverse {
        ($pos:expr) => {{
            // SAFETY: pos always points to a valid node in flat_tree by construction.
            let node = unsafe { flat_tree.get_unchecked($pos) };
            if node.property0 < 0 {
                let pred = node.predictor.predict_one(prediction_data, 0);
                return PredictionResult {
                    guess: pred + node.properties_or_offset[0] as i32 as i64,
                    multiplier: node.splitvals_or_multiplier[0] as u32,
                    context: node.child_id,
                };
            }
            // SAFETY: node.property0 is a validated property index for property_buffer.
            let p0 = unsafe {
                *property_buffer.get_unchecked(node.property0 as usize) <= node.splitval0
            };
            // SAFETY: properties_or_offset[0] is a validated property index.
            let off0 = unsafe {
                (*property_buffer.get_unchecked(node.properties_or_offset[0] as usize)
                    <= node.splitvals_or_multiplier[0]) as u32
            };
            // SAFETY: properties_or_offset[1] is a validated property index.
            let off1 = unsafe {
                2 | (*property_buffer.get_unchecked(node.properties_or_offset[1] as usize)
                    <= node.splitvals_or_multiplier[1]) as u32
            };
            (node.child_id + if p0 { off1 } else { off0 }) as usize
        }};
    }
    loop {
        pos = traverse!(pos);
        pos = traverse!(pos);
    }
}

/// Specialized predict_flat with weighted predictor (avoids Option overhead).
#[inline(always)]
#[allow(clippy::too_many_arguments, unsafe_code)]
pub(super) fn predict_flat_with_wp(
    flat_tree: &[FlatTreeNode],
    prediction_data: PredictionData,
    xsize: usize,
    wp_state: &mut WeightedPredictorState,
    x: usize,
    y: usize,
    references: &Image<i32>,
    property_buffer: &mut [i32],
    used_mask: u32,
) -> PredictionResult {
    let wp_pred = compute_properties_with_wp(
        prediction_data,
        xsize,
        wp_state,
        references,
        property_buffer,
        x,
        y,
        used_mask,
    );

    let mut pos = 0;
    macro_rules! traverse {
        ($pos:expr) => {{
            // SAFETY: pos always points to a valid node in flat_tree by construction.
            let node = unsafe { flat_tree.get_unchecked($pos) };
            if node.property0 < 0 {
                let pred = node.predictor.predict_one(prediction_data, wp_pred);
                return PredictionResult {
                    guess: pred + node.properties_or_offset[0] as i32 as i64,
                    multiplier: node.splitvals_or_multiplier[0] as u32,
                    context: node.child_id,
                };
            }
            // SAFETY: node.property0 is a validated property index for property_buffer.
            let p0 = unsafe {
                *property_buffer.get_unchecked(node.property0 as usize) <= node.splitval0
            };
            // SAFETY: properties_or_offset[0] is a validated property index.
            let off0 = unsafe {
                (*property_buffer.get_unchecked(node.properties_or_offset[0] as usize)
                    <= node.splitvals_or_multiplier[0]) as u32
            };
            // SAFETY: properties_or_offset[1] is a validated property index.
            let off1 = unsafe {
                2 | (*property_buffer.get_unchecked(node.properties_or_offset[1] as usize)
                    <= node.splitvals_or_multiplier[1]) as u32
            };
            (node.child_id + if p0 { off1 } else { off0 }) as usize
        }};
    }
    loop {
        pos = traverse!(pos);
        pos = traverse!(pos);
    }
}

/// Interior version of predict_flat_with_wp. No edge checks for WP predictor.
/// x > 0 and x < xsize-1 guaranteed.
#[inline(always)]
#[allow(clippy::too_many_arguments, unsafe_code)]
pub(super) fn predict_flat_with_wp_interior(
    flat_tree: &[FlatTreeNode],
    prediction_data: PredictionData,
    xsize: usize,
    wp_state: &mut WeightedPredictorState,
    x: usize,
    y: usize,
    references: &Image<i32>,
    property_buffer: &mut [i32],
    used_mask: u32,
) -> PredictionResult {
    let wp_pred = compute_properties_with_wp_interior(
        prediction_data,
        xsize,
        wp_state,
        references,
        property_buffer,
        x,
        y,
        used_mask,
    );

    let mut pos = 0;
    macro_rules! traverse {
        ($pos:expr) => {{
            // SAFETY: pos always points to a valid node in flat_tree by construction.
            let node = unsafe { flat_tree.get_unchecked($pos) };
            if node.property0 < 0 {
                let pred = node.predictor.predict_one(prediction_data, wp_pred);
                return PredictionResult {
                    guess: pred + node.properties_or_offset[0] as i32 as i64,
                    multiplier: node.splitvals_or_multiplier[0] as u32,
                    context: node.child_id,
                };
            }
            // SAFETY: node.property0 is a validated property index for property_buffer.
            let p0 = unsafe {
                *property_buffer.get_unchecked(node.property0 as usize) <= node.splitval0
            };
            // SAFETY: properties_or_offset[0] is a validated property index.
            let off0 = unsafe {
                (*property_buffer.get_unchecked(node.properties_or_offset[0] as usize)
                    <= node.splitvals_or_multiplier[0]) as u32
            };
            // SAFETY: properties_or_offset[1] is a validated property index.
            let off1 = unsafe {
                2 | (*property_buffer.get_unchecked(node.properties_or_offset[1] as usize)
                    <= node.splitvals_or_multiplier[1]) as u32
            };
            (node.child_id + if p0 { off1 } else { off0 }) as usize
        }};
    }
    loop {
        pos = traverse!(pos);
        pos = traverse!(pos);
    }
}

impl Tree {
    #[instrument(level = "debug", skip(br), err)]
    pub fn read(br: &mut BitReader, size_limit: usize) -> Result<Tree> {
        assert!(size_limit <= u32::MAX as usize);
        trace!(pos = br.total_bits_read());
        let tree_histograms = Histograms::decode(NUM_TREE_CONTEXTS, br, true)?;
        let mut tree_reader = SymbolReader::new(&tree_histograms, br, None)?;
        // TODO(veluca): consider early-exiting for trees known to be infinite.
        let mut tree: Vec<TreeNode> = vec![];
        let mut to_decode = 1;
        let mut leaf_id = 0;
        let mut max_property = 0;
        while to_decode > 0 {
            if tree.len() > size_limit {
                return Err(Error::TreeTooLarge(tree.len(), size_limit));
            }
            if tree.len() >= tree.capacity() {
                tree.try_reserve(tree.len() * 2 + 1)?;
            }
            to_decode -= 1;
            let property = tree_reader.read_unsigned(&tree_histograms, br, PROPERTY_CONTEXT);
            trace!(property);
            if let Some(property) = property.checked_sub(1) {
                // inner node.
                if property > 255 {
                    return Err(Error::InvalidProperty(property));
                }
                max_property = max_property.max(property);
                let splitval = tree_reader.read_signed(&tree_histograms, br, SPLIT_VAL_CONTEXT);
                let left_child = (tree.len() + to_decode + 1) as u32;
                let node = TreeNode::Split {
                    property: property as u8,
                    val: splitval,
                    left: left_child,
                    right: left_child + 1,
                };
                trace!("split node {:?}", node);
                to_decode += 2;
                tree.push(node);
            } else {
                let predictor = Predictor::try_from(tree_reader.read_unsigned(
                    &tree_histograms,
                    br,
                    PREDICTOR_CONTEXT,
                ))?;
                let offset = tree_reader.read_signed(&tree_histograms, br, OFFSET_CONTEXT);
                let mul_log =
                    tree_reader.read_unsigned(&tree_histograms, br, MULTIPLIER_LOG_CONTEXT);
                if mul_log >= 31 {
                    return Err(Error::TreeMultiplierTooLarge(mul_log, 31));
                }
                let mul_bits =
                    tree_reader.read_unsigned(&tree_histograms, br, MULTIPLIER_BITS_CONTEXT);
                let multiplier = (mul_bits as u64 + 1) << mul_log;
                if multiplier > (u32::MAX as u64) {
                    return Err(Error::TreeMultiplierBitsTooLarge(mul_bits, mul_log));
                }
                let node = TreeNode::Leaf {
                    predictor,
                    offset,
                    id: leaf_id,
                    multiplier: multiplier as u32,
                };
                leaf_id += 1;
                trace!("leaf node {:?}", node);
                tree.push(node);
            }
        }
        tree_reader.check_final_state(&tree_histograms, br)?;

        let num_properties = max_property as usize + 1;
        validate_tree(&tree, num_properties)?;

        let histograms = Histograms::decode(tree.len().div_ceil(2), br, true)?;

        Ok(Tree {
            nodes: tree,
            histograms,
        })
    }

    /// Build flat tree using BFS traversal (matches C++ encoding.cc:81-144).
    /// Each flat node stores parent + both children info to reduce branches.
    /// Returns (flat_nodes, used_properties_mask).
    /// The mask has bit `i` set if property `i` is used in any split node.
    pub(super) fn build_flat_tree(nodes: &[TreeNode]) -> Result<(Vec<FlatTreeNode>, u32)> {
        use std::collections::VecDeque;

        if nodes.is_empty() {
            return Ok((vec![], 0));
        }

        let mut flat_nodes = Vec::new_with_capacity(nodes.len())?;
        let mut used_mask: u32 = 0;
        let mut queue: VecDeque<usize> = VecDeque::new();
        queue.push_back(0); // Start with root

        while let Some(cur_idx) = queue.pop_front() {
            match &nodes[cur_idx] {
                TreeNode::Leaf {
                    predictor,
                    offset,
                    multiplier,
                    id,
                } => {
                    flat_nodes.push(FlatTreeNode::leaf(*predictor, *offset, *multiplier, *id));
                }
                TreeNode::Split {
                    property,
                    val,
                    left,
                    right,
                } => {
                    used_mask |= 1u32 << (*property as u32);
                    // childID points to first of 4 grandchildren in output
                    let child_id = (flat_nodes.len() + queue.len() + 1) as u32;

                    let mut properties = [0i16; 2];
                    let mut splitvals = [0i32; 2];

                    // Process left (i=0) and right (i=1) children
                    for (i, &child_idx) in [*left as usize, *right as usize].iter().enumerate() {
                        match &nodes[child_idx] {
                            TreeNode::Leaf { .. } => {
                                // Child is leaf: set property=0 and enqueue leaf twice
                                properties[i] = 0;
                                splitvals[i] = 0;
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
                                used_mask |= 1u32 << (*cp as u32);
                                properties[i] = *cp as i16;
                                splitvals[i] = *cv;
                                queue.push_back(*cl as usize);
                                queue.push_back(*cr as usize);
                            }
                        }
                    }

                    flat_nodes.push(FlatTreeNode::split(
                        *property as i32,
                        *val,
                        child_id,
                        properties,
                        splitvals,
                    ));
                }
            }
        }

        Ok((flat_nodes, used_mask))
    }

    pub fn max_property_count(&self) -> usize {
        self.nodes
            .iter()
            .map(|x| match x {
                TreeNode::Leaf { .. } => 0,
                TreeNode::Split { property, .. } => *property,
            })
            .max()
            .unwrap_or_default() as usize
            + 1
    }

    pub fn num_prev_channels(&self) -> usize {
        self.max_property_count()
            .saturating_sub(NUM_NONREF_PROPERTIES)
            .div_ceil(PROPERTIES_PER_PREVCHAN)
    }
}
