// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::fmt::Debug;

use super::{predict::WeightedPredictorState, ModularBufferInfo, Predictor};
use crate::{
    bit_reader::BitReader,
    entropy_coding::decode::Histograms,
    error::{Error, Result},
    util::{tracing_wrappers::*, NewWithCapacity},
};

#[allow(dead_code)]
#[derive(Debug)]
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

#[allow(dead_code)]
pub struct Tree {
    pub nodes: Vec<TreeNode>,
    pub histograms: Histograms,
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

impl Tree {
    #[instrument(level = "debug", skip(br), ret, err)]
    pub fn read(br: &mut BitReader, size_limit: usize) -> Result<Tree> {
        assert!(size_limit <= u32::MAX as usize);
        trace!(pos = br.total_bits_read());
        let tree_histograms = Histograms::decode(NUM_TREE_CONTEXTS, br, true)?;
        let mut tree_reader = tree_histograms.make_reader(br)?;
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
            let property = tree_reader.read(br, PROPERTY_CONTEXT)?;
            trace!(property);
            if let Some(property) = property.checked_sub(1) {
                // inner node.
                if property > 255 {
                    return Err(Error::InvalidProperty(property));
                }
                max_property = max_property.max(property);
                let splitval = tree_reader.read_signed(br, SPLIT_VAL_CONTEXT)?;
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
                let predictor = Predictor::try_from(tree_reader.read(br, PREDICTOR_CONTEXT)?)?;
                let offset = tree_reader.read_signed(br, OFFSET_CONTEXT)?;
                let mul_log = tree_reader.read(br, MULTIPLIER_LOG_CONTEXT)?;
                if mul_log >= 31 {
                    return Err(Error::TreeMultiplierTooLarge(mul_log, 31));
                }
                let mul_bits = tree_reader.read(br, MULTIPLIER_BITS_CONTEXT)?;
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
        tree_reader.check_final_state()?;

        let num_properties = max_property as usize + 1;
        let mut property_ranges = Vec::new_with_capacity(num_properties * tree.len())?;
        property_ranges.resize(num_properties * tree.len(), (i32::MIN, i32::MAX));
        let mut height = Vec::new_with_capacity(tree.len())?;
        height.resize(tree.len(), 0);
        for i in 0..tree.len() {
            const HEIGHT_LIMIT: usize = 2048;
            if height[i] > HEIGHT_LIMIT {
                return Err(Error::TreeTooLarge(height[i], HEIGHT_LIMIT));
            }
            if let TreeNode::Split {
                property,
                val,
                left,
                right,
            } = tree[i]
            {
                height[left as usize] = height[i] + 1;
                height[right as usize] = height[i] + 1;
                for p in 0..num_properties {
                    if p == property as usize {
                        let (l, u) = property_ranges[i * num_properties + p];
                        if l > val || u <= val {
                            return Err(Error::TreeSplitOnEmptyRange(p as u8, val, l, u));
                        }
                        trace!("splitting at node {i} on property {p}, range [{l}, {u}] at position {val}");
                        property_ranges[left as usize * num_properties + p] = (val + 1, u);
                        property_ranges[right as usize * num_properties + p] = (l, val);
                    } else {
                        property_ranges[left as usize * num_properties + p] =
                            property_ranges[i * num_properties + p];
                        property_ranges[right as usize * num_properties + p] =
                            property_ranges[i * num_properties + p];
                    }
                }
            } else {
                #[cfg(feature = "tracing")]
                {
                    for p in 0..num_properties {
                        let (l, u) = property_ranges[i * num_properties + p];
                        trace!("final range at node {i} property {p}: [{l}, {u}]");
                    }
                }
            }
        }

        let histograms = Histograms::decode(tree.len().div_ceil(2), br, true)?;

        Ok(Tree {
            nodes: tree,
            histograms,
        })
    }

    pub fn max_property(&self) -> usize {
        self.nodes
            .iter()
            .map(|x| match x {
                TreeNode::Leaf { .. } => 0,
                TreeNode::Split { property, .. } => *property,
            })
            .max()
            .unwrap() as usize
    }

    pub fn num_prev_channels(&self) -> usize {
        self.max_property().saturating_sub(NUM_NONREF_PROPERTIES) / PROPERTIES_PER_PREVCHAN
    }

    // Note: `property_buffer` is passed as input because this implementation relies on having the
    // previous values available for computing the local gradient property.
    // Also, the first two properties (the static properties) should be already set by the caller.
    // All other properties should be 0 on the first call in a row.
    #[allow(clippy::too_many_arguments)]
    #[instrument(level = "trace", skip(buffers), ret)]
    pub(super) fn predict(
        &self,
        buffers: &mut [ModularBufferInfo],
        buffer_indices: &[usize],
        index: usize,
        grid_index: usize,
        wp_state: &mut WeightedPredictorState,
        x: usize,
        y: usize,
        property_buffer: &mut [i32; 256],
    ) -> PredictionResult {
        let get_pixel = |x: usize, y: usize| -> i32 {
            buffers[buffer_indices[index]].buffer_grid[grid_index]
                .data
                .as_ref()
                .unwrap()
                .as_rect()
                .row(y)[x]
        };

        let (w, _) = buffers[buffer_indices[index]].buffer_grid[grid_index].size;

        let left = if x > 0 {
            get_pixel(x - 1, y)
        } else if y > 0 {
            get_pixel(x, y - 1)
        } else {
            0
        };
        let top = if y > 0 { get_pixel(x, y - 1) } else { left };
        let topleft = if x > 0 && y > 0 {
            get_pixel(x - 1, y - 1)
        } else {
            left
        };
        let topright = if x + 1 < w && y > 0 {
            get_pixel(x + 1, y - 1)
        } else {
            top
        };
        let leftleft = if x > 1 { get_pixel(x - 2, y) } else { left };
        let toptop = if y > 1 { get_pixel(x, y - 2) } else { top };
        let toprightright = if x + 2 < w && y > 0 {
            get_pixel(x + 2, y - 1)
        } else {
            topright
        };

        trace!(
            left,
            top,
            topleft,
            topright,
            leftleft,
            toptop,
            toprightright
        );

        // Position
        property_buffer[2] = y as i32;
        property_buffer[3] = x as i32;

        // Neighbours
        property_buffer[4] = top.abs();
        property_buffer[5] = left.abs();
        property_buffer[6] = top;
        property_buffer[7] = left;

        // Local gradient
        property_buffer[8] = left - property_buffer[9];
        property_buffer[9] = left + top - topleft;

        // FFV1 context properties
        property_buffer[10] = left - topleft;
        property_buffer[11] = topleft - top;
        property_buffer[12] = top - topright;
        property_buffer[13] = top - toptop;
        property_buffer[14] = left - leftleft;

        // Weighted predictor property.
        let (wp_pred, property) = wp_state.predict_and_property();
        property_buffer[15] = property;

        // TODO(veluca): reference properties.

        trace!(?property_buffer, "new properties");

        let mut tree_node = 0;
        while let TreeNode::Split {
            property,
            val,
            left,
            right,
        } = self.nodes[tree_node]
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

        trace!(leaf = ?self.nodes[tree_node]);

        let TreeNode::Leaf {
            predictor,
            offset,
            multiplier,
            id,
        } = self.nodes[tree_node]
        else {
            unreachable!();
        };

        let pred = match predictor {
            Predictor::Zero => 0,
            Predictor::West => left as i64,
            Predictor::North => top as i64,
            Predictor::Select => select(left as i64, top as i64, topleft as i64),
            Predictor::Gradient => clamped_gradient(left as i64, top as i64, topleft as i64),
            Predictor::Weighted => wp_pred,
            Predictor::WestWest => leftleft as i64,
            Predictor::NorthEast => topright as i64,
            Predictor::NorthWest => topleft as i64,
            Predictor::AverageWestAndNorth => (top as i64 + left as i64) / 2,
            Predictor::AverageWestAndNorthWest => (left as i64 + topleft as i64) / 2,
            Predictor::AverageNorthAndNorthWest => (top as i64 + topleft as i64) / 2,
            Predictor::AverageNorthAndNorthEast => (top as i64 + topright as i64) / 2,
            Predictor::AverageAll => {
                (6 * top as i64 - 2 * toptop as i64
                    + 7 * left as i64
                    + leftleft as i64
                    + toprightright as i64
                    + 3 * topright as i64
                    + 8)
                    / 16
            }
        };

        PredictionResult {
            guess: pred + offset as i64,
            multiplier,
            context: id,
        }
    }
}

fn select(left: i64, top: i64, topleft: i64) -> i64 {
    let p = left + top - topleft;
    if (p - left).abs() < (p - top).abs() {
        left
    } else {
        top
    }
}

fn clamped_gradient(left: i64, top: i64, topleft: i64) -> i64 {
    // Same code/logic as libjxl.
    let min = left.min(top);
    let max = left.max(top);
    let grad = left + top - topleft;
    let grad_clamp_max = if topleft < min { max } else { grad };
    if topleft > max {
        min
    } else {
        grad_clamp_max
    }
}
