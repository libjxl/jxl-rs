// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use super::Predictor;
use crate::{
    bit_reader::BitReader,
    entropy_coding::decode::Histograms,
    error::{Error, Result},
    util::tracing_wrappers::*,
};

#[allow(dead_code)]
#[derive(Debug)]
enum TreeNode {
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
#[derive(Debug)]
pub struct Tree {
    nodes: Vec<TreeNode>,
    histograms: Histograms,
}

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
        let mut property_ranges = vec![];
        property_ranges.try_reserve(num_properties * tree.len())?;
        property_ranges.resize(num_properties * tree.len(), (i32::MIN, i32::MAX));
        let mut height = vec![];
        height.try_reserve(tree.len())?;
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

        let histograms = Histograms::decode((tree.len() + 1) / 2, br, true)?;

        Ok(Tree {
            nodes: tree,
            histograms,
        })
    }
}
