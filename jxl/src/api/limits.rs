// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Security limits and cancellation support for JXL decoding.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Configurable security limits for the JXL decoder.
///
/// These limits help protect against denial-of-service attacks from malicious
/// images that attempt to exhaust system resources.
///
/// # Backwards Compatibility
///
/// By default, all limits are set to `None` (unlimited), which preserves the
/// existing behavior. Use [`JxlDecoderLimits::default_safe()`] for recommended
/// security limits, or [`JxlDecoderLimits::restrictive()`] for untrusted content.
///
/// # Example
///
/// ```
/// use jxl::api::JxlDecoderLimits;
///
/// // Use default (unlimited) for trusted content
/// let limits = JxlDecoderLimits::default();
///
/// // Use safe defaults for general use
/// let limits = JxlDecoderLimits::default_safe();
///
/// // Use restrictive limits for untrusted web content
/// let limits = JxlDecoderLimits::restrictive();
/// ```
#[derive(Clone, Debug)]
pub struct JxlDecoderLimits {
    /// Maximum total pixels allowed (width * height).
    /// Default: `None` (unlimited).
    /// Recommended safe: `1 << 30` (about 1 billion pixels).
    pub max_pixels: Option<usize>,

    /// Maximum number of extra channels allowed.
    /// Default: `None` (unlimited).
    /// Recommended safe: `256`.
    pub max_extra_channels: Option<usize>,

    /// Maximum ICC profile size in bytes.
    /// Default: `None` (unlimited).
    /// Recommended safe: `1 << 28` (256 MB).
    pub max_icc_size: Option<usize>,

    /// Maximum modular tree size.
    /// Default: `None` (unlimited).
    /// Recommended safe: `1 << 22`.
    pub max_tree_size: Option<usize>,

    /// Maximum number of patches allowed.
    /// Default: `None` (unlimited).
    pub max_patches: Option<usize>,

    /// Maximum number of spline control points.
    /// Default: `None` (unlimited).
    /// Recommended safe: `1 << 20`.
    pub max_spline_points: Option<u32>,

    /// Maximum number of reference frames that can be stored.
    /// Default: `None` (unlimited).
    /// Recommended safe: `4`.
    pub max_reference_frames: Option<usize>,

    /// Maximum memory budget in bytes for decode operations.
    /// Default: `None` (unlimited).
    /// When set, allocations will fail if they would exceed this budget.
    pub max_memory_bytes: Option<u64>,
}

impl Default for JxlDecoderLimits {
    /// Returns limits with all values set to `None` (unlimited).
    ///
    /// This preserves backwards compatibility - decoding will work exactly
    /// as before without any new restrictions.
    fn default() -> Self {
        Self {
            max_pixels: None,
            max_extra_channels: None,
            max_icc_size: None,
            max_tree_size: None,
            max_patches: None,
            max_spline_points: None,
            max_reference_frames: None,
            max_memory_bytes: None,
        }
    }
}

impl JxlDecoderLimits {
    /// Returns limits with recommended safe defaults.
    ///
    /// These limits are suitable for general use and protect against
    /// most resource exhaustion attacks while allowing legitimate images.
    pub fn default_safe() -> Self {
        Self {
            max_pixels: Some(1 << 30), // ~1 billion pixels
            max_extra_channels: Some(256),
            max_icc_size: Some(1 << 28), // 256 MB
            max_tree_size: Some(1 << 22),
            max_patches: None, // No default limit
            max_spline_points: Some(1 << 20),
            max_reference_frames: Some(4),
            max_memory_bytes: None, // No memory tracking by default
        }
    }

    /// Returns restrictive limits suitable for untrusted content.
    ///
    /// These limits are appropriate for web browsers or other contexts
    /// where images may come from untrusted sources.
    pub fn restrictive() -> Self {
        Self {
            max_pixels: Some(100_000_000), // 100 megapixels
            max_extra_channels: Some(16),
            max_icc_size: Some(1 << 20), // 1 MB
            max_tree_size: Some(1 << 20),
            max_patches: Some(1 << 16), // 64K patches
            max_spline_points: Some(1 << 18),
            max_reference_frames: Some(4),
            max_memory_bytes: Some(1 << 30), // 1 GB memory budget
        }
    }

    /// Returns limits with all restrictions disabled.
    ///
    /// Use only for trusted content where resource limits are not a concern.
    pub fn unlimited() -> Self {
        Self::default()
    }
}

/// A thread-safe token for cooperative cancellation of decode operations.
///
/// The decoder will check this token at various points during decoding and
/// return an error if cancellation has been requested.
///
/// # Example
///
/// ```
/// use jxl::api::CancellationToken;
/// use std::thread;
///
/// let token = CancellationToken::new();
/// let token_clone = token.clone();
///
/// // In another thread or async context:
/// // token_clone.cancel();
///
/// // The decoder will check token.is_cancelled() and abort if true
/// ```
#[derive(Clone, Debug, Default)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    /// Creates a new cancellation token.
    pub fn new() -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Signals that the operation should be cancelled.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    /// Returns `true` if cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    /// Resets the token to allow reuse.
    pub fn reset(&self) {
        self.cancelled.store(false, Ordering::Release);
    }
}
