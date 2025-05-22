// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::error::Error;
use std::{fs, path::Path};

/// Convenience function which does what std::fs::write does, but also
/// creates the full directory path if it does not exist.
pub fn write_output_file(output_filename: &Path, output_bytes: &[u8]) -> Result<(), Error> {
    let parent = output_filename.parent();

    // If there's a parent directory, create it and then write the file.
    // Otherwise, just write the file (it implies the file is in the current directory).
    parent
        .map_or(Ok(()), fs::create_dir_all)
        .and_then(|_| fs::write(output_filename, output_bytes))
        .map_err(|_| Error::OutputWriteFailure)
}
