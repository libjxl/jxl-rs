// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#[cfg(test)]
pub mod test_utils {
    use std::fs::{metadata, File};
    use std::io::{Read, Result};
    use std::path::PathBuf;

    pub fn read_test_file(name: &str) -> Result<Vec<u8>> {
        let mut file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        file_path.push("resources/test/");
        file_path.push(name);
        let mut file = File::open(&file_path)?;
        let file_metadata = metadata(&file_path)?;
        let mut buf = vec![0; file_metadata.len() as usize];
        file.read(&mut buf)?;
        Ok(buf)
    }
}
