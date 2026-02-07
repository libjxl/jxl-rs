// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use color_eyre::eyre::{Result, WrapErr};
use jxl::api::JxlMetadataBox;
use std::path::{Path, PathBuf};

/// Decompress a Brotli-compressed metadata box, or return the data as-is if not compressed.
pub fn decompress_metadata(metadata_box: &JxlMetadataBox) -> Result<Vec<u8>> {
    if metadata_box.is_brotli_compressed {
        let mut decompressed = Vec::new();
        let mut reader = brotli::Decompressor::new(std::io::Cursor::new(&metadata_box.data), 4096);
        std::io::Read::read_to_end(&mut reader, &mut decompressed)
            .wrap_err("Failed to decompress Brotli-compressed metadata box")?;
        Ok(decompressed)
    } else {
        Ok(metadata_box.data.clone())
    }
}

/// Generate the output path for the Nth metadata box (0-indexed).
/// If there's only one box, use the base path as-is.
/// For additional boxes (index >= 1), insert a 1-based number before the extension.
pub fn numbered_path(base: &Path, index: usize, total: usize) -> PathBuf {
    if total == 1 {
        return base.to_path_buf();
    }
    let stem = base.file_stem().unwrap_or_default().to_string_lossy();
    let ext = base.extension();
    let name = if index == 0 {
        match ext {
            Some(e) => format!("{}.{}", stem, e.to_string_lossy()),
            None => stem.into_owned(),
        }
    } else {
        let num = index + 1; // 1-based, so second box is _2
        match ext {
            Some(e) => format!("{}_{}.{}", stem, num, e.to_string_lossy()),
            None => format!("{}_{}", stem, num),
        }
    };
    base.with_file_name(name)
}

/// Strip the 4-byte TIFF header offset prefix from EXIF box data.
/// The JXL Exif box prepends a 4-byte big-endian offset before the TIFF data.
/// In practice this is always zero, but we warn if it's not.
fn strip_exif_tiff_offset(data: &[u8]) -> &[u8] {
    if data.len() < 4 {
        return data;
    }
    let offset = u32::from_be_bytes(data[..4].try_into().unwrap());
    if offset != 0 {
        eprintln!(
            "Warning: EXIF box has non-zero TIFF header offset ({offset}), stripping 4-byte prefix anyway"
        );
    }
    &data[4..]
}

/// Save metadata boxes to files at the given base path.
pub fn save_metadata_boxes(
    base_path: Option<&PathBuf>,
    boxes: &Option<Vec<JxlMetadataBox>>,
    is_exif: bool,
) -> Result<()> {
    let Some(base) = base_path else {
        return Ok(());
    };
    let Some(metadata_boxes) = boxes else {
        return Ok(());
    };
    let total = metadata_boxes.len();
    for (i, metadata_box) in metadata_boxes.iter().enumerate() {
        let data = decompress_metadata(metadata_box)?;
        let data = if is_exif {
            strip_exif_tiff_offset(&data).to_vec()
        } else {
            data
        };
        let path = numbered_path(base, i, total);
        std::fs::write(&path, &data)
            .wrap_err_with(|| format!("Failed to write metadata to {:?}", path))?;
    }
    Ok(())
}

/// Print metadata box info for --info output.
pub fn print_metadata_info(label: &str, boxes: &Option<Vec<JxlMetadataBox>>) {
    match boxes {
        Some(b) if !b.is_empty() => {
            let sizes: Vec<String> = b
                .iter()
                .map(|m| {
                    if m.is_brotli_compressed {
                        format!("{} bytes (brotli)", m.data.len())
                    } else {
                        format!("{} bytes", m.data.len())
                    }
                })
                .collect();
            println!("{}: {} box(es): {}", label, b.len(), sizes.join(", "));
        }
        _ => println!("{}: none", label),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numbered_path_single() {
        let base = Path::new("/tmp/metadata.bin");
        assert_eq!(
            numbered_path(base, 0, 1),
            PathBuf::from("/tmp/metadata.bin")
        );
    }

    #[test]
    fn test_numbered_path_multi_first() {
        let base = Path::new("/tmp/metadata.bin");
        assert_eq!(
            numbered_path(base, 0, 3),
            PathBuf::from("/tmp/metadata.bin")
        );
    }

    #[test]
    fn test_numbered_path_multi_second() {
        let base = Path::new("/tmp/metadata.bin");
        assert_eq!(
            numbered_path(base, 1, 3),
            PathBuf::from("/tmp/metadata_2.bin")
        );
    }

    #[test]
    fn test_numbered_path_multi_third() {
        let base = Path::new("/tmp/metadata.bin");
        assert_eq!(
            numbered_path(base, 2, 3),
            PathBuf::from("/tmp/metadata_3.bin")
        );
    }

    #[test]
    fn test_numbered_path_no_extension() {
        let base = Path::new("/tmp/metadata");
        assert_eq!(numbered_path(base, 0, 1), PathBuf::from("/tmp/metadata"));
        assert_eq!(numbered_path(base, 1, 2), PathBuf::from("/tmp/metadata_2"));
    }
}
