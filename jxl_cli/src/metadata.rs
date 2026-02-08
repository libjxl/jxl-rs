// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use color_eyre::eyre::{Result, WrapErr, eyre};
use jxl::api::JxlMetadataBox;
use jxl::container::{BitstreamKind, ContainerBoxType, ContainerParser, ParseEvent};
use std::io::{BufWriter, Write};
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

/// JXL container signature: the first box is always `JXL ` (12 bytes).
const JXL_CONTAINER_SIGNATURE: &[u8] = b"\x00\x00\x00\x0cJXL \x0d\x0a\x87\x0a";

/// Brotli-compress the given data at maximum quality.
fn brotli_compress(data: &[u8]) -> Result<Vec<u8>> {
    let mut compressed = Vec::new();
    let mut compressor = brotli::CompressorWriter::new(&mut compressed, 4096, 11, 22);
    compressor.write_all(data)?;
    drop(compressor);
    Ok(compressed)
}

/// Write a standard or extended box header.
fn write_box_header(
    out: &mut impl Write,
    box_type: ContainerBoxType,
    content_size: u64,
) -> Result<()> {
    let total_size = 8 + content_size;
    if total_size > u32::MAX as u64 {
        // Extended size header (16 bytes)
        out.write_all(&1u32.to_be_bytes())?;
        out.write_all(&box_type.0)?;
        out.write_all(&(16 + content_size).to_be_bytes())?;
    } else {
        // Standard header (8 bytes)
        out.write_all(&(total_size as u32).to_be_bytes())?;
        out.write_all(&box_type.0)?;
    }
    Ok(())
}

/// Brotli-compress metadata boxes in a JXL container file.
///
/// Reads the input file, parses boxes using `ContainerParser`, compresses `Exif`, `xml `,
/// and `jumb` boxes as `brob` boxes, and writes the result to the output path.
/// Already-compressed `brob` boxes and all other box types are reconstructed as-is.
pub fn compress_metadata_boxes(input_path: &Path, output_path: &Path) -> Result<()> {
    let data = std::fs::read(input_path)?;

    let mut parser = ContainerParser::new();
    let mut output = BufWriter::new(
        std::fs::File::create(output_path)
            .wrap_err_with(|| format!("Failed to create output file {:?}", output_path))?,
    );
    let mut codestream_buf: Vec<u8> = Vec::new();
    let mut summary: Vec<(String, usize, usize)> = Vec::new();
    // Accumulator for auxiliary boxes that arrive in multiple chunks.
    let mut aux_acc: Option<(ContainerBoxType, Vec<u8>)> = None;

    for event in parser.process_bytes(&data) {
        let event = event.map_err(|e| eyre!("Failed to parse JXL container: {:?}", e))?;

        // When we leave codestream events, flush the accumulated codestream as a jxlc box.
        if !matches!(event, ParseEvent::Codestream(_)) && !codestream_buf.is_empty() {
            write_box_header(
                &mut output,
                ContainerBoxType::CODESTREAM,
                codestream_buf.len() as u64,
            )?;
            output.write_all(&codestream_buf)?;
            codestream_buf.clear();
        }

        match event {
            ParseEvent::BitstreamKind(BitstreamKind::Container) => {
                output.write_all(JXL_CONTAINER_SIGNATURE)?;
            }
            ParseEvent::BitstreamKind(_) => {
                return Err(eyre!(
                    "Not a JXL container file (bare codestream?). Cannot compress metadata."
                ));
            }
            ParseEvent::Codestream(chunk) => {
                codestream_buf.extend_from_slice(chunk);
            }
            ParseEvent::AuxiliaryBox {
                box_type,
                data: chunk,
                is_last,
            } => {
                // Accumulate partial box data until we have the complete box.
                let (bt, full_data) = if let Some((acc_type, mut acc_data)) = aux_acc.take() {
                    acc_data.extend_from_slice(chunk);
                    if !is_last {
                        aux_acc = Some((acc_type, acc_data));
                        continue;
                    }
                    (acc_type, acc_data)
                } else if is_last {
                    (box_type, chunk.to_vec())
                } else {
                    aux_acc = Some((box_type, chunk.to_vec()));
                    continue;
                };

                if bt == ContainerBoxType::EXIF
                    || bt == ContainerBoxType::XML
                    || bt == ContainerBoxType::JUMBF
                {
                    let compressed = brotli_compress(&full_data)?;
                    let brob_content_size = 4 + compressed.len() as u64;
                    write_box_header(
                        &mut output,
                        ContainerBoxType::BROTLI_COMPRESSED,
                        brob_content_size,
                    )?;
                    output.write_all(&bt.0)?;
                    output.write_all(&compressed)?;

                    let type_str = String::from_utf8_lossy(&bt.0);
                    summary.push((type_str.into_owned(), full_data.len(), compressed.len()));
                } else {
                    write_box_header(&mut output, bt, full_data.len() as u64)?;
                    output.write_all(&full_data)?;
                }
            }
        }
    }

    // Flush any remaining codestream data.
    if !codestream_buf.is_empty() {
        write_box_header(
            &mut output,
            ContainerBoxType::CODESTREAM,
            codestream_buf.len() as u64,
        )?;
        output.write_all(&codestream_buf)?;
    }

    output.flush()?;

    if !summary.is_empty() {
        println!("Compressed metadata boxes:");
        for (type_name, orig, comp) in &summary {
            println!("  {}: {} bytes -> {} bytes (brotli)", type_name, orig, comp);
        }
    } else {
        println!("No metadata boxes found to compress.");
    }
    println!("Output written to: {}", output_path.display());

    Ok(())
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

    fn get_metadata_test_file(name: &str) -> PathBuf {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        root.parent()
            .unwrap()
            .join("jxl/resources/test/metadata_test_images")
            .join(name)
    }

    #[test]
    fn test_compress_metadata_boxes() {
        let input = get_metadata_test_file("single_exif.jxl");
        if !input.exists() {
            eprintln!("Skipping (metadata test images not found)");
            return;
        }

        let dir = std::env::temp_dir().join("jxl_test_compress_metadata");
        std::fs::create_dir_all(&dir).unwrap();
        let output = dir.join("compressed.jxl");

        compress_metadata_boxes(&input, &output).unwrap();

        // Output should exist and be smaller (compressed metadata)
        let input_size = std::fs::metadata(&input).unwrap().len();
        let output_size = std::fs::metadata(&output).unwrap().len();
        assert!(output_size < input_size);

        // Output should start with JXL container signature
        let output_data = std::fs::read(&output).unwrap();
        assert!(output_data.starts_with(JXL_CONTAINER_SIGNATURE));

        // Output should contain brob boxes, not Exif boxes
        assert!(!contains_box_type(&output_data, b"Exif"));
        assert!(contains_box_type(&output_data, b"brob"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_compress_already_compressed_is_noop() {
        let input = get_metadata_test_file("single_exif_brob.jxl");
        if !input.exists() {
            eprintln!("Skipping (metadata test images not found)");
            return;
        }

        let dir = std::env::temp_dir().join("jxl_test_compress_noop");
        std::fs::create_dir_all(&dir).unwrap();
        let output = dir.join("compressed.jxl");

        compress_metadata_boxes(&input, &output).unwrap();

        // Output should be identical to input (no uncompressed metadata to compress)
        let input_data = std::fs::read(&input).unwrap();
        let output_data = std::fs::read(&output).unwrap();
        assert_eq!(input_data, output_data);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_compress_bare_codestream_errors() {
        let input = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("jxl/resources/test/3x3_srgb_lossless.jxl");
        if !input.exists() {
            eprintln!("Skipping (test images not found)");
            return;
        }

        let dir = std::env::temp_dir().join("jxl_test_compress_bare");
        std::fs::create_dir_all(&dir).unwrap();
        let output = dir.join("output.jxl");

        let result = compress_metadata_boxes(&input, &output);
        assert!(result.is_err());

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Check if raw file data contains a box with the given 4-byte type.
    fn contains_box_type(data: &[u8], box_type: &[u8; 4]) -> bool {
        let mut pos = 0;
        while pos + 8 <= data.len() {
            let size_field = u32::from_be_bytes(data[pos..pos + 4].try_into().unwrap());
            let ty: [u8; 4] = data[pos + 4..pos + 8].try_into().unwrap();
            if &ty == box_type {
                return true;
            }
            let content_start;
            let content_size: u64;
            if size_field == 1 && pos + 16 <= data.len() {
                let total = u64::from_be_bytes(data[pos + 8..pos + 16].try_into().unwrap());
                content_start = pos + 16;
                content_size = total - 16;
            } else if size_field == 0 {
                break;
            } else {
                content_start = pos + 8;
                content_size = (size_field as u64) - 8;
            }
            pos = content_start + content_size as usize;
        }
        false
    }
}
