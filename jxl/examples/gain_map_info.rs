// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Example: Display gain map information from a JPEG XL file
//!
//! This example demonstrates how to check if a JPEG XL file contains
//! an ISO 21496-1 gain map and display its metadata.
//!
//! Usage:
//!   cargo run --example gain_map_info -- input.jxl

use jxl::api::{JxlDecoder, JxlDecoderOptions, ProcessingResult};
use jxl::error::Error;
use std::env;
use std::fs;

fn main() -> Result<(), Error> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <input.jxl>", args[0]);
        std::process::exit(1);
    }

    let filename = &args[1];
    let data = fs::read(filename).expect("Failed to read file");

    // Create decoder
    let decoder = JxlDecoder::new(JxlDecoderOptions::default());

    // Process the file to get image info
    let mut input = &data[..];
    let decoder = match decoder.process(&mut input)? {
        ProcessingResult::Complete { result } => result,
        ProcessingResult::NeedsMoreInput { .. } => {
            return Err(Error::OutOfBounds(0));
        }
    };

    // Check for gain map
    println!("=== JPEG XL Gain Map Info ===");
    println!("File: {}", filename);
    println!();

    match decoder.gain_map() {
        Some(gain_map) => {
            println!("✓ Gain map found!");
            println!();
            println!("  Version: {}", gain_map.jhgm_version);
            println!(
                "  Metadata size: {} bytes",
                gain_map.gain_map_metadata.len()
            );

            if !gain_map.alt_icc.is_empty() {
                println!("  ICC profile size: {} bytes", gain_map.alt_icc.len());
            }

            if gain_map.color_encoding.is_some() {
                println!("  Color encoding: present");
            }

            println!(
                "  Gain map codestream size: {} bytes",
                gain_map.gain_map.len()
            );
            println!();

            // Display ISO 21496-1 metadata as hex (first 64 bytes)
            println!(
                "  ISO 21496-1 metadata (first {} bytes):",
                gain_map.gain_map_metadata.len().min(64)
            );
            print!("  ");
            for (i, byte) in gain_map.gain_map_metadata.iter().take(64).enumerate() {
                if i > 0 && i % 16 == 0 {
                    print!("\n  ");
                }
                print!("{:02x} ", byte);
            }
            println!();

            if gain_map.gain_map_metadata.len() > 64 {
                println!(
                    "  ... ({} more bytes)",
                    gain_map.gain_map_metadata.len() - 64
                );
            }
        }
        None => {
            println!("✗ No gain map found in this file");
            println!();
            println!("This file does not contain an ISO 21496-1 gain map (jhgm box).");
            println!("Gain maps allow HDR/SDR tone mapping for images.");
        }
    }

    println!();
    println!("Basic image info:");
    let info = decoder.basic_info();
    println!("  Dimensions: {}x{}", info.size.0, info.size.1);
    println!("  Bit depth: {:?}", info.bit_depth);

    Ok(())
}
