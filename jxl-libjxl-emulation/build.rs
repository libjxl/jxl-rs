// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::path::{Path, PathBuf};
use std::{env, fs};

use bindgen::callbacks::{IntKind, ParseCallbacks};

const WRAPPER_H: &str = "
#include <jxl/cms_interface.h>
#include <jxl/codestream_header.h>
#include <jxl/color_encoding.h>
#include <jxl/compressed_icc.h>
#include <jxl/decode.h>
#include <jxl/encode.h>
#include <jxl/gain_map.h>
#include <jxl/memory_manager.h>
#include <jxl/parallel_runner.h>
#include <jxl/stats.h>
#include <jxl/types.h>
#include <jxl/version.h>
";

#[derive(Debug)]
struct JxlCallbacks;

impl ParseCallbacks for JxlCallbacks {
    fn include_file(&self, filename: &str) {
        println!("cargo::rerun-if-changed={filename}");
    }

    fn int_macro(&self, name: &str, _value: i64) -> Option<IntKind> {
        match name {
            "JXL_TRUE"
            | "JXL_FALSE"
            | "JXL_PARALLEL_RET_SUCCESS"
            | "JXL_PARALLEL_RET_RUNNER_ERROR" => Some(IntKind::Int),
            "JPEGXL_MAJOR_VERSION"
            | "JPEGXL_MINOR_VERSION"
            | "JPEGXL_PATCH_VERSION"
            | "JPEGXL_NUMERIC_VERSION" => Some(IntKind::U32),
            _ => None,
        }
    }
}

fn parse_cmake_version(cmake_text: &str, var: &str) -> Option<String> {
    let prefix = format!("set({var} ");
    cmake_text.lines().find_map(|line| {
        let rest = line.trim().strip_prefix(&prefix)?;
        Some(rest.trim_end_matches(')').trim().to_string())
    })
}

fn write_generated_headers(libjxl_dir: &Path, out_include_jxl: &Path) {
    fs::create_dir_all(out_include_jxl).expect("Failed to create OUT_DIR/include/jxl");

    fs::write(
        out_include_jxl.join("jxl_export.h"),
        "#ifndef JXL_EXPORT_H_\n#define JXL_EXPORT_H_\n#define JXL_EXPORT\n#define JXL_NO_EXPORT\n#define JXL_DEPRECATED\n#endif\n",
    )
    .expect("Failed to write jxl_export.h");

    fs::write(
        out_include_jxl.join("jxl_cms_export.h"),
        "#ifndef JXL_CMS_EXPORT_H_\n#define JXL_CMS_EXPORT_H_\n#define JXL_CMS_EXPORT\n#define JXL_CMS_NO_EXPORT\n#endif\n",
    )
    .expect("Failed to write jxl_cms_export.h");

    fs::write(
        out_include_jxl.join("jxl_threads_export.h"),
        "#ifndef JXL_THREADS_EXPORT_H_\n#define JXL_THREADS_EXPORT_H_\n#define JXL_THREADS_EXPORT\n#define JXL_THREADS_NO_EXPORT\n#endif\n",
    )
    .expect("Failed to write jxl_threads_export.h");

    let cmake_path = libjxl_dir.join("lib/CMakeLists.txt");
    let version_in_path = libjxl_dir.join("lib/jxl/version.h.in");
    println!("cargo::rerun-if-changed={}", cmake_path.display());
    println!("cargo::rerun-if-changed={}", version_in_path.display());

    let cmake_text = fs::read_to_string(&cmake_path).expect("Failed to read lib/CMakeLists.txt");
    let major =
        parse_cmake_version(&cmake_text, "JPEGXL_MAJOR_VERSION").unwrap_or_else(|| "0".into());
    let minor =
        parse_cmake_version(&cmake_text, "JPEGXL_MINOR_VERSION").unwrap_or_else(|| "13".into());
    let patch =
        parse_cmake_version(&cmake_text, "JPEGXL_PATCH_VERSION").unwrap_or_else(|| "0".into());

    let version_h = fs::read_to_string(&version_in_path)
        .expect("Failed to read lib/jxl/version.h.in")
        .replace("@JPEGXL_MAJOR_VERSION@", &major)
        .replace("@JPEGXL_MINOR_VERSION@", &minor)
        .replace("@JPEGXL_PATCH_VERSION@", &patch);

    fs::write(out_include_jxl.join("version.h"), version_h).expect("Failed to write version.h");
}

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let libjxl_dir = manifest_dir.join("../third_party/libjxl");
    let libjxl_include = libjxl_dir.join("lib/include");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let out_include = out_dir.join("include");
    write_generated_headers(&libjxl_dir, &out_include.join("jxl"));

    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();

    let mut builder = bindgen::Builder::default()
        .header_contents("wrapper.h", WRAPPER_H)
        .clang_arg(format!("-I{}", out_include.display()))
        .clang_arg(format!("-I{}", libjxl_include.display()))
        .parse_callbacks(Box::new(JxlCallbacks))
        .ignore_functions()
        .default_enum_style(bindgen::EnumVariation::NewType {
            is_bitfield: false,
            is_global: true,
        })
        .prepend_enum_name(false)
        .derive_default(true)
        .derive_eq(true)
        .derive_partialeq(true)
        .derive_hash(true)
        .allowlist_type("Jxl.*")
        .allowlist_type("jpegxl_.*")
        .allowlist_var("JXL_.*")
        .allowlist_var("JPEGXL_.*")
        .allowlist_var("kFrames|kDC|kLastPasses|kPasses|kDCProgressive|kDCGroups|kGroups")
        .blocklist_type("JxlDecoder")
        .blocklist_type("JxlDecoderStruct");

    if target_arch == "wasm32" {
        builder = builder.clang_arg("-ffreestanding");
    }

    let bindings = builder
        .generate()
        .expect("Failed to generate libjxl C API bindings");

    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("Failed to write bindings.rs");
}
