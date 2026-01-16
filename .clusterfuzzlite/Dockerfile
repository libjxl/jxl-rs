# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

FROM gcr.io/oss-fuzz-base/base-builder-rust:v1

# Copy the project source
COPY . $SRC/jxl-rs
WORKDIR $SRC/jxl-rs

# Copy build script
COPY .clusterfuzzlite/build.sh $SRC/
