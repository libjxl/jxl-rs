#!/bin/bash
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

for hash in $(git log --format='%h' | head -n 100); do
  email=$(git log --format='%ae' "$hash^!")
  name=$(git log --format='%an' "$hash^!")
  "$(dirname $(realpath "$0"))/check_author.py" "${email}" "${name}" || return
done
