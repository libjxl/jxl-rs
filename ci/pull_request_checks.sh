#!/bin/bash
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Tests implemented in bash. These typically will run checks about the source
# code rather than the compiled one.

MYDIR=$(dirname $(realpath "$0"))

TEXT_FILES='(Dockerfile.*|\.c|\.cc|\.cpp|\.gni|\.h|\.java|\.sh|\.m|\.py|\.ui|\.yml|\.rs)$'

set -u

# Check for copyright / license markers.
test_copyright() {
  local ret=0
  local f
  for f in $(git ls-files .. | grep -E ${TEXT_FILES}); do
    if [[ "${f#third_party/}" == "$f" ]]; then
      # $f is not in third_party/
      if ! head -n 10 "$f" |
          grep -F 'Copyright (c) the JPEG XL Project Authors.' >/dev/null ; then
        echo "$f: Missing Copyright blob near the top of the file." >&2
        ret=1
      fi
      if ! head -n 10 "$f" |
          grep -F 'Use of this source code is governed by a BSD-style' \
            >/dev/null ; then
        echo "$f: Missing License blob near the top of the file." >&2
        ret=1
      fi
    fi
  done
  return ${ret}
}

test_author() {
  local hash
  for hash in $(git log --format='%h' | head -n 100)
  do
    local email=$(git log --format='%ae' "$hash^!")
    local name=$(git log --format='%an' "$hash^!")
    "${MYDIR}"/check_author.py "${email}" "${name}" || return
  done
}

# Check for git merge conflict markers.
test_merge_conflict() {
  local ret=0
  for f in $(git ls-files .. | grep -E "${TEXT_FILES}"); do
    if grep -E '^<<<<<<< ' "$f"; then
      echo "$f: Found git merge conflict marker. Please resolve." >&2
      ret=1
    fi
  done
  return ${ret}
}

main() {
  local ret=0
  cd "${MYDIR}"

  if ! git rev-parse >/dev/null 2>/dev/null; then
    echo "Not a git checkout, skipping bash_test"
    return 0
  fi

  IFS=$'\n'
  for f in $(declare -F); do
    local test_name=$(echo "$f" | cut -f 3 -d ' ')
    # Runs all the local bash functions that start with "test_".
    if [[ "${test_name}" == test_* ]]; then
      echo "Test ${test_name}: Start"
      if ${test_name}; then
        echo "Test ${test_name}: PASS"
      else
        echo "Test ${test_name}: FAIL"
        ret=1
      fi
    fi
  done
  return ${ret}
}

main "$@"
