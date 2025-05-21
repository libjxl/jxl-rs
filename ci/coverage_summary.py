#!/usr/bin/env python3
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import json


def format_cov(entry):
    cov, count = entry['covered'], entry['count']
    bad = cov * 100 < count * 75
    good = cov * 100 >= count * 90
    mark = '🔴' if bad else '🟢' if good else '🟡'
    return f'{cov} / {count} {mark}'


def main():
    with open('coverage.json', 'r') as file:
        raw_json = file.read()
    coverage = json.loads(raw_json)
    print('| File | Function | Line |')
    print('| :--- | ---: | ---: |')
    for entry in coverage['data'][0]['files']:
        path = entry['filename'].rsplit('/src/')[1]
        summary = entry['summary']
        fn = summary['functions']
        ln = summary['lines']
        print(f'| {path} | {format_cov(fn)} | {format_cov(ln)} |')


if __name__ == '__main__':
    main()
