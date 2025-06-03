#!/usr/bin/env python3
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import json
import sys


def test_status(test: dict[str, any]) -> str:
    if test['success']:
        return '🟢 Success'
    elif test.get('num_frames', 0) > 0:
        msg = ''
        for frame_no in range(test['num_frames']):
            frame_info = test[f'frame{frame_no}_compare_npy']
            if not frame_info['success']:
                msg = frame_info['message']
        if msg != '':
            return f'🟡 [Failure](## "{msg}")'
        return '🟡 Failure'
    else:
        return f'🔴 [Failure](## "{test["message"]}")'


def main():
    icc_json_path = sys.argv[1]
    png_json_path = sys.argv[2]
    print('\n')
    print('| Name | ICC result | PNG result |')
    print('|------|------------|------------|')
    results = {}
    with open(icc_json_path) as icc_json_input:
        for test in json.load(icc_json_input):
            results[test['test_id']] = dict(icc_success=test_status(test))
    with open(png_json_path) as png_json_input:
        for test in json.load(png_json_input):
            results[test['test_id']]['png_success'] = test_status(test)
    for test_id, values in results.items():
        print(f'| {test_id} | {values["icc_success"]} | {values["png_success"]} |')
                                                                        

if __name__ == '__main__':
    main()
