#!/usr/bin/env python3
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import glob
import json
import re
import sys

def main():
    docs_path = sys.argv[1]
    file_parts = [];
    for json_path in glob.glob(f'{docs_path}/dumps/*.json'):
        with open(json_path) as f:
            file_parts.append(f'"{json_path[len(docs_path)+1:]}": {f.read()}')
    file_const = 'const files = {' + ", ".join(file_parts) + '};'
    with open(f'{docs_path}/index.html') as f:
        content = f.read()
        print(re.sub(r'// <REPLACE_WHEN_LOCAL_FILE>.*// </REPLACE_WHEN_LOCAL_FILE>', f'''
{file_const};
return new Promise(resolve => resolve(files[path]));
''', content, flags=re.M | re.DOTALL))
                                                                        

if __name__ == '__main__':
    main()
