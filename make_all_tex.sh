#!/bin/bash
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.


rm -rf tex/all
mkdir -p tex/all/

cat > tex/all/main.tex << EOF
\documentclass[10pt]{article}
\usepackage{array}
\usepackage[colorlinks,linkcolor=blue]{hyperref}
\usepackage{fullpage}
\usepackage{minted}
\usepackage{booktabs}

\begin{document}
EOF

for i in $(ls tex/*.tex)
do
  echo "\\input{../$(basename $i)}" >> tex/all/main.tex
done

echo "\end{document}" >> tex/all/main.tex

cd tex/all
latexmk -pdf --shell-escape main
