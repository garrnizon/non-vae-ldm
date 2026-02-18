#! /bin/bash

git clone https://github.com/facebookresearch/dinov3.git
rm -rf dinov3/.git

git clone https://github.com/shiml20/SVG.git
rm -rf SVG/.git

python utils/load.py
