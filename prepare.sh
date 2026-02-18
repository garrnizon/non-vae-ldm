#! /bin/bash

git clone https://github.com/facebookresearch/dinov3.git
rm -rf dinov3/.git

git clone https://github.com/shiml20/SVG.git
rm -rf SVG/.git

mkdir data
python3 utils/load.py

python3 -m venv venv_vis
venv_vis/bin/pip install -r visualizations/requirements.txt


