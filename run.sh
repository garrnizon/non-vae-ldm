venv_vis/bin/python visualizations/visualize_svg_features.py --imagenet_path ./data/imagenet-10/  --num_classes 10 --samples_per_class 10

python dataset/download_imagenet100.py
python dataset/generate_small_imagenet.py     --source-path data/imagenet100/     --output-path data/imgnet100     --num-classes 100     --num-train 100   --num-val 5


venv_vis/bin/python visualizations/visualize_svg_features.py --imagenet_path data/imgnet100/train  --num_classes 10 --samples_per_class 100
