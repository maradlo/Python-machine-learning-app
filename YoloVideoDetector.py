"""
Nazov: Online/online

Online -> spracovanie 
online -> pouzitie

2. verzia aplikacie

git clone https://github.com/ultralytics/yolov5

cd yolov5

pip install -qr requirements.txt

import torch
from yolov5 import utils
display = utils.notebook_init()

python train.py --img 640 --batch 3 --epochs 80 --data custom_coco128.yaml --weights yolov5s.pt --cache

python detect.py --weights runs/train/exp3/weights/last.pt --img 640 --conf 0.25 --source ../suciastky_video.mp4
"""

