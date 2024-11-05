# STN-Retina-Face

### Training STN
Dataset: There are 2 types of dataset: straight LPs & Pseudo LPs in Retina format
```
- StrightLPs
  - image_1.jpg
  - image_2.jpg
  - ...
- Pseudo_LPs
  - image_1.jpg
  - image_1.txt
  - ...
image_1.txt: 4 corners start at top-left in clockwise order
```
Train
```
CUDA_VISIBLE_DEVICES=1 python train_STN.py --root "add you root in here" \
    --batch_size 64 --lr 5e-7 \
    --float16 True
```

### Training Retina-LP
```
train.sh
```
