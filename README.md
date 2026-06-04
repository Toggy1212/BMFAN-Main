# BMFAN

This repository provides the PyTorch implementation of **BMFAN** for lightweight single image super-resolution (SISR).

## Environment

Please install the required packages before running the code.

```bash
pip install torch torchvision numpy imageio scikit-image tqdm
```

If your code uses extra modules such as `einops`, `timm`, or `opencv-python`, install them as needed:

```bash
pip install einops timm opencv-python
```

## Dataset Preparation

Please organize the datasets as follows:

```bash
dataset/
├── DIV2K/
│   ├── DIV2K_train_HR/
│   └── DIV2K_train_LR_bicubic/
│       └── X2/
└── benchmark/
    ├── Set5/
    │   ├── HR/
    │   └── LR_bicubic/
    │       └── X2/
    ├── Set14/
    │   ├── HR/
    │   └── LR_bicubic/
    │       └── X2/
    ├── B100/
    │   ├── HR/
    │   └── LR_bicubic/
    │       └── X2/
    ├── Urban100/
    │   ├── HR/
    │   └── LR_bicubic/
    │       └── X2/
    └── Manga109/
        ├── HR/
        └── LR_bicubic/
            └── X2/
```

For benchmark testing, the dataset path should be:

```bash
../dataset/benchmark/Set5
../dataset/benchmark/Set14
../dataset/benchmark/B100
../dataset/benchmark/Urban100
../dataset/benchmark/Manga109
```

For example, the Set5 dataset should be placed as:

```bash
../dataset/benchmark/Set5/HR
../dataset/benchmark/Set5/LR_bicubic/X2
```

The LR image names should follow the EDSR-style naming format:

```bash
babyx2.png
birdx2.png
butterflyx2.png
headx2.png
womanx2.png
```

## Pre-trained Model

Please place the pretrained model at:

```bash
../experiment/x2.pt
```

Or modify the `--pre_train` path in `demo.sh` according to your own model location.

For example:

```bash
--pre_train ../experiment/x2.pt
```

## Training

To train BMFAN on DIV2K for ×2 super-resolution, run:

```bash
CUDA_VISIBLE_DEVICES=0,1 python main.py \
  --model BMFAN \
  --scale 2 \
  --data_train DIV2K \
  --data_test Set5 \
  --n_feats 48 \
  --n_colors 3 \
  --batch_size 64 \
  --patch_size 64 \
  --epochs 600 \
  --lr 5e-4 \
  --decay 300-450-550-575 \
  --gamma 0.5 \
  --data_range 1-800 \
  --rgb_range 1 \
  --gclip 0.5 \
  --dir_data ../dataset \
  --n_GPUs 2 \
  --save BMFAN_x2_DIV2K_ep600_nf48
```

The trained model and logs will be saved under:

```bash
../experiment/BMFAN_x2_DIV2K_ep600_nf48
```

## Testing

The testing command has been written in `demo.sh`.

Run the following command:

```bash
bash demo.sh
```

If `demo.sh` does not have execution permission, you can also run:

```bash
chmod +x demo.sh
./demo.sh
```

A typical `demo.sh` for ×2 testing is:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --model BMFAN \
  --scale 2 \
  --data_test Set5+Set14+B100+Urban100+Manga109 \
  --n_feats 48 \
  --rgb_range 1 \
  --dir_data ../dataset \
  --pre_train ../experiment/x2.pt \
  --test_only
```

Please make sure there are no spaces after the line-continuation symbol `\`.
