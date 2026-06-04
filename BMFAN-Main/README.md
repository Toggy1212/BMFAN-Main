# BMFAN

This repository provides the PyTorch implementation of **BMFAN** for lightweight single image super-resolution (SISR).  
The code is based on the EDSR-style testing framework and supports benchmark evaluation on Set5, Set14, B100, Urban100, and Manga109.

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

The benchmark datasets should be organized in the following format:

```bash
dataset/
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

For example, the Set5 path should be:

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

Please make sure the LR images are not named like:

```bash
baby_x2.png
babyX2.png
baby_LRBI_x2.png
```

## Pre-trained Model

Place the pretrained model at:

```bash
../experiment/x2.pt
```

Or modify the `--pre_train` path in the testing command according to your own file location.

For example, if the pretrained model is stored in the upper-level `experiment` directory, use:

```bash
--pre_train ../experiment/x2.pt
```

## Testing

To test BMFAN on Set5, Set14, B100, Urban100, and Manga109 for ×2 super-resolution, run:

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

If you want to use another GPU, modify `CUDA_VISIBLE_DEVICES`. For example:

```bash
CUDA_VISIBLE_DEVICES=1 python main.py \
  --model BMFAN \
  --scale 2 \
  --data_test Set5+Set14+B100+Urban100+Manga109 \
  --n_feats 48 \
  --rgb_range 1 \
  --dir_data ../dataset \
  --pre_train ../experiment/x2.pt \
  --test_only
```

## Running with demo.sh

You can also save the testing command into `demo.sh`:

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

Then run:

```bash
bash demo.sh
```

Alternatively, you can give the script execution permission:

```bash
chmod +x demo.sh
./demo.sh
```

If the script was edited on Windows and reports a line-ending error such as `/bin/bash^M`, run:

```bash
sed -i 's/\r$//' demo.sh
bash demo.sh
```

Note that there must be no spaces after the line-continuation symbol `\`.  
The following format is correct:

```bash
--model BMFAN \
```

The following format is incorrect:

```bash
--model BMFAN \ 
```

## Important Code Check

For benchmark testing, `Set5` must be included in the benchmark dataset list in `data/__init__.py`.

Please make sure the testing dataset selection contains `Set5`:

```python
if d in ['HR', 'Set5', 'Set14', 'B100', 'Urban100', 'Manga109']:
    m = import_module('data.benchmark')
    testset = getattr(m, 'Benchmark')(args, train=False, name=d)
else:
    module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
    m = import_module('data.' + module_name.lower())
    testset = getattr(m, module_name)(args, train=False, name=d)
```

The benchmark filesystem should be defined as follows in `data/benchmark.py`:

```python
def _set_filesystem(self, dir_data):
    self.apath = os.path.join(dir_data, 'benchmark', self.name)
    self.dir_hr = os.path.join(self.apath, 'HR')
    if self.input_large:
        self.dir_lr = os.path.join(self.apath, 'LR_bicubicL')
    else:
        self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
    self.ext = ('', '.png')
```

With this setting, the code will read Set5 from:

```bash
../dataset/benchmark/Set5
```

instead of:

```bash
../dataset/Set5
```

## Common Problems

### 1. Set5 shows `0it` or `PSNR: nan`

This means Set5 was not correctly loaded.

Please check:

```bash
ls ../dataset/benchmark/Set5/HR
ls ../dataset/benchmark/Set5/LR_bicubic/X2
```

The number of HR and LR images should both be 5:

```bash
find ../dataset/benchmark/Set5/HR -type f | wc -l
find ../dataset/benchmark/Set5/LR_bicubic/X2 -type f | wc -l
```

If an empty folder `../dataset/Set5` was created by mistake, remove it:

```bash
rm -rf ../dataset/Set5
```

Then make sure `Set5` is included in the benchmark list in `data/__init__.py`.

### 2. The pretrained model cannot be found

Check whether the model exists:

```bash
ls ../experiment/x2.pt
```

If the file does not exist, modify `--pre_train` to the correct path.

### 3. The command fails when copied into demo.sh

Check whether there are spaces after `\`:

```bash
cat -n demo.sh
```

The line-continuation symbol `\` must be the last character of the line.

## Expected Result

When the dataset path, pretrained model, and benchmark loader are correct, the Set5 ×2 PSNR should be close to the expected result, for example around:

```bash
Set5 x2 PSNR: 38.222
```

Small differences may occur due to dataset versions, RGB/Y-channel evaluation settings, crop border settings, or image naming differences.

## Citation

If this project is helpful for your research, please cite our paper:

```bibtex
@article{bmfan,
  title={BMFAN: ...},
  author={...},
  journal={...},
  year={2026}
}
```
