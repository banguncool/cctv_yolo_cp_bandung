# Python & Yolo Ultralytics v11 Settings

## Install Python Conda Env

```bash
conda create --name cp_bandung python=3.10
conda activate cp_bandung
```

## Install Python Libraries

```bash
pip install opencv-python
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -U ultralytics
```

Check if Torch using cuda:

```bash
$ python
$ >>> import torch
$ >>> torch.__verison__
$ >>> torch.cuda.is_available()
$ >>> exit()
```
