# Neural-Symbolic Recursive Machine for Systematic Generalization

<!-- <div align="center">
  <img src="ans.png" width="750px">
</div> -->

## Prerequisites
* Ubuntu 20.04
* Python 3.6
* NVIDIA TITAN Xp
* PyTorch 1.7.0

## Getting started
1. Download the datasets
```
cd datasets
./download.sh
```

2. Download the pre-trained [ResNet-18](https://drive.google.com/file/d/1vDB88m50BMtcnyA8LOD7Uem2Q05nlg9q/view?usp=sharing) to the `data/perception-pretrain/` folder:


3. Download the [DreamCoder](https://drive.google.com/file/d/1HUi69T1LkrH7baxgXzbgE24bvXKgdIJt/view?usp=sharing) and unzip it to the `semantics/` folder.

4. Create an environment and install all packages from `requirements.txt`:
```
conda create -y -n ans python=3.6
source activate ans
pip install -r requirements.txt
```
