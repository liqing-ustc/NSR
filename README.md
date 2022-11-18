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
conda create -y -n nsr python=3.6
source activate nsr
pip install -r requirements.txt
```

5. Run experiments on :
```
python run_sweep.py sweeps/[DATASET].yaml
```

## Usage
The code uses Weights &  Biases for experiment tracking. In the [sweeps](./sweeps/) directory, we provide sweep configurations for all experiments we have performed. The sweeps are officially meant for hyperparameter optimization, but we use them to run multiple configurations and seeds.

To reproduce our results, start a sweep for each of the YAML files in the [sweeps](./sweeps/) directory.


For example,run the experiments on the SCAN dataset: 
```
./run_sweep.py sweeps/scan.yaml --n_agent 3 --gpus=0,1,2
```

You can change the number of sweep agents and GPUs used, according to your computing resource.

More details on how to run W&B sweeps can be found at https://docs.wandb.com/sweeps/quickstart.
