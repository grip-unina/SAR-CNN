# SAR-CNN : SAR image despeckling through convolutional neural networks
This is the implementation in Python/Torch of the paper "[SAR image despeckling through convolutional neural networks](https://ieeexplore.ieee.org/document/8128234)".Results differ slightly from those of the paper, obtained using Matlab/MatConvNet.

## Team members
 Giovanni Chierchia (giovanni (dot) chierchia (at) esiee (dot) fr);
 Davide Cozzolino (davide (dot) cozzolino@unina (dot) it);
 Luisa Verdoliva  (verdoliv (at) unina (dot)it);
 Giovanni Poggi   (poggi (at) unina (dot)it).
 
## License
Copyright (c) 2020 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
All rights reserved.
This software should be used, reproduced and modified only for informational and nonprofit purposes.

By downloading and/or using any of these files, you implicitly agree to all the
terms of the license, as specified in the document LICENSE.txt
(included in this package) 

## Prerequisits
All the functions and scripts were tested on Python 3.6, PyTorch 0.4.1 and Cuda 9.2,
the operation is not guaranteed with other configurations.
The command to create the CONDA environment: 
```
conda env create -n env_cnn_nlm -f environment.yml
```

The command to anctivate the CONDA environment:
```
conda activate env_cnn_nlm
```

Please download the datasets using the provided script:
```
bash download_sets.sh
python generate_noisy_synthetics.py
```

## Usage

### Demo
Use `demo_sync.py` to execute a demo.

### Training and Testing
The command to train the network on synthetic data:

```
CUDA_VISIBLE_DEVICES=0 python experiment_sarcnn17.py --exp_name new_train
```

The command to test the network on synthetic data:

```
CUDA_VISIBLE_DEVICES=0 python experiment_sarcnn17.py --eval --eval_epoch 50 --exp_name new_train
```

NOTE: the SSIM of the paper is little different because it was computed using Matlab instead of Python. 
