# You Are Your Own Best Teacher: Achieving Centralized-level Performance in Federated Learning under Heterogeneous and Long-tailed Data

## Abstract
Data heterogeneity, stemming from local non-IID data
and global long-tailed distributions, is a major challenge
in federated learning (FL), leading to significant performance gaps compared to centralized learning. Previous
research found that poor representations and biased classifiers are the main problems and proposed neural-collapseinspired synthetic simplex ETF to help representations be
closer to neural collapse optima. However, we find that the
neural-collapse-inspired methods are not strong enough to
reach neural collapse and still have huge gaps to centralized training. In this paper, we rethink this issue from a selfbootstrap perspective and propose FedYoYo (You Are Your
Own Best Teacher), introducing Augmented Self-bootstrap
Distillation (ASD) to improve representation learning by
distilling knowledge between weakly and strongly augmented local samples, without needing extra datasets or
models. We further introduce Distribution-aware Logit Adjustment (DLA) to balance the self-bootstrap process and
correct biased feature representations. FedYoYo nearly
eliminates the performance gap, achieving centralized-level
performance even under mixed heterogeneity. It enhances
local representation learning, reducing model drift and improving convergence, with feature prototypes closer to neural collapse optimality. Extensive experiments show FedYoYo achieves state-of-the-art results, even surpassing centralized logit adjustment methods by 5.4% under global
long-tailed settings.

## Dependencies

- python 3.7
- pyTorch 1.7.0
- torchvision 0.8.1
- CUDA 11.2
- cuDNN 8.0.4

## Dataset

- CIFAR-10
- CIFAR-100
- SVHN
- ImageNet-LT

# run

```shell
nohup bash run fedyoyo.sh
```

```python
device_id=2
noniid=0.5
imb_factor=0.1
dst='cifar10'
arch="resnet8"
method="fedyoyo"
num_rounds=300
lamda=4.0
gamma=0.1
warmup=50

CUDA_VISIBLE_DEVICES=$device_id python -u main_fedyoyo.py \
    --noniid $noniid \
    --imb_factor $imb_factor \
    --dst $dst \
    --num_rounds $num_rounds \
    --arch $arch \
    --method $method \
    --gamma $gamma \
    --warmup $warmup \
    --lamda $lamda \
# nohup bash run fedyoyo.sh
```

## citation

> @article{yan2025you,  
title={You Are Your Own Best Teacher: Achieving Centralized-level Performance in Federated Learning under Heterogeneous and Long-tailed Data},  
author={Yan, Shanshan and Li, Zexi and Wu, Chao and Pang, Meng and Lu, Yang and Yan, Yan and Wang, Hanzi},  
journal={arXiv preprint arXiv:2503.06916},  
year={2025}
}

