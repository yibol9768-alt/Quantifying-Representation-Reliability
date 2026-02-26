# 2024 UAI - Representation Reliability

[![License: MIT](https://img.shields.io/badge/License-MIT-g.svg)](https://opensource.org/licenses/MIT)

Code for the paper "Quantifying Representation Reliability in Self-Supervised Learning Models"

Paper: https://arxiv.org/abs/2306.00206  
Authors: [Young-Jin Park](https://young-j-park.github.io/) $^1$, [Hao Wang](https://haowang94.github.io/) $^2$, [Shervin Ardeshir](https://sites.google.com/view/shervinardeshir/home) , [Navid Azizan](https://azizan.mit.edu/) $^1$  
$^1$ Massachusetts Institute of Technology, $^2$ MIT-IBM Watson AI Lab

## Summary

![NC](nc.jpg)

- We introduce a formal definition of __representation reliability (Reli)__: the representation for a given test point is considered to be reliable if the downstream models built on top of that representation can consistently generate accurate predictions for that test point. However, accessing downstream data to quantify the representation reliability is often infeasible or restricted due to privacy concerns.
- We propose an ensemble-based method for estimating the representation reliability without knowing the downstream tasks a priori. Our method is based on the concept of __neighborhood consistency (NC)__ across distinct pre-trained representation spaces. The key insight is to find shared neighboring points as anchors to align these representation spaces before comparing them.
- We demonstrate through comprehensive numerical experiments that our method effectively captures the representation reliability with a high degree of correlation, achieving robust and favorable performance compared with baseline methods.

## Install

```
conda create -n repreli python=3.9
conda activate repreli

pip install PyYAML>=6.0.1
pip install torch>=2.3.0
pip install numpy>=1.26.4
pip install pandas>=2.2.2
pip install scikit-learn>=1.4.2
pip install numba>=0.59.1
```

## Preparing Representations and YAML config

To conduct the following experiments, you will need to first generate embeddings from pre-trained models such as SimCLR, MoCo, and BYOL. Please ensure that the output is saved in a pickle file, which should contain a dictionary structured as follows:

```python
{
  "emb": np.ndarray,  # Array shape: (N, D)
  "label": np.ndarray  # Array shape: (N,)
}
```

Additionally, you are required to create a YAML configuration file in the `./configs` directory. Two example configuration files have been provided in this repository for your reference.

## Experiments

#### Ex 1) In-Distribution Setting
```python
python main.py --verbose --output_dir ./results --pretrain cifar100 --downstream cifar100 --seed 0
```

#### Ex 2) Transfer Learning Setting
```python
python main.py --verbose --output_dir ./results --pretrain imagenet32 --downstream cifar100 --seed 0
```

## Evaluation

Check out the [Jupyter notebook!](https://github.com/young-j-park/repreli/blob/main/Parse%20Results.ipynb)
