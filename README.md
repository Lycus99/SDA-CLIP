# SDA-CLIP: Surgical Visual Domain Adaptation Using Video and Text Labels
Created by Yuchong Li

This repository contains PyTorch implementation for SDA-CLIP.

We introduce a Surgical Domain Adaptation method based on the Contrastive Language-Image Pretraining model (SDA-CLIP) to recognize cross-domain surgical action. 
Specifically, we utilize the Vision Transformer(ViT) and Transformer based on CLIP pre-trained parameters to extract video and text embeddings, respectively. 
Text embedding is developed as a bridge between VR and clinical domains.
Inter- and intra- modality loss functions are employed to enhance the consistency of embeddings of the same class.

Our code is based on [CLIP](https://github.com/openai/CLIP) and [ActionCLIP](https://github.com/sallymmx/ActionCLIP).

## Prerequisites

### Requirements

- [PyTorch](https://pytorch.org/) >= 1.8
- wandb~=0.13.1
- yaml~=0.2.5
- pyyaml~=6.0
- tqdm~=4.64.0
- dotmap~=1.3.30
- pillow~=9.0.1
- torchvision~=0.13.0
- numpy~=1.22.4
- ftfy~=6.1.1
- regex~=2022.3.15
- pandas~=1.4.2
- scikit-learn~=1.0.2
- opencv-python~=4.6.0.66
- setuptools~=61.2.0
- matplotlib~=3.5.1
- seaborn~=0.11.2


The environment is also recorded in *requirements.txt*.

## Pretrained models

We use the base model (ViT-B/16 for image encoder & text encoder) pre-trained by [ActionCLIP](https://github.com/sallymmx/ActionCLIP) based on Kinetics-400. The model can be downloaded in [link](https://drive.google.com/drive/folders/1WhU_9hPcnTd3EwaMQxS1M0TUMmXqUTbj?usp=sharing). The pre-trained model should be saved in ./models/.

## Model weights

Our model weights for the hard and soft domain adaptation tasks can be downloaded in [link](https://drive.google.com/drive/folders/1WhU_9hPcnTd3EwaMQxS1M0TUMmXqUTbj?usp=sharing).