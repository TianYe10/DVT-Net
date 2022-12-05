[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


# DVT-Net
This repo covers the full implementation of  ‘DVT-Net’ - [Deep Vascular Topology Network](https://github.com/TianYe10/DVT-Net/), a multimodal deep vascular topology network for disease prediction.

![Hybrid_RGB_VSI_TDA_new](https://user-images.githubusercontent.com/117670714/205598352-355f5a4f-cf25-4c87-b90b-1f58b787d801.png)


## Introduction

The architechture of DVT-Net leverages clinical knowledge pertaining to altered vessel morphology and propose a multimodal pipeline that combines retinal fundus RGB images, vessel segmentation images (VSIs), and Topological Data Analysis (TDA) to achieve accurate and interpretable disease detection. 

We build a customized deep learning model for each of our inputs and concatenate features into a final unified model via late fusion. Specifically, we implemented pretrained swin-transformer-tiny on [ImageNet-1K](https://huggingface.co/datasets/imagenet-1k) as our image feature encoder for each one of the three input modalities. [Swin Transformer](https://arxiv.org/pdf/2103.14030.pdf) (the name Swin stands for Shifted window) is initially described by Microsoft, and the pretrained weights are imported from [TorchVision](https://pytorch.org/vision/main/models/generated/torchvision.models.swin_t.html#torchvision.models.swin_t) models

### For a complete pipeline, please refer to our [DVT-Net](https://github.com/TianYe10/DVT-Net/tree/main/DVT-NET).



## Data Preparation

The second modality of input, vessel segmentation images (VSIs), can be generated from original RGB images using image segmentation methods.
