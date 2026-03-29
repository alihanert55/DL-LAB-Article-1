# A Multi-Attention Approach for Person Re-Identification Using Deep Learning

This repository contains the PyTorch implementation of the architecture described in the paper *"A Multi-Attention Approach for Person Re-Identification Using Deep Learning"* published in **Sensors (2023)**. 

📄 **Paper Reference & DOI:** [10.3390/s23073678](https://doi.org/10.3390/s23073678)

## 📌 Overview
Person Re-Identification (Re-ID) is the challenging task of matching individuals across non-overlapping camera views under varying illuminations, poses, and occlusions. This project implements a **Multi-Attention** framework that integrates both spatial and channel attention mechanisms to extract highly robust and discriminative features from human images, making it resilient to misalignment and partial occlusions.

## 🏗️ Model Architecture (`MultiAttentionRe_id`)
The proposed architecture consists of several customized components:
1. **Backbone Network:** A lightweight and powerful **OSNet** (`osnet_ain_x1_0`) initialized with ImageNet weights is utilized to extract base feature maps.
2. **Position Attention Module (PAM):** Captures long-range spatial contextual information. It utilizes Query (Q), Key (K), and Value (V) branches to calculate a spatial attention matrix, refining the feature map based on pixel-to-pixel relationships.
3. **Efficient Channel Attention (ECA):** A lightweight channel attention mechanism that captures local cross-channel interaction without dimensionality reduction (using Global Average Pooling and a 1D Convolution).
4. **Multi-Feature Extraction:** The refined feature map is horizontally partitioned into 6 equal local strips. These local features, along with 1 global feature vector, are flattened and passed through independent classifiers to form a comprehensive concatenated feature representation (fingerprint) for each image.

## 🧮 Loss Functions
The network is optimized end-to-end using a combination of three distinct loss functions to ensure both class separability and intra-class compactness:
* **AMSoftmax Loss (Additive Margin Softmax):** Enforces strict identity classification with an angular margin.
* **Triplet Margin Loss:** A metric learning loss that minimizes the distance between positive pairs while pushing negative pairs apart (Margin = 0.3).
* **Center Loss:** Minimizes the intra-class variance by penalizing the distance between features and their corresponding class centers.

## 💾 Datasets & Augmentation
The model is configured to be trained and evaluated on standard Re-ID benchmark datasets:
* **Market-1501**
* **DukeMTMC-reID**

Images are resized to `384x128`. To prevent overfitting and improve generalization, a robust data augmentation pipeline is applied, including *Random Flip*, *Color Jitter*, and *Random Erasing*.

## ⚙️ Requirements
* Python 3.x
* PyTorch & Torchvision
* [Torchreid (deep-person-reid)](https://github.com/KaiyangZhou/deep-person-reid)
* NumPy, SciPy

## 🚀 Usage

### Installation
To install the required dependencies (specifically the `torchreid` library) in your environment (e.g., Google Colab):
```bash
pip install git+[https://github.com/KaiyangZhou/deep-person-reid.git](https://github.com/KaiyangZhou/deep-person-reid.git)
pip install gdown
