# Image Classification Experiments: Augmentation, Imbalance, and Transfer Learning

## Overview
This repository contains a series of deep learning experiments investigating three fundamental challenges in computer vision and image classification. The implementations utilize PyTorch and various convolutional neural networks to evaluate different training methodologies, focusing on data augmentation, class imbalance, and transfer learning strategies.

---

## 🛠 Technologies Used
* **Language:** Python
* **Framework:** PyTorch, Torchvision
* **Libraries:** Albumentations (for data augmentation), dlvsp-utils (for dataset manipulation)
* **Environment:** Jupyter Notebook / Kaggle

---

## 📂 Project Structure

### 1. Data Augmentation 
This section explores how data augmentation affects the generalization of a SimpleCNN model.
* **Dataset:** Scenes 15 (indoor and outdoor environments).
* **Techniques:** Evaluates transformations using the Albumentations library, including `HorizontalFlip`, `RandomResizedCrop`, `Rotate`, `ColorJitter`, and `InvertedImage`.
* **MixUp:** Includes a custom `mixup_data` function to blend two images from the same batch using a Beta distribution.
* **Key Findings:** The combination of `HorizontalFlip` and `RandomResizedCrop` yielded the best overall performance. Complex combinations or excessive augmentation ultimately distorted the data and reduced model performance.

### 2. Class Imbalance 
This study addresses how uneven data distribution impacts a SimpleCNN classifier.
* **Dataset:** Scenes 15.
* **Scenarios:** Simulates two main data distributions: a uniform minority setting and a long-tail distribution setting.
* **Technique:** Applies a weighted cross-entropy loss function, assigning larger weights to minority classes to penalize misclassifications heavily.
* **Key Findings:** The weighted loss strategy significantly improves the accuracy of minority classes compared to the baseline. Long-tail distributions proved much harder to handle than uniform imbalance, as the model had to compensate for a continuous range of varying class frequencies.

### 3. Transfer Learning & Fine-Tuning
This project evaluates the adaptation of pre-trained convolutional neural networks to new target domains.
* **Dataset:** CIFAR-10 (full and reduced subsets).
* **Architectures:** Compares multiple ResNet models (ResNet18, ResNet50, and ResNet101).
* **Strategies:** Tests training from scratch, feature extraction (frozen backbone), selective fine-tuning, and full fine-tuning.
* **Key Findings:** Full fine-tuning achieved the best generalization. On a reduced four-class dataset, the lightweight ResNet18 achieved higher accuracy and much faster training times than the heavyweight ResNet101. Adding a deeper, more complex classifier head provided limited benefits compared to a simple linear head.

---
