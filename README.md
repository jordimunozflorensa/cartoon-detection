# Cartoon Detection

This repository contains a computer vision project for detecting cartoons in images. The project leverages image processing techniques and machine learning models to classify input images as cartoons or non-cartoons.

## Table of Contents
- [Overview](#overview)
- [Project Files](#project-files)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

---

## Overview

The goal of this project is to build a system that can detect and classify cartoon images. Using various image processing techniques such as HOG, LBP, and color histograms, combined with machine learning models, this project achieves accurate predictions.

### Key Features:
- **Image Processing:** Extracts features like color histograms, HOG (Histogram of Oriented Gradients), and LBP (Local Binary Patterns).
- **Machine Learning:** Uses optimized models for classification.
- **Oversampling:** Balances the dataset using oversampling techniques to improve model performance.

---

## Project Files

The repository contains the following files:

- **`main.m`**: The main script to run the cartoon detection system.
- **`fusio_models.m`**: Combines models for enhanced performance.
- **`histogrames_color_HSV.m`** and **`histogrames_color_RGB.m`**: Scripts for extracting color histograms in HSV and RGB color spaces.
- **`hog.m`**: Extracts HOG features from images.
- **`lbp.m`**: Extracts LBP features for texture analysis.
- **`optimtzacio_models.m`**: Script for optimizing and training machine learning models.
- **`split_train_test.m`**: Splits the dataset into training and testing sets.
- **`sift.m`**: SIFT feature extraction for keypoint detection.
- **`substracció_soroll.m`**: Preprocessing script for noise reduction in images.
- **`treeBaggerModel.mat`**: Pre-trained Tree Bagger model for prediction.
- **`Informe_final.pdf`**: Detailed project report.
- **`LICENSE`**: License information for this repository.
- **`OVERSAM/`**: Directory containing images used for oversampling.
- **`TRAIN/`**: Directory containing raw images

---

## Installation

To set up this project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/username/cartoon-detection.git
   cd cartoon-detection
