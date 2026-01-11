# Deep Learning 2026: LXMERT Implementation

**Students**: Nathan CLAUDE, Abel AUBRON  
**School**: EPITA  
**Date**: January 2026  
**Supervisor**: Alessio RAGNO  

## Overview

This project implements a simplified version of the **LXMERT** (Learning Cross-Modality Encoder Representations from Transformers) architecture "from scratch" using PyTorch. The model is designed to handle multimodal tasks involving vision and language, specifically focusing on the **Image-Text Matching (ITM)** task using the **Flickr30k** dataset.

The implementation follows the "Two-Stream" architecture paradigm where visual and textual modalities are first encoded independently before being fused via a Cross-Modality Encoder.

## Project Structure

The project relies primarily on two main files:

### 1. `extract_features.py` (Preprocessing)
This script performs offline extraction of visual features from the Flickr30k images to speed up training.

- **Model**: Uses a pre-trained **Faster R-CNN** with a **ResNet-50 FPN** backbone (pretrained on COCO).
- **Process**: 
    1. Detects objects in each image.
    2. Selects the top $K=36$ regions of interest (RoIs) based on confidence scores.
    3. Extracts visual features (1024-dim vectors from RoI Align) and normalized spatial coordinates.
- **Output**: Saves `.pt` tensors in the `flickr30k_features/` directory.

### 2. `LXMERT.ipynb` (Model & Training)
This Jupyter Notebook contains the core logic of the project:

- **Architecture Definition**:
    - **Language Encoder**: Transformers processing text embeddings (WordPiece tokenization).
    - **Object Encoder**: Transformers processing visual features + position embeddings.
    - **Cross-Modality Encoder**: Bidirectional Cross-Attention (Vision-to-Language and Language-to-Vision).
- **Data Loading**: Custom DataLoader that pairs images (features) with captions (positive samples) or mismatched captions (negative samples 50% of the time).
- **Training Loop**: Trains the model on the ITM task using Binary Cross Entropy loss.
- **Inference**: Visualizes attention scores and predictions on test pairs.

## Architecture Simplifications

To fit within student computational constraints (single GPU, limited time), the following simplifications were made compared to the original paper (Tan & Bansal, 2019):

| Component | Original Paper | Our Implementation |
|-----------|---------------|--------------------|
| **Visual Backbone** | ResNet-101 (Visual Genome) | ResNet-50 FPN (COCO) |
| **Hidden Dimension** | 768 | 256 |
| **Layers (Lang/Obj/Cross)** | 9 / 5 / 5 | 2 / 2 / 2 |
| **Training Task** | Masked LM + Masked Obj + VQA | ITM (Matching) only |

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch & Torchvision
- Transformers (Hugging Face)
- Pillow
- NumPy
- Tqdm
- Matplotlib (for visualization)

### Usage

1. **Extract Visual Features**:
   Ensure your Flickr30k images are in the correct directory (e.g., `flickr30k_images/`). Run the extraction script:
   ```bash
   python extract_features.py
   ```
   *Note: This process may take some time depending on your GPU.*

2. **Train the Model**:
   Open `LXMERT.ipynb` in Jupyter or VS Code. Run the cells sequentially to:
   - Load the pre-extracted features.
   - Initialize the simplified LXMERT model.
   - Train for the defined number of epochs (default: 10).
   - Visualize the inference results.

## Results

- **Dynamics**: The model achieves rapid convergence on the training set, stabilizing around a binary cross-entropy loss of 0.3-0.4.
- **Performance**: Despite the simplifications and lack of extensive pre-training (MLM), the model successfully learns to discriminate between matching and non-matching image-caption pairs with high confidence.

## References

- Tan, H., & Bansal, M. (2019). *LXMERT: Learning Cross-Modality Encoder Representations from Transformers*. EMNLP.
- Dataset: [Flickr30k Entities](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset).
