# DTIQA: A Dual-Path Transformer Framework for Robust No-Reference Image Quality Assessment
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18842363.svg)](https://doi.org/10.5281/zenodo.18842363)
![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)

> **Notice:** This repository contains the official PyTorch implementation and evaluation code for the manuscript **"DTIQA: A Dual-Path Transformer Framework for Robust No-Reference Image Quality Assessment"**, currently submitted to *The Visual Computer*.

---

## 📖 Introduction

No-Reference Image Quality Assessment (NR-IQA) is a challenging field, particularly when dealing with images that suffer from diverse and complex authentic distortions. **DTIQA** introduces a robust dual-path architecture that explicitly separates semantic content representation from degradation-aware representation. By uniting global context with local distortion sensitivity via our proposed **Global-Local Gated Decomposition (GLGD)** and adaptively fusing them using **Dual-Stream Attention Modulation (DSAM)**, DTIQA achieves state-of-the-art performance across both synthetic and authentically distorted datasets.

## 🌟 Key Algorithms & Contributions

To facilitate replication and further research, we provide clean, explicit implementations of our core modules (found in `models/components.py`):

1. **Global-Local Gated Decomposition (GLGD)**: Dynamically separates the input feature maps into complementary content-aware (semantic) and distortion-aware (fine-grained) representations.
2. **Self-Attention Feature Enrichment**: Enhances feature robustness within both the global and local streams prior to cross-scale interactions.
3. **Dual-Stream Attention Modulation (DSAM)**: Employs a bidirectional cross-scale attention mechanism to seamlessly fuse contextual importance (Top-Down flow) with degradation severity (Bottom-Up flow).
4. **Self-Attention Pooling (SAPool)**: Aggregates the heavily enriched sequence features to robustly predict the final quality score.

## ⚙️ Installation & Dependencies

To set up the environment, simply clone the repository and install the dependencies. *Note: Official `torchvision` weights are utilized for stability.*

```bash
# Clone the repository
git clone https://github.com/algaradi/DTIQA.git
cd DTIQA

# Install required dependencies
pip install -r requirements.txt
```

---

## 📂 Repository Structure

The DTIQA codebase has been meticulously reorganized into a highly modular, professional structure:

- `config/`: Centralized configuration (`config.py`) managing dataset routing and evaluation boundaries.
- `core/`: Core generic network solver (`solver.py`) for streamlined training and testing.
- `datasets/`: DataLoader wrappers and spatial transformations for 6 top-tier benchmark datasets.
- `models/`: The PyTorch architectures backing DTIQA (`dtiqa.py`, `backbone.py`, `components.py`).
- `demos/`: Out-of-the-box demo scripts to run single-image inference and architecture forward-pass checks.
- `evaluation/`: Scripts for specialized per-distortion evaluations.
- `data/`: Information regarding the datasets (see `data/README.md`) and sample images.

---

## 🚀 Quick Start & Usage

### 1. Dataset Preparation & Configuration

Before running any training or inference scripts, you must acquire the datasets:

1. **Download the Datasets**: Download the official datasets using the instructions provided in their specific `download.md` files (e.g., `data/LIVEIQA_release2/download.md`).
2. **Preserve Structure**: Keep the extracted folders and structures exactly as provided by the official sources.
3. **Configure Paths**: Open `config/config.py` and update the absolute paths to match where you extracted each dataset.

For full details, please refer to [**`data/README.md`**](data/README.md).

### 2. Simple Image Inference

You can immediately test DTIQA on a single image without running a full training loop:

```bash
python demos/demo_inference.py --image_path data/samples/sample.png --model_path /path/to/weights.pth --backbone_type vit16
```

### 3. Individual Dataset (Intra-Dataset) Evaluation

Train and test the network natively using a standard 10-fold 80/20 train/test split on a specific benchmark dataset:

```bash
python train_test.py --dataset livec --backbone_type vit16 --model_type direct --train_test_num 10
```

### 4. Cross-Dataset & Cross-Domain Evaluation

To test robustness, you can provide the `--cross_dataset` argument. The framework will gracefully train on 80% of the *source dataset* and evaluate strictly on 100% of the *target dataset*—repeating for the standard 10-fold median.

**Same Domain (e.g., Synthetic -> Synthetic):**

```bash
python train_test.py --dataset csiq --cross_dataset live --backbone_type vit16
```

**Cross Domain (e.g., Synthetic -> Authentic):**

```bash
python train_test.py --dataset live --cross_dataset koniq-10k --backbone_type vit16
```

### 5. Per-Distortion Evaluation

To track performance strictly isolated by specific synthetic distortion typologies (e.g., JPEG, White Noise, Fast Fading):

```bash
python evaluation/evaluate_per_distortion.py --dataset live --backbone_type vit16
```

---

## 📊 Datasets Configurations

Evaluating on local hardware is extremely straightforward. The exact absolute paths for all 6 benchmark datasets are explicitly defined inside `config/config.py`.

Modify the dictionary inside `config/config.py` to point to your local dataset directories. For a full breakdown of the datasets and where to download them, please refer to [**`data/README.md`**](data/README.md).

---

## 📄 Citation

If you use this repository in your research, please cite the associated paper.

The BibTeX entry will be updated upon publication.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

