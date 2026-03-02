# DTIQA Evaluation Protocols

This directory and its associated logic are designed to rigorously test the DTIQA framework across standard, cross-dataset, and cross-domain boundaries.

All evaluations inherently use **10-fold cross-validation** with an 80/20 train/test split. The median SRCC and PLCC over the 10 rounds are automatically calculated and logged natively without the need for manual index recovery.

---

## 📌 1. Individual Dataset (Intra-Dataset) Evaluation

To evaluate the model's performance on a single dataset, simply use the primary `train_test.py` script from the root directory without providing a cross-dataset flag.

By default, the script trains on 80% of the specified dataset and tests on the remaining 20%.

**Example Commands:**

```bash
# Standard 10-fold test on LIVE 
python train_test.py --dataset live --model_type direct --backbone_type vit16

# Standard 10-fold test on CSIQ
python train_test.py --dataset csiq --model_type direct --backbone_type vit16
```

*(Supported datasets: `live`, `csiq`, `tid2013`, `livec`, `koniq-10k`, `bid`)*

---

## 📌 2. Cross-Dataset Evaluation (Same Domain)

To test generalizability across different datasets of the *same domain* (e.g., Synthetic → Synthetic, or Authentic → Authentic), use the `--cross_dataset` argument.

The framework will automatically train on 80% of the *source dataset*, restore the optimal semantic weights, and then robustly test on 100% of the *target dataset*. It performs this seamlessly for 10 rounds to acquire the true median.

**Example Commands (Synthetic to Synthetic):**

```bash
# Train on LIVE, Test on CSIQ
python train_test.py --dataset live --cross_dataset csiq --backbone_type vit16

# Train on CSIQ, Test on LIVE
python train_test.py --dataset csiq --cross_dataset live --backbone_type vit16
```

**Example Commands (Authentic to Authentic):**

```bash
# Train on LIVE Challenge, Test on BID
python train_test.py --dataset livec --cross_dataset bid --backbone_type vit16
```

---

## 📌 3. Cross-Domain Evaluation

Cross-Domain evaluation is executed identically to Cross-Dataset evaluation. You use the exact same script and exact same argument (`--cross_dataset`). The distinction is conceptual (e.g., Synthetic → Authentic).

**Example Commands (Synthetic to Authentic / Authentic to Synthetic):**

```bash
# Train on LIVE (Synthetic), Test on LIVE Challenge (Authentic)
python train_test.py --dataset live --cross_dataset livec --backbone_type vit16

# Train on KonIQ-10k (Authentic), Test on TID2013 (Synthetic)
python train_test.py --dataset koniq-10k --cross_dataset tid2013 --backbone_type vit16
```

---

## 📌 4. Per-Distortion Evaluation

To evaluate precisely how the model handles specific explicit distortion classes (e.g., JPEG, Gaussian Blur, Fast Fading), we provide a dedicated standalone script.

This script tests ONLY the datasets that have explicit categorical distortion breakdowns (currently `LIVE` and `CSIQ`). It iterates through the dataset logic internally and reports isolated matrices for each category natively.

**Example Commands:**

```bash
# Evaluate explicitly on distortion sub-categories within LIVE
python evaluation/evaluate_per_distortion.py --dataset live --backbone_type vit16

# Evaluate explicitly on distortion sub-categories within CSIQ
python evaluation/evaluate_per_distortion.py --dataset csiq --backbone_type vit16
```
