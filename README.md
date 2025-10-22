# Histopathologic Cancer Detection — Binary Image Classification

This repository contains my work for the **Kaggle Histopathologic Cancer Detection** challenge — a binary image classification task to identify **metastatic cancer** in small histopathology image patches from lymph node tissue.  

The project demonstrates a complete applied deep learning workflow: data exploration, transfer learning model design, training and tuning, evaluation, and final analysis.  
It was completed as part of a graded machine learning mini-project with **125 total points**, emphasizing clarity, reproducibility, and analytical depth rather than leaderboard placement.

---

## 1. Problem Description

**Objective:** Detect the presence of metastatic cancer in histopathology image patches.  
**Type:** Binary image classification (`label = 1` → tumor, `label = 0` → normal).  
**Dataset Source:** [Kaggle Histopathologic Cancer Detection](https://www.kaggle.com/competitions/histopathologic-cancer-detection)

Detecting metastases in pathology slides helps pathologists identify cancer spread more efficiently and accurately, supporting faster and more reliable diagnoses.

---

## 2. Dataset Overview

- **train_labels.csv** — two columns: `id` (image name) and `label` (0 or 1).  
- **train/** — training image tiles (`<id>.tif`) with labels.  
- **test/** — unlabeled images used for leaderboard evaluation.  
- **Image dimensions:** 96×96 RGB patches.  
- **Data volume:** ~220,000 tiles (≈100k positive, ≈120k negative).  
- **Challenge:** class imbalance and significant intra-class variability in tissue texture and color.

---

## 3. Exploratory Data Analysis (EDA)

EDA was used to inspect data structure, label balance, and sample image quality.

### Steps:
1. Verified no missing or duplicate IDs in `train_labels.csv`.
2. Computed class balance showing a slight skew toward normal (label 0).
3. Visualized random grids of tumor and normal patches.
4. Observed high variance in staining intensity and tissue texture.
5. Defined augmentation and normalization strategies.

### Sample Findings
- **Tumor patches** often have darker nuclei regions and denser tissue clusters.  
- **Normal patches** appear lighter and more uniform.  
- Augmentations such as rotations and color jittering are beneficial for generalization.

### Plan of Analysis
1. Apply transfer learning using standard CNN backbones (ResNet18, ResNet34).  
2. Address class imbalance with sampling and loss weighting.  
3. Use image augmentations to increase robustness.  
4. Run stratified K-Fold validation (one fold reported).  
5. Perform lightweight hyperparameter tuning on learning rate and weight decay.  
6. Report validation metrics (AUC, F1-score) and analyze results.

---

## 4. Model Architecture (25 pts)

Two convolutional neural network (CNN) backbones were evaluated using **transfer learning** from ImageNet:

| Architecture | Parameters | Pretrained | Final Head | Notes |
|---------------|-------------|-------------|-------------|--------|
| **ResNet18** | ~11M | Yes | 1×1 output (binary) | Fast, reliable baseline |
| **ResNet34** | ~21M | Yes | 1×1 output (binary) | Deeper, better representation |

**Training details**
- **Loss:** `BCEWithLogitsLoss(pos_weight)` to handle imbalance  
- **Optimizer:** AdamW  
- **Scheduler:** CosineAnnealingLR  
- **Augmentations:** random flips, rotations, color jitter, normalization  
- **Mixed precision:** enabled for GPU acceleration  
- **Validation metric:** ROC AUC  

---

## 5. Results and Analysis (35 pts)

### Hyperparameter Search

A small grid search over learning rates and weight decay values was performed for both ResNet18 and ResNet34.

| Model | LR | Weight Decay | Best Validation AUC |
|--------|----|---------------|---------------------|
| ResNet18 | 3e-4 | 1e-4 | 0.952 |
| ResNet18 | 1e-4 | 3e-5 | 0.947 |
| ResNet34 | 3e-4 | 1e-4 | **0.958** |
| ResNet34 | 1e-4 | 3e-5 | 0.956 |

**Best Model:** ResNet34 with learning rate `3e-4`, weight decay `1e-4`.

### Validation Metrics
- **Best Validation AUC:** 0.958  
- **Best Validation F1-Score:** ~0.82 (optimized threshold ≈ 0.45)  
- **Confusion Matrix Analysis:** Improved recall for tumor patches using tuned threshold.  

### Kaggle Leaderboard
- **Public Score:** **0.9552 AUC**  
- Confirms solid generalization and successful end-to-end pipeline.  

### Insights
- **Weighted sampling** and **loss weighting** helped mitigate imbalance.  
- **ResNet34** achieved slightly higher AUC but required longer training.  
- **Data augmentations** improved generalization by ~0.02 AUC.  
- **CosineAnnealingLR** improved stability vs. fixed learning rate.

---

## 6. Conclusion and Discussion (15 pts)

This project successfully implemented and analyzed a CNN-based approach for cancer metastasis detection in pathology images.

### Key Takeaways
- Transfer learning with moderate architectures yields strong performance even on small tiles.  
- Proper handling of class imbalance is crucial to stable learning.  
- Augmentations meaningfully improve model robustness.  
- Validation AUC ≈ 0.958 and Kaggle AUC = **0.9552** demonstrate high-quality work for a mini-project.

### Future Improvements
- Extend training to all 5 folds and average predictions.  
- Incorporate more architectures (EfficientNet, ConvNeXt, ViT).  
- Explore stain normalization or self-supervised pretraining.  
- Experiment with attention-based pooling for better spatial awareness.

---

## 7. Deliverables Summary

| Deliverable | Description |
|--------------|-------------|
| **1. Jupyter Notebook** | Full workflow: EDA, model training, analysis, results, and conclusion |
| **2. GitHub Repository** | This repo, including all code, README, and setup instructions |
| **3. Kaggle Leaderboard Screenshot** | Image file included (`leaderboard_screenshot.png`) |

---

## 8. Reproducibility Notes

- All random seeds fixed for reproducibility.  
- Controlled configuration via dataclass (`Config`) structure.  
- Compatible with both Kaggle Notebook and local environments.  
- To reproduce locally:
  ```bash
  git clone https://github.com/<your-username>/histopathologic-cancer-detection.git
  cd histopathologic-cancer-detection
  jupyter notebook histopathologic_cancer_detection_baseline.ipynb
  ```
  Ensure the Kaggle dataset is downloaded into `./input/histopathologic-cancer-detection/`.

---

## 9. Repository Structure

```
histopathologic-cancer-detection/
│
├── histopathologic_cancer_detection_baseline.ipynb   # main notebook (EDA + modeling)
├── submission.csv                                    # generated Kaggle submission
├── leaderboard_screenshot.png                        # Kaggle leaderboard proof
├── README.md                                         # project report (this file)
└── input/
    └── histopathologic-cancer-detection/             # Kaggle dataset directory
```

---

## 10. References

- **Kaggle Competition:** [Histopathologic Cancer Detection](https://www.kaggle.com/competitions/histopathologic-cancer-detection)  
- **ResNet Paper:** He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016  
- **EfficientNet Paper:** Tan & Le, *EfficientNet: Rethinking Model Scaling for CNNs*, ICML 2019  
