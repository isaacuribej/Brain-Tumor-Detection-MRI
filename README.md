# 🧠 Brain Tumor Detection — MRI Image Classification

> Binary classification of brain MRI scans using a custom Convolutional Neural Network built with PyTorch.

---

## 📌 Overview

This project implements a deep learning pipeline to classify brain MRI images as **Tumor** or **Healthy** using a LeNet-inspired CNN trained from scratch. The full workflow covers exploratory data analysis, MRI-safe data augmentation, model training with train/val split, evaluation, feature map visualization, and a hyperparameter grid search.

Built as a portfolio project to demonstrate practical skills in computer vision, PyTorch, and medical image analysis.

---

## 🗂️ Dataset

**[Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)** — Kaggle

| Split | Healthy | Tumor | Total |
|-------|---------|-------|-------|
| Full dataset | ~98 | ~155 | ~253 |

Images are JPG, resized to **128 × 128** pixels. Labels come from the folder structure (`yes/` = tumor, `no/` = healthy).

### Download the dataset

1. Create a free account at [kaggle.com](https://kaggle.com)
2. Go to the dataset page and click **Download**
3. Unzip and place it following this structure:

```
data/
└── brain_tumor_dataset/
    ├── yes/   ← tumor MRI images
    └── no/    ← healthy MRI images
```

Or use the Kaggle CLI:

```bash
pip install kaggle
kaggle datasets download -d navoneel/brain-mri-images-for-brain-tumor-detection
unzip brain-mri-images-for-brain-tumor-detection.zip -d data/
```

---

## 📊 Results

First training run — **50 epochs**, Adam (`lr=1e-4`), BCELoss, 80/20 stratified split:

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | **81.63%** |
| Macro F1-score | 0.75 |
| Weighted F1-score | 0.76 |

### Classification Report

```
              precision    recall  f1-score   support

     Healthy       0.76      0.72      0.74        18
       Tumor       0.84      0.87      0.86        31

    accuracy                           0.82        49
   macro avg       0.80      0.80      0.80        49
weighted avg       0.81      0.82      0.82        49
```

The model shows stronger recall for the **Tumor** class (87%), which is the clinically more critical class — false negatives (missing a tumor) are more costly than false positives.

---

## 🔬 What We Built

| Component | Details |
|-----------|---------|
| **CNN Architecture** | LeNet-inspired: 2 Conv blocks (Tanh + AvgPool) + 3 FC layers + Sigmoid |
| **Data Augmentation** | MRI-safe transforms: horizontal/vertical flip, ±15° rotation, mild brightness/contrast jitter, Gaussian blur |
| **Training** | Adam optimizer, BCELoss, stratified 80/20 split, 200 epochs |
| **Evaluation** | Accuracy, confusion matrix, classification report, prediction score plot |
| **Visualization** | Sample images, class distribution, pixel intensity histograms, feature maps |
| **Hyperparameter Search** | Grid search over learning rates `[1e-3, 1e-4]`, filters `[6, 12]`, neurons `[84, 120]` |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/brain-tumor-detection.git
cd brain-tumor-detection
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Follow the [dataset download instructions](#download-the-dataset) above and place the images under `data/brain_tumor_dataset/`.

### 5. Open the notebook

```bash
jupyter notebook brain_MRI_tumor_detection.ipynb
```

Run all cells from top to bottom — paths are set via the `DATA_ROOT` variable at the top of the notebook.

---

## 🧰 Tech Stack

- **Python 3.10+**
- **PyTorch** — model, training loop, DataLoader
- **torchvision** — data augmentation transforms
- **OpenCV** — image loading and preprocessing
- **scikit-learn** — train/val split, metrics
- **Matplotlib / Seaborn** — visualization

---

## 📁 Project Structure

```
brain-tumor-detection/
├── data/
│   └── brain_tumor_dataset/
│       ├── yes/
│       └── no/
├── notebooks/
│   └── brain_MRI_tumor_detectionV2.ipynb
├── requirements.txt
└── README.md
```

---

## ⚠️ Disclaimer

This project is for **educational and portfolio purposes only**. It is not intended for clinical use or medical diagnosis.

---

## 📄 License

MIT License — feel free to fork, adapt, and build on this work.
