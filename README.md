# ECG Image Classifier using HOG + Random Forest

![Python](https://img.shields.io/badge/Python-3.x-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Random%20Forest-orange)
![Computer Vision](https://img.shields.io/badge/Feature%20Extraction-HOG-green)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter%20%2F%20Colab-orange)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-94.94%25-brightgreen)

A classical machine-learning pipeline for **four-class ECG image classification** using **Histogram of Oriented Gradients (HOG)** for image feature extraction and a **Random Forest classifier** for multiclass prediction.

The complete workflow is implemented in a reproducible Jupyter/Google Colab notebook and includes dataset loading, image visualization, HOG feature extraction, model training, multiclass evaluation, ROC analysis, bootstrap uncertainty estimation, model export, and inference on new ECG images.

> **Research and educational use only.** This project has not been clinically validated and should not be used for diagnosis, treatment, or patient-management decisions.

---

## Project Objective

The model classifies ECG images into four categories:

1. **Myocardial Infarction**
2. **History of Myocardial Infarction**
3. **Abnormal Heartbeat**
4. **Normal**

The pipeline follows this structure:

```text
ECG Image
    │
    ▼
Grayscale Conversion
    │
    ▼
Resize to 128 × 128
    │
    ▼
Histogram of Oriented Gradients (HOG)
    │
    ▼
8,100-Dimensional Feature Vector
    │
    ▼
Random Forest Classifier
    │
    ▼
Four-Class ECG Prediction
```

---

## Dataset

The notebook uses the Kaggle dataset:

```text
kanishkarathore1604/ecg-image
```

and downloads it with `kagglehub`.

### Dataset distribution

| Class | Training | Test |
|---|---:|---:|
| Myocardial Infarction | 956 | 239 |
| History of MI | 516 | 172 |
| Abnormal Heartbeat | 699 | 233 |
| Normal | 852 | 284 |
| **Total** | **3,023** | **928** |

![Class Distribution](results/class_distribution.png)

### Representative ECG images

![Representative ECG Images](results/representative_ecg_images.png)

The dataset itself is not included in this repository.

---

## Image Preprocessing and HOG Features

Each ECG image is:

- loaded in grayscale,
- resized to **128 × 128 pixels**,
- converted into a HOG feature vector.

### HOG configuration

| Parameter | Value |
|---|---:|
| Orientations | 9 |
| Pixels per cell | 8 × 8 |
| Cells per block | 2 × 2 |
| Block normalization | L2-Hys |
| Final feature dimensions | **8,100** |

HOG represents local gradient orientation and edge structure, making it useful for capturing line morphology in ECG images.

### HOG visualization

![HOG Visualization](results/hog_visualization.png)

---

## Random Forest Model

The classifier was trained using:

```python
RandomForestClassifier(
    n_estimators=1000,
    max_depth=4,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
```

`class_weight="balanced"` adjusts training weights according to class frequencies, while the fixed random seed supports reproducibility.

---

# Experimental Results

The executed notebook produced the following held-out test results.

## Overall performance

| Metric | Result |
|---|---:|
| **Accuracy** | **94.94%** |
| **Balanced Accuracy** | **94.27%** |
| **Macro F1-score** | **0.9460** |
| **Weighted F1-score** | **0.9492** |
| **Macro One-vs-Rest ROC-AUC** | **0.9958** |
| **Image-level bootstrap 95% CI for accuracy** | **93.43% – 96.34%** |

The classifier correctly predicted **881 of 928 test images**.

---

## Per-Class Performance

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| **Myocardial Infarction** | 1.0000 | 1.0000 | 1.0000 | 239 |
| **History of MI** | 0.9259 | 0.8721 | 0.8982 | 172 |
| **Abnormal Heartbeat** | 0.9643 | 0.9270 | 0.9453 | 233 |
| **Normal** | 0.9109 | 0.9718 | 0.9404 | 284 |

![Per-Class Metrics](results/per_class_metrics.png)

The strongest performance was observed for **Myocardial Infarction**, for which all 239 test images were classified correctly in this dataset split.

The **History of MI** class showed the lowest recall among the four classes, indicating that this category accounted for a larger share of the remaining classification errors.

---

## Confusion Matrix

![Confusion Matrix](results/confusion_matrix.png)

The confusion matrix from the executed run was:

| Actual \ Predicted | Myocardial Infarction | History of MI | Abnormal Heartbeat | Normal |
|---|---:|---:|---:|---:|
| **Myocardial Infarction** | **239** | 0 | 0 | 0 |
| **History of MI** | 0 | **150** | 4 | 18 |
| **Abnormal Heartbeat** | 0 | 8 | **216** | 9 |
| **Normal** | 0 | 4 | 4 | **276** |

The main error pattern was between **History of MI** and **Normal**, with 18 History-of-MI images predicted as Normal.

---

## Multiclass ROC Analysis

A one-vs-rest ROC analysis was performed using Random Forest class probabilities.

| Class | One-vs-Rest AUC |
|---|---:|
| Myocardial Infarction | **1.000** |
| History of MI | **0.994** |
| Abnormal Heartbeat | **0.996** |
| Normal | **0.993** |
| **Macro AUC** | **0.9958** |

![ROC Curves](results/roc_curves.png)

The ROC results indicate strong class separability on this test split.

---

## Bootstrap Accuracy Interval

The notebook also estimated uncertainty around test accuracy using **2,000 bootstrap resamples** of the test images.

```text
Observed accuracy: 94.94%
Image-level bootstrap 95% CI: 93.43% – 96.34%
```

This interval is based on resampling individual test images. It should not be interpreted as a patient-level confidence interval when multiple images may originate from the same patient.

---

# Repository Structure

```text
ECG-Image-Classifier-RF/
│
├── README.md
├── ECG_Image_Classifier_RF.ipynb
│
└── results/
    ├── metrics.json
    ├── class_distribution.png
    ├── representative_ecg_images.png
    ├── hog_visualization.png
    ├── confusion_matrix.png
    ├── per_class_metrics.png
    └── roc_curves.png
```

The trained notebook can additionally create:

```text
rf_ecg_model.joblib
```

for local model reuse and inference.

---

# Running the Project

## Google Colab

Open:

```text
ECG_Image_Classifier_RF.ipynb
```

and run the notebook from top to bottom.

The notebook installs the required packages and downloads the dataset through KaggleHub.

---

## Local Jupyter Environment

Install the main dependencies:

```bash
pip install kagglehub opencv-python scikit-image scikit-learn matplotlib seaborn pandas joblib tqdm jupyter
```

Launch Jupyter:

```bash
jupyter notebook ECG_Image_Classifier_RF.ipynb
```

---

# Model Inference

After training, the model bundle is saved as:

```text
rf_ecg_model.joblib
```

A new ECG image can then be evaluated using:

```python
predict_ecg_image("path/to/ecg_image.png")
```

The inference function reports:

- the predicted class,
- class probability scores,
- and the input ECG image.

Random Forest probability outputs should be interpreted as model scores rather than calibrated clinical probabilities.

---

# Machine-Readable Results

The repository includes:

```text
results/metrics.json
```

containing the main executed-run configuration and evaluation results, including:

- train/test sample counts,
- HOG configuration,
- Random Forest configuration,
- accuracy,
- balanced accuracy,
- F1 scores,
- multiclass ROC-AUC,
- bootstrap confidence interval,
- per-class metrics,
- confusion matrix.

---

# Technical Stack

- **Python**
- **scikit-learn**
- **scikit-image**
- **OpenCV**
- **NumPy**
- **pandas**
- **Matplotlib**
- **Seaborn**
- **KaggleHub**
- **Joblib**
- **Jupyter Notebook / Google Colab**

---

# Limitations

The reported results should be interpreted within the scope of this dataset and experimental setup.

### ECG images rather than raw signals

The model operates on rendered ECG images rather than the original ECG time-series signals. Image-level characteristics such as gridlines, plotting layout, text, resolution, compression, or export format may therefore influence predictions.

### Dataset-provided split

Evaluation uses the dataset's existing train and test directories.

Patient identifiers are not available in the implemented workflow, so **patient-level independence between training and test images cannot be independently verified from this experiment**.

### External validation

No independent external clinical dataset was used.

Performance on ECGs from different hospitals, devices, acquisition systems, or plotting formats may differ.

### Bootstrap interval

The reported confidence interval resamples test images rather than patients. If multiple test images are correlated because they originate from the same patient, the effective uncertainty may be larger than the image-level interval suggests.

### Probability calibration

The Random Forest probability outputs were used for ROC analysis but were not calibrated for clinical probability interpretation.

### Clinical use

The model has not undergone prospective clinical validation, regulatory review, or deployment testing.

---

# Future Work

Possible extensions include:

- verify **patient-level train/test separation**,
- perform grouped or patient-level cross-validation,
- evaluate the classifier on an **independent external ECG dataset**,
- compare Random Forest with **SVM, Logistic Regression, XGBoost, and CNN baselines**,
- analyze HOG and Random Forest feature importance,
- test robustness to image scaling, rotation, compression, noise, and grid/background changes,
- isolate waveform regions to reduce dependence on non-waveform image features,
- calibrate predicted probabilities,
- calculate patient-level confidence intervals where patient identifiers are available,
- compare image-based classification with models trained on **raw ECG waveforms**.

---

# Key Results at a Glance

```text
Training images       : 3,023
Test images           : 928
HOG features/image    : 8,100

Accuracy              : 94.94%
Balanced accuracy     : 94.27%
Macro F1              : 0.9460
Weighted F1           : 0.9492
Macro ROC-AUC         : 0.9958
Bootstrap 95% CI      : 93.43% – 96.34%
```

---

# Author

**Saadalishah107**

GitHub: [github.com/Saadalishah107](https://github.com/Saadalishah107)

Repository: [ECG-Image-Classifier-RF](https://github.com/Saadalishah107/ECG-Image-Classifier-RF)

---

# Disclaimer

This project is intended for **research, learning, and portfolio demonstration only**. It is not a substitute for professional medical evaluation and must not be used for diagnosis, treatment decisions, or patient management.
