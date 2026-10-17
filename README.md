# Thyroid-Nodule-Risk-Classifier
# Thyroid Nodule Classifier

A student research project that predicts whether a thyroid nodule is **benign** or **malignant** from ultrasound images and from basic patient details. It focuses on Bethesda III nodules, where doctors are unsure and many patients get surgery they didn't need. It uses no genetic tests, so it could work in smaller hospitals.

> **This is a pilot. It is not tested on real patients and must not be used for medical decisions.**

## What it does

- **Ultrasound model:** DenseNet-201 sorts ultrasound images into benign or malignant. Accuracy is 86% on 1,001 validation images.
- **Patient-data model:** uses age, sex and Bethesda category from 76 hospital records.
- **Slide-image model:** a small demo on 53 microscope images.

## Results

| Model | Result |
| --- | --- |
| Ultrasound (validation) | 86.0% accuracy, AUC 0.91 |
| Ultrasound (5-fold check) | 82.2% ± 1.3% |
| Patient data, logistic regression | 79.7% accuracy, finds 5 of 9 cancers |
| Patient data, random forest | 85.1% accuracy, finds 3 of 9 cancers |

The slide-image model scored 100% on only 10 images. That number is not reliable.

## Slide-image (histopathology) demo

A second DenseNet-201 model looks at microscope images of thyroid tissue and labels them benign or malignant (papillary and follicular cancer vs. benign). It was trained on 53 small image patches (43 for training, 10 for checking).
```bash
# put images in histopathology_data/staging/benign and /malignant
python histopathology_split_data.py
python histopathology_train.py      # saves models/histopathology_densenet201.pth
```

## Setup

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m venv venv
venv\Scripts\Activate.ps1        # Windows PowerShell
pip install -r requirements.txt
```

## How to use

**Predict on images:** put pictures in `test_images/`, then run

```bash
python inference.py
```

**Train on new ultrasound images:**

```bash
python split_data.py    # splits staging_area/ into train and val
python train.py         # saves models/densenet201_updated.pth
```

**Patient-data model:**

```bash
python tabular_train.py
```

## Folders

```text
models/              saved model weights
staging_area/        raw ultrasound images (benign/, malignant/)
new_training_data/   train/val split made by split_data.py
test_images/         images to predict
```

## Important limits

- The data sets are small and there is no separate test set, so scores are probably too high.
- Only 9 of the 74 patients had cancer, so those results are very uncertain.
- Ultrasound images and patient records are not linked to the same people.


