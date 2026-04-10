# Task 2: Deep Learning Project — Sentiment Analysis (NLP)

## Overview
This project implements a **Deep Learning model (LSTM)** for **Natural Language Processing (NLP)** — specifically **Sentiment Analysis** on the IMDB Movie Reviews dataset using **TensorFlow/Keras**.

The model classifies movie reviews as **Positive** or **Negative** with ~88% accuracy.

---

## Files

| File | Description |
|------|-------------|
| `Task2_deep_learning_nlp.py` | Main Python script — full deep learning pipeline |
| `task2_output/best_model.keras` | Saved best model weights (generated on run) |
| `task2_output/results_visualization.png` | Training plots and evaluation charts (generated on run) |
| `README.md` | Project documentation |

---

## Dataset — IMDB Movie Reviews

- **Source**: Built into Keras (`tensorflow.keras.datasets.imdb`) — downloads automatically
- **Size**: 50,000 movie reviews (25,000 train + 25,000 test)
- **Labels**: `0` = Negative, `1` = Positive (balanced dataset)
- **Vocabulary**: Top 10,000 most frequent words

No separate dataset file needed — Keras downloads it automatically on first run.

---

## Model Architecture

```
Input (padded word sequences, length=200)
        ↓
Embedding Layer (10000 vocab → 64-dim vectors)
        ↓
SpatialDropout1D (0.3)
        ↓
LSTM (64 units, dropout=0.2)
        ↓
Dropout (0.3)
        ↓
Dense (32 units, ReLU)
        ↓
Dropout (0.2)
        ↓
Dense (1 unit, Sigmoid)  ← Binary output
```

---

## Pipeline Steps

### 1. Load Data
- Loads IMDB dataset via Keras (auto-downloads ~17MB)

### 2. Preprocess
- Pads all reviews to fixed length of 200 words
- Shorter reviews → zero-padded | Longer reviews → truncated

### 3. Build Model
- LSTM-based sequential model compiled with Adam optimizer
- Binary cross-entropy loss for binary classification

### 4. Train
- EarlyStopping to prevent overfitting (patience=3)
- ModelCheckpoint saves best model automatically

### 5. Evaluate
- Test accuracy, loss
- Classification report (Precision, Recall, F1-Score)

### 6. Visualize
Generates 4 plots saved as `results_visualization.png`:
- Training vs Validation Accuracy
- Training vs Validation Loss
- Confusion Matrix
- ROC Curve (with AUC score)

### 7. Custom Predictions
- Tests the trained model on 5 hand-written reviews
- Shows sentiment label + confidence score

---

## How to Run

```bash
# 1. Install dependencies
pip install tensorflow scikit-learn matplotlib numpy

# 2. Run the script
python Task2_deep_learning_nlp.py
```

Expected output:
```
Test Accuracy : ~88%
AUC Score     : ~0.95
Output saved  : task2_output/
```

---

## Dependencies

```
tensorflow >= 2.x
numpy
matplotlib
scikit-learn
```

---

## Internship Details
- **Organization**: CODTECH IT Solutions
- **Task**: Task 2 — Deep Learning Project
- **Domain**: Data Science / NLP
