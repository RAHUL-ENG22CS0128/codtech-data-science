"""
================================================================================
CODTECH INTERNSHIP - TASK 2: DEEP LEARNING PROJECT
================================================================================
Author      : Data Science Intern
Description : Sentiment Analysis using a Deep Learning model (LSTM) built
              with TensorFlow/Keras on the IMDB Movie Reviews dataset.
              Includes training, evaluation, and result visualizations.
Task Type   : Natural Language Processing (NLP) — Text Classification
================================================================================
"""

# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, LSTM, Dense, Dropout,
    SpatialDropout1D, GlobalMaxPooling1D
)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Scikit-learn for evaluation metrics
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, ConfusionMatrixDisplay
)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
VOCAB_SIZE   = 10000      # top 10,000 most frequent words
MAX_LEN      = 200        # pad/truncate all reviews to 200 words
EMBED_DIM    = 64         # embedding vector size
LSTM_UNITS   = 64         # LSTM hidden units
BATCH_SIZE   = 128
EPOCHS       = 10
OUTPUT_DIR   = "task2_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────
np.random.seed(42)
tf.random.set_seed(42)


# ================================================================================
# STEP 1: LOAD & EXPLORE DATA
# ================================================================================

def load_data():
    """
    Loads the built-in IMDB dataset from Keras.
    25,000 training reviews and 25,000 test reviews.
    Labels: 0 = Negative, 1 = Positive
    """
    print("\n" + "="*60)
    print("  STEP 1: LOADING IMDB DATASET")
    print("="*60)

    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)

    print(f"  ✔ Training samples : {len(X_train)}")
    print(f"  ✔ Test samples     : {len(X_test)}")
    print(f"  ✔ Classes          : 0 = Negative, 1 = Positive")
    print(f"  ✔ Positive reviews in train : {y_train.sum()} ({y_train.mean()*100:.1f}%)")
    print(f"  ✔ Avg review length        : {np.mean([len(r) for r in X_train]):.0f} words")

    return (X_train, y_train), (X_test, y_test)


# ================================================================================
# STEP 2: PREPROCESS — Pad Sequences
# ================================================================================

def preprocess(X_train, X_test):
    """
    Pads all reviews to the same length (MAX_LEN) so they can be
    fed into the Embedding layer as fixed-size input tensors.
    Shorter reviews are zero-padded; longer ones are truncated.
    """
    print("\n" + "="*60)
    print("  STEP 2: PREPROCESSING — Padding Sequences")
    print("="*60)

    X_train_pad = pad_sequences(X_train, maxlen=MAX_LEN, padding="post", truncating="post")
    X_test_pad  = pad_sequences(X_test,  maxlen=MAX_LEN, padding="post", truncating="post")

    print(f"  ✔ Train shape after padding : {X_train_pad.shape}")
    print(f"  ✔ Test  shape after padding : {X_test_pad.shape}")

    return X_train_pad, X_test_pad


# ================================================================================
# STEP 3: BUILD THE DEEP LEARNING MODEL (LSTM)
# ================================================================================

def build_model():
    """
    Architecture:
      Embedding → SpatialDropout → LSTM → Dropout → Dense(ReLU) → Dense(Sigmoid)

    - Embedding  : converts word indices to dense vectors
    - LSTM       : captures sequential/contextual patterns in text
    - Dense      : final binary classification (positive / negative)
    """
    print("\n" + "="*60)
    print("  STEP 3: BUILDING LSTM MODEL")
    print("="*60)

    model = Sequential([
        # Word Embedding layer
        Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_DIM, input_length=MAX_LEN),

        # Spatial dropout — drops entire embedding dimensions (better for NLP)
        SpatialDropout1D(0.3),

        # LSTM layer — learns sequential patterns
        LSTM(LSTM_UNITS, dropout=0.2, recurrent_dropout=0.2, return_sequences=False),

        # Regularisation
        Dropout(0.3),

        # Dense hidden layer
        Dense(32, activation="relu"),
        Dropout(0.2),

        # Output layer — sigmoid for binary classification
        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()
    return model


# ================================================================================
# STEP 4: TRAIN THE MODEL
# ================================================================================

def train_model(model, X_train, y_train, X_test, y_test):
    """
    Trains the LSTM model with:
      - EarlyStopping   : stops if val_loss doesn't improve for 3 epochs
      - ModelCheckpoint : saves the best model weights automatically
    """
    print("\n" + "="*60)
    print("  STEP 4: TRAINING THE MODEL")
    print("="*60)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ModelCheckpoint(
            filepath=os.path.join(OUTPUT_DIR, "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        )
    ]

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )

    print("\n  ✔ Training complete!")
    return history


# ================================================================================
# STEP 5: EVALUATE THE MODEL
# ================================================================================

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model and prints:
      - Test loss and accuracy
      - Full classification report (precision, recall, F1)
    """
    print("\n" + "="*60)
    print("  STEP 5: EVALUATING THE MODEL")
    print("="*60)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"  ✔ Test Loss     : {loss:.4f}")
    print(f"  ✔ Test Accuracy : {accuracy*100:.2f}%")

    # Predictions
    y_pred_prob = model.predict(X_test, verbose=0).flatten()
    y_pred      = (y_pred_prob >= 0.5).astype(int)

    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

    return y_pred, y_pred_prob


# ================================================================================
# STEP 6: VISUALIZE RESULTS
# ================================================================================

def visualize_results(history, y_test, y_pred, y_pred_prob):
    """
    Creates 4 plots:
      1. Training vs Validation Accuracy
      2. Training vs Validation Loss
      3. Confusion Matrix
      4. ROC Curve
    Saves all to task2_output/
    """
    print("\n" + "="*60)
    print("  STEP 6: VISUALIZATIONS")
    print("="*60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Task 2 — LSTM Sentiment Analysis Results", fontsize=16, fontweight="bold")

    # ── Plot 1: Accuracy ────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(history.history["accuracy"],     label="Train Accuracy", color="steelblue", linewidth=2)
    ax.plot(history.history["val_accuracy"], label="Val Accuracy",   color="darkorange", linewidth=2, linestyle="--")
    ax.set_title("Training vs Validation Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Plot 2: Loss ─────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(history.history["loss"],     label="Train Loss", color="steelblue", linewidth=2)
    ax.plot(history.history["val_loss"], label="Val Loss",   color="darkorange", linewidth=2, linestyle="--")
    ax.set_title("Training vs Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Plot 3: Confusion Matrix ─────────────────────────────────
    ax = axes[1, 0]
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix")

    # ── Plot 4: ROC Curve ────────────────────────────────────────
    ax = axes[1, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc     = auc(fpr, tpr)
    ax.plot(fpr, tpr, color="steelblue", linewidth=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "results_visualization.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  ✔ Visualization saved → {save_path}")


# ================================================================================
# STEP 7: PREDICT ON CUSTOM REVIEWS
# ================================================================================

def predict_custom(model):
    """
    Demonstrates the model on a few hand-written movie reviews.
    Shows how to preprocess new text and run inference.
    """
    print("\n" + "="*60)
    print("  STEP 7: CUSTOM REVIEW PREDICTIONS")
    print("="*60)

    # Get the word index from IMDB dataset
    word_index = imdb.get_word_index()

    def encode_review(text):
        """Converts a raw text review into a padded integer sequence."""
        tokens = text.lower().split()
        # +3 offset is the IMDB convention (reserved indices)
        encoded = [word_index.get(word, 2) + 3 for word in tokens]
        encoded = [idx if idx < VOCAB_SIZE else 2 for idx in encoded]
        return pad_sequences([encoded], maxlen=MAX_LEN, padding="post")

    custom_reviews = [
        "This movie was absolutely fantastic! The acting was brilliant and the story was captivating.",
        "Terrible film. Complete waste of time. Boring plot and awful acting.",
        "It was okay. Some good moments but overall pretty average experience.",
        "One of the best movies I have ever seen. A masterpiece of cinema.",
        "I fell asleep halfway through. Nothing interesting happens at all.",
    ]

    print(f"\n  {'Review':<65} {'Sentiment':<12} {'Confidence'}")
    print("  " + "-"*90)
    for review in custom_reviews:
        encoded = encode_review(review)
        prob    = model.predict(encoded, verbose=0)[0][0]
        label   = "POSITIVE 😊" if prob >= 0.5 else "NEGATIVE 😞"
        conf    = prob if prob >= 0.5 else 1 - prob
        short   = review[:62] + "..." if len(review) > 62 else review
        print(f"  {short:<65} {label:<12} {conf*100:.1f}%")


# ================================================================================
# MAIN — Run Full Pipeline
# ================================================================================

if __name__ == "__main__":

    print("\n" + "█"*60)
    print("  CODTECH INTERNSHIP — TASK 2: DEEP LEARNING (NLP)")
    print("  Sentiment Analysis with LSTM on IMDB Dataset")
    print("█"*60)

    # Step 1: Load data
    (X_train, y_train), (X_test, y_test) = load_data()

    # Step 2: Preprocess
    X_train_pad, X_test_pad = preprocess(X_train, X_test)

    # Step 3: Build model
    model = build_model()

    # Step 4: Train
    history = train_model(model, X_train_pad, y_train, X_test_pad, y_test)

    # Step 5: Evaluate
    y_pred, y_pred_prob = evaluate_model(model, X_test_pad, y_test)

    # Step 6: Visualize
    visualize_results(history, y_test, y_pred, y_pred_prob)

    # Step 7: Custom predictions
    predict_custom(model)

    print("\n" + "█"*60)
    print("  ✅ Task 2 Complete! Output saved to task2_output/")
    print("█"*60 + "\n")
