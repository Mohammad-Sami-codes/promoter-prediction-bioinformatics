"""
Promoter Region Prediction in DNA Sequences
Comparing classical ML (Logistic Regression, Random Forest) vs a CNN
on the UCI Molecular Biology Promoter Gene Sequences dataset (E. coli).

Authors: Mohammad Sami, Shanzida Hasan Esha
Course: CSE443 Bioinformatics
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, roc_auc_score,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "molecular-biology/promoter-gene-sequences/promoters.data"
)


def load_and_clean_data():
    raw_data = pd.read_csv(DATA_URL, header=None)
    raw_data.columns = ["label", "name", "sequence"]
    raw_data["sequence"] = raw_data["sequence"].str.strip()
    raw_data["label"] = raw_data["label"].map({"+": 1, "-": 0})
    print(f"Loaded {raw_data.shape[0]} sequences")
    print(raw_data["label"].value_counts())
    return raw_data


def build_kmer_features(raw_data, k=4):
    vectorizer = CountVectorizer(analyzer="char", ngram_range=(k, k))
    kmer_features = vectorizer.fit_transform(raw_data["sequence"])
    print(f"k-mer feature matrix: {kmer_features.shape}")
    return kmer_features


def train_classical_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_predictions = lr_model.predict(X_test)
    print("Logistic Regression accuracy:", accuracy_score(y_test, lr_predictions))
    print(classification_report(y_test, lr_predictions))

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)
    print("Random Forest accuracy:", accuracy_score(y_test, rf_predictions))
    print(classification_report(y_test, rf_predictions))

    lr_scores = cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=5)
    rf_scores = cross_val_score(
        RandomForestClassifier(n_estimators=100, random_state=42), X, y, cv=5
    )
    print("Logistic Regression 5-fold CV:", lr_scores.mean(), lr_scores.std())
    print("Random Forest 5-fold CV:", rf_scores.mean(), rf_scores.std())

    return {
        "lr_model": lr_model, "rf_model": rf_model,
        "X_test": X_test, "y_test": y_test,
        "lr_predictions": lr_predictions, "rf_predictions": rf_predictions,
        "lr_cv": lr_scores, "rf_cv": rf_scores,
    }


def one_hot_encode(seq, base_to_index={"a": 0, "c": 1, "g": 2, "t": 3}):
    encoded = np.zeros((len(seq), 4))
    for i, base in enumerate(seq):
        encoded[i, base_to_index[base]] = 1
    return encoded


def train_cnn(raw_data):
    X_cnn = np.array([one_hot_encode(seq) for seq in raw_data["sequence"]])
    y_cnn = raw_data["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X_cnn, y_cnn, test_size=0.2, random_state=42, stratify=y_cnn
    )

    tf.random.set_seed(42)
    cnn_model = Sequential([
        Conv1D(filters=32, kernel_size=4, activation="relu", input_shape=(57, 4)),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(16, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ])
    cnn_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    cnn_model.summary()

    history = cnn_model.fit(
        X_train, y_train, epochs=30, batch_size=8, validation_split=0.2, verbose=1
    )

    test_loss, test_accuracy = cnn_model.evaluate(X_test, y_test, verbose=0)
    print("CNN test accuracy:", test_accuracy)

    cnn_predictions = (cnn_model.predict(X_test) > 0.5).astype(int).flatten()
    print(classification_report(y_test, cnn_predictions))

    return {
        "model": cnn_model, "history": history,
        "X_test": X_test, "y_test": y_test, "predictions": cnn_predictions,
    }


def plot_confusion_matrices(lr_cm, rf_cm, cnn_cm):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    labels = ["Non-promoter", "Promoter"]
    for ax, cm, cmap, title in zip(
        axes, [lr_cm, rf_cm, cnn_cm],
        ["Blues", "Greens", "Oranges"],
        ["Logistic Regression", "Random Forest", "CNN"],
    ):
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix_all3.png"), dpi=200)
    plt.close()


def plot_roc_curves(y_test, lr_probs, rf_probs, y_test_cnn, cnn_probs):
    lr_auc = roc_auc_score(y_test, lr_probs)
    rf_auc = roc_auc_score(y_test, rf_probs)
    cnn_auc = roc_auc_score(y_test_cnn, cnn_probs)

    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
    cnn_fpr, cnn_tpr, _ = roc_curve(y_test_cnn, cnn_probs)

    plt.figure(figsize=(6, 5))
    plt.plot(lr_fpr, lr_tpr, label=f"Logistic Regression (AUC = {lr_auc:.2f})")
    plt.plot(rf_fpr, rf_tpr, label=f"Random Forest (AUC = {rf_auc:.2f})")
    plt.plot(cnn_fpr, cnn_tpr, label=f"CNN (AUC = {cnn_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Model Comparison")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, "roc_curves.png"), dpi=200)
    plt.close()

    print("Logistic Regression AUC:", lr_auc)
    print("Random Forest AUC:", rf_auc)
    print("CNN AUC:", cnn_auc)


def plot_cnn_training_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="Train accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Validation accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("CNN Accuracy over Epochs")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train loss")
    axes[1].plot(history.history["val_loss"], label="Validation loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("CNN Loss over Epochs")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "cnn_training_curves.png"), dpi=200)
    plt.close()


def main():
    raw_data = load_and_clean_data()
    kmer_features = build_kmer_features(raw_data, k=4)
    y = raw_data["label"]

    classical = train_classical_models(kmer_features, y)
    cnn = train_cnn(raw_data)

    lr_cm = confusion_matrix(classical["y_test"], classical["lr_predictions"])
    rf_cm = confusion_matrix(classical["y_test"], classical["rf_predictions"])
    cnn_cm = confusion_matrix(cnn["y_test"], cnn["predictions"])
    plot_confusion_matrices(lr_cm, rf_cm, cnn_cm)

    lr_probs = classical["lr_model"].predict_proba(classical["X_test"])[:, 1]
    rf_probs = classical["rf_model"].predict_proba(classical["X_test"])[:, 1]
    cnn_probs = cnn["model"].predict(cnn["X_test"]).flatten()
    plot_roc_curves(classical["y_test"], lr_probs, rf_probs, cnn["y_test"], cnn_probs)

    plot_cnn_training_curves(cnn["history"])

    print(f"\nAll figures saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
