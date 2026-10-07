# Promoter Region Prediction in DNA Sequences

Comparing classical machine learning and deep learning for identifying
promoter regions in *E. coli* DNA sequences — a binary classification
problem framed around a core bioinformatics task: finding where genes
are marked for transcription.

**Course project — CSE443 Bioinformatics, BRAC University**
Authors: Mohammad Sami, Shanzida Hasan Esha

## Problem

Promoters are short DNA regions that signal where transcription should
begin. Identifying them is useful for genome annotation, and classical
sequence-matching approaches don't always generalize well across
organisms. This project asks: for a *small*, well-studied dataset, does
a more complex deep learning model actually outperform simpler
classical methods?

## Approach

Two pipelines were built and evaluated on the same data:

- **Classical ML** — DNA sequences converted to 4-mer frequency
  features (256 features per sequence), fed into Logistic Regression
  and Random Forest.
- **Deep learning** — sequences one-hot encoded (57×4 per sequence) and
  fed into a small 1D CNN (Conv1D → MaxPooling → Dense → Dropout →
  Sigmoid output).

Both pipelines used the same 80/20 train-test split (fixed seed) for a
fair comparison; classical models were additionally validated with
5-fold cross-validation.

## Dataset

[UCI Molecular Biology Promoter Gene Sequences](https://archive.ics.uci.edu/dataset/67/molecular+biology+promoter+gene+sequences)
— 106 *E. coli* DNA sequences (53 promoter, 53 non-promoter), each 57
bases long. Loaded directly from the UCI repository at runtime.

## Results

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.864    | 0.87      | 0.86   | 0.86     | 0.983   |
| Random Forest       | 0.909    | 0.91      | 0.91   | 0.91     | 0.971   |
| CNN                 | 0.909    | 0.92      | 0.91   | 0.91     | 0.958   |

**Key finding:** the CNN matched Random Forest on accuracy but did not
outperform the classical models, and showed clear signs of overfitting
(training accuracy ≈100% vs. validation accuracy ≈88%) — unsurprising
given the CNN has 14,000+ trainable parameters against only 84 training
sequences. On a small, well-structured biological dataset like this,
simpler, more interpretable models (especially Logistic Regression,
which had the highest ROC-AUC) were competitive with — and more
consistent than — the deep learning approach.

Full methodology, discussion, and figures are in
[`report/CSE443_Project_Report.pdf`](report/CSE443_Project_Report.pdf).

## Repository structure

```
.
├── src/
│   └── promoter_classification.py   # full pipeline: data → features → models → evaluation plots
├── results/                         # generated figures (ROC curves, confusion matrices, training curves)
├── report/
│   └── CSE443_Project_Report.pdf    # full written report
└── requirements.txt
```

## Running it

```bash
pip install -r requirements.txt
python src/promoter_classification.py
```

This downloads the dataset, trains all three models, prints evaluation
metrics, and saves `roc_curves.png`, `confusion_matrix_all3.png`, and
`cnn_training_curves.png` to `results/`.

## Limitations

- Only 106 sequences total (22 in the test set) — results may shift
  with retraining given the CNN's sensitivity to random initialization.
- Cross-validation was applied to the classical models only; applying
  it to the CNN was out of scope due to time constraints.
- Only one CNN architecture was tested.

## References

1. S. Anveshrithaa, B. Aathavan, N. Jaisankar, "Promoter Prediction in
   DNA Sequences of *Escherichia Coli* Using Machine Learning
   Algorithms," 2019.
2. N. Bhandari, S. Khare, R. Walambe, K. Kotecha, "Comparison of
   machine learning and deep learning techniques in promoter prediction
   across diverse species," *PeerJ Computer Science*, 7:e365, 2021.
3. F. Anwar et al., "Pol II promoter prediction using characteristic
   4-Mer Motifs: a machine learning approach," *BMC Bioinformatics*,
   9:1–8, 2008.
4. C. Harley, R. Reynolds, M. Noordewier, "Molecular Biology (Promoter
   Gene Sequences)," UCI Machine Learning Repository, 1990.
