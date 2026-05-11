# Training Results and Evaluation Artifacts

This document explains the model training outputs and how to interpret the charts produced by `traingnn.py`.

## Training and Prediction Pipeline (Summary)
1. Load `MMORPG_Medium_Cleaned.csv` and engineer node features (sent/received totals, trade counts, ratios, PageRank, velocity).
2. Build the transaction graph (`edge_index`) linking sender and receiver nodes.
3. Train `HyperEliteSAGE` with biased sampling + Focal Loss to handle class imbalance.
4. Evaluate on the full graph to compute accuracy, precision, recall, F1, and an optimal threshold.
5. Save the trained weights (`hyper_elite_medium_model.pth`) and threshold (`threshold.txt`).
6. Run `inference.py` to score every player and export `players.json`, `dashboard.json`, and `graph.json` for the API.

## Outputs Produced by traingnn.py
All artifacts are saved in `reports/` when you run training.

- `training_history.json` / `training_history.csv`
  Per-epoch metrics for loss, accuracy, precision, recall, and F1.
- `training_loss.png`
  Line chart showing how the training loss decreases each epoch.
- `training_metrics.png`
  Line chart showing accuracy, precision, recall, and F1 across epochs.
- `precision_recall_curve.png`
  Precision vs. recall curve across decision thresholds.
- `precision_recall_curve.csv`
  Threshold-level precision, recall, and F1 for auditing tradeoffs.
- `confusion_matrix.png`
  Final confusion matrix for Safe vs Hacker classification.
- `classification_report.json` / `classification_report.txt`
  Final classification report with class-level metrics.
- `final_metrics.json`
  Final accuracy, precision, recall, F1, and the selected threshold.
- `sample_predictions.csv`
  Random sample of player IDs with true labels and predicted probabilities.

## How to Read the Charts
- Training Loss
  Lower is better. A smooth downward trend indicates stable convergence.
- Training Metrics (Accuracy / Precision / Recall / F1)
  These show how the model improves per epoch on a fixed evaluation sample.
- Precision-Recall Curve
  Visualizes the tradeoff: higher precision reduces false alarms, higher recall catches more fraud.
- Confusion Matrix
  Shows true positives, true negatives, false positives, and false negatives at the chosen threshold.

## Tuning Knobs (Optional)
You can control output sizes via environment variables:
- `EPOCHS` (default 20)
- `EVAL_SAMPLE_SIZE` (default 20000)
- `SAMPLE_PREDICTIONS` (default 200)
