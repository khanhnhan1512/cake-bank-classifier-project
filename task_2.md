## Section 2: Model Performance Evaluation

### 1. Evaluation Metrics

The binary liveness detection model based on **EfficientNet-B2** was evaluated using the following metrics:

- **Cross-Entropy Loss**: Measures probabilistic classification error during optimization.
- **Accuracy (94.66%)**: Overall proportion of correctly classified samples.
- **Precision (95.11%)**: Measures how many predicted _live_ samples are truly live.
- **Recall (93.86%)**: Measures the model’s ability to detect actual live samples.
- **F1-score (94.48%)**: Harmonic mean of precision and recall, balancing false positives and false negatives.
- **ROC-AUC (0.9877)**: Evaluates class separability across all decision thresholds.
- **Average Precision (0.9895)**: Summarizes the precision–recall trade-off, robust under class imbalance.

---

### 2. Justification of the Evaluation Approach

Although the training and test datasets are class-balanced, real-world liveness detection is inherently imbalanced, with a large dominance of normal/live samples and relatively few spoof attempts.

Therefore:

- **Accuracy** alone is misleading and insufficient.
- **Precision, Recall, and F1-score** capture security-critical error trade-offs.
- **ROC-AUC and Average Precision** are emphasized because they remain informative under skewed class distributions and allow flexible threshold selection.
- **Cross-Entropy Loss** ensures stable optimization and probabilistic outputs suitable for post-training calibration.

This evaluation strategy anticipates deployment conditions rather than merely reflecting offline dataset statistics.

---

### 3. Limitations and Areas for Improvement

Despite strong performance, the model has several limitations:

- **Generalization risk**: Performance may degrade under unseen spoofing techniques, lighting conditions, or camera domains.
- **Temporal ignorance**: Frame-based classification ignores motion cues critical for liveness.
- **Calibration**: High ROC-AUC does not guarantee well-calibrated confidence scores.
- **Dataset bias**: Potential imbalance or lack of spoof diversity can inflate evaluation metrics.

---

### 4. Proposed Improvements and Alternatives

To enhance robustness and real-world reliability, the following techniques are recommended:

- **Temporal modeling**: Incorporate video-based architectures (CNN + LSTM, 3D CNNs, or Vision Transformers).
- **Multi-modal signals**: Fuse RGB with depth, infrared, or frequency-domain features.
- **Advanced losses**: Use Focal Loss or Class-Balanced Loss to handle hard samples.
- **Domain generalization**: Apply strong data augmentation, adversarial training, or domain adaptation.
- **Score calibration**: Use temperature scaling or Platt scaling before deployment.
- **Ensemble methods**: Combine multiple backbones or resolutions for improved stability.

---

**Summary**:
The EfficientNet-B2 classifier demonstrates **excellent discriminative ability** (ROC-AUC ≈ 0.99) and strong balanced performance. However, addressing temporal dynamics, generalization, and calibration is essential before production deployment.
