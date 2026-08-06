# Wafer Map Defect Patterns and Yield Engineering

## Overview
In semiconductor manufacturing, a wafer map records the pass/fail result of each
die on a silicon wafer after electrical test. When failing dies cluster into
recognizable spatial shapes, these clusters are called defect patterns. Each
pattern is a fingerprint that points back to a specific process problem, so
classifying wafer maps is a core part of yield engineering and root-cause
analysis.

## Common Defect Patterns
- **Center**: Failures concentrated in the middle of the wafer. Often linked to
  chemical-mechanical planarization (CMP) non-uniformity or spin-coating issues
  where the center receives uneven material thickness.
- **Donut**: A ring of failures surrounding a healthy center. Frequently caused
  by uneven film deposition or thermal gradients during rapid thermal processing.
- **Edge-Ring**: Failures forming a band around the wafer edge. Common causes
  include edge-bead removal problems, wafer-handling contact damage, and
  non-uniform etch at the wafer perimeter.
- **Scratch**: A line or streak of failing dies, typically from mechanical
  handling damage or particles dragged across the surface during transport.
- **Random**: Failures with no spatial correlation, usually attributed to random
  particle contamination rather than a systematic process fault.

## Class Imbalance
Real wafer datasets such as WM-811K are heavily imbalanced: the vast majority of
wafers are defect-free or fall into a few common patterns, while patterns like
Scratch and Donut are rare. Training a classifier without addressing imbalance
leads to a model that ignores rare-but-critical patterns. Techniques include
class weighting, oversampling the minority classes, focal loss, and data
augmentation such as rotation and flipping, which are label-preserving for most
wafer patterns.

## Evaluation Metrics
Accuracy alone is misleading under class imbalance because a model can score
high by always predicting the majority class. Precision, recall, and the F1
score per class give a truer picture. A confusion matrix shows exactly which
patterns get confused with each other, which often maps back to physically
similar failure mechanisms.

## Anomaly Detection for Rare Defects
Because some defect patterns are too rare to train a supervised classifier on,
an unsupervised anomaly-detection baseline is useful. An autoencoder trained
only on normal wafers learns to reconstruct them well; wafers it reconstructs
poorly (high reconstruction error) are flagged as anomalous and routed to an
engineer for review.
