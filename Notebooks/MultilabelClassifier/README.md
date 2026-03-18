# Ring Multilabel Classification

Multilabel classification of ring structures in galaxy images using a fine-tuned [Zoobot](https://github.com/mwalmsley/zoobot) encoder. The model detects the presence of **inner rings** and **outer rings** independently in FITS-format astronomical images, framing the problem as binary multilabel classification rather than mutually exclusive multiclass.

## Dataset

The catalog contains **8,346 galaxy images** in FITS format (3-band: r, g, z) sourced from the[ DESI Legacy Imaging](https://www.legacysurvey.org/dr9/description/) and MaNGA surveys. Each galaxy is labeled with two independent labels: `inner_ring` and `outer_ring`, which are added
on preprocessing:

| ring_class | Inner Ring | Outer Ring | Description       | Count |
|:----------:|:----------:|:----------:|-------------------|------:|
| 0          | 0          | 0          | No rings          | 6,327 |
| 1          | 1          | 0          | Inner ring only   | 1,266 |
| 2          | 0          | 1          | Outer ring only   |   333 |
| 3          | 1          | 1          | Both rings        |   420 |

The dataset is split 80/10/10 (train/val/test) with stratification by `ring_class` to preserve class proportions across splits.

## Model Architecture

The model (`RingDetectionZoobot`) is a PyTorch Lightning module built on top of a pretrained **Zoobot ConvNeXt Tiny** encoder (~27.8M parameters, encoder dim = 768) loaded from `hf_hub:mwalmsley/zoobot-encoder-convnext_tiny`. A custom classification head maps the encoder features to two sigmoid outputs:

```
Encoder (ConvNeXt Tiny, 27.8M params)
  └─ Dropout(0.4)
  └─ Linear(768 → 256) + ReLU
  └─ Dropout(0.2)
  └─ Linear(256 → 2)   →  [inner_ring_logit, outer_ring_logit]
```

Per-label decision thresholds are tuned independently via 2D grid search on the validation set, optimizing macro-averaged **F2 score** (FBeta with beta=2.0) over both labels. F2 was chosen over F1 to weight recall more heavily than precision, prioritizing the detection of rare ring structures at the cost of slightly more false positives.

## Image Preprocessing

Raw FITS data passes through an astrophysical preprocessing pipeline before entering the model:

1. **Resize** to 224x224
2. **Lupton RGB** compositing (`stretch=0.5`, `Q=10`) via [`astropy.visualization.make_lupton_rgb`](https://docs.astropy.org/en/latest/api/astropy.visualization.make_lupton_rgb.html) (see [Lupton et al. (2004)](https://ui.adsabs.harvard.edu/abs/2004PASP..116..133L/abstract))
3. **Scale to [0, 1]** interval

Training data is further augmented with random horizontal/vertical flips, rotation (up to 180 degrees), small affine translations and scaling, and mild color jitter to exploit the approximate rotational invariance of galaxy morphology.

## Training Pipeline (`main.ipynb`)

The notebook walks through five sections:

### 1. Load Data
Reads the galaxy catalog CSV and inspects class distribution, coordinate ranges, and redshift statistics.

### 2. Preprocessing
Defines and visualizes the image transform pipeline (Lupton RGB, optional sky subtraction and multi-scale unsharp masking). Includes 3D channel visualizations and per-channel intensity range inspection.

### 3. Model & Data Catalog
- Instantiates the `ZoobotFitsDataModule` with `WeightedRandomSampler` (inverse-frequency weighting) to counteract the heavy class imbalance.
- Computes per-label `pos_weight` for loss weighting.
- Creates the `RingDetectionZoobot` model with **Asymmetric Loss** (ASL, `gamma_neg=2.0`, `gamma_pos=0.0`, `clip=0.05`) to suppress easy-negative dominance in the multilabel setting.
- Uses a **cosine annealing** LR schedule with linear warmup.

### 4. Training (Two-Stage)

**Stage 1 -- Head-only training (encoder frozen):**
- Trains only the classification head for up to 10 epochs.
- Head learning rate: `1e-4`, cosine schedule with 3-epoch warmup.
- Early stopping on `val_f2_macro` (patience 5).
- 16-bit mixed precision on GPU.

**Stage 2 -- Full fine-tuning (encoder unfrozen):**
- Loads the best Stage 1 checkpoint and unfreezes the encoder.
- Encoder learning rate: `5e-6` (20x smaller than head LR) to preserve pretrained representations.
- Weight decay: `1e-2` for regularization.
- Trains for up to 30 epochs with early stopping (patience 25).
- Saves the best checkpoint by `val_f2_macro`.

### 5. Metrics
- Loads the best Stage 2 checkpoint and tunes per-label thresholds on the validation set.
- Evaluates on the test set with **test-time augmentation** (TTA): predictions are averaged over 8 views (4 rotations x 2 flip states).
- Produces ROC curves with AUC, per-label confusion matrices, and a combined 4-class confusion matrix.

## Checkpoint Comparer (`checkpoint_visualizer.ipynb`)

A standalone notebook for side-by-side evaluation of any two training checkpoints. It provides:

1. **Checkpoint loading** -- Loads two model versions from `lightning_logs/version_*/`, reconstructs the encoder, restores weights, and tunes per-label decision thresholds on the validation set.
2. **Interactive probability explorer** -- Scatter plots of predicted probabilities alongside 2D galaxy images and 3D channel surface plots. Supports TTA predictions.
3. **ROC comparison** -- Overlaid ROC curves (inner ring, outer ring, and micro-average) for both checkpoints with AUC values.
4. **Confusion matrix comparison** -- Per-label (inner, outer) and combined 4-class confusion matrices displayed side-by-side with row-normalized percentages.

## Final Model (Checkpoint v37)

The production model is **version 37** (`lightning_logs/version_37/checkpoints/stage2-best-epoch=20.ckpt`), selected as the best-performing checkpoint after Stage 2 fine-tuning. Decision thresholds optimized on the validation set using F2 macro:

| Label      | Threshold |
|------------|:---------:|
| Inner Ring |   0.580   |
| Outer Ring |   0.480   |

The lower thresholds compared to earlier F1-tuned checkpoints reflect the F2 objective's emphasis on recall: the model accepts a higher false-positive rate to avoid missing true ring detections.

## Project Structure

```
RingMultilabelClassification/
├── main.ipynb                  # Full training pipeline
├── checkpoint_visualizer.ipynb # Checkpoint comparison tool
├── ring_detection_model.py     # RingDetectionZoobot model & threshold tuning
├── datasets.py                 # FitsDataset & ZoobotFitsDataModule
├── galaxy_transforms.py        # Lupton RGB, unsharp mask, sky subtraction
├── visualizations.py           # Plotting & visualization utilities
├── Data/
│   └── dataset_all_with_files_patched.csv
└── lightning_logs/             # Training checkpoints (gitignored)
```

## Requirements

Key dependencies:

- Python 3.10+
- PyTorch
- Lightning (PyTorch Lightning)
- Zoobot (`zoobot`)
- timm
- torchmetrics
- torchvision
- astropy
- scikit-learn
- pandas, numpy, matplotlib, seaborn, scipy
