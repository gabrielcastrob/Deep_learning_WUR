# Deep Learning — Multilabel Land-Use Classification

Multilabel classification on **UC Merced** (17 classes, 2100 images) — WUR Assignment (GRS-34806 / FTE-35306).

## Goal

Compare three architectures for multilabel aerial image classification:

| Level | Model | Role |
|---|---|---|
| 1 | ResNet-50 (fine-tuned) | Base architecture |
| 2 | ViT-B/16 (fine-tuned) | Newer architecture |

## Structure

```
├── notebooks/
│   ├── utils.py                        # Dataset, dataloaders, Lightning module, metrics & plots
│   ├── 00_ML_Resnet.ipynb              # ResNet-50 training
│   ├── 00_ML_ViT.ipynb                 # ViT-B/16 training
│   └── 02_ModelComparison_F1_mAP.ipynb # Final model comparison
```

## Training Pipeline

1. Mount Drive / clone repo and download `ucmdata`.
2. `build_dataloaders()` → stratified train/val/test (70/15/15) + `pos_weight`.
3. Build backbone (`resnet50` or `vit_b_16`) and replace head with `num_classes = 17`.
4. Train with `ModelCheckpoint(monitor="val_f1")` + `EarlyStopping(patience=5)`.
5. Evaluate on test → save metrics to `outputs/ModelComparisons.csv`.

## Hyperparameters

| Parameter | ResNet-50 | ViT-B/16 |
|---|---|---|
| LR | 1e-4 | 1e-5 |
| Weight decay | 1e-4 | 0.01 |
| Batch size | 32 | 32 |
| Max epochs | 25 (early stop) | 25 (early stop) |
| Loss | BCEWithLogitsLoss + pos_weight | idem |

## Results

| Model | Macro F1 | Macro mAP | Exact Match |
|---|---|---|---|
| ResNet-50 | 0.933 | 0.978 | 0.513 |
| ViT-B/16 | **0.950** | **0.981** | **0.570** |

## Requirements

```bash
pip install torch torchvision lightning torchmetrics iterative-stratification scikit-learn pandas matplotlib
```
