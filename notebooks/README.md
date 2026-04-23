# Deep Learning – Multi-label Land Use Classification

Comparison of **ResNet-50** and **ViT-B/16** for multi-label classification on the [UCMerced Land Use](https://git.wur.nl/lobry001/ucmdata) dataset using PyTorch Lightning.

## Structure

| File | Description |
|---|---|
| `00_ML_Resnet.ipynb` | Training & evaluation with ResNet-50 |
| `00_ML_ViT.ipynb` | Training & evaluation with ViT-B/16 |
| `02_ModelComparison_F1_mAP.ipynb` | Side-by-side comparison (F1, mAP) |
| `utils.py` | Dataloaders, Lightning module, metrics & plots |
| `ucmdata/` | UCMerced dataset (images + multi-label file) |

## Quickstart (Google Colab)

1. Mount Google Drive and set `PROJECT_DIR`.
2. Clone or pull the repo automatically (first cell of each notebook).
3. Run all cells — the dataset downloads itself if not present.

## Models & Hyperparameters

| | ResNet-50 | ViT-B/16 |
|---|---|---|
| Pretrained weights | ImageNet1K V2 | ImageNet1K V1 |
| LR | 1e-4 | 1e-5 |
| Weight decay | 1e-4 | 0.01 |
| Max epochs | 25 | 25 |
| Early stopping | 5 epochs | 5 epochs |

Both models use the same data split (train/val/test), batch size 32, and image size 224×224.

## Metrics

Macro F1, micro F1, samples F1, and macro mAP evaluated on the held-out test set.

## Dependencies

```
torch torchvision pytorch-lightning scikit-learn matplotlib seaborn pandas numpy
```
