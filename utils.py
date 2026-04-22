from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, average_precision_score, hamming_loss
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import torch

class UCMMultilabelDataset(Dataset):
    """
    UCM multilabel land-use dataset.
 
    Folder layout:
        ucmdata/
            Images/
                agricultural/   agricultural00.tif … agricultural99.tif
                airplane/       airplane00.tif      … airplane99.tif
                …               (21 subfolders, already in correct order)
            LandUse_Multilabeled.txt
 
    Label file format (tab-separated):
        IMAGE\LABEL   airplane   bare-soil   buildings   …   water
        agricultural00    0           0            0      …     0
        …
 
    Strategy
    --------
    1. Parse the label file → class names from the header (cols 1…end,
       skipping "IMAGE\LABEL") + label matrix (N × C).
    2. Walk ucmdata/Images/ subfolder-by-subfolder in sorted order, collecting
       every image path into a flat list — same traversal order as the txt file.
    3. Pair image_paths[i]  ↔  label_matrix[i]  by position (no name matching).
    """
 
    def __init__(
        self,
        root_dir: str = "ucmdata",
        label_file: str = "LandUse_Multilabeled.txt",
        transform=None,
        image_ext: str = ".tif",
    ):
        self.root_dir   = root_dir
        self.images_dir = os.path.join(root_dir, "Images")
        self.transform  = transform
        self.image_ext  = image_ext
 
        # 1. Parse label file 
        label_path = os.path.join(root_dir, label_file)
        self.class_names, self.label_matrix = self._parse_labels(label_path)
        self.num_classes = len(self.class_names)
 
        # 2. Collect image paths in sorted subfolder order 
        self.image_paths = self._collect_image_paths()
 
        # 3. Sanity check
        assert len(self.image_paths) == len(self.label_matrix), (
            f"Mismatch: {len(self.image_paths)} images found but "
            f"{len(self.label_matrix)} label rows in the txt file."
        )
 
    # ------------------------------------------------------------------ 
    def _parse_labels(self, label_path: str):
        """
        Returns:
            class_names  – list[str], length C  (column headers, cols 1…end)
            label_matrix – torch.FloatTensor, shape (N, C)
        """
        with open(label_path, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
 
        # Header row → class names (skip the first column "IMAGE\LABEL")
        header      = lines[0].split("\t")
        class_names = header[1:]          # ['airplane', 'bare-soil', …, 'water']
 
        # Data rows → label matrix (parts[0] is the image name, ignored here)
        rows = []
        for line in lines[1:]:
            parts = line.split("\t")
            label_vals = list(map(int, parts[1:]))
            rows.append(label_vals)
 
        label_matrix = torch.tensor(rows, dtype=torch.float32)  # (N, C)
        return class_names, label_matrix
 
    # ------------------------------------------------------------------ 
    def _collect_image_paths(self) -> list:
        """
        Walks ucmdata/Images/ subfolder-by-subfolder in sorted order.
        Within each subfolder images are also sorted — matching the txt order.
        Returns a flat list of absolute image paths.
        """
        image_paths = []
 
        subfolders = sorted( entry.name for entry in os.scandir(self.images_dir)) 
        
 
        for subfolder in subfolders:
            folder_path = os.path.join(self.images_dir, subfolder)
            files = sorted(
                fname
                for fname in os.listdir(folder_path)
                if fname.lower().endswith(self.image_ext)
            )
            for fname in files:
                image_paths.append(os.path.join(folder_path, fname))
 
        return image_paths
 
    # ------------------------------------------------------------------ 
    def __len__(self) -> int:
        return len(self.image_paths)
 
    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        labels   = self.label_matrix[idx]          # float32 tensor, shape (C,)
 
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
 
        return image, labels
 
    # Utility helpers

    def get_class_names(self) -> list:
        return self.class_names
 
    def get_class_weights(self) -> torch.Tensor:
        """
        Inverse-frequency pos_weight per class for BCEWithLogitsLoss. Combines a Sigmoid layer and the Binary Cross Entropy (BCE) loss into one single class
        Shape: (num_classes,)
        """
        pos = self.label_matrix.sum(dim=0).clamp(min=1)
        neg = (len(self) - self.label_matrix.sum(dim=0)).clamp(min=1)
        return neg / pos
 
    def verify_alignment(self, n_samples: int = 10):
        """
        Prints the first n_samples rows so you can visually cross-check
        image names against the txt file.
        """
        print(f"{'Image path':<55}  {'Active labels'}")
        print("-" * 80)
        for i in range(min(n_samples, len(self))):
            active = [self.class_names[j]
                      for j, v in enumerate(self.label_matrix[i]) if v == 1]
            short = os.path.join(*self.image_paths[i].split(os.sep)[-2:])
            print(f"{short:<55}  {active}")
 

## Augmentation

def get_transforms(split: str = "train", image_size: tuple = (224, 224)):
    """Standard augmentation."""
    if split == "train":
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(45),
            transforms.ColorJitter(brightness=0.2, contrast=0.4, saturation=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], # to prevent exploding gradients 
                                std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225]),
        ])
 
 
### Dataloader and data splitting
def build_dataloaders(
    root_dir:    str   = "ucmdata",
    label_file:  str   = "LandUse_Multilabeled.txt",
    image_size:  tuple = (224, 224), # Default but needs to be specified! 
    batch_size:  int   = 32,
    num_workers: int   = 2,
    val_frac:    float = 0.15,
    test_frac:   float = 0.15,
    seed:        int   = 42,
    image_ext:   str   = ".tif",
):
    """
    Returns (train_loader, val_loader, test_loader, class_names, pos_weights).
 
    Usage
    -----
    train_loader, val_loader, test_loader, classes, pos_w = build_dataloaders()
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
    """

    full_ds = UCMMultilabelDataset(
        root_dir=root_dir, label_file=label_file,
        transform=None, image_ext=image_ext,
    )
 
    n = len(full_ds)
    labels_array = full_ds.label_matrix.numpy()  # Convert to numpy
    
    # First split: train+val vs test
    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=test_frac, random_state=seed
    )
    train_val_idx, test_idx = next(splitter.split(np.zeros(n), labels_array))
    
    # Second split: train vs val from train+val
    train_val_labels = labels_array[train_val_idx]
    val_frac_adjusted = val_frac / (1 - test_frac)  # Adjust fraction for remaining data
    
    splitter2 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=val_frac_adjusted, random_state=seed
    )
    train_idx_local, val_idx_local = next(splitter2.split(
        np.zeros(len(train_val_idx)), train_val_labels
    ))
    
    # Map back to original indices
    train_idx = train_val_idx[train_idx_local]
    val_idx = train_val_idx[val_idx_local]
    
    def make_subset(split_name, indices):
        ds = UCMMultilabelDataset(
            root_dir=root_dir, label_file=label_file,
            transform=get_transforms(split_name, image_size), image_ext=image_ext,
        )
        return Subset(ds, indices)
    
    train_ds = make_subset("train", train_idx)
    val_ds   = make_subset("val",   val_idx)
    test_ds  = make_subset("test",  test_idx)
    
    kw = dict(batch_size=batch_size, num_workers=num_workers,
              pin_memory=torch.cuda.is_available())
 
    return (
        DataLoader(train_ds, shuffle=True,  **kw),
        DataLoader(val_ds,   shuffle=False, **kw),
        DataLoader(test_ds,  shuffle=False, **kw),
        full_ds.get_class_names(),
        full_ds.get_class_weights(),
    )

 
### Lightning Module Pretrained model and Head for multilabel classification
import torch.nn as nn
import torchmetrics
import pytorch_lightning as L

class LightningModuleMultilabel(L.LightningModule):
    def __init__(self, model, num_classes, lr=1e-4, weight_decay=1e-4,
                 max_epochs=15, threshold=0.5, pos_weight=None):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        pos_weight = torch.as_tensor(pos_weight, dtype=torch.float32) if pos_weight is not None else None

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.threshold = threshold

        mk = dict(task="multilabel", num_labels=num_classes, threshold=threshold)
        self.train_f1 = torchmetrics.F1Score(average="macro", **mk)
        self.val_f1   = torchmetrics.F1Score(average="macro", **mk)
        self.test_f1  = torchmetrics.F1Score(average="macro", **mk)
        self.val_acc  = torchmetrics.Accuracy(average="macro", **mk)
        self.test_acc = torchmetrics.Accuracy(average="macro", **mk)
        self.val_map  = torchmetrics.AveragePrecision(task="multilabel", num_labels=num_classes, average="macro")
        self.test_map = torchmetrics.AveragePrecision(task="multilabel", num_labels=num_classes, average="macro")


    def forward(self, x):
        return self.model(x)

    def _step(self, batch):
        imgs, labels = batch          # labels: float tensor (B, C)
        logits = self(imgs)
        loss = self.criterion(logits, labels)
        return loss, logits, labels

    def training_step(self, batch, batch_idx):
        loss, logits, labels = self._step(batch)
        self.train_f1.update(logits, labels.int())
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_f1",   self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, labels = self._step(batch)
        self.val_f1.update(logits, labels.int())
        self.val_acc.update(logits, labels.int())
        self.val_map.update(logits, labels.int())
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1",   self.val_f1,  on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc",  self.val_acc, on_step=False, on_epoch=True)
        self.log("val_map",  self.val_map, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        loss, logits, labels = self._step(batch)
        self.test_f1.update(logits, labels.int())
        self.test_acc.update(logits, labels.int())
        self.test_map.update(logits, labels.int())
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_f1",   self.test_f1,  on_step=False, on_epoch=True)
        self.log("test_acc",  self.test_acc, on_step=False, on_epoch=True)
        self.log("test_map",  self.test_map, on_step=False, on_epoch=True)

    def predict_step(self, batch, batch_idx):
        imgs, labels = batch
        logits = self(imgs)
        probs  = torch.sigmoid(logits)
        preds  = (probs >= self.threshold).int()
        return {"probs": probs, "preds": preds, "labels": labels.int()}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.hparams.lr,
                                      weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs)
        return [optimizer], [scheduler]
 

### EVALUATIONS and VISUALIZATIONS 
from sklearn.metrics import f1_score, average_precision_score, hamming_loss
import matplotlib.patches as mpatches
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


def save_model_results(test_probs, test_preds, test_labels, classes, pretrained_model):
    output_df = pd.DataFrame(test_probs, columns=[f"{c}_prob" for c in classes])
    output_df[[f"{c}_pred" for c in classes]] = test_preds
    output_df[[f"{c}_label" for c in classes]] = test_labels
    output_df.to_csv(f"outputs/{pretrained_model}_multilabel_test_results.csv", index=False)


def compute_test_metrics(test_preds, test_labels, test_probs):
    """
    Compute multilabel classification metrics on test data.
    
    Args:
        test_preds: Predicted labels, shape (N, C)
        test_labels: Ground truth labels, shape (N, C)
        test_probs: Predicted probabilities, shape (N, C)
    
    Returns:
        dict: Dictionary containing all computed metrics
    """
    
    metrics = {
        "accuracy": (test_preds == test_labels).mean(),
        "macro_f1": f1_score(test_labels, test_preds, average="macro", zero_division=0),
        "micro_f1": f1_score(test_labels, test_preds, average="micro", zero_division=0),
        "samples_f1": f1_score(test_labels, test_preds, average="samples", zero_division=0),
        "macro_map": average_precision_score(test_labels, test_probs, average="macro"),
        "hamming_loss": hamming_loss(test_labels, test_preds),
        "subset_acc": (test_preds == test_labels).all(axis=1).mean(),
    }
    
    return metrics


def append_metrics_to_csv(metrics, model_name: str, csv_path="outputs/ModelComparisons.csv"):
    results_df = pd.DataFrame({
        "model": [model_name],
        "accuracy": [metrics['accuracy']],
        "macro_f1": [metrics['macro_f1']],
        "micro_f1": [metrics['micro_f1']],
        "samples_f1": [metrics['samples_f1']],
        "macro_map": [metrics['macro_map']],
        "hamming_loss": [metrics['hamming_loss']],
        "subset_acc": [metrics['subset_acc']]
    })
    results_csv_path = csv_path
    if os.path.exists(results_csv_path):
        results_df.to_csv(results_csv_path, mode='a', header=False, index=False)
    else:
        results_df.to_csv(results_csv_path, index=False)


def plot_training_curves(csv_logger, model_name: str = "ResNet-50 multilabel", save_path: str = None):
    """
    Plot training and validation loss/F1 curves from CSV logger.
    
    Args:
        csv_logger: Lightning CSVLogger instance
        model_name: Name for the plot title
        save_path: Optional path to save the figure (e.g., "outputs/curves.png")
    
    Returns:
        tuple: (fig, axes) matplotlib objects
    """
    metrics_csv = Path(csv_logger.log_dir) / "metrics.csv"
    df = pd.read_csv(metrics_csv)
    epoch_df = df.groupby("epoch").mean(numeric_only=True).reset_index()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss subplot
    if "train_loss" in epoch_df.columns:
        axes[0].plot(epoch_df["epoch"], epoch_df["train_loss"], label="train", marker="o", markersize=3)
    if "val_loss" in epoch_df.columns:
        axes[0].plot(epoch_df["epoch"], epoch_df["val_loss"], label="val", marker="s", markersize=3)
    axes[0].set_xlabel("epoch")
    axes[0].set_xlim(0, 20)
    axes[0].set_ylabel("BCE loss")
    axes[0].set_ylim(0, 1)
    axes[0].set_title(f"{model_name} — Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # F1 subplot
    if "train_f1" in epoch_df.columns:
        axes[1].plot(epoch_df["epoch"], epoch_df["train_f1"], label="train macro-F1", marker="o", markersize=3)
    if "val_f1" in epoch_df.columns:
        axes[1].plot(epoch_df["epoch"], epoch_df["val_f1"], label="val macro-F1", marker="s", markersize=3)
    axes[1].set_xlabel("epoch")
    axes[1].set_xlim(0, 20)
    axes[1].set_ylabel("macro F1")
    axes[1].set_ylim(0, 1)
    axes[1].set_title(f"{model_name} — macro F1")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved training curves to: {save_path}")
    
    plt.show()
    return fig, axes


def plot_prediction_grid(test_preds, test_labels, test_probs, classes, 
                         test_loader, root_dir: str = "ucmdata", 
                         label_file: str = "LandUse_Multilabeled.txt",
                         n_show: int = 9, seed: int = 4,
                         save_path: str = None):
    """
    Plot a grid of test images with ground-truth vs predicted labels.
    
    Returns:
        fig: matplotlib figure object
    """

    
    # Load full dataset to access image paths
    test_ds_full = UCMMultilabelDataset(root_dir=root_dir, label_file=label_file, 
                                        transform=None)
    test_ds_subset = test_loader.dataset
    test_indices = test_ds_subset.indices
    test_items = [(test_ds_full.image_paths[idx], None) for idx in test_indices]
    
    # Setup grid
    cols = 3
    rows = (n_show + cols - 1) // cols
    
    # Pick images with 3 or more classes
    np.random.seed(seed)
    multi_class_indices = [i for i in range(len(test_labels)) if test_labels[i].sum() >= 3]
    chosen = np.random.choice(multi_class_indices, size=min(n_show, len(multi_class_indices)), replace=False)

    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 5.2))
    axes = axes.flatten()
    
    # Plot each sample
    for ax, sample_i in zip(axes, chosen):
        rel_path, _ = test_items[sample_i]
        img = Image.open(rel_path).convert("RGB")
        ax.imshow(img)
        ax.axis("off")
        
        gt = test_labels[sample_i].astype(bool)
        pred = test_preds[sample_i].astype(bool)
        probs = test_probs[sample_i]
        
        tp = gt & pred
        fp = ~gt & pred
        fn = gt & ~pred
        
        # Title with counts
        fname = Path(rel_path).name
        ax.set_title(f"{fname}\nTP={tp.sum()}  FP={fp.sum()}  FN={fn.sum()}",
                     fontsize=10, loc="right")
        
        # Build colored label captions
        lines = []
        for i in np.where(tp)[0]:
            lines.append(("green", f"✓ {classes[i]}({probs[i]:.2f})"))
        for i in np.where(fn)[0]:
            lines.append(("red", f"✗ {classes[i]}({probs[i]:.2f})  [missed]"))
        for i in np.where(fp)[0]:
            lines.append(("orange", f"+ {classes[i]}({probs[i]:.2f})  [extra]"))
        
        # Render colored text
        for j, (color, text) in enumerate(lines):
            ax.text(0.0, -0.04 - 0.06 * j, text, transform=ax.transAxes,
                    fontsize=9, color=color, va="top", ha="left", family="monospace")
    
    # Hide unused axes
    for ax in axes[len(chosen):]:
        ax.axis("off")
    
    # Add legend
    legend_handles = [
        mpatches.Patch(color="green", label="TP — correctly predicted"),
        mpatches.Patch(color="red", label="FN — missed (in GT, not predicted)"),
        mpatches.Patch(color="orange", label="FP — extra (predicted, not in GT)"),
    ]
    fig.legend(handles=legend_handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved prediction grid to: {save_path}")
    
    plt.show()
    return fig


def plot_exact_match_by_class_count(preds, labels, model_name="Model",
                                     color="steelblue", ax=None, save_path=None):
    """
    Calcula y grafica el Exact Match Accuracy agrupado por número de
    clases presentes por imagen.

    Args:
        preds:      np.ndarray (N, C) — predicciones binarias
        labels:     np.ndarray (N, C) — etiquetas ground truth
        model_name: str — nombre del modelo para el título
        color:      str — color de las barras
        ax:         matplotlib Axes (opcional, para subplots)
        save_path:  str — ruta para guardar la figura (opcional)
    """
    # Número de clases presentes por imagen (suma de la fila)
    classes_per_image = labels.sum(axis=1).astype(int)   # shape (N,)

    # Exact match por imagen: 1 si toda la fila coincide exactamente
    exact_match = (preds == labels).all(axis=1).astype(int)  # shape (N,)

    # Agrupar por número de clases
    unique_counts = sorted(set(classes_per_image))
    em_per_count  = []
    n_per_count   = []

    for k in unique_counts:
        mask = classes_per_image == k
        em_per_count.append(exact_match[mask].mean())
        n_per_count.append(mask.sum())

    # ── Plot ─────────────────────────────────────────────────────────────────
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(9, 4))

    bars = ax.bar(unique_counts, em_per_count, color=color, alpha=0.85, edgecolor="white")

    # Anotar cada barra con el número de muestras del grupo
    for bar, n in zip(bars, n_per_count):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"n={n}",
            ha="center", va="bottom", fontsize=8, color="gray"
        )

    ax.set_xlabel("Number of classes per image", fontsize=11)
    ax.set_ylabel("Exact Match Accuracy", fontsize=11)
    ax.set_title(f"{model_name} — Exact Match Acc. by class count", fontsize=12)
    ax.set_xticks(unique_counts)
    ax.set_ylim(0, 1.15)
    ax.axhline(exact_match.mean(), color="red", linestyle="--", linewidth=1.2,
               label=f"Overall EM acc = {exact_match.mean():.3f}")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    if standalone:
        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved to: {save_path}")
        try:
            from IPython.display import display
            display(fig)
        except Exception:
            plt.show()
        plt.close(fig)
        return fig, ax

    return ax


def plot_per_class_metrics_comparison(
    test_labels1, test_preds1, test_probs1, 
    test_labels2, test_preds2, test_probs2,
    class_counts_df,
    model_name1: str = "Model 1",
    model_name2: str = "Model 2",
    save_path: str = None):
    """
    Plot per-class F1 and AP metrics for 2 models with grouped bars (separate figures).

    Returns:
        tuple: (fig_f1, ax_f1, fig_ap, ax_ap, summary_df) with comparison results
    """
    
    # Compute per-class metrics for both models
    per_class_f1_1 = f1_score(test_labels1, test_preds1, average=None, zero_division=0)
    per_class_ap_1 = average_precision_score(test_labels1, test_probs1, average=None)
    
    per_class_f1_2 = f1_score(test_labels2, test_preds2, average=None, zero_division=0)
    per_class_ap_2 = average_precision_score(test_labels2, test_probs2, average=None)
    
    classes = class_counts_df["Class"].tolist()
    num_classes = len(classes)
    
    # Create a mapping of class names to indices
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    # Sort by counts from class_counts_df (descending)
    class_counts_df_sorted = class_counts_df.sort_values("Count", ascending=False).reset_index(drop=True)
    
    # Get sorted indices based on sorted class names
    sorted_indices = [class_to_idx[c] for c in class_counts_df_sorted["Class"]]
    
    classes_sorted = class_counts_df_sorted["Class"].tolist()
    counts_sorted = class_counts_df_sorted["Count"].values
    f1_1_sorted = per_class_f1_1[sorted_indices]
    ap_1_sorted = per_class_ap_1[sorted_indices]
    f1_2_sorted = per_class_f1_2[sorted_indices]
    ap_2_sorted = per_class_ap_2[sorted_indices]
    
    x = np.arange(num_classes)
    width = 0.35  # Width of each bar
    
    # Figure 1: F1 Scores
    fig_f1, ax_f1 = plt.subplots(figsize=(max(12, 0.8 * num_classes), 6))
    ax_f1.bar(x - width/2, f1_1_sorted, width, label=f"{model_name1} F1", color="lightblue")
    ax_f1.bar(x + width/2, f1_2_sorted, width, label=f"{model_name2} F1", color="lightsalmon")
    
    # Add count labels on top of bars
    for i, count in enumerate(counts_sorted):
        ax_f1.text(i, 1.05, f"n={int(count)}", ha="center", va="bottom", fontsize=8, color="black")
    
    ax_f1.set_xticks(x)
    ax_f1.set_xticklabels(classes_sorted, rotation=45, ha="right")
    ax_f1.set_ylim(0, 1.15)
    ax_f1.set_ylabel("F1 Score")
    ax_f1.set_title(f"{model_name1} vs {model_name2} — Per-class F1 Comparison (sorted by sample count)")
    ax_f1.legend(loc="lower right", fontsize=10)
    ax_f1.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        f1_path = save_path.replace(".png", "_f1.png")
        Path(f1_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(f1_path, dpi=150, bbox_inches="tight")
        print(f"Saved F1 comparison plot to: {f1_path}")
    
    plt.show()
    
    # Figure 2: AP Scores
    fig_ap, ax_ap = plt.subplots(figsize=(max(12, 0.8 * num_classes), 6))
    ax_ap.bar(x - width/2, ap_1_sorted, width, label=f"{model_name1} AP", color="lightblue")
    ax_ap.bar(x + width/2, ap_2_sorted, width, label=f"{model_name2} AP", color="lightsalmon")
    
    # Add count labels on top of bars
    for i, count in enumerate(counts_sorted):
        ax_ap.text(i, 1.05, f"n={int(count)}", ha="center", va="bottom", fontsize=8, color="black")
    
    ax_ap.set_xticks(x)
    ax_ap.set_xticklabels(classes_sorted, rotation=45, ha="right")
    ax_ap.set_ylim(0, 1.15)
    ax_ap.set_ylabel("Average Precision")
    ax_ap.set_title(f"{model_name1} vs {model_name2} — Per-class AP Comparison (sorted by sample count)")
    ax_ap.legend(loc="lower right", fontsize=10)
    ax_ap.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        ap_path = save_path.replace(".png", "_ap.png")
        Path(ap_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(ap_path, dpi=150, bbox_inches="tight")
        print(f"Saved AP comparison plot to: {ap_path}")
    
    plt.show()
    

    
    return fig_f1, ax_f1, fig_ap, ax_ap