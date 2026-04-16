from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import pandas as pd
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
 
        # ── 1. Parse label file ──────────────────────────────────────────
        label_path = os.path.join(root_dir, label_file)
        self.class_names, self.label_matrix = self._parse_labels(label_path)
        self.num_classes = len(self.class_names)
 
        # ── 2. Collect image paths in sorted subfolder order ─────────────
        self.image_paths = self._collect_image_paths()
 
        # ── 3. Sanity check ──────────────────────────────────────────────
        assert len(self.image_paths) == len(self.label_matrix), (
            f"Mismatch: {len(self.image_paths)} images found but "
            f"{len(self.label_matrix)} label rows in the txt file."
        )
 
    # ------------------------------------------------------------------ #
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
 
    # ------------------------------------------------------------------ #
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
 
    # ------------------------------------------------------------------ #
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

def get_transforms(split: str = "train", image_size: tuple = IMAGE_SIZE):
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
    image_size:  tuple = IMAGE_SIZE,
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
 
 
 

### EVALUATIONS and VISUALIZATIONS 



def evaluate_multilabel_model(trainer, lit_model, test_loader, checkpoint_cb):
    """
    Evaluates a multilabel classification model on test data.
    
    Args:
        trainer: Lightning Trainer instance
        lit_model: LightningModule model
        test_loader: DataLoader for test set
        checkpoint_cb: ModelCheckpoint callback with best model path
    
    Returns:
        dict: Dictionary containing test metrics and predictions
    """
    
    # Load best checkpoint
    trainer.test(lit_model, dataloaders=test_loader, ckpt_path="best")
    best_path = checkpoint_cb.best_model_path
    lit_model = LitResNetMultilabel.load_from_checkpoint(best_path, model=lit_model.model)
    
    # Collect predictions
    preds_out = trainer.predict(lit_model, dataloaders=test_loader)
    test_probs = torch.cat([b["probs"] for b in preds_out], dim=0).cpu().numpy()
    test_preds = torch.cat([b["preds"] for b in preds_out], dim=0).cpu().numpy()
    test_labels = torch.cat([b["labels"] for b in preds_out], dim=0).cpu().numpy()
    
    # Compute metrics
    metrics = {
        "macro_f1": f1_score(test_labels, test_preds, average="macro", zero_division=0),
        "micro_f1": f1_score(test_labels, test_preds, average="micro", zero_division=0),
        "samples_f1": f1_score(test_labels, test_preds, average="samples", zero_division=0),
        "macro_map": average_precision_score(test_labels, test_probs, average="macro"),
        "hamming_loss": hamming_loss(test_labels, test_preds),
        "subset_acc": (test_preds == test_labels).all(axis=1).mean(),
    }
    
    # Print metrics
    print(f"\nTest macro F1   : {metrics['macro_f1']:.4f}")
    print(f"Test micro F1   : {metrics['micro_f1']:.4f}")
    print(f"Test samples F1 : {metrics['samples_f1']:.4f}")
    print(f"Test macro mAP  : {metrics['macro_map']:.4f}")
    print(f"Hamming loss    : {metrics['hamming_loss']:.4f}")
    print(f"Exact-match acc : {metrics['subset_acc']:.4f}")
    
    return {
        "metrics": metrics,
        "test_probs": test_probs,
        "test_preds": test_preds,
        "test_labels": test_labels,
    }


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
    axes[0].set_ylabel("BCE loss")
    axes[0].set_title(f"{model_name} — Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # F1 subplot
    if "train_f1" in epoch_df.columns:
        axes[1].plot(epoch_df["epoch"], epoch_df["train_f1"], label="train macro-F1", marker="o", markersize=3)
    if "val_f1" in epoch_df.columns:
        axes[1].plot(epoch_df["epoch"], epoch_df["val_f1"], label="val macro-F1", marker="s", markersize=3)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("macro F1")
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


def plot_per_class_metrics(test_labels, test_preds, test_probs, classes, 
                           macro_f1=None, macro_map=None, 
                           model_name: str = "Model", 
                           save_path: str = None, 
                           csv_output: str = None):
    """
    Plot per-class F1 and Average Precision metrics with a summary table.
    
    Args:
        test_labels: Ground truth labels, shape (N, C)
        test_preds: Predicted labels, shape (N, C)
        test_probs: Predicted probabilities, shape (N, C)
        classes: List of class names
        macro_f1: Optional macro F1 score for title
        macro_map: Optional macro mAP score for title
        model_name: Name for the plot title
        save_path: Optional path to save the figure (e.g., "outputs/per_class.png")
        csv_output: Optional path to save the summary table as CSV
    
    Returns:
        tuple: (fig, ax, summary_df) matplotlib objects and summary DataFrame
    """
    from sklearn.metrics import f1_score, average_precision_score
    
    # Compute per-class metrics
    per_class_f1 = f1_score(test_labels, test_preds, average=None, zero_division=0)
    per_class_ap = average_precision_score(test_labels, test_probs, average=None)
    positives = test_labels.sum(axis=0).astype(int)
    num_classes = len(classes)
    
    # Sort by F1 score (ascending, weakest first)
    order = np.argsort(per_class_f1)
    classes_sorted = [classes[i] for i in order]
    f1_sorted = per_class_f1[order]
    ap_sorted = per_class_ap[order]
    pos_sorted = positives[order]
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * num_classes)))
    y = np.arange(num_classes)
    ax.barh(y - 0.2, f1_sorted, height=0.4, label="F1")
    ax.barh(y + 0.2, ap_sorted, height=0.4, label="AP")
    ax.set_yticks(y)
    ax.set_yticklabels(classes_sorted)
    ax.set_xlim(0, 1)
    ax.set_xlabel("score")
    
    # Build title with optional macro metrics
    title = f"{model_name} — per-class F1 & AP"
    if macro_f1 is not None and macro_map is not None:
        title += f"  (macro F1 = {macro_f1:.3f}, macro mAP = {macro_map:.3f})"
    ax.set_title(title)
    
    # Add sample counts
    for i, (f, a, p) in enumerate(zip(f1_sorted, ap_sorted, pos_sorted)):
        ax.text(max(f, a) + 0.01, i, f"n={p}", va="center", fontsize=8, color="gray")
    
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved per-class metrics plot to: {save_path}")
    
    plt.show()
    
    # Create summary DataFrame
    summary = pd.DataFrame({
        "class": classes_sorted,
        "positives": pos_sorted,
        "F1": f1_sorted.round(4),
        "AP": ap_sorted.round(4),
    })
    
    print("\nPer-class Summary:")
    print(summary.to_string(index=False))
    
    # Save CSV if requested
    if csv_output:
        Path(csv_output).parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(csv_output, index=False)
        print(f"Saved summary table to: {csv_output}")
    
    return fig, ax, summary

