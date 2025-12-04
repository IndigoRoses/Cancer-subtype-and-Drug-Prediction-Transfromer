#!/usr/bin/env python3
"""
Train PathOmics model for BOTH cervical cancer tasks:
1. Treatment outcome prediction
2. Histological subtype classification

Usage:
    python train_pathomics_both_tasks.py --task treatment      # Treatment outcome
    python train_pathomics_both_tasks.py --task subtype        # Subtype classification
    python train_pathomics_both_tasks.py --task treatment --no-ph   # Without PH
"""

import os
import sys
import json
import random
import pickle
import argparse
import numpy as np
import pandas as pd
from glob import glob
from functools import reduce
from sklearn.utils import resample

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Add PathOmics to path
sys.path.append("/users/chantelle/PathOmics/PathOmics/")
from PathOmics_Survival_model import PathOmics_Surv
from train_and_eval_utils_core import train_and_evaluate

# -------------------------
# Parse Arguments
# -------------------------
parser = argparse.ArgumentParser(description='Train PathOmics for treatment outcome or subtype')
parser.add_argument('--task', type=str, required=True, choices=['treatment', 'subtype'],
                   help='Task: treatment outcome or subtype classification')
parser.add_argument('--no-ph', action='store_true', help='Train without PH features')
parser.add_argument('--eval-only', action='store_true', help='Skip training, only evaluate')
parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
args = parser.parse_args()

TASK = args.task
USE_PH_FEATURES = not args.no_ph
EVAL_ONLY = args.eval_only
EPOCHS = args.epochs
SEED = args.seed

# Set random seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -------------------------
# CONFIG
# -------------------------
ROOT = "/scratch3/users/chantelle/tcga_cesc_data/processed"
WSI_EMB_DIR = os.path.join(ROOT, "wsi_patch_embeddings")
CNV_PCA = os.path.join(ROOT, "integration", "tcga_cesc_cnv_pca50.tsv")
RNA_PCA = os.path.join(ROOT, "integration", "tcga_cesc_rna_pca50.tsv")
SNV_PCA = os.path.join(ROOT, "integration", "tcga_cesc_snv_pca25.tsv")
METHYL_PCA = os.path.join(ROOT, "integration", "tcga_cesc_methylation_pca50.tsv")
CLINICAL = os.path.join(ROOT, "clinical_outcome.tsv")
PH_FEATURES = os.path.join(ROOT, "ph_features", "ph_features.tsv")

# Output directory based on task
task_suffix = "treatment" if TASK == "treatment" else "subtype"
ph_suffix = "_with_ph" if USE_PH_FEATURES else "_no_ph"
OUTDIR = os.path.join(ROOT, f"training_out_{task_suffix}{ph_suffix}")
os.makedirs(OUTDIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("\n" + "="*70)
print("PATHOMICS TRAINING CONFIGURATION")
print("="*70)
print(f"Task: {TASK.upper()}")
print(f"Device: {device}")
print(f"PH features: {'YES' if USE_PH_FEATURES else 'NO'}")
print(f"Eval only: {EVAL_ONLY}")
print(f"Epochs: {EPOCHS}")
print(f"Output: {OUTDIR}")
print("="*70 + "\n")

# -------------------------
# Load data
# -------------------------
def read_pca(path):
    if not os.path.exists(path):
        print(f"  Warning: {os.path.basename(path)} not found")
        return None
    df = pd.read_csv(path, sep="\t", engine="python")
    if "case_id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "case_id"})
    print(f"  Loaded: {os.path.basename(path)}")
    return df

print("Loading omics data...")
cnv_pca = read_pca(CNV_PCA)
rna_pca = read_pca(RNA_PCA)
snv_pca = read_pca(SNV_PCA)
methyl_pca = read_pca(METHYL_PCA)
clin = pd.read_csv(CLINICAL, sep="\t", engine="python")
print(f"  Loaded: clinical_outcome.tsv")

# Load PH features if requested
ph_feats = None
if USE_PH_FEATURES and os.path.exists(PH_FEATURES):
    ph_feats = read_pca(PH_FEATURES)
    ph_cols = [c for c in ph_feats.columns if c != 'case_id' and c != 'n_patches']
    ph_feats = ph_feats.rename(columns={c: f"PH_{c}" for c in ph_cols})
    print(f"  Loaded: PH features ({len(ph_cols)} features)")
elif USE_PH_FEATURES:
    print(f"  Warning: PH features not found at {PH_FEATURES}")
    USE_PH_FEATURES = False

# Merge data
dfs = [df for df in [cnv_pca, rna_pca, snv_pca, methyl_pca, clin] if df is not None]
if USE_PH_FEATURES and ph_feats is not None:
    dfs.append(ph_feats)
meta = reduce(lambda a, b: pd.merge(a, b, on="case_id", how="inner"), dfs)
print(f"\nMerged dataset: {meta.shape}")

# -------------------------
# Attach WSI embeddings
# -------------------------
def find_wsi(case_id):
    matches = glob(os.path.join(WSI_EMB_DIR, f"{case_id}*.npy"))
    return matches[0] if matches else None

meta["wsi_emb_path"] = meta["case_id"].apply(find_wsi)
meta = meta.dropna(subset=["wsi_emb_path"]).reset_index(drop=True)
print(f"After WSI attach: {meta.shape}")

# -------------------------
# Task-specific filtering and labeling
# -------------------------
if TASK == "treatment":
    # Filter for treatment outcome
    meta = meta.dropna(subset=["treatment_outcome"])
    meta = meta[meta["treatment_outcome"].str.strip() != ""].reset_index(drop=True)
    
    # Encode treatment outcome as label
    label_encoder = {lbl: i for i, lbl in enumerate(sorted(meta["treatment_outcome"].unique()))}
    meta["label"] = meta["treatment_outcome"].map(label_encoder)
    
    # Process treatment type for auxiliary features
    def simplify_treatment_type(x):
        if pd.isna(x):
            return "Unknown"
        x = x.lower()
        if "radiation" in x:
            return "Radiation"
        elif "chemo" in x:
            return "Chemotherapy"
        elif "hysterectomy" in x or "surgery" in x:
            return "Surgery"
        elif "pharma" in x:
            return "Pharmaceutical"
        else:
            return "Other"
    
    meta["treatment_type_clean"] = meta["treatment_type"].apply(simplify_treatment_type)
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    treat_type_encoded = enc.fit_transform(meta[["treatment_type_clean"]])
    treat_type_cols = [f"treat_{c}" for c in enc.categories_[0]]
    treat_type_df = pd.DataFrame(treat_type_encoded, columns=treat_type_cols)
    meta = pd.concat([meta, treat_type_df], axis=1)
    
    print(f"\nTask: Treatment Outcome Prediction")
    print(f"After filtering: {meta.shape}")
    print(f"Classes: {label_encoder}")
    
elif TASK == "subtype":
    # Filter for histological subtype
    meta = meta.dropna(subset=["subtype"])
    meta = meta[meta["subtype"].str.strip() != ""].reset_index(drop=True)
    
    # Simplify subtypes first
    def simplify_subtype(x):
        x_lower = x.lower()
        if "squamous" in x_lower:
            return "Squamous"
        elif "adenocarcinoma" in x_lower or "adeno" in x_lower:
            return "Adenocarcinoma"
        else:
            return "Other"
    
    meta["subtype_simplified"] = meta["subtype"].apply(simplify_subtype)
    
    # Check class distribution
    subtype_counts = meta["subtype_simplified"].value_counts()
    print(f"\nInitial subtype distribution:")
    for subtype, count in subtype_counts.items():
        print(f"  {subtype}: {count} samples")

    
    # Oversample Adenocarcinoma moderately
    adeno_count = subtype_counts.get("Adenocarcinoma", 0)
    squamous_count = subtype_counts.get("Squamous", 0)
    target_adeno_count = int(min(squamous_count * 0.6, squamous_count))  # configurable ratio
    if adeno_count < target_adeno_count:
        n_to_add = target_adeno_count - adeno_count
        adeno_df = meta[meta["subtype_simplified"] == "Adenocarcinoma"]
        adeno_oversampled = resample(
            adeno_df,
            replace=True,
            n_samples=n_to_add,
            random_state=SEED
        )
        meta = pd.concat([meta, adeno_oversampled], axis=0).reset_index(drop=True)
        print(f"\n✅ Oversampled Adenocarcinoma from {adeno_count} → {target_adeno_count} samples")
    else:
        print("\nAdenocarcinoma count sufficient; no oversampling applied.")

        # Print final distribution
        subtype_counts = meta["subtype_simplified"].value_counts()
        print(f"\nFinal subtype distribution after oversampling:")
        for subtype, count in subtype_counts.items():
            print(f"  {subtype}: {count} samples")

    # Encode subtype as label
    label_encoder = {lbl: i for i, lbl in enumerate(sorted(meta["subtype_simplified"].unique()))}
    meta["label"] = meta["subtype_simplified"].map(label_encoder)

    treat_type_cols = []  # No treatment type for subtype classification
    
    print(f"\nTask: Histological Subtype Classification")
    print(f"After filtering: {meta.shape}")
    print(f"Classes: {label_encoder}")

# -------------------------
# Build omic groups
# -------------------------
cnv_cols = [c for c in meta.columns if c.startswith("CNV_")]
rna_cols = [c for c in meta.columns if c.startswith("RNA_")]
snv_cols = [c for c in meta.columns if c.startswith("SNV_")]
methyl_cols = [c for c in meta.columns if c.startswith("METHYLATION_")]
ph_cols = [c for c in meta.columns if c.startswith("PH_")]

omic_groups = []
group_names = []

if cnv_cols: 
    omic_groups.append(cnv_cols)
    group_names.append("CNV")
if rna_cols: 
    omic_groups.append(rna_cols)
    group_names.append("RNA")
if snv_cols: 
    omic_groups.append(snv_cols)
    group_names.append("SNV")
if methyl_cols:
    omic_groups.append(methyl_cols)
    group_names.append("Methylation")
if treat_type_cols: 
    omic_groups.append(treat_type_cols)
    group_names.append("Treatment")
if USE_PH_FEATURES and ph_cols: 
    omic_groups.append(ph_cols)
    group_names.append("PH")

print(f"\nOmic groups: {group_names}")
print(f"Omic sizes: {[len(g) for g in omic_groups]}")

# -------------------------
# Dataset class
# -------------------------
class PathomicsDataset(Dataset):
    def __init__(self, df, omic_cols_groups):
        self.df = df.reset_index(drop=True)
        self.omic_groups = omic_cols_groups

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        case_id = row["case_id"]

        # Load WSI embeddings
        emb = np.load(row["wsi_emb_path"])
        
        # Ensure 2D shape
        if len(emb.shape) == 1:
            emb = emb.reshape(1, -1)
        elif len(emb.shape) == 3:
            emb = emb.squeeze(0)
        
        # Downsample to 1024 if needed
        if emb.shape[-1] == 2048:
            emb = emb[:, :1024]
        
        if len(emb.shape) == 1:
            emb = emb.reshape(1, -1)
        
        # Duplicate single patches
        if emb.shape[0] == 1:
            emb = np.repeat(emb, 2, axis=0)
        
        wsi_tensor = torch.tensor(emb, dtype=torch.float32)

        # Omic tensors
        omic_tensors = []
        for cols in self.omic_groups:
            omic_data = row[cols].values.astype(np.float32)
            if len(omic_data.shape) > 1:
                omic_data = omic_data.flatten()
            omic_tensors.append(torch.tensor(omic_data, dtype=torch.float32))

        label = torch.tensor(int(row["label"]), dtype=torch.long)
        c = torch.tensor(0, dtype=torch.long)
        event_time = torch.tensor(0.0, dtype=torch.float32)

        return wsi_tensor, omic_tensors, c, event_time, label, case_id

# -------------------------
# Collate function
# -------------------------
def pathomics_collate_fn(batch):
    assert len(batch) == 1, "batch_size must be 1"
    wsi_tensor, omic_tensors, c, event_time, label, case_id = batch[0]
    
    wsi_batched = wsi_tensor.unsqueeze(0)
    omic_batched = [o.unsqueeze(0) for o in omic_tensors]
    
    return (
        wsi_batched,
        omic_batched,
        c.unsqueeze(0),
        event_time.unsqueeze(0),
        label.unsqueeze(0),
        [case_id]
    )

# -------------------------
# Train/val/test split
# -------------------------
print("\n" + "="*70)
print("DATA SPLIT")
print("="*70)
print(f"Total samples: {len(meta)}")
print(f"Class distribution:")
for cls in sorted(meta["label"].unique()):
    count = (meta["label"] == cls).sum()
    class_name = [k for k, v in label_encoder.items() if v == cls][0]
    print(f"  {class_name} (class {cls}): {count}")

min_class_count = meta["label"].value_counts().min()

if min_class_count < 2:
    print(f"\nWarning: Smallest class has {min_class_count} sample(s). Random split.")
    train_df, test_df = train_test_split(meta, test_size=0.2, random_state=SEED)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=SEED)
elif min_class_count < 5:
    print(f"\nWarning: Small dataset. Combined val/test set.")
    train_df, test_df = train_test_split(meta, test_size=0.25, random_state=SEED, stratify=meta["label"])
    val_df = test_df.copy()
else:
    train_df, test_df = train_test_split(meta, test_size=0.2, random_state=SEED, stratify=meta["label"])
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=SEED, stratify=train_df["label"])

print(f"\nSplit sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

# -------------------------
# Create datasets/loaders
# -------------------------
train_ds = PathomicsDataset(train_df, omic_groups)
val_ds = PathomicsDataset(val_df, omic_groups)
test_ds = PathomicsDataset(test_df, omic_groups)

train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0, collate_fn=pathomics_collate_fn)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=pathomics_collate_fn)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=pathomics_collate_fn)

# -------------------------
# Model setup
# -------------------------
omic_sizes = [len(g) for g in omic_groups]
num_classes = len(label_encoder)

print("\n" + "="*70)
print("MODEL CONFIGURATION")
print("="*70)
print(f"Task: {TASK.upper()}")
print(f"Omic groups: {group_names}")
print(f"Omic sizes: {omic_sizes}")
print(f"Number of classes: {num_classes}")
print(f"Device: {device}")

model = PathOmics_Surv(
    device=device,
    fusion='concat',
    omic_sizes=omic_sizes,
    model_size_wsi='small',
    model_size_omic='multi_omics',
    n_classes=num_classes
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

# Calculate class weights for imbalanced datasets
class_counts = meta["label"].value_counts().sort_index()
total_samples = len(meta)
class_weights = torch.tensor([total_samples / (len(class_counts) * count) 
                              for count in class_counts], dtype=torch.float32).to(device)

print(f"\nClass distribution:")
for cls, count in class_counts.items():
    class_name = [k for k, v in label_encoder.items() if v == cls][0]
    weight = class_weights[cls].item()
    print(f"  {class_name}: {count} samples (weight: {weight:.2f})")

class ClassificationLossWrapper(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    
    def forward(self, hazards, S, Y, c):
        return self.ce_loss(hazards, Y)

criterion = ClassificationLossWrapper(class_weights=class_weights)

# -------------------------
# Train or Evaluate
# -------------------------
save_path = OUTDIR

if EVAL_ONLY:
    print("\n" + "="*70)
    print("EVALUATION-ONLY MODE")
    print("="*70 + "\n")
    
    checkpoint_path = os.path.join(save_path, "s_0_checkpoint.pt")
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    print("✓ Model loaded\n")
    
else:
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70 + "\n")
    
    # Track training metrics
    training_metrics = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epochs': []
    }
    
    try:
        logs, best_c_index, best_epoch = train_and_evaluate(
            fold=0,
            epochs=EPOCHS,
            model=model,
            loader_list=(train_loader, val_loader, test_loader),
            optimizer=optimizer,
            loss_fn=criterion,
            reg_fn=None,
            device=device,
            save_path=save_path,
            lambda_reg=0.0,
            gc=8,
            save_model=True,
            model_mode='classification',
            fold_mode='train_val_test'
        )
        
        print("\n✅ Training complete!")
        
        # Extract metrics from logs if available
        if isinstance(logs, dict):
            if 'train' in logs:
                training_metrics['train_loss'] = logs['train'].get('loss', [])
                training_metrics['train_acc'] = logs['train'].get('accuracy', [])
            if 'val' in logs:
                training_metrics['val_loss'] = logs['val'].get('loss', [])
                training_metrics['val_acc'] = logs['val'].get('accuracy', [])
            if training_metrics['train_loss']:
                training_metrics['epochs'] = list(range(1, len(training_metrics['train_loss']) + 1))
        
        # Save training logs
        logs_file = os.path.join(save_path, "s_0_training_logs.pkl")
        with open(logs_file, 'wb') as f:
            pickle.dump(logs, f)
        print(f"✓ Saved training logs: {logs_file}")
        
        # Ensure checkpoint was saved
        checkpoint_path = os.path.join(save_path, "s_0_checkpoint.pt")
        if not os.path.exists(checkpoint_path):
            print(f"⚠️  train_and_evaluate didn't save checkpoint, saving manually...")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✓ Saved checkpoint: {checkpoint_path}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Save model even if training crashed
        checkpoint_path = os.path.join(save_path, "s_0_checkpoint.pt")
        print(f"Attempting to save model checkpoint anyway...")
        try:
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✓ Saved checkpoint: {checkpoint_path}")
        except:
            print(f"✗ Could not save checkpoint")

# -------------------------
# Test Evaluation
# -------------------------
print("\n" + "="*70)
print("TEST SET EVALUATION")
print("="*70 + "\n")

model.eval()
all_preds = []
all_labels = []
all_probs = []

print(f"Evaluating {len(test_ds)} test samples...")

with torch.no_grad():
    for i in range(len(test_ds)):
        if (i+1) % 10 == 0:
            print(f"  {i+1}/{len(test_ds)}...")
        
        wsi_tensor, omic_tensors, c, event_time, label, case_id = test_ds[i]
        wsi_tensor = wsi_tensor.to(device)
        omic_tensors = [o.to(device) for o in omic_tensors]
        
        dummy_cluster = [0] * len(omic_tensors)
        hazards, S, Y_hat, _, _ = model(
            x_path=wsi_tensor,
            x_omic=omic_tensors,
            x_cluster=dummy_cluster,
            mode='classification'
        )
        
        probs = torch.softmax(hazards, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        all_preds.append(preds.cpu().numpy())
        all_labels.append(label.cpu().numpy().reshape(-1))
        all_probs.append(probs.cpu().numpy())

all_preds = np.concatenate(all_preds).flatten()
all_labels = np.concatenate(all_labels).flatten()
all_probs = np.vstack(all_probs)

test_accuracy = accuracy_score(all_labels, all_preds)
print(f"\n✓ Test Accuracy: {test_accuracy:.4f}")

class_names = [k for k, v in sorted(label_encoder.items(), key=lambda x: x[1])]
print("\nClassification Report:")
report_str = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
print(report_str)

# Get classification report as dict for saving
from sklearn.metrics import classification_report as cr_dict
report_dict = cr_dict(all_labels, all_preds, target_names=class_names, zero_division=0, output_dict=True)

print("\nConfusion Matrix:")
cm = confusion_matrix(all_labels, all_preds)
print(cm)

# Per-class metrics
print("\n" + "="*70)
print("PER-CLASS DETAILED METRICS")
print("="*70)
from sklearn.metrics import precision_recall_fscore_support
precision, recall, f1, support = precision_recall_fscore_support(
    all_labels, all_preds, labels=range(len(class_names)), zero_division=0
)

per_class_metrics = {}
for i, class_name in enumerate(class_names):
    per_class_metrics[class_name] = {
        'precision': float(precision[i]),
        'recall': float(recall[i]),
        'f1_score': float(f1[i]),
        'support': int(support[i])
    }
    print(f"\n{class_name}:")
    print(f"  Precision: {precision[i]:.4f}")
    print(f"  Recall:    {recall[i]:.4f}")
    print(f"  F1-Score:  {f1[i]:.4f}")
    print(f"  Support:   {support[i]}")

# Calculate additional metrics
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
balanced_acc = balanced_accuracy_score(all_labels, all_preds)
kappa = cohen_kappa_score(all_labels, all_preds)

print("\n" + "="*70)
print("ADDITIONAL METRICS")
print("="*70)
print(f"Balanced Accuracy: {balanced_acc:.4f}")
print(f"Cohen's Kappa:     {kappa:.4f}")
print("="*70)

# -------------------------
# Save Results
# -------------------------
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70 + "\n")

# Save predictions
predictions_dict = {
    'y_true': all_labels.tolist(),
    'y_pred': all_preds.tolist(),
    'y_prob': all_probs.tolist(),
    'test_accuracy': float(test_accuracy),
    'balanced_accuracy': float(balanced_acc),
    'cohen_kappa': float(kappa),
    'class_names': class_names,
    'confusion_matrix': cm.tolist(),
    'per_class_metrics': per_class_metrics,
    'classification_report': report_dict
}

pred_file = os.path.join(save_path, "s_0_test_predictions.pkl")
with open(pred_file, 'wb') as f:
    pickle.dump(predictions_dict, f)
print(f"✓ Saved predictions: {pred_file}")

# Save classification report as text
report_file = os.path.join(save_path, "classification_report.txt")
with open(report_file, 'w') as f:
    f.write("="*70 + "\n")
    f.write(f"CLASSIFICATION REPORT - {TASK.upper()}\n")
    f.write("="*70 + "\n\n")
    f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
    f.write(f"Balanced Accuracy: {balanced_acc:.4f}\n")
    f.write(f"Cohen's Kappa: {kappa:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report_str)
    f.write("\n\nConfusion Matrix:\n")
    f.write(str(cm))
    f.write("\n\nPer-Class Metrics:\n")
    for class_name, metrics in per_class_metrics.items():
        f.write(f"\n{class_name}:\n")
        f.write(f"  Precision: {metrics['precision']:.4f}\n")
        f.write(f"  Recall:    {metrics['recall']:.4f}\n")
        f.write(f"  F1-Score:  {metrics['f1_score']:.4f}\n")
        f.write(f"  Support:   {metrics['support']}\n")
print(f"✓ Saved classification report: {report_file}")

# Save confusion matrix as CSV
cm_file = os.path.join(save_path, "confusion_matrix.csv")
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
cm_df.to_csv(cm_file)
print(f"✓ Saved confusion matrix: {cm_file}")

# Save training metrics if available
if 'training_metrics' in locals() and training_metrics['epochs']:
    metrics_file = os.path.join(save_path, "training_metrics.csv")
    metrics_df = pd.DataFrame(training_metrics)
    metrics_df.to_csv(metrics_file, index=False)
    print(f"✓ Saved training metrics: {metrics_file}")

# Save/update experiment summary
summary = {
    "experiment": f"{TASK.upper()}_{'WITH_PH' if USE_PH_FEATURES else 'NO_PH'}",
    "task": TASK,
    "ph_features_used": USE_PH_FEATURES,
    "num_ph_features": len(ph_cols) if USE_PH_FEATURES and 'ph_cols' in locals() else 0,
    "omic_groups": group_names,
    "omic_sizes": omic_sizes,
    "num_classes": num_classes,
    "train_samples": len(train_df),
    "val_samples": len(val_df),
    "test_samples": len(test_df),
    "test_accuracy": float(test_accuracy),
    "balanced_accuracy": float(balanced_acc),
    "cohen_kappa": float(kappa),
    "class_names": class_names,
    "label_encoder": label_encoder,
    "per_class_metrics": per_class_metrics,
    "best_epoch": best_epoch if 'best_epoch' in locals() else 0,
    "best_c_index": float(best_c_index) if 'best_c_index' in locals() and best_c_index != -float('inf') else None
}

summary_file = os.path.join(save_path, "experiment_summary.json")
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"✓ Saved experiment summary: {summary_file}")

# Save model checkpoint if not already saved
checkpoint_path = os.path.join(save_path, "s_0_checkpoint.pt")
if not os.path.exists(checkpoint_path):
    print(f"\n⚠️  Checkpoint not found, saving current model state...")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"✓ Saved checkpoint: {checkpoint_path}")
else:
    print(f"✓ Checkpoint already exists: {checkpoint_path}")

print("\n" + "="*70)
print("COMPLETE!")
print("="*70)
print(f"\nTask: {TASK.upper()}")
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Balanced accuracy: {balanced_acc:.4f}")
print(f"Output: {save_path}")
print("\nGenerated files:")
print(f"  • {pred_file}")
print(f"  • {summary_file}")
print(f"  • {report_file}")
print(f"  • {cm_file}")
if 'training_metrics' in locals() and training_metrics['epochs']:
    print(f"  • {metrics_file}")
print(f"  • {os.path.join(save_path, 's_0_checkpoint.pt')}")
if 'logs_file' in locals():
    print(f"  • {logs_file}")
print("="*70)