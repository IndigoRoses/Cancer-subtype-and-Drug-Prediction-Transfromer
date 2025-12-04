#!/usr/bin/env python3
"""

Compute persistent-homology (PH) features for each WSI case from per-patch embeddings
(or compute embeddings from images on-the-fly if requested).

Outputs:
 - <out_dir>/ph_features.tsv  (one row per case with vectorized PH features)
 - <out_dir>/diagrams/<case_id>_dgms.npy (optional - raw diagrams)
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc

# Ripser + persim (for diagrams -> vector features)
try:
    from ripser import ripser
    from persim import PersistenceImager
    from persim.persistent_entropy import persistent_entropy
except ImportError:
    try:
        from ripser import ripser
        from persim import PersistenceImager
        from persim import persistent_entropy
    except ImportError:
        from ripser import ripser
        from persim.persistent_images import PersistenceImager
        from persim.persistent_entropy import persistent_entropy

# Optional: for on-the-fly embeddings
def load_cnn(device="cpu"):
    import torch
    from torchvision import models, transforms
    cnn = models.resnet50(pretrained=True)
    cnn = torch.nn.Sequential(*list(cnn.children())[:-1])
    cnn = cnn.to(device).eval()
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return cnn, transform

def compute_embeddings_from_images(case_dir, cnn, transform, device="cpu", batch_size=128, num_workers=4):
    """Return numpy array shape (n_patches, emb_dim) computed from images in case_dir"""
    import torch
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image
    from pathlib import Path

    files = sorted([str(p) for p in Path(case_dir).glob("*.png")] + [str(p) for p in Path(case_dir).glob("*.jpg")])
    if len(files) == 0:
        return None
    class DS(Dataset):
        def __init__(self, files):
            self.files = files
        def __len__(self): return len(self.files)
        def __getitem__(self, i):
            img = Image.open(self.files[i]).convert("RGB")
            return transform(img)
    dl = DataLoader(DS(files), batch_size=batch_size, shuffle=False, num_workers=min(num_workers, os.cpu_count()))
    embs = []
    with torch.no_grad():
        for xb in dl:
            xb = xb.to(device)
            out = cnn(xb).squeeze(-1).squeeze(-1)  # B x 2048
            embs.append(out.cpu().numpy())
    embs = np.vstack(embs)
    return embs


def create_persistence_imager(pi_res, pi_spread):
    """Create PersistenceImager with version-compatible parameters"""
    # Try different API versions
    try:
        # Newer API: birth_range, pers_range, pixel_size
        imager = PersistenceImager(birth_range=(0, 1), pers_range=(0, 1), 
                                   pixel_size=1.0/pi_res)
        imager.fit(np.array([[0., 1.]]))
        return imager, 'birth_range'
    except TypeError:
        pass
    
    try:
        # Alternative: pixels parameter
        imager = PersistenceImager(pixels=(pi_res, pi_res), spread=pi_spread)
        imager.fit(np.array([[0., 1.]]))
        return imager, 'pixels'
    except TypeError:
        pass
    
    try:
        # Older API: no parameters, set later
        imager = PersistenceImager()
        # Try to set attributes if they exist
        if hasattr(imager, 'resolution'):
            imager.resolution = (pi_res, pi_res)
        if hasattr(imager, 'spread'):
            imager.spread = pi_spread
        imager.fit(np.array([[0., 1.]]))
        return imager, 'default'
    except:
        pass
    
    # Last resort: return None and we'll skip PI features
    print("WARNING: Could not initialize PersistenceImager. PI features will be skipped.")
    return None, None


def get_pi_size(imager, api_type, pi_res):
    """Get the flattened size of persistence image"""
    if imager is None:
        return 0
    try:
        # Try to infer from a test transform
        test_dgm = np.array([[0., 0.5]])
        pi = imager.transform(test_dgm)
        return np.asarray(pi).size
    except:
        # Fallback to expected size
        return pi_res * pi_res


def vectorize_diagrams(dgms, imager=None, pi_size=256):
    """
    Given ripser dgms (list of arrays for H0,H1,...), return a dict of vector features:
      - entropy_dim0, entropy_dim1, ...
      - nfeatures_dim0, nfeatures_dim1, ...
      - sum_lifetime_dim1, mean_lifetime_dim1, median_lifetime_dim1, ...
      - flattened persistence image for H1: pi_h1_*
    """
    feats = {}
    for dim, dgm in enumerate(dgms):
        # dgm is an (n_points x 2) array of (birth, death) (or inf)
        # remove infinite death entries for stats (ripser uses np.inf)
        finite_mask = np.isfinite(dgm[:,1])
        finite = dgm[finite_mask]
        lifetimes = (finite[:,1] - finite[:,0]) if finite.shape[0] > 0 else np.array([])
        
        # Compute persistent entropy
        try:
            ent = persistent_entropy(dgm) if dgm.size > 0 else 0.0
        except:
            ent = 0.0
        
        feats[f"entropy_h{dim}"] = float(ent)
        feats[f"nfeatures_h{dim}"] = int(len(dgm))
        feats[f"sum_lifetime_h{dim}"] = float(np.sum(lifetimes)) if lifetimes.size>0 else 0.0
        feats[f"mean_lifetime_h{dim}"] = float(np.mean(lifetimes)) if lifetimes.size>0 else 0.0
        feats[f"median_lifetime_h{dim}"] = float(np.median(lifetimes)) if lifetimes.size>0 else 0.0

    # Persistence image for H1 (if present)
    if len(dgms) > 1 and imager is not None:
        try:
            D1 = dgms[1]
            # remove infinities
            finite_mask = np.isfinite(D1[:,1])
            D1finite = D1[finite_mask]
            if D1finite.shape[0] == 0:
                pi_vec = np.zeros(pi_size)
            else:
                pi = imager.transform(D1finite)
                pi_vec = np.asarray(pi).ravel()
                # Ensure correct size
                if pi_vec.size != pi_size:
                    # Pad or truncate
                    if pi_vec.size < pi_size:
                        pi_vec = np.pad(pi_vec, (0, pi_size - pi_vec.size))
                    else:
                        pi_vec = pi_vec[:pi_size]
            # attach flattened vector as features
            for i, val in enumerate(pi_vec):
                feats[f"pi_h1_{i}"] = float(val)
        except Exception as e:
            # fallback: don't fail pipeline
            print(f"    Warning: PI computation failed ({e}), filling with zeros")
            for i in range(pi_size):
                feats[f"pi_h1_{i}"] = 0.0
    return feats


def main(args):
    embeddings_dir = Path(args.embeddings_dir) if args.embeddings_dir else None
    patch_root = Path(args.patch_root) if args.patch_root else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    diag_dir = out_dir / "diagrams"
    diag_dir.mkdir(exist_ok=True)

    # gather case ids
    case_ids = []
    # prefer embeddings_dir listing; else use patch_root directories
    if embeddings_dir and embeddings_dir.exists():
        case_files = sorted([p for p in embeddings_dir.glob("*.npy")])
        case_ids = [p.stem for p in case_files]
    elif patch_root and patch_root.exists():
        case_dirs = sorted([p for p in patch_root.iterdir() if p.is_dir()])
        case_ids = [p.name for p in case_dirs]
    else:
        raise FileNotFoundError("Provide either --embeddings-dir (per-case .npy) or --patch-root (patch directories).")

    print(f"Found {len(case_ids)} cases to process")
    rows = []
    
    # Prepare imager for vectorization (for H1)
    imager, api_type = create_persistence_imager(args.pi_res, args.pi_spread)
    if imager is not None:
        print(f"PersistenceImager initialized successfully (API type: {api_type})")
        pi_size = get_pi_size(imager, api_type, args.pi_res)
        print(f"PI feature vector size: {pi_size}")
    else:
        print("Skipping persistence image features")
        pi_size = 0

    # optional on-the-fly CNN
    cnn = None
    transform = None
    device = "cuda" if args.use_cuda and (os.environ.get("CUDA_VISIBLE_DEVICES") or False) else "cpu"
    if args.compute_from_images:
        print("Loading CNN for on-the-fly embeddings (may take a minute)...")
        cnn, transform = load_cnn(device)
        print("CNN loaded. Device:", device)

    for cid in tqdm(case_ids, desc="cases"):
        try:
            # 1) Load per-patch embeddings if available
            embs = None
            if embeddings_dir and (embeddings_dir / f"{cid}.npy").exists():
                embs = np.load(embeddings_dir / f"{cid}.npy")
                # shape check
                if embs.ndim == 1:
                    # maybe mean vector saved, not per-patch -> cannot compute PH
                    print(f"  SKIP {cid}: found 1D array (probably mean embedding). Use per-patch embeddings or compute from images.")
                    continue
            elif patch_root and (patch_root / cid).exists() and args.compute_from_images:
                # compute from images on-the-fly
                embs = compute_embeddings_from_images(patch_root / cid, cnn, transform, device=device,
                                                     batch_size=args.embed_batch, num_workers=args.embed_workers)
                if embs is None:
                    print(f"  SKIP {cid}: no images found at {patch_root/cid}")
                    continue
            else:
                print(f"  SKIP {cid}: no embeddings and compute_from_images not set")
                continue

            n_patches = embs.shape[0]
            # optional subsample
            if args.subsample and n_patches > args.subsample:
                rng = np.random.default_rng(seed=args.seed)
                idx = rng.choice(n_patches, size=args.subsample, replace=False)
                embs = embs[idx]

            # optionally whiten / PCA-reduce to speed-up PH
            if args.pre_pca_dim and embs.shape[1] > args.pre_pca_dim:
                # compute a small PCA via SVD (center first)
                from sklearn.decomposition import TruncatedSVD
                svd = TruncatedSVD(n_components=args.pre_pca_dim, random_state=0)
                embs = svd.fit_transform(embs)

            # Run ripser on point cloud (uses euclidean metric by default)
            # Use pairwise distances mode if requested (args.use_distance_matrix)
            if args.use_distance_matrix:
                from scipy.spatial.distance import pdist, squareform
                D = squareform(pdist(embs, metric=args.metric))
                rip = ripser(D, maxdim=args.maxdim, distance_matrix=True)
            else:
                rip = ripser(embs, maxdim=args.maxdim, metric=args.metric)
            dgms = rip['dgms']  # list of arrays

            # Optionally save raw diagrams
            if args.save_diagrams:
                np.save(diag_dir / f"{cid}_dgms.npy", dgms, allow_pickle=True)

            # Vectorize diagrams
            feats = vectorize_diagrams(dgms, imager=imager, pi_size=pi_size)
            feats['case_id'] = cid
            feats['n_patches'] = int(n_patches)
            rows.append(feats)

            # optional GC
            del embs, dgms, rip
            gc.collect()

        except Exception as e:
            print(f"ERROR processing {cid}: {e}")
            import traceback; traceback.print_exc()
            continue

    # assemble DataFrame and save
    if len(rows) == 0:
        print("No features computed.")
        return

    df = pd.DataFrame(rows).set_index('case_id')
    out_tsv = out_dir / "ph_features.tsv"
    df.to_csv(out_tsv, sep="\t")
    print("Saved PH features to:", out_tsv)
    print("Feature matrix shape:", df.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute persistent homology features per WSI")
    parser.add_argument("--embeddings-dir", type=str, default=None,
                        help="Directory with per-case per-patch embeddings saved as <case_id>.npy (shape: n_patches x emb_dim)")
    parser.add_argument("--patch-root", type=str, default=None,
                        help="Directory of patch subdirectories per case (only used if --compute-from-images)")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory to write ph features")
    parser.add_argument("--subsample", type=int, default=1000,
                        help="Max patches to sample per slide (reduces ripser cost). Set 0 for no subsampling")
    parser.add_argument("--maxdim", type=int, default=1, help="Max homology dimension to compute (0,1 or 2)")
    parser.add_argument("--use-distance-matrix", dest="use_distance_matrix", action="store_true",
                        help="Compute distance matrix first and feed ripser(distance_matrix=True) (slower mem heavy)")
    parser.add_argument("--metric", type=str, default="euclidean", help="Metric for ripser (euclidean by default)")
    parser.add_argument("--pre-pca-dim", type=int, default=64,
                        help="Optional TruncatedSVD dim to reduce embedding dimension before PH (speeds ripser)")
    parser.add_argument("--pi-res", type=int, default=16, help="persistence image resolution (per side)")
    parser.add_argument("--pi-spread", type=float, default=1.0, help="persistence imager spread")
    parser.add_argument("--save-diagrams", action="store_true", help="Save raw diagrams for each case")
    parser.add_argument("--compute-from-images", action="store_true",
                        help="If no per-patch embeddings exist, compute patch embeddings from images in patch directories (slow)")
    parser.add_argument("--use-cuda", action="store_true", help="Use CUDA if available and computing embeddings from images")
    parser.add_argument("--embed-batch", type=int, default=64, help="Batch size when computing embeddings from images")
    parser.add_argument("--embed-workers", type=int, default=4, help="Workers for DataLoader when computing embeddings")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    # Normalize arguments
    args.subsample = args.subsample if args.subsample and args.subsample > 0 else None
    args.maxdim = max(0, min(2, args.maxdim))
    main(args)