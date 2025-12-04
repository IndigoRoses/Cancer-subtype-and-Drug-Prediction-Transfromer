#!/usr/bin/env python3
"""
compute_pointclouds_and_diagrams.py

For each WSI case: load per-patch embeddings (.npy: shape n_patches x emb_dim),
optionally subsample and reduce (TruncatedSVD), compute persistent diagrams via ripser,
and save:
  - <out_dir>/pointclouds/<case_id>_pc.npy       (n_sub x dim)
  - <out_dir>/diagrams/<case_id>_dgms.npy       (list of arrays: H0, H1, ...)
  - <out_dir>/plots/<case_id>_pc.png            (optional 2D scatter)
  - <out_dir>/plots/<case_id>_dgms.png          (optional persistence diagram plot)

Usage example:
python compute_pointclouds_and_diagrams.py \
  --emb-dir /scratch3/users/chantelle/tcga_cesc_data/processed/wsi_patch_embeddings \
  --out-dir /scratch3/users/chantelle/tcga_cesc_data/processed/ph_out \
  --subsample 1000 --pre-pca 64 --maxdim 1 --plot

Dependencies:
 pip install numpy pandas ripser persim scikit-learn matplotlib tqdm
"""

import os
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# ripser + persim
try:
    from ripser import ripser
    from persim import plot_diagrams
except Exception:
    # fallback imports (different versions / module paths)
    from ripser import ripser
    try:
        from persim import plot_diagrams
    except Exception:
        from persim import plot as plot_diagrams  # older variants might differ

# sklearn for fast dimensionality reduction
from sklearn.decomposition import TruncatedSVD
from sklearn.utils import resample

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def load_embeddings(path):
    arr = np.load(path, allow_pickle=True)
    # ensure 2D
    if arr.ndim == 1:
        # maybe saved mean vector
        return arr.reshape(1, -1)
    if arr.ndim == 3 and arr.shape[0] == 1:  # some files saved as (1, n, d)
        return arr.squeeze(0)
    return arr

def compute_and_save(case_id, emb_path, out_dir, subsample=None, pre_pca_dim=None, maxdim=1, metric='euclidean', plot=False, random_state=0):
    # load
    emb = load_embeddings(emb_path)
    n_patches, emb_dim = emb.shape
    if n_patches == 0:
        print(f"SKIP {case_id}: zero patches")
        return None

    # subsample if requested
    if subsample is not None and n_patches > subsample:
        idx = np.random.default_rng(seed=random_state).choice(n_patches, size=subsample, replace=False)
        pc = emb[idx]
    else:
        pc = emb

    # optionally reduce dimension via TruncatedSVD (works with dense arrays)
    if pre_pca_dim is not None and pre_pca_dim > 0 and pc.shape[1] > pre_pca_dim:
        svd = TruncatedSVD(n_components=pre_pca_dim, random_state=random_state)
        pc = svd.fit_transform(pc)

    # ensure float64 (ripser prefers float)
    pc = np.asarray(pc, dtype=float)

    # directories
    point_dir = Path(out_dir) / "pointclouds"
    diag_dir = Path(out_dir) / "diagrams"
    plot_dir = Path(out_dir) / "plots"
    ensure_dir(point_dir)
    ensure_dir(diag_dir)
    ensure_dir(plot_dir)

    # save point cloud
    pc_path = point_dir / f"{case_id}_pc.npy"
    np.save(pc_path, pc)

    # compute ripser diagrams
    try:
        # if user wants to compute via distance matrix set distance_matrix=True and compute pairwise distances beforehand
        rip = ripser(pc, maxdim=maxdim, metric=metric)
        dgms = rip['dgms']
    except Exception as e:
        # retry with distance matrix fallback (slower)
        from scipy.spatial.distance import pdist, squareform
        D = squareform(pdist(pc, metric=metric))
        rip = ripser(D, maxdim=maxdim, distance_matrix=True)
        dgms = rip['dgms']

    # save diagrams (must wrap in object array to handle variable shapes)
    dgm_path = diag_dir / f"{case_id}_dgms.npy"
    np.save(dgm_path, np.array(dgms, dtype=object), allow_pickle=True)


    # optional plotting
    if plot:
        try:
            # point cloud 2D scatter (use first two PCA dims if >2)
            fig, ax = plt.subplots(1, 2, figsize=(10,4))
            if pc.shape[1] >= 2:
                ax[0].scatter(pc[:,0], pc[:,1], s=6, alpha=0.6)
                ax[0].set_title(f"{case_id} point cloud (dim1 vs dim2)")
            else:
                ax[0].scatter(range(pc.shape[0]), pc[:,0], s=6, alpha=0.6)
                ax[0].set_title(f"{case_id} point cloud (1D)")

            # persistence diagrams plot (H0/H1)
            try:
                plot_diagrams(dgms, show=False, ax=ax[1])
            except TypeError:
                # some API differences: plot_diagrams returns fig
                _ = plot_diagrams(dgms, ax=ax[1])
            ax[1].set_title(f"{case_id} diagrams")
            plt.tight_layout()
            plot_path = plot_dir / f"{case_id}_pc_dgms.png"
            plt.savefig(plot_path, dpi=150)
            plt.close(fig)
        except Exception as e:
            print(f"Warning: could not save plots for {case_id}: {e}")

    return {
        "case_id": case_id,
        "n_original_patches": int(n_patches),
        "n_used_patches": int(pc.shape[0]),
        "emb_dim_used": int(pc.shape[1]),
        "pointcloud_path": str(pc_path),
        "diag_path": str(dgm_path)
    }

def main(args):
    emb_dir = Path(args.emb_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # gather candidate files
    # accept: <case_id>.npy OR <case_id>_patches.npy OR <case_id>*.npy
    files = sorted([p for p in emb_dir.glob("*.npy")])
        raise FileNotFoundError(f"No .npy files in {emb_dir}")

    results = []
    for p in tqdm(files, desc="cases"):
        # derive case_id from filename
        case_id = p.stem
        try:
            res = compute_and_save(case_id, p, out_dir,
                                   subsample=args.subsample,
                                   pre_pca_dim=args.pre_pca_dim,
                                   maxdim=args.maxdim,
                                   metric=args.metric,
                                   plot=args.plot,
                                   random_state=args.seed)
            if res:
                results.append(res)
        except Exception as e:
            print(f"ERROR {case_id}: {e}")
            import traceback; traceback.print_exc()
            continue

    # write summary TSV
    import pandas as pd
    df = pd.DataFrame(results)
    out_tsv = out_dir / "ph_pointclouds_and_diagrams_summary.tsv"
    df.to_csv(out_tsv, sep="\t", index=False)
    print("Saved summary to:", out_tsv)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute point clouds and persistent diagrams from per-patch embeddings (.npy)")
    parser.add_argument("--emb-dir", type=str, required=True, help="Directory with per-case embeddings (.npy)")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--subsample", type=int, default=1000, help="Max patches to subsample per slide (None=no subsample).")
    parser.add_argument("--pre-pca-dim", type=int, default=64, help="Reduce embedding dim before PH (TruncatedSVD). Set 0 or None to skip.")
    parser.add_argument("--maxdim", type=int, default=1, help="Max homology dimension (0,1 or 2).")
    parser.add_argument("--metric", type=str, default="euclidean", help="Metric for ripser.")
    parser.add_argument("--plot", action="store_true", help="Save diagnostic plots (point cloud 2D + diagram plot).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # normalize None/0
    if args.pre_pca_dim is not None and args.pre_pca_dim <= 0:
        args.pre_pca_dim = None
    if args.subsample is not None and args.subsample <= 0:
        args.subsample = None

    main(args)
