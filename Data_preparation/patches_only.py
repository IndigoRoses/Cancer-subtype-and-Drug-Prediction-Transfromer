import os
import sys
import glob
import openslide
import numpy as np
from PIL import Image
from multiprocessing import Pool
from tqdm import tqdm
import warnings
import time

# Completely suppress all warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# === CONFIGURATION ===
base_dir = "/scratch3/users/chantelle/tcga_cesc_data"
output_dir = os.path.join(base_dir, "processed/patches_macenko")
os.makedirs(output_dir, exist_ok=True)

paths = {
    "clinical": os.path.join(base_dir, "mutation"),
    "cnv": os.path.join(base_dir, "Clincal"),
    "snv": os.path.join(base_dir, "MAF"),
    "transcriptome": os.path.join(base_dir, "CNV"),
    "wsi": os.path.join(base_dir, "wsi"),
    "methylation": os.path.join(base_dir, "methylation")
}

# === COLLECT UNIQUE WSI FILES ===
wsi_files = (
    glob.glob(os.path.join(paths["wsi"], "**/*.svs"), recursive=True)
    + glob.glob(os.path.join(paths["cnv"], "**/*.svs"), recursive=True)
    + glob.glob(os.path.join(paths["clinical"], "**/*.svs"), recursive=True)
    + glob.glob(os.path.join(paths["transcriptome"], "**/*.svs"), recursive=True)
    + glob.glob(os.path.join(paths["snv"], "**/*.svs"), recursive=True)
)

unique_wsi = {os.path.basename(f): f for f in wsi_files}
wsi_files = list(unique_wsi.values())
print(f"Found {len(wsi_files)} unique .svs files")

# === SETTINGS ===
patch_size = 256
overlap = 0
tissue_threshold = 0.7
n_workers = 16
quick_check_level = 3  # Pyramid level for fast tissue check

# Global normalizer per worker
normalizer = None

def is_tissue(patch_np, thresh=tissue_threshold):
    """Ultra-fast tissue detection"""
    gray = np.mean(patch_np[:, :, :3], axis=2)
    return np.mean(gray < 220) > thresh

def safe_normalize(patch_np, norm):
    """Robust normalization with failure handling"""
    try:
        # Skip if too white (background)
        if np.mean(patch_np) > 210:
            return patch_np
        
        import staintools
        patch_np = staintools.LuminosityStandardizer.standardize(patch_np)
        result = norm.transform(patch_np)
        
        # Validate
        if np.any(np.isnan(result)) or np.any(np.isinf(result)):
            return patch_np
        
        return np.clip(result, 0, 255).astype(np.uint8)
    except:
        return patch_np

def init_worker():
    """Initialize normalizer per worker"""
    global normalizer
    import staintools
    try:
        ref_path = "/scratch3/users/chantelle/tcga_cesc_data/wsi/ref_thumbnail.png"
        target = staintools.read_image(ref_path)
        normalizer = staintools.StainNormalizer(method='macenko')
        normalizer.fit(target)
    except:
        normalizer = None

def process_slide(wsi_path):
    """Extract patches from one slide"""
    global normalizer
    start_time = time.time()
    
    try:
        filename = os.path.basename(wsi_path)
        case_id = os.path.splitext(filename)[0]
        out_dir = os.path.join(output_dir, case_id)
        os.makedirs(out_dir, exist_ok=True)

        # Skip if processed
        existing = [f for f in os.listdir(out_dir) if f.endswith('.png')]
        if existing:
            return None  # Silent skip

        slide = openslide.OpenSlide(wsi_path)
        w, h = slide.dimensions
        
        # Use low-res level for quick checks
        check_level = min(quick_check_level, len(slide.level_dimensions) - 1)
        ds = int(slide.level_downsamples[check_level])
        
        saved = 0
        batch = []
        
        # Grid scan with pre-filtering
        for y in range(0, h - patch_size + 1, patch_size - overlap):
            for x in range(0, w - patch_size + 1, patch_size - overlap):
                
                # Quick low-res check
                try:
                    qx, qy = x // ds, y // ds
                    qs = max(8, patch_size // ds)
                    quick = slide.read_region((qx * ds, qy * ds), check_level, (qs, qs))
                    quick_np = np.array(quick)[:, :, :3]
                    if not is_tissue(quick_np):
                        continue
                except:
                    continue
                
                # Full resolution extraction
                try:
                    patch = slide.read_region((x, y), 0, (patch_size, patch_size))
                    patch_np = np.array(patch)[:, :, :3]
                except:
                    continue
                
                # Verify tissue at full res
                if not is_tissue(patch_np):
                    continue
                
                # Normalize
                if normalizer:
                    patch_np = safe_normalize(patch_np, normalizer)
                
                # Save
                path = os.path.join(out_dir, f"patch_{x}_{y}.png")
                batch.append((patch_np, path))
                saved += 1
                
                # Batch write every 150 patches
                if len(batch) >= 64:
                    for data, p in batch:
                        Image.fromarray(data).save(p, compress_level=1)
                    batch = []
        
        # Write remainder
        for data, p in batch:
            Image.fromarray(data).save(p, compress_level=1)
        
        slide.close()
        elapsed = time.time() - start_time
        
        if saved > 0:
            return f"{case_id}: {saved} patches ({elapsed:.1f}s)"
        return None
        
    except Exception as e:
        return f"{case_id}: ERROR - {str(e)[:40]}"

def main():
    print(f"\nProcessing {len(wsi_files)} slides with {n_workers} workers\n")
    
    start = time.time()
    results = []
    
    with Pool(processes=n_workers, initializer=init_worker) as pool:
        for result in tqdm(pool.imap(process_slide, wsi_files, chunksize=1),
                          total=len(wsi_files),
                          desc="Progress",
                          unit="slide",
                          ncols=80):
            if result:  # Only show non-skipped
                results.append(result)
                if len(results) % 10 == 0:  # Print every 10
                    tqdm.write(result)
    
    # Final summary
    total_time = time.time() - start
    processed = sum(1 for r in results if "patches" in r)
    errors = sum(1 for r in results if "ERROR" in r)
    
    print(f"\n{'='*60}")
    print(f"Completed in {total_time/60:.1f} minutes")
    print(f"Processed: {processed} | Errors: {errors}")
    print(f"Average: {total_time/len(wsi_files):.1f}s per slide")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()