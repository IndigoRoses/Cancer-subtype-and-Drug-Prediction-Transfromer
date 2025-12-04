# Save as e.g. compute_wsi_embeddings.py and run in terminal 
import torch
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import numpy as np
import os
import sys
import gc
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

ROOT = Path("/scratch3/users/chantelle/tcga_cesc_data")
OUTDIR = ROOT / "processed/integration"
PATCH_ROOT = "/scratch3/users/chantelle/tcga_cesc_data/processed/patches_macenko"
OUTDIR = ROOT / "processed/wsi_embeddings"
OUTDIR_PATCHES = ROOT / "processed/wsi_patch_embeddings"  # For individual patch embeddings
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(OUTDIR_PATCHES, exist_ok=True)

# Global model - will be loaded in each worker process
_model = None
_device = None
_transform = None

def init_worker():
    """Initialize model in each worker process"""
    global _model, _device, _transform
    
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    cnn = models.resnet50(pretrained=True)
    cnn = torch.nn.Sequential(*list(cnn.children())[:-1])
    _model = cnn.to(_device).eval()
    
    # Define transform
    _transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    
    # Set torch to use single thread per process
    torch.set_num_threads(1)

def process_batch(img_paths):
    """Process a batch of images at once"""
    batch_tensors = []
    
    for img_path in img_paths:
        try:
            img = Image.open(img_path).convert("RGB")
            batch_tensors.append(_transform(img))
            img.close()
        except Exception as e:
            continue
    
    if len(batch_tensors) == 0:
        return None
    
    # Stack and process batch
    batch = torch.stack(batch_tensors).to(_device)
    
    with torch.no_grad():
        embs = _model(batch).squeeze(-1).squeeze(-1).cpu().numpy()
    
    del batch, batch_tensors
    return embs

def process_case_worker(case_dir_path, batch_size=32):
    """Worker function to process a single case"""
    try:
        p = Path(case_dir_path)
        case_id = p.name
        
        # Check if already done
        output_path = OUTDIR / f"{case_id}.npy"
        output_path_patches = OUTDIR_PATCHES / f"{case_id}_patches.npy"
        
        if output_path.exists() and output_path_patches.exists():
            return {"status": "skip", "case_id": case_id, "patches": 0}
        
        patch_files = sorted(list(p.glob("*.png")) + list(p.glob("*.jpg")))
        
        if len(patch_files) == 0:
            return {"status": "skip", "case_id": case_id, "patches": 0}
        
        # Use running mean and collect all embeddings
        running_mean = None
        all_embeddings = []
        processed = 0
        
        for batch_idx in range(0, len(patch_files), batch_size):
            batch_paths = [str(p) for p in patch_files[batch_idx:batch_idx + batch_size]]
            
            embs = process_batch(batch_paths)
            
            if embs is not None:
                # Store embeddings for patch-level matrix
                all_embeddings.append(embs)
                
                # Update running mean with batch
                for emb in embs:
                    if running_mean is None:
                        running_mean = emb.astype(np.float64)
                    else:
                        running_mean += (emb - running_mean) / (processed + 1)
                    processed += 1
                
                del embs
            
            # Clear cache periodically
            if (batch_idx // batch_size) % 5 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
        
        if running_mean is None:
            return {"status": "error", "case_id": case_id, "error": "No valid embeddings"}
        
        # Save aggregated mean embedding
        np.save(output_path, running_mean.astype(np.float32))
        
        # Save patch-level embeddings matrix
        all_embeddings_matrix = np.vstack(all_embeddings)
        np.save(output_path_patches, all_embeddings_matrix)
        
        # Cleanup
        del running_mean, all_embeddings, all_embeddings_matrix
        gc.collect()
        
        return {"status": "success", "case_id": case_id, "patches": processed}
        
    except Exception as e:
        return {"status": "error", "case_id": Path(case_dir_path).name, "error": str(e)}

def main():
    """Main function with parallel processing"""
    print("="*60, flush=True)
    print("STARTING PARALLEL PROCESSING", flush=True)
    print("="*60 + "\n", flush=True)
    
    # Get all case directories
    case_dirs = sorted([d for d in Path(PATCH_ROOT).iterdir() if d.is_dir()])
    print(f"Found {len(case_dirs)} case directories\n", flush=True)
    
    # Check what's already done
    existing_agg = set([f.stem for f in Path(OUTDIR).glob("*.npy")])
    existing_patches = set([f.stem.replace("_patches", "") for f in Path(OUTDIR_PATCHES).glob("*_patches.npy")])
    existing = existing_agg & existing_patches
    print(f"Already processed (both files): {len(existing)} cases", flush=True)
    
    # Filter to remaining cases
    remaining_cases = [c for c in case_dirs if c.name not in existing]
    print(f"Remaining to process: {len(remaining_cases)} cases\n", flush=True)
    
    if len(remaining_cases) == 0:
        print("All cases already processed!", flush=True)
        return
    
    # Determine number of workers
    num_workers = min(12, multiprocessing.cpu_count())
    print(f"Using {num_workers} parallel workers", flush=True)
    print(f"Batch size per worker: 64\n", flush=True)
    
    # Determine batch size based on device availability
    batch_size = 128 if torch.cuda.is_available() else 64
    
    # Process in parallel
    completed = 0
    errors = 0
    skipped = 0
    start_time = time.time()
    
    print("Starting parallel processing...\n", flush=True)
    
    with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker) as executor:
        # Submit all jobs
        future_to_case = {executor.submit(process_case_worker, str(case_dir), batch_size): case_dir 
                         for case_dir in remaining_cases}
        
        # Process completed jobs as they finish
        for future in as_completed(future_to_case):
            case_dir = future_to_case[future]
            
            try:
                result = future.result()
                
                if result["status"] == "success":
                    completed += 1
                    elapsed = time.time() - start_time
                    avg_time = elapsed / completed if completed > 0 else 0
                    remaining_time = avg_time * (len(remaining_cases) - completed - errors - skipped)
                    
                    print(f"[{completed + errors + skipped}/{len(remaining_cases)}] "
                          f"✓ {result['case_id']}: {result['patches']} patches | "
                          f"ETA: {remaining_time/3600:.1f}h", flush=True)
                
                elif result["status"] == "skip":
                    skipped += 1
                    print(f"[{completed + errors + skipped}/{len(remaining_cases)}] "
                          f"⊘ {result['case_id']}: skipped", flush=True)
                
                elif result["status"] == "error":
                    errors += 1
                    print(f"[{completed + errors + skipped}/{len(remaining_cases)}] "
                          f"✗ {result['case_id']}: {result.get('error', 'Unknown error')}", flush=True)
                    
            except Exception as e:
                errors += 1
                print(f"✗ {case_dir.name}: Fatal error: {e}", flush=True)
    
    # Final summary
    total_time = time.time() - start_time
    print("\n" + "="*60, flush=True)
    print("PROCESSING COMPLETE", flush=True)
    print("="*60, flush=True)
    print(f"Total time: {total_time/3600:.2f} hours", flush=True)
    print(f"Completed: {completed}", flush=True)
    print(f"Errors: {errors}", flush=True)
    print(f"Skipped: {skipped}", flush=True)
    print(f"Average time per case: {total_time/completed:.1f}s" if completed > 0 else "", flush=True)

if __name__ == "__main__":
    # Prevent numpy from using multiple threads (conflicts with multiprocessing)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    
    main()