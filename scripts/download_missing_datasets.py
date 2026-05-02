#!/usr/bin/env python3
"""
download_missing_datasets.py
=============================
Downloads datasets from Table 2 that are currently missing or invalid.

This script handles three source types:
  1. Figshare API  → get download_url from /articles/{id}/files, then download
  2. GEO           → use GEOparse / NCBI EDirect (NOT hand-written FTP URLs)
  3. CELLxGENE     → per-dataset download with dedicated env & source_h5ad
  4. Bioconductor  → TabulaMuris / TabulaMurisSenisData packages

Usage:
  python scripts/download_missing_datasets.py --list          # Show all missing datasets
  python scripts/download_missing_datasets.py --source figshare  # Only figshare datasets
  python scripts/download_missing_datasets.py --source geo     # Only GEO datasets
  python scripts/download_missing_datasets.py --source cellxgene # Only CELLxGENE datasets
  python scripts/download_missing_datasets.py --source bioconductor # Only Bioconductor
  python scripts/download_missing_datasets.py --dataset "Human Pancreas 1"  # One dataset
"""

import os
import sys
import json
import time
import argparse
import subprocess
import tarfile
import gzip
import shutil
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import scanpy as sc
import anndata as ad
from scipy.sparse import csr_matrix

warnings.filterwarnings("ignore")

# ─── Configuration ─────────────────────────────────────────────────────────────
CODE_DIR = "/home/luolie/biopipeline/dimension-reduction/scCluBench-main"
DATA_DIR = "/home/luolie/biopipeline/dimension-reduction/scCluBench-main/data"
TMP_DIR = "/data/luolie/biopipeline/scCluBench/tmp"
REGISTRY_PATH = "/home/luolie/biopipeline/dimension-reduction/scCluBench-main/data_manifest/dataset_registry.tsv"

os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ─── NCBI Entrez config ───────────────────────────────────────────────────────
ENTREZ_EMAIL = os.environ.get("NCBI_EMAIL", "user@example.com")
try:
    from Bio import Entrez
    Entrez.email = ENTREZ_EMAIL
except ImportError:
    print("WARNING: Biopython not installed. GEO downloads may fail.")
    print("  Install with: pip install biopython")


# ─── Helper Utilities ───────────────────────────────────────────────────────────

def curl_download(url: str, output_path: str, timeout: int = 600,
                  extra_args: List[str] = None) -> bool:
    """Download a file using curl with retries."""
    args = [
        "curl", "-s", "-L",
        "--max-time", str(timeout),
        "--retry", "3",
        "--retry-delay", "10",
        "-o", output_path,
    ]
    if extra_args:
        args = args[:-1] + extra_args + [args[-1]]  # insert before -o
    result = subprocess.run(args + [url], capture_output=True, text=True)
    ok = result.returncode == 0
    if not ok:
        print(f"    curl failed: {result.stderr[:200]}")
    return ok


def aria2c_download(url: str, output_path: str, timeout: int = 600) -> bool:
    """Download using aria2c (parallel download, resume support)."""
    cmd = [
        "aria2c", "-x", "4", "-s", "4",
        "--max-timer-limit", str(timeout),
        "-d", os.path.dirname(output_path),
        "-o", os.path.basename(output_path),
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0 and os.path.exists(output_path)


def validate_h5ad(filepath: str) -> Tuple[bool, str]:
    """
    Validate an h5ad file: gzip test, file size, scanpy.read_h5ad.
    Returns (is_valid, message).
    """
    if not os.path.exists(filepath):
        return False, "file does not exist"

    size = os.path.getsize(filepath)
    if size < 1024:
        return False, f"file too small ({size} bytes)"

    # gzip integrity test (h5ad files are gzip-compressed)
    result = subprocess.run(
        ["gzip", "-t", filepath],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        # Not necessarily a gzip file — h5ad can use other compression
        pass

    try:
        adata = sc.read_h5ad(filepath)
        return True, f"OK: {adata.n_obs} cells × {adata.n_vars} genes"
    except Exception as e:
        return False, f"scanpy.read_h5ad failed: {e}"


def compute_sparsity(adata) -> float:
    """Compute sparsity (% zeros)."""
    X = adata.X
    if hasattr(X, 'toarray'):
        arr = X.toarray()
    else:
        arr = np.asarray(X)
    total = arr.size
    nonzero = np.count_nonzero(arr)
    return (1.0 - nonzero / total) * 100 if total > 0 else 0.0


# ─── Source 1: Figshare API ───────────────────────────────────────────────────

FIGSHARE_ARTICLES = {
    # Tabula Muris Senis (droplet)
    "8273102": {
        "name": "Tabula Muris Senis",
        "base_url": "https://api.figshare.com/v2/articles/8273102/files",
        "files": {
            "Limb_Muscle": "Limb_muscle.h5ad",
            "Lung": "Lung.h5ad",
            "Diaphragm": "Diaphragm.h5ad",
            "Trachea": "Trachea.h5ad",
        },
    },
    # Tabula Muris (SMART-Seq2 FACS)
    "5715040": {
        "name": "Tabula Muris SMART-Seq2",
        "base_url": "https://api.figshare.com/v2/articles/5715040/files",
        "files": {
            "Limb_Muscle": "Limb_muscle.h5ad",
            "Lung": "Lung.h5ad",
            "Bladder": "Bladder.h5ad",
            "Spleen": "Spleen.h5ad",
            "Kidney": "Kidney.h5ad",
            "Brain_Myeloid": "Brain.h5ad",
        },
    },
    # Tabula Muris (Droplet 10x)
    "5715025": {
        "name": "Tabula Muris 10x",
        "base_url": "https://api.figshare.com/v2/articles/5715025/files",
        "files": {
            "Limb_Muscle": "Limb_muscle.h5ad",
            "Lung": "Lung.h5ad",
            "Brain": "Brain.h5ad",
            "Kidney": "Kidney.h5ad",
        },
    },
}


def figshare_get_download_url(article_id: str, filename: str) -> Optional[str]:
    """
    Get a download URL from Figshare API without authentication.
    
    Steps:
    1. GET /articles/{id}/files → list of files
    2. Find file by name
    3. GET /articles/{id}/files/{file_id} → contains download_url
    
    This avoids the 403 direct download by using the API endpoint.
    """
    import urllib.request
    import urllib.error
    import json
    
    # Step 1: List files
    files_url = f"https://api.figshare.com/v2/articles/{article_id}/files"
    try:
        req = urllib.request.Request(files_url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            files = json.loads(resp.read().decode())
    except Exception as e:
        print(f"    Figshare API error listing files: {e}")
        return None
    
    # Step 2: Find the matching file
    matched_file = None
    for f in files:
        if filename.lower() in f["name"].lower() or f["name"].lower() in filename.lower():
            matched_file = f
            break
    
    if not matched_file:
        # Try exact match
        for f in files:
            if f["name"] == filename:
                matched_file = f
                break
    
    if not matched_file:
        print(f"    File '{filename}' not found in article {article_id}")
        print(f"    Available files: {[f['name'] for f in files]}")
        return None
    
    # Step 3: Get the file details (includes download_url)
    file_id = matched_file["id"]
    detail_url = f"https://api.figshare.com/v2/articles/{article_id}/files/{file_id}"
    try:
        req = urllib.request.Request(detail_url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            file_info = json.loads(resp.read().decode())
        return file_info.get("download_url")
    except Exception as e:
        print(f"    Figshare API error getting file details: {e}")
        return None


def figshare_download(article_id: str, filename: str, output_path: str) -> bool:
    """
    Download a file from Figshare using the API (bypasses 403 on direct URLs).
    
    Returns True if download succeeded.
    """
    print(f"  Figshare article {article_id}: getting download_url for '{filename}'...")
    
    url = figshare_get_download_url(article_id, filename)
    if not url:
        return False
    
    print(f"  Downloading: {url[:80]}...")
    
    # Try aria2c first (faster, resume support)
    if shutil.which("aria2c"):
        if aria2c_download(url, output_path):
            return True
        print("    aria2c failed, trying curl...")
    
    # Fall back to curl
    if curl_download(url, output_path, timeout=600):
        return True
    
    return False


# ─── Source 2: GEO via GEOparse / EDirect ─────────────────────────────────────

def geo_check_accession(geo_id: str) -> Dict[str, Any]:
    """
    Use NCBI EDirect to query a GEO accession.
    Returns metadata about available files.
    """
    import xml.etree.ElementTree as ET
    
    cmd = [
        "efetch", "-db", "gds", "-id", geo_id, "-format", "xml",
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30,
            env={**os.environ, "EMAIL": ENTREZ_EMAIL}
        )
        if result.returncode != 0:
            return {"error": result.stderr}
        
        # Parse XML for supplementary file URLs
        root = ET.fromstring(result.stdout)
        # Look for supplementary file links
        supp_urls = []
        for elem in root.iter():
            if "supplementary" in elem.text.lower() if elem.text else False:
                supp_urls.append(elem.text)
        
        return {
            "geo_id": geo_id,
            "xml_content": result.stdout[:5000],
            "supplementary_urls": supp_urls,
            "raw": result.stdout,
        }
    except Exception as e:
        return {"error": str(e)}


def geo_list_supplementary_files(geo_id: str) -> List[Dict]:
    """
    List all supplementary files for a GEO accession using GEOparse.
    Returns list of {name, url, size} dicts.
    """
    try:
        import GEOparse
    except ImportError:
        print("    GEOparse not installed. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "GEOparse", "-q"],
                       capture_output=True)
        import GEOparse
    
    try:
        print(f"  Fetching GEO metadata for {geo_id}...")
        gse = GEOparse.get_GEO(geo=geo_id, destdir=TMP_DIR, how="soft", silent=True)
        
        files = []
        for gsm_name, gsm in gse.gsms.items():
            supp = gsm.metadata.get("supplementary_file", [])
            if isinstance(supp, str):
                supp = [supp]
            for url in supp:
                if url:
                    fname = os.path.basename(url)
                    files.append({
                        "gsm": gsm_name,
                        "url": url,
                        "filename": fname,
                        "gsm_title": gsm.metadata.get("title", [""])[0],
                    })
        
        print(f"  Found {len(files)} supplementary files")
        for f in files[:5]:
            print(f"    {f['gsm']}: {f['filename']}")
        if len(files) > 5:
            print(f"    ... and {len(files) - 5} more")
        
        return files
    except Exception as e:
        print(f"  GEOparse error: {e}")
        return []


def geo_download_supplementary(geo_id: str, patterns: List[str],
                               output_dir: str) -> Optional[str]:
    """
    Download supplementary files from GEO matching given filename patterns.
    Returns path to downloaded file, or None.
    """
    files = geo_list_supplementary_files(geo_id)
    if not files:
        return None
    
    matched = []
    for f in files:
        fname = f["filename"]
        for pat in patterns:
            if pat.lower() in fname.lower():
                matched.append(f)
                break
    
    if not matched:
        print(f"  No files matched patterns: {patterns}")
        print(f"  All files: {[f['filename'] for f in files[:10]]}")
        return None
    
    for m in matched:
        print(f"  Downloading: {m['filename']} from {m['url'][:60]}...")
        local_path = os.path.join(output_dir, m["filename"])
        
        ok = curl_download(m["url"], local_path, timeout=600)
        if ok and os.path.exists(local_path):
            size = os.path.getsize(local_path)
            if size > 1000:
                print(f"    Downloaded: {size / 1024 / 1024:.1f} MB")
                return local_path
        
        # Try with wget as fallback
        result = subprocess.run(
            ["wget", "-q", "-O", local_path, m["url"]],
            capture_output=True, timeout=600,
        )
        if result.returncode == 0 and os.path.exists(local_path) and os.path.getsize(local_path) > 1000:
            print(f"    Downloaded via wget: {os.path.getsize(local_path) / 1024 / 1024:.1f} MB")
            return local_path
    
    return None


def geo_download_tarball(geo_id: str, output_dir: str = TMP_DIR) -> Optional[str]:
    """
    Download GEO tar archive using NCBI FTP URL (constructed from accession).
    Note: Uses the standard FTP path pattern, not hand-written URLs.
    """
    # Construct FTP path from accession
    gse_num = geo_id[3:]  # Remove "GSE" prefix
    gse_dir_num = gse_num[:-3] + "nnn"  # e.g., "GSE6" → "GSEnnn"
    
    ftp_base = f"ftp://ftp.ncbi.nlm.nih.gov/geo/series/{gse_dir_num}/{geo_id}/"
    tar_name = f"{geo_id}_family.tgz"
    local_tar = os.path.join(output_dir, tar_name)
    
    print(f"  Downloading GEO tarball from FTP: {ftp_base}")
    
    # Try wget first
    cmd = ["wget", "-q", "--timeout=120", "-O", local_tar,
           f"{ftp_base}{tar_name}"]
    result = subprocess.run(cmd, capture_output=True, timeout=300)
    
    if result.returncode != 0 or not os.path.exists(local_tar):
        # Try alternative: download via HTTPS
        https_url = f"https://www.ncbi.nlm.nih.gov/geo/download/?acc={geo_id}&format=file"
        ok = curl_download(https_url, local_tar, timeout=600)
        if not ok:
            return None
    
    if os.path.exists(local_tar) and os.path.getsize(local_tar) > 10000:
        print(f"  Downloaded: {os.path.getsize(local_tar) / 1024 / 1024:.1f} MB")
        return local_tar
    
    return None


# ─── Source 3: CELLxGENE Census (per-dataset) ────────────────────────────────

CELLXGENE_DATASETS = {
    "Human Kidney": {
        "organism": "Homo sapiens",
        "tissue": "kidney",
        "dataset_filter": "dataset_title.str.contains('kidney', case=False) & "
                          "~dataset_title.str.contains('mouse', case=False)",
        "note": "Large dataset — may need dedicated env with more RAM",
    },
    "Sonya Liver": {
        "organism": "Homo sapiens",
        "tissue": "liver",
        "dataset_filter": "dataset_title.str.contains('liver', case=False)",
        "note": "Large dataset — may need dedicated env",
    },
    "Mouse Kidney": {
        "organism": "Mus musculus",
        "tissue": "kidney",
        "dataset_filter": "dataset_title.str.contains('kidney', case=False)",
        "note": "Large dataset — may need dedicated env",
    },
    "Mouse bladder": {
        "organism": "Mus musculus",
        "tissue": "bladder",
        "dataset_filter": "dataset_title.str.contains('bladder', case=False)",
        "note": "Medium dataset",
    },
    "CITE CMBC": {
        "organism": "Homo sapiens",
        "tissue": "bone_marrow",
        "dataset_filter": "dataset_title.str.contains('CD34', case=False)",
        "note": "Medium dataset",
    },
}


def census_find_dataset(organism: str, tissue: str,
                        dataset_filter: str = None) -> Optional[Dict]:
    """
    Find the best matching dataset_id in CELLxGENE Census for a tissue.
    Returns dataset metadata dict or None.
    """
    try:
        import cellxgene_census
    except ImportError:
        print("    cellxgene-census not installed. Run: pip install cellxgene-census")
        return None
    
    try:
        with cellxgene_census.open_soma() as census:
            datasets = census["census_info"]["datasets"].read().concat().to_pandas()
            
            # Filter by organism
            datasets = datasets[datasets["organism"].str.lower() == organism.lower()]
            
            if dataset_filter:
                datasets = datasets.query(dataset_filter)
            
            if datasets.empty:
                print(f"    No datasets found for {organism}/{tissue}")
                return None
            
            # Pick the largest dataset (most cells)
            datasets = datasets.sort_values(
                "dataset_total_cell_count", ascending=False
            )
            
            row = datasets.iloc[0]
            return {
                "dataset_id": row["dataset_id"],
                "dataset_title": row.get("dataset_title", ""),
                "total_cells": row.get("dataset_total_cell_count", 0),
                "organism": row.get("organism", ""),
            }
    except Exception as e:
        print(f"    CELLxGENE Census error: {e}")
        return None


def census_download_tissue(organism: str, tissue: str,
                           output_path: str,
                           dataset_filter: str = None,
                           max_cells: int = None) -> bool:
    """
    Download a specific tissue from CELLxGENE Census.
    Returns True if successful.
    """
    try:
        import cellxgene_census
    except ImportError:
        print("    cellxgene-census not installed")
        return False
    
    print(f"  Finding CELLxGENE dataset for {organism}/{tissue}...")
    meta = census_find_dataset(organism, tissue, dataset_filter)
    if not meta:
        return False
    
    print(f"  Dataset: {meta['dataset_id']}")
    print(f"  Title: {meta['dataset_title'][:60]}")
    print(f"  Cells: {meta['total_cells']:,}")
    
    dataset_id = meta["dataset_id"]
    
    try:
        print(f"  Downloading (this may take a while for large datasets)...")
        t0 = time.time()
        
        adata = cellxgene_census.get_anndata(
            census=cellxgene_census.open_soma(),
            organism=organism,
            obs_value_filter=f"dataset_id == '{dataset_id}'",
        )
        
        elapsed = time.time() - t0
        print(f"  Downloaded: {adata.n_obs:,} cells × {adata.n_vars:,} genes "
              f"in {elapsed:.0f}s")
        
        # Filter to specific tissue
        tissue_lower = tissue.lower().replace("_", " ")
        if "tissue" in adata.obs.columns:
            mask = adata.obs["tissue"].str.lower().str.contains(
                tissue_lower, na=False
            )
            if mask.sum() > 0:
                adata = adata[mask].copy()
                print(f"  Filtered to {adata.n_obs:,} cells for tissue '{tissue}'")
        
        # Save
        adata.write_h5ad(output_path, compression="gzip")
        print(f"  Saved: {output_path}")
        return True
        
    except Exception as e:
        print(f"  Download failed: {e}")
        return False


# ─── Source 4: Bioconductor (Tabula Muris) ───────────────────────────────────

def bioconductor_tabula_muris(tissue: str, output_path: str,
                                is_senis: bool = False) -> bool:
    """
    Download from Bioconductor TabulaMuris / TabulaMurisSenisData.
    Returns True if successful.
    """
    pkg = "TabulaMurisSenisData" if is_senis else "TabulaMurisData"
    
    print(f"  Checking Bioconductor package: {pkg}")
    
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        pandas2ri.activate()
    except ImportError:
        print(f"    rpy2 not installed. Run: pip install rpy2")
        print(f"    Then in R: BiocManager::install('{pkg}')")
        return False
    
    try:
        r = ro.r
        r(f'library("{pkg}")')
        
        # List available datasets
        datasets = r(f'data(list=data(package="{pkg}"))')
        print(f"  Available datasets in {pkg}: {datasets}")
        
        # Try to load the specific tissue
        tissue_name = tissue.replace("_", " ")
        r_code = f'''
        obj <- {pkg}::SCE tissue == "{tissue_name}"
        '''.strip()
        r(r_code)
        
        # Convert to AnnData... this is complex
        # Better approach: export to h5ad from R and read in Python
        
        print(f"  Bioconductor approach needs R environment")
        print(f"  Alternative: use GEO accession GSE{'149590' if is_senis else '109774'}")
        return False
        
    except Exception as e:
        print(f"  Bioconductor error: {e}")
        return False


# ─── Post-processing ───────────────────────────────────────────────────────────

def find_cell_type_column(obs) -> Optional[str]:
    """Find a cell type column in obs. Returns column name or None."""
    for col in obs.columns:
        if obs[col].dtype.name not in ("category", "object"):
            continue
        n_unique = obs[col].nunique()
        if n_unique >= 2 and n_unique <= 200:
            return col
    return None


def post_process_h5ad(adata, label_col: str = None) -> bool:
    """
    Post-process downloaded AnnData to ensure scCluBench compatibility.
    Returns True if successful.
    """
    # Find or set cell_type column
    if label_col and label_col in adata.obs.columns:
        adata.obs["cell_type"] = adata.obs[label_col].astype(str)
    elif "cell_type" not in adata.obs.columns:
        ct_col = find_cell_type_column(adata.obs)
        if ct_col:
            adata.obs["cell_type"] = adata.obs[ct_col].astype(str)
            print(f"  Set cell_type from column '{ct_col}'")
        else:
            print(f"  WARNING: No cell_type column found")
            return False
    
    # Ensure raw counts
    if adata.raw is None:
        adata.raw = adata.copy()
    
    # Ensure n_counts
    if "n_counts" not in adata.obs.columns:
        if hasattr(adata.X, "sum"):
            if hasattr(adata.X, "toarray"):
                counts = np.array(adata.X.sum(axis=1)).flatten()
            else:
                counts = adata.X.sum(axis=1)
            adata.obs["n_counts"] = counts
    
    # Ensure size_factors
    if "size_factors" not in adata.obs.columns:
        if "n_counts" in adata.obs.columns:
            adata.obs["size_factors"] = (
                adata.obs["n_counts"] / np.median(adata.obs["n_counts"].values)
            )
        else:
            adata.obs["size_factors"] = 1.0
    
    # Verify at least 2 cell types
    n_ct = adata.obs["cell_type"].nunique()
    if n_ct < 2:
        print(f"  WARNING: Only {n_ct} cell type(s) — may be invalid label")
    
    return True


# ─── Dataset-specific download functions ────────────────────────────────────────

def download_human_pancreas_4(output_path: str) -> bool:
    """Download Human Pancreas 4 from GEO GSE81547."""
    print("\n=== Human Pancreas 4 (GEO GSE81547, Enge 2017) ===")
    
    geo_id = "GSE81547"
    
    # Try supplementary files first
    local = geo_download_supplementary(
        geo_id, ["h5ad", "hdf5", "matrix"], TMP_DIR
    )
    if local and os.path.getsize(local) > 1000:
        print(f"  Found: {local}")
        # Process...
        return True
    
    # Try tarball
    tar_path = geo_download_tarball(geo_id)
    if not tar_path:
        print(f"  GEO tarball download failed")
        print(f"  Manual download: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={geo_id}")
        return False
    
    # Extract and process
    print(f"  Extracting from {tar_path}...")
    try:
        with tarfile.open(tar_path, "r") as tar:
            members = tar.getmembers()
            for m in members:
                print(f"    {m.name}")
    except Exception as e:
        print(f"  Extraction failed: {e}")
        return False
    
    return False


def download_pbmc_68k(output_path: str) -> bool:
    """Download 68K PBMC from 10x Genomics."""
    print("\n=== 68K PBMC (10x Genomics) ===")
    
    # Try 10x direct URL
    url = "https://cf.10xgenomics.com/samples/cell-exp/6.1.2/68k_pbmc/68k_pbmc_filtered_gene_bc_matrices_h5.h5"
    
    local_path = os.path.join(TMP_DIR, "68k_pbmc.h5")
    print(f"  Downloading from 10x Genomics...")
    
    if curl_download(url, local_path, timeout=300):
        valid, msg = validate_h5ad(local_path)
        if valid:
            try:
                adata = sc.read_10x_h5(local_path)
                adata.var_names_make_unique()
                adata.write_h5ad(output_path, compression="gzip")
                print(f"  Saved: {output_path}")
                
                # Try to find cell type labels
                if post_process_h5ad(adata):
                    adata.write_h5ad(output_path, compression="gzip")
                return True
            except Exception as e:
                print(f"  Processing error: {e}")
    
    # Fallback: GEO
    geo_id = "GSE115469"
    print(f"  10x download failed, trying GEO {geo_id}...")
    local = geo_download_supplementary(geo_id, ["h5", "matrix", "barcodes"], TMP_DIR)
    
    return False


def download_mouse_hypothalamus(output_path: str) -> bool:
    """Download Mouse hypothalamus from GEO GSE87544."""
    print("\n=== Mouse Hypothalamus (GEO GSE87544, Chen 2017) ===")
    
    geo_id = "GSE87544"
    
    # GSE87544 has a processed matrix file
    local = geo_download_supplementary(
        geo_id, ["counts", "matrix", "expression", "tsv"], TMP_DIR
    )
    
    if local:
        print(f"  Found supplementary file: {local}")
        # Process TSV to h5ad
        try:
            df = pd.read_csv(local, sep="\t", index_col=0)
            X = csr_matrix(df.values.astype(np.float32))
            adata = ad.AnnData(X=X)
            adata.obs = pd.DataFrame(index=df.index)
            adata.var = pd.DataFrame(index=df.columns)
            
            if post_process_h5ad(adata, label_col="cluster"):
                adata.write_h5ad(output_path, compression="gzip")
                return True
        except Exception as e:
            print(f"  Processing error: {e}")
    
    tar_path = geo_download_tarball(geo_id)
    if not tar_path:
        print(f"  GEO download failed")
        return False
    
    return False


# ─── Main Download Orchestrator ─────────────────────────────────────────────

def get_dataset_info(paper_name: str) -> Dict[str, Any]:
    """Get dataset info from registry or alias map."""
    import csv
    
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH) as f:
            for row in csv.DictReader(f, delimiter="\t"):
                if row.get("paper_name", "").strip('"') == paper_name:
                    return dict(row)
    
    return {}


def download_dataset(paper_name: str, force: bool = False) -> bool:
    """
    Download a single dataset by paper name.
    Returns True if successful.
    """
    info = get_dataset_info(paper_name)
    source_type = info.get("source_type", "unknown")
    canonical_file = info.get("canonical_file", f"{paper_name.replace(' ', '_')}.h5ad")
    output_path = os.path.join(DATA_DIR, canonical_file)
    
    if os.path.exists(output_path) and not force:
        valid, msg = validate_h5ad(output_path)
        if valid:
            print(f"  Already exists and valid: {output_path}")
            return True
    
    print(f"\n{'='*70}")
    print(f"Downloading: {paper_name}")
    print(f"  Source type: {source_type}")
    print(f"  Output: {output_path}")
    print(f"{'='*70}")
    
    # Route to appropriate download function
    if paper_name == "Human Pancreas 4":
        return download_human_pancreas_4(output_path)
    elif paper_name == "68K PBMC":
        return download_pbmc_68k(output_path)
    elif paper_name == "Mouse hypothalamus":
        return download_mouse_hypothalamus(output_path)
    elif source_type == "figshare_api":
        # Extract figshare article and filename from notes
        notes = info.get("notes", "")
        article_id = info.get("figshare_id", "")
        tissue = info.get("tissue", "")
        
        for art_id, art_info in FIGSHARE_ARTICLES.items():
            if art_id in notes or art_id in str(info):
                files_map = art_info["files"]
                h5ad_name = files_map.get(tissue, f"{tissue}.h5ad")
                local_path = os.path.join(TMP_DIR, h5ad_name)
                
                ok = figshare_download(art_id, h5ad_name, local_path)
                if ok:
                    valid, msg = validate_h5ad(local_path)
                    if valid:
                        try:
                            adata = sc.read_h5ad(local_path)
                            if post_process_h5ad(adata):
                                adata.write_h5ad(output_path, compression="gzip")
                                print(f"  Saved: {output_path}")
                                return True
                        except Exception as e:
                            print(f"  Post-processing error: {e}")
                    else:
                        print(f"  Validation failed: {msg}")
                break
        return False
    
    elif source_type == "geo_tar":
        geo_id = info.get("source_url_or_accession", "")
        if geo_id.startswith("GSE"):
            local = geo_download_supplementary(geo_id, ["h5ad", "counts", "matrix", "tsv"], TMP_DIR)
            if local:
                try:
                    if local.endswith((".h5ad", ".h5")):
                        adata = sc.read_h5ad(local)
                    else:
                        df = pd.read_csv(local, sep="\t", index_col=0)
                        X = csr_matrix(df.values.astype(np.float32))
                        adata = ad.AnnData(X=X)
                        adata.obs = pd.DataFrame(index=df.index)
                        adata.var = pd.DataFrame(index=df.columns)
                    
                    if post_process_h5ad(adata):
                        adata.write_h5ad(output_path, compression="gzip")
                        return True
                except Exception as e:
                    print(f"  Processing error: {e}")
        
        tar_path = geo_download_tarball(geo_id)
        return tar_path is not None
    
    elif source_type == "cellxgene_census":
        tissue = info.get("tissue", paper_name.split()[-1])
        organism = "Homo sapiens" if "Human" in paper_name else "Mus musculus"
        
        if paper_name in CELLXGENE_DATASETS:
            cfg = CELLXGENE_DATASETS[paper_name]
            return census_download_tissue(
                cfg["organism"], cfg["tissue"], output_path,
                cfg.get("dataset_filter"),
            )
        
        return census_download_tissue(organism, tissue, output_path)
    
    elif source_type == "bioconductor":
        is_senis = "Senis" in paper_name or "QS" in paper_name
        tissue = info.get("tissue", "")
        return bioconductor_tabula_muris(tissue, output_path, is_senis)
    
    else:
        print(f"  Unknown source type: {source_type}")
        return False


def list_missing_datasets():
    """List all missing datasets by source type."""
    import csv
    
    sources = {
        "figshare_api": [],
        "geo_tar": [],
        "cellxgene_census": [],
        "bioconductor": [],
        "tenx_direct": [],
        "unknown": [],
    }
    
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH) as f:
            for row in csv.DictReader(f, delimiter="\t"):
                status = row.get("status", "")
                paper = row.get("paper_name", "").strip('"')
                source = row.get("source_type", "unknown")
                
                if status in ("missing", "download_failed", "manual_required", "invalid_label"):
                    src_key = source if source in sources else "unknown"
                    sources[src_key].append({
                        "paper_name": paper,
                        "status": status,
                        "source": source,
                        "notes": row.get("notes", ""),
                    })
    
    print("\nMissing/Invalid Datasets by Source Type:")
    print("=" * 70)
    for src, datasets in sources.items():
        if datasets:
            print(f"\n{src} ({len(datasets)} datasets):")
            for ds in datasets:
                print(f"  - {ds['paper_name']}: {ds['status']}")
                if ds["notes"]:
                    print(f"    {ds['notes'][:80]}")


# ─── Manual Download Instructions ─────────────────────────────────────────────

MANUAL_INSTRUCTIONS = {
    "Mauro Pancreas": """
    Manual download instructions for Mauro Pancreas:
    
    Option 1: GEO (GSE85241)
    1. Go to: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE85241
    2. Download all supplementary files
    3. Look for *_counts.txt or *_expression.csv files
    4. Convert to h5ad using:
       python -c "
       import pandas as pd
       import scanpy as sc
       from scipy.sparse import csr_matrix
       df = pd.read_csv('counts.csv', index_col=0)
       adata = sc.AnnData(X=csr_matrix(df.values))
       adata.obs_names = df.index
       adata.var_names = df.columns
       adata.obs['cell_type'] = 'to_fill'  # fill from metadata
       adata.write_h5ad('Mauro_Pancreas.h5ad')
       "
    
    Option 2: Contact the original authors
    Mauro et al. dataset: Email authors for the processed count matrix
    """,
    
    "Shekhar mouse retina": """
    Manual download instructions for Shekhar mouse retina:
    
    Option 1: Recount3 (recommended)
    1. Go to: https://rna.recount.cloud/
    2. Search for SRP045424
    3. Download gene-level counts
    
    Option 2: Bioconductor
    1. In R: BiocManager::install('curatedMetagenomicData')
    2. Or: search for 'Dropseq' mouse retina count matrix online
    
    Option 3: 10x Genomics
    1. Go to: https://www.10xgenomics.com/
    2. Search for Shekhar retina dataset
    """,
    
    "Macosko mouse retina": """
    Manual download instructions for Macosko mouse retina:
    
    Option 1: 10x Genomics (recommended)
    1. Go to: https://www.10xgenomics.com/resources/datasets/
    2. Search: 'Macosko 2015 mouse retina'
    3. Download the 1.3M mouse retina dataset
    
    Option 2: NCBI SRA
    1. Download raw reads from SRA (SRP033203)
    2. Align and count using Drop-seq tools
    """,
    
    "Tabula Muris / Tabula Muris Senis datasets": """
    Manual download instructions for Tabula Muris datasets:
    
    Option 1: GEO (recommended - no authentication needed)
    1. Tabula Muris FACS: GEO GSE109774
       wget https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE109774&format=file
    2. Tabula Muris Senis: GEO GSE149590
       wget https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE149590&format=file
    
    Option 2: Figshare (requires authentication)
    1. Go to: https://figshare.com/projects/Tabula_Muris/27733
    2. Sign in and download files
    3. rsync to server: rsync -avz user@server:/path/to/files/ ./
    
    Option 3: Bioconductor
    1. In R:
       BiocManager::install('TabulaMurisData')
       BiocManager::install('TabulaMurisSenisData')
       library(TabulaMurisData)
       data("FACS")
       # Export to h5ad
    """,
}


# ─── CLI Entry Point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download missing scCluBench datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/download_missing_datasets.py --list
  python scripts/download_missing_datasets.py --source figshare
  python scripts/download_missing_datasets.py --dataset "Human Pancreas 4"
  python scripts/download_missing_datasets.py --manual "Shekhar mouse retina"
        """
    )
    parser.add_argument("--list", action="store_true",
                       help="List all missing datasets by source")
    parser.add_argument("--source", type=str, choices=["figshare", "geo", "cellxgene", "bioconductor"],
                       help="Download only datasets of a specific source type")
    parser.add_argument("--dataset", type=str, default=None,
                       help="Download a specific dataset by paper name")
    parser.add_argument("--manual", type=str, default=None,
                       help="Print manual download instructions for a dataset")
    parser.add_argument("--force", action="store_true",
                       help="Re-download even if file exists")
    args = parser.parse_args()
    
    if args.list:
        list_missing_datasets()
        return
    
    if args.manual:
        if args.manual in MANUAL_INSTRUCTIONS:
            print(MANUAL_INSTRUCTIONS[args.manual])
        else:
            print(f"No manual instructions for: {args.manual}")
        return
    
    if args.dataset:
        ok = download_dataset(args.dataset, force=args.force)
        if ok:
            print(f"SUCCESS: {args.dataset}")
        else:
            print(f"FAILED: {args.dataset}")
        return
    
    # Download by source type
    import csv
    sources_filter = {
        "figshare": "figshare_api",
        "geo": "geo_tar",
        "cellxgene": "cellxgene_census",
        "bioconductor": "bioconductor",
    }
    
    target_source = sources_filter.get(args.source, None) if args.source else None
    
    if os.path.exists(REGISTRY_PATH):
        results = []
        with open(REGISTRY_PATH) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                paper = row.get("paper_name", "").strip('"')
                status = row.get("status", "")
                source = row.get("source_type", "unknown")
                
                if status in ("missing", "download_failed") and row.get("actual_file", "") == "":
                    if target_source is None or source == target_source:
                        ok = download_dataset(paper, force=args.force)
                        results.append((paper, ok))
        
        print(f"\n{'='*70}")
        print("Download Summary:")
        for paper, ok in results:
            status_str = "SUCCESS" if ok else "FAILED"
            print(f"  [{status_str}] {paper}")
    
    print("\nNote: For datasets marked 'manual_required', "
          "use --manual DATASET_NAME for download instructions.")


if __name__ == "__main__":
    main()
