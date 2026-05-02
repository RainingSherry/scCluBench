#!/usr/bin/env python3
"""
audit_h5ad_labels.py
====================
Scans every .h5ad file in the data directory and checks:
  1. Which obs columns could be cell-type labels (categorical/object, 2-200 unique values)
  2. Whether raw counts / n_counts / size_factors are present
  3. Whether the actual n_clusters matches the paper's Table 2 expected value
  4. Whether the file should be marked status=invalid_label

Output:
  - Console table of all datasets
  - Updates data_manifest/dataset_registry.tsv with actual stats
  - Warns about files that must NOT be added to run_full_benchmark.py

Usage:
  python scripts/audit_h5ad_labels.py [--data_dir DIR] [--registry TSV]
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

import numpy as np
import scanpy as sc
import scipy.sparse as sp


# ─── Paper Table 2 expected n_clusters ────────────────────────────────────────
PAPER_N_CLUSTERS = {
    "Human Pancreas 1": 7,
    "Human Pancreas 2": 10,
    "Human Pancreas 3": 14,
    "Human Pancreas 4": 7,
    "Mauro Pancreas": 10,
    "68K PBMC": 8,
    "CITE CMBC": 8,
    "Human Kidney": 10,
    "Sonya Liver": 10,
    "Sapiens Liver": 15,
    "Sapiens Ear Crista Ampullaris": 7,
    "Sapiens Ear Utricle": 5,
    "Sapiens Lung": 25,
    "Sapiens Testis": 8,
    "Sapiens Trachea": 20,
    "Mouse cerebral cortex": 9,
    "Mouse embryonic stem": 4,
    "Mouse hypothalamus": 10,
    "Mouse Pancreas 1": 13,
    "Mouse Pancreas 2": 9,
    "Shekhar mouse retina": 19,
    "Macosko mouse retina": 39,
    "Mouse Kidney": 7,
    "Mouse bladder": 8,
    "QS Diaphragm": 5,
    "QS Lung": 11,
    "QS Trachea": 4,
    "QS Limb Muscle": 6,
    "Ox Limb Muscle": 6,
    "Ox Bladder": 5,
    "Ox Spleen": 5,
    "Muris Limb Muscle": 6,
    "Muris Brain": 16,
    "Muris Kidney": 9,
    "Muris Liver": 11,
    "Muris Lung": 25,
}

# Map canonical filenames → paper names
CANONICAL_TO_PAPER = {
    "Human_Pancreas_1.h5ad": "Human Pancreas 1",
    "Human_Pancreas_2.h5ad": "Human Pancreas 2",
    "Human_Pancreas_3.h5ad": "Human Pancreas 3",
    "Human_Pancreas_4.h5ad": "Human Pancreas 4",
    "Mauro_Pancreas.h5ad": "Mauro Pancreas",
    "68K_PBMC.h5ad": "68K PBMC",
    "PBMC_68K.h5ad": "68K PBMC",
    "CITE_CMBC.h5ad": "CITE CMBC",
    "Human_Kidney.h5ad": "Human Kidney",
    "Sonya_Liver.h5ad": "Sonya Liver",
    "Sapiens_Liver.h5ad": "Sapiens Liver",
    "Sapiens_Ear_Crista_Ampullaris.h5ad": "Sapiens Ear Crista Ampullaris",
    "Sapiens_Ear_Utricle.h5ad": "Sapiens Ear Utricle",
    "Sapiens_Lung.h5ad": "Sapiens Lung",
    "Sapiens_Testis.h5ad": "Sapiens Testis",
    "Sapiens_Trachea.h5ad": "Sapiens Trachea",
    "Mouse_cerebral_cortex.h5ad": "Mouse cerebral cortex",
    "Mouse_embryonic_stem.h5ad": "Mouse embryonic stem",
    "Mouse_hypothalamus.h5ad": "Mouse hypothalamus",
    "Mouse_Pancreas_1.h5ad": "Mouse Pancreas 1",
    "Mouse_Pancreas_2.h5ad": "Mouse Pancreas 2",
    "Shekhar_mouse_retina.h5ad": "Shekhar mouse retina",
    "Macosko_mouse_retina.h5ad": "Macosko mouse retina",
    "Mouse_Kidney.h5ad": "Mouse Kidney",
    "Mouse_bladder.h5ad": "Mouse bladder",
    "QS_Diaphragm.h5ad": "QS Diaphragm",
    "QS_Lung.h5ad": "QS Lung",
    "QS_Trachea.h5ad": "QS Trachea",
    "QS_Limb_Muscle.h5ad": "QS Limb Muscle",
    "Ox_Limb_Muscle.h5ad": "Ox Limb Muscle",
    "Ox_Bladder.h5ad": "Ox Bladder",
    "Ox_Spleen.h5ad": "Ox Spleen",
    "Muris_Limb_Muscle.h5ad": "Muris Limb Muscle",
    "Muris_Brain.h5ad": "Muris Brain",
    "Muris_Kidney.h5ad": "Muris Kidney",
    "Muris_Liver.h5ad": "Muris Liver",
    "Muris_Lung.h5ad": "Muris Lung",
}

# Extra (non-Table 2) datasets
EXTRA_FILES = {
    "Arabidopsis_scRNA_synthetic.h5ad": "extra",
    "Arabidopsis_Stereo-seq_leaf.h5ad": "extra",
    "Arabidopsis_Stereo-seq_leaf_S1-2.h5ad": "extra",
    "Blood_BoneMarrow.h5ad": "extra",
    "Bone_Marrow.h5ad": "extra",
    "Plant_scRNA_synthetic.h5ad": "extra",
    "TabulaSapiens_Pancreas.h5ad": "extra",
    "PBMC3K.h5ad": "extra",
    "pbmc3k_raw.h5ad": "extra",
}


def compute_sparsity(adata) -> float:
    """Compute sparsity (% zeros) of the expression matrix."""
    X = adata.X
    if sp.issparse(X):
        arr = X.toarray()
    else:
        arr = np.asarray(X)
    total = arr.size
    nonzero = np.count_nonzero(arr)
    return (1.0 - nonzero / total) * 100 if total > 0 else 0.0


def find_cell_type_column(obs) -> tuple:
    """
    Scan obs columns for a likely cell-type label.
    Returns (col_name, n_unique, cell_types_dict, is_valid_label).

    Rules:
      - column must be categorical or object dtype
      - n_unique must be >= 1 (1 = INVALID, 2-200 = valid)
      - If n_unique == 1: INVALID (all cells same label = broken parsing)
      - If n_unique >= 200: probably NOT a cell type (donor ID, etc.)
      - Also check if all labels are numeric-looking (IDs, not cell types)
    """
    candidates = []
    for col in obs.columns:
        dtype = obs[col].dtype
        if dtype.name not in ("category", "object"):
            continue
        n_unique = obs[col].nunique()
        if n_unique < 1 or n_unique > 200:
            continue
        # Check if labels look like cell type names (not numeric IDs)
        sample = obs[col].dropna().astype(str).head(20)
        numeric_ratio = sum(v.replace(".", "").replace("-", "").replace("+", "").isdigit() for v in sample) / max(len(sample), 1)
        is_likely_label = numeric_ratio < 0.5  # not mostly numeric
        candidates.append({
            "col": col,
            "n_unique": n_unique,
            "counts": obs[col].value_counts().to_dict(),
            "is_likely_label": is_likely_label,
        })

    # Sort: prefer columns with "cell_type" in name, then by n_unique, then likely_label
    candidates.sort(
        key=lambda x: (
            0 if "cell_type" in x["col"].lower() else 1,
            0 if x["is_likely_label"] else 1,
            -x["n_unique"],  # prefer more clusters (higher is better for cell type)
        )
    )

    if candidates:
        best = candidates[0]
        # n_unique == 1 means all cells same label = invalid
        is_valid = best["n_unique"] >= 2
        return best["col"], best["n_unique"], best["counts"], is_valid
    return None, 0, {}, False


def audit_file(filepath: str) -> dict:
    """Audit a single h5ad file. Returns a dict of findings."""
    result = {
        "filepath": filepath,
        "filename": os.path.basename(filepath),
        "exists": os.path.exists(filepath),
        "readable": False,
        "n_obs": None,
        "n_vars": None,
        "sparsity": None,
        "obs_columns": [],
        "ct_col": None,
        "ct_n_unique": None,
        "ct_counts": {},
        "ct_is_valid": False,
        "has_raw": False,
        "has_ncounts": False,
        "has_sizefactors": False,
        "paper_name": None,
        "paper_n_clusters": None,
        "cluster_match": None,
        "status": "missing",
        "warnings": [],
        "errors": [],
    }

    if not result["exists"]:
        result["status"] = "missing"
        return result

    # Try to read
    try:
        ad = sc.read_h5ad(filepath, backed="r")
        result["readable"] = True
    except Exception as e:
        result["readable"] = False
        result["errors"].append(f"read_h5ad failed: {e}")
        result["status"] = "corrupt"
        return result

    result["n_obs"] = ad.n_obs
    result["n_vars"] = ad.n_vars
    result["obs_columns"] = list(ad.obs.columns)

    # Compute sparsity (need to load data for this)
    try:
        ad_full = sc.read_h5ad(filepath)  # load fully for sparsity
        result["sparsity"] = compute_sparsity(ad_full)
    except Exception as e:
        result["warnings"].append(f"Could not compute sparsity: {e}")
        result["sparsity"] = None

    # Check raw, n_counts, size_factors
    try:
        ad_full2 = sc.read_h5ad(filepath)
        result["has_raw"] = ad_full2.raw is not None
        result["has_ncounts"] = "n_counts" in ad_full2.obs.columns
        result["has_sizefactors"] = "size_factors" in ad_full2.obs.columns
        del ad_full2
    except Exception:
        pass

    # Find cell type column
    try:
        ad_full3 = sc.read_h5ad(filepath)
        ct_col, ct_n, ct_counts, ct_valid = find_cell_type_column(ad_full3.obs)
        result["ct_col"] = ct_col
        result["ct_n_unique"] = ct_n
        result["ct_counts"] = ct_counts
        result["ct_is_valid"] = ct_valid
        del ad_full3
    except Exception as e:
        result["warnings"].append(f"Could not scan obs columns: {e}")

    # Map to paper name (case-insensitive lookup)
    fname = result["filename"]
    result["paper_name"] = CANONICAL_TO_PAPER.get(fname, "unknown")
    if result["paper_name"] == "unknown":
        # Try lowercase match
        for key, val in CANONICAL_TO_PAPER.items():
            if key.lower() == fname.lower():
                result["paper_name"] = val
                break
        if result["paper_name"] == "unknown" and fname in EXTRA_FILES:
            result["paper_name"] = f"extra: {fname}"
            result["status"] = "extra"
    
    # Check cluster match
    if result["paper_name"] in PAPER_N_CLUSTERS:
        result["paper_n_clusters"] = PAPER_N_CLUSTERS[result["paper_name"]]
        if result["ct_n_unique"] is not None:
            if result["ct_n_unique"] == result["paper_n_clusters"]:
                result["cluster_match"] = "exact"
                result["status"] = "valid"
            elif result["ct_n_unique"] == 1:
                result["cluster_match"] = "INVALID: only 1 type"
                result["status"] = "invalid_label"
            else:
                result["cluster_match"] = f"MISMATCH: got {result['ct_n_unique']}, expected {result['paper_n_clusters']}"
                result["status"] = "invalid_label"
        else:
            result["status"] = "no_ct_column"
    elif result["status"] == "extra":
        pass  # don't check extra files
    else:
        result["status"] = "no_paper_entry"
    
    return result


def print_report(results: list):
    """Print a formatted console report."""
    print("\n" + "=" * 120)
    print("H5AD LABEL AUDIT REPORT")
    print("=" * 120)
    
    # Table header
    hdr = (f"{'File':<42} {'Status':<15} {'Cells':>7} {'Genes':>7} "
           f"{'Types':>6} {'Exp':>4} {'Match':<30} {'raw':>4} {'n_cnt':>6} {'sf':>3}")
    print(hdr)
    print("-" * 120)
    
    # Group by status
    status_order = ["valid", "invalid_label", "no_ct_column", "corrupt", "missing", "extra", "no_paper_entry"]
    status_labels = {
        "valid": "VALID",
        "invalid_label": "INVALID_LABEL",
        "no_ct_column": "NO_CT_COLUMN",
        "corrupt": "CORRUPT",
        "missing": "MISSING",
        "extra": "EXTRA",
        "no_paper_entry": "NO_PAPER_ENTRY",
    }
    
    for status in status_order:
        group = [r for r in results if r["status"] == status]
        if not group:
            continue
        for r in group:
            fname = r["filename"][:40]
            cells = r["n_obs"] if r["n_obs"] else "-"
            genes = r["n_vars"] if r["n_vars"] else "-"
            n_ct = r["ct_n_unique"] if r["ct_n_unique"] is not None else "-"
            exp_ct = r["paper_n_clusters"] if r["paper_n_clusters"] else "-"
            match = r.get("cluster_match", "-") or "-"
            if len(str(match)) > 28:
                match = str(match)[:25] + "..."
            raw = "Y" if r["has_raw"] else "N"
            ncnt = "Y" if r["has_ncounts"] else "N"
            sf = "Y" if r["has_sizefactors"] else "N"
            
            print(f"{fname:<42} {status_labels[status]:<15} "
                  f"{str(cells):>7} {str(genes):>7} "
                  f"{str(n_ct):>6} {str(exp_ct):>4} "
                  f"{str(match):<30} {raw:>4} {ncnt:>6} {sf:>3}")
    
    print("-" * 120)
    
    # Summary counts
    counts = {}
    for r in results:
        s = r["status"]
        counts[s] = counts.get(s, 0) + 1
    
    summary = "  ".join(f"{status_labels.get(s, s)}={n}" for s, n in sorted(counts.items()))
    print(f"Total files scanned: {len(results)}")
    print(f"Summary: {summary}")
    
    # Warnings / errors
    problems = [(r, w) for r in results for w in r["warnings"]]
    errors = [(r, e) for r in results for e in r["errors"]]
    
    if errors:
        print(f"\n{len(errors)} ERRORS:")
        for r, e in errors:
            print(f"  [{r['filename']}] {e}")
    
    if problems:
        print(f"\n{len(problems)} WARNINGS:")
        for r, w in problems:
            print(f"  [{r['filename']}] {w}")
    
    # Blocked datasets warning
    blocked = [r for r in results if r["status"] == "invalid_label"]
    if blocked:
        print(f"\n{'='*120}")
        print(f"BLOCKED ({len(blocked)} datasets with invalid labels — MUST NOT be added to run_full_benchmark.py):")
        for r in blocked:
            match = r.get("cluster_match", "unknown")
            print(f"  - {r['filename']}: {match}")
    
    # Valid datasets
    valid = [r for r in results if r["status"] == "valid"]
    if valid:
        print(f"\n{'='*120}")
        print(f"VALID ({len(valid)} datasets confirmed correct — safe to add to run_full_benchmark.py):")
        for r in valid:
            print(f"  - {r['filename']}: {r['n_obs']} cells × {r['n_vars']} genes, "
                  f"{r['ct_n_unique']} types, sparsity={r['sparsity']:.1f}%")
            if r["has_raw"] and r["has_ncounts"] and r["has_sizefactors"]:
                print(f"    ✓ has raw, n_counts, size_factors")


def update_registry(registry_path: str, results: list):
    """Update the dataset_registry.tsv with actual values from audit results."""
    import csv

    # Load existing registry
    with open(registry_path, "r") as f:
        content = f.read()

    # Build lookup: canonical_file → result
    result_map = {}
    for r in results:
        fname = r["filename"]
        result_map[fname] = r

    # Process line by line to preserve comments and structure
    lines = content.splitlines()
    header = lines[0]  # '#' comment header
    tsv_header = lines[1].split("\t")

    updated_lines = [header, lines[1]]  # keep header + column header

    for line in lines[2:]:
        cols = line.split("\t")
        if len(cols) < len(tsv_header):
            updated_lines.append(line)
            continue

        canonical = cols[0]  # first column is canonical_file
        r = result_map.get(canonical)

        # Find column indices
        def col_idx(name):
            try:
                return tsv_header.index(name)
            except ValueError:
                return -1

        if r and r["readable"]:
            ci_nobs = col_idx("actual_n_obs")
            ci_nvar = col_idx("actual_n_vars")
            ci_nclu = col_idx("actual_n_clusters")
            ci_spar = col_idx("actual_sparsity")
            ci_raw = col_idx("has_raw")
            ci_ncnt = col_idx("has_ncounts")
            ci_sf = col_idx("has_size_factors")
            ci_stat = col_idx("status")

            if ci_nobs >= 0 and r["n_obs"] is not None:
                cols[ci_nobs] = str(r["n_obs"])
            if ci_nvar >= 0 and r["n_vars"] is not None:
                cols[ci_nvar] = str(r["n_vars"])
            if ci_nclu >= 0 and r["ct_n_unique"] is not None:
                cols[ci_nclu] = str(r["ct_n_unique"])
            if ci_spar >= 0 and r["sparsity"] is not None:
                cols[ci_spar] = f"{r['sparsity']:.2f}%"
            if ci_raw >= 0:
                cols[ci_raw] = "yes" if r["has_raw"] else "no"
            if ci_ncnt >= 0:
                cols[ci_ncnt] = "yes" if r["has_ncounts"] else "no"
            if ci_sf >= 0:
                cols[ci_sf] = "yes" if r["has_sizefactors"] else "no"
            if ci_stat >= 0 and r["status"] in ("valid", "invalid_label", "no_ct_column", "corrupt"):
                cols[ci_stat] = r["status"]

        updated_lines.append("\t".join(cols))

    # Write back
    backup_path = registry_path + ".bak"
    with open(backup_path, "w") as f:
        f.write("\n".join(updated_lines) + "\n")

    with open(registry_path, "w") as f:
        f.write("\n".join(updated_lines) + "\n")

    updated = sum(
        1 for r in results
        if r["readable"] and r["filename"] in result_map
    )
    print(f"\nRegistry updated: {updated} rows updated (backup: {backup_path})")
    return updated_lines


def main():
    parser = argparse.ArgumentParser(description="Audit h5ad files for cell-type label validity")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/luolie/biopipeline/dimension-reduction/scCluBench-main/data",
        help="Path to data directory containing .h5ad files",
    )
    parser.add_argument(
        "--registry",
        type=str,
        default="/home/luolie/biopipeline/dimension-reduction/scCluBench-main/data_manifest/dataset_registry.tsv",
        help="Path to dataset_registry.tsv",
    )
    parser.add_argument(
        "--skip_update",
        action="store_true",
        help="Skip updating the registry TSV",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Audit a single file instead of all files",
    )
    args = parser.parse_args()
    
    if args.file:
        results = [audit_file(args.file)]
    else:
        data_dir = Path(args.data_dir)
        files = sorted(data_dir.glob("*.h5ad"))
        results = []
        for fp in files:
            print(f"Scanning: {fp.name}...", end=" ", flush=True)
            r = audit_file(str(fp))
            print(f"  {r['status']} | {r['n_obs'] or '?'} cells | "
                  f"{r['ct_n_unique'] or '?'} types | raw={r['has_raw']}")
            results.append(r)
    
    print_report(results)
    
    if not args.skip_update and not args.file:
        if os.path.exists(args.registry):
            update_registry(args.registry, results)
        else:
            print(f"\nRegistry not found at {args.registry}, skipping update.")
    
    # Exit code: 1 if any invalid_label found
    blocked = [r for r in results if r["status"] == "invalid_label"]
    if blocked:
        print(f"\n⚠ {len(blocked)} INVALID_LABEL datasets found — see above")
        sys.exit(1)
    
    print("\n✓ All audited datasets passed label validation")
    sys.exit(0)


if __name__ == "__main__":
    main()
