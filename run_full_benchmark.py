#!/usr/bin/env python3
"""
Comprehensive scCluBench Reproduction Script
=============================================
Runs all models on all available datasets with full epochs.
Based on paper methodology (AAAI 2026).

CRITICAL: This script reads dataset list from data_manifest/dataset_registry.tsv.
Only datasets with status=valid are included.
DO NOT add invalid_label, missing, download_failed, or manual_required datasets.

Usage:
    python run_full_benchmark.py [--model MODEL] [--dataset DATASET] [--epochs EPOCHS]

Examples:
    python run_full_benchmark.py                          # Run all models on valid datasets only
    python run_full_benchmark.py --model Leiden          # Run only Leiden
    python run_full_benchmark.py --dataset Human_Pancreas_1  # Run all models on one dataset
    python run_full_benchmark.py --epochs 200            # Custom epochs
"""
import os
import sys
import argparse
import subprocess
import json
import time
import glob
import csv

# ─── Configuration ───────────────────────────────────────────────────────────

DATA_DIR = "/data/luolie/biopipeline/scCluBench/data"
RESULTS_DIR = "/data/luolie/biopipeline/dimension-reduction/scCluBench/results"
CODE_DIR = "/home/luolie/biopipeline/dimension-reduction/scCluBench-main"
REGISTRY_PATH = os.path.join(CODE_DIR, "data_manifest", "dataset_registry.tsv")


def load_registry() -> dict:
    """
    Load dataset registry from TSV. Only returns datasets with status=valid.

    The registry is the authoritative source of truth. If a dataset is not
    in the registry with status=valid, it MUST NOT be run.
    """
    datasets = {}

    if not os.path.exists(REGISTRY_PATH):
        print(f"WARNING: Registry not found at {REGISTRY_PATH}")
        print("  Falling back to hardcoded dataset list")
        return _hardcoded_datasets()

    # Read all lines, skip comments and empty lines
    header = None
    data_lines = []

    with open(REGISTRY_PATH) as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            if header is None:
                header = stripped
                continue
            data_lines.append(stripped)

    if header is None:
        print(f"ERROR: No header found in registry")
        return _hardcoded_datasets()

    headers = header.split('\t')

    for line in data_lines:
        values = line.split('\t')
        if len(values) < len(headers):
            continue

        row = dict(zip(headers, values))

        paper_name = row.get("paper_name", "").strip('"').strip()
        if not paper_name:
            continue

        status = row.get("status", "").strip()
        canonical_file = row.get("canonical_file", "").strip('"').strip()
        actual_file = row.get("actual_file", "").strip('"').strip()
        expected_n_clusters = row.get("expected_n_clusters", "").strip()
        label_field = row.get("label_field", "cell_type").strip()

        # Only include VALID datasets
        if status != "valid":
            continue

        # Determine actual filename
        filename = actual_file if actual_file and actual_file != "—" else canonical_file
        if not filename:
            continue

        key = paper_name.replace(" ", "_")

        datasets[key] = {
            "paper_name": paper_name,
            "file": filename,
            "canonical_file": canonical_file,
            "n_types": int(expected_n_clusters) if expected_n_clusters.isdigit() else None,
            "label_field": label_field,
            "status": status,
        }

    print(f"Loaded {len(datasets)} VALID datasets from registry:")
    for name, info in sorted(datasets.items()):
        print(f"  ✓ {name}: {info['file']} (n_clusters={info['n_types']})")

    return datasets


def _hardcoded_datasets() -> dict:
    """Fallback dataset list (only used if registry is missing)."""
    return {
        "Human_Pancreas_1": {
            "paper_name": "Human Pancreas 1",
            "file": "Human_Pancreas_1.h5ad",
            "n_types": 7,
            "label_field": "cell_type",
        },
        "Human_Pancreas_2": {
            "paper_name": "Human Pancreas 2",
            "file": "Human_Pancreas_2.h5ad",
            "n_types": 10,
            "label_field": "cell_type",
        },
        "Human_Pancreas_3": {
            "paper_name": "Human Pancreas 3",
            "file": "Human_Pancreas_3.h5ad",
            "n_types": 14,
            "label_field": "cell_type",
        },
        "Mouse_Pancreas_1": {
            "paper_name": "Mouse Pancreas 1",
            "file": "Mouse_Pancreas_1.h5ad",
            "n_types": 13,
            "label_field": "cell_type",
        },
        "Mouse_Pancreas_2": {
            "paper_name": "Mouse Pancreas 2",
            "file": "Mouse_Pancreas_2.h5ad",
            "n_types": 9,
            "label_field": "cell_type",
        },
    }


# Load datasets from registry
DATASETS = load_registry()

# Default epochs
DEFAULT_EPOCHS = {"scCDCG": 200, "scMAE": 100}

# Model configurations
MODELS = {
    "Leiden": {
        "script": "run_leiden_improved.py",
        "gpu": False,
    },
    "scCDCG": {
        "script": "GNN/scCDCG/run.py",
        "gpu": True,
    },
    "scMAE": {
        "script": "DeepLearning/scMAE/run.py",
        "gpu": True,
    },
}

GPU_ID = "0"


def run_command(cmd, env=None, timeout=None):
    """Run a shell command and return output."""
    result = subprocess.run(
        cmd, capture_output=True, text=True, env=env, timeout=timeout, cwd=CODE_DIR,
    )
    return result


def check_existing(model, dataset):
    """Check if results already exist."""
    path = os.path.join(RESULTS_DIR, model, dataset, "metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return True, json.load(f)
    return False, None


def run_leiden(dataset_name, dataset_info):
    """Run Leiden baseline."""
    data_path = os.path.join(DATA_DIR, dataset_info["file"])
    if not os.path.exists(data_path):
        print(f"  [SKIP] {dataset_name}: file not found")
        return "skip", None

    exists, m = check_existing("Leiden", dataset_name)
    if exists:
        print(f"  [SKIP] Leiden/{dataset_name}: ACC={m['acc']:.4f}")
        return "skip", m

    n_types = dataset_info.get("n_types")
    if not n_types:
        return "skip", None

    save_dir = os.path.join(RESULTS_DIR, "Leiden", dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    cmd = [
        sys.executable, os.path.join(CODE_DIR, "run_leiden_improved.py"),
        data_path, RESULTS_DIR, dataset_name, str(n_types),
    ]

    result = run_command(cmd, env=os.environ.copy(), timeout=3600)
    if result.returncode == 0:
        _, m = check_existing("Leiden", dataset_name)
        if m:
            print(f"  [OK] Leiden/{dataset_name}: ACC={m['acc']:.4f}")
            return "ok", m
    print(f"  [FAIL] Leiden/{dataset_name}")
    return "fail", None


def run_scCDCG(dataset_name, dataset_info, epochs=None):
    """Run scCDCG."""
    data_path = os.path.join(DATA_DIR, dataset_info["file"])
    if not os.path.exists(data_path):
        print(f"  [SKIP] {dataset_name}: file not found")
        return "skip", None

    exists, m = check_existing("scCDCG", dataset_name)
    if exists:
        print(f"  [SKIP] scCDCG/{dataset_name}: ACC={m['acc']:.4f}")
        return "skip", m

    n_types = dataset_info.get("n_types")
    if not n_types:
        return "skip", None

    n_epochs = epochs or DEFAULT_EPOCHS["scCDCG"]
    save_dir = os.path.join(RESULTS_DIR, "scCDCG", dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    cmd = [
        sys.executable, os.path.join(CODE_DIR, "GNN/scCDCG/run.py"),
        "--data_path", data_path, "--save_dir", save_dir,
        "--n_clusters", str(n_types), "--epochs", str(n_epochs),
        "--embedding_dim", "16", "--hidden_dim", "256",
        "--gpu", GPU_ID,
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPU_ID
    start = time.time()
    result = run_command(cmd, env=env, timeout=n_epochs * 60 + 600)
    elapsed = time.time() - start

    if result.returncode == 0:
        _, m = check_existing("scCDCG", dataset_name)
        if m:
            print(f"  [OK] scCDCG/{dataset_name}: ACC={m['acc']:.4f} ({elapsed:.0f}s)")
            return "ok", m
    print(f"  [FAIL] scCDCG/{dataset_name}")
    return "fail", None


def run_scMAE(dataset_name, dataset_info, epochs=None):
    """Run scMAE."""
    data_path = os.path.join(DATA_DIR, dataset_info["file"])
    if not os.path.exists(data_path):
        print(f"  [SKIP] {dataset_name}: file not found")
        return "skip", None

    exists, m = check_existing("scMAE", dataset_name)
    if exists:
        print(f"  [SKIP] scMAE/{dataset_name}: ACC={m['acc']:.4f}")
        return "skip", m

    n_types = dataset_info.get("n_types")
    if not n_types:
        return "skip", None

    n_epochs = epochs or DEFAULT_EPOCHS["scMAE"]
    save_dir = os.path.join(RESULTS_DIR, "scMAE", dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    cmd = [
        sys.executable, os.path.join(CODE_DIR, "DeepLearning/scMAE/run.py"),
        "--data_path", data_path, "--save_dir", save_dir,
        "--n_clusters", str(n_types), "--epochs", str(n_epochs),
        "--hidden_size", "128", "--batch_size", "256",
        "--gpu", GPU_ID,
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPU_ID
    start = time.time()
    result = run_command(cmd, env=env, timeout=n_epochs * 60 + 600)
    elapsed = time.time() - start

    if result.returncode == 0:
        _, m = check_existing("scMAE", dataset_name)
        if m:
            print(f"  [OK] scMAE/{dataset_name}: ACC={m['acc']:.4f} ({elapsed:.0f}s)")
            return "ok", m
    print(f"  [FAIL] scMAE/{dataset_name}")
    return "fail", None


def generate_summary():
    """Generate summary CSV from all results."""
    rows = []
    for model_dir in glob.glob(os.path.join(RESULTS_DIR, "*")):
        if not os.path.isdir(model_dir):
            continue
        model_name = os.path.basename(model_dir)
        for dataset_dir in glob.glob(os.path.join(model_dir, "*")):
            if not os.path.isdir(dataset_dir):
                continue
            dataset_name = os.path.basename(dataset_dir)
            metrics_path = os.path.join(dataset_dir, "metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path) as f:
                    m = json.load(f)
                rows.append({
                    "dataset": dataset_name, "model": model_name,
                    "ACC": m.get("acc", ""), "NMI": m.get("nmi", ""),
                    "ARI": m.get("ari", ""),
                })

    if rows:
        csv_path = os.path.join(CODE_DIR, "results_summary.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSummary saved to: {csv_path}")

    return rows


def main():
    parser = argparse.ArgumentParser(description="scCluBench Full Reproduction")
    parser.add_argument("--model", type=str, default=None,
                        choices=["Leiden", "scCDCG", "scMAE", "all"])
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("scCluBench Comprehensive Reproduction")
    print("=" * 70)
    print(f"Data dir:   {DATA_DIR}")
    print(f"Results dir: {RESULTS_DIR}")
    print()

    # Filter datasets
    datasets_to_run = DATASETS
    if args.dataset:
        if args.dataset not in DATASETS:
            print(f"Unknown dataset: {args.dataset}")
            print(f"Available: {list(DATASETS.keys())}")
            return
        datasets_to_run = {args.dataset: DATASETS[args.dataset]}

    # Filter models
    models_to_run = list(MODELS.keys())
    if args.model and args.model != "all":
        if args.model in MODELS:
            models_to_run = [args.model]

    summary = {"ok": 0, "skip": 0, "fail": 0}

    if "Leiden" in models_to_run:
        print("\n" + "=" * 70)
        print("PHASE 1: Leiden Baseline")
        print("=" * 70)
        for ds_name, ds_info in sorted(datasets_to_run.items()):
            status, _ = run_leiden(ds_name, ds_info)
            summary[status] = summary.get(status, 0) + 1

    if "scCDCG" in models_to_run:
        print("\n" + "=" * 70)
        print("PHASE 2: scCDCG")
        print("=" * 70)
        for ds_name, ds_info in sorted(datasets_to_run.items()):
            status, _ = run_scCDCG(ds_name, ds_info, epochs=args.epochs)
            summary[status] = summary.get(status, 0) + 1

    if "scMAE" in models_to_run:
        print("\n" + "=" * 70)
        print("PHASE 3: scMAE")
        print("=" * 70)
        for ds_name, ds_info in sorted(datasets_to_run.items()):
            status, _ = run_scMAE(ds_name, ds_info, epochs=args.epochs)
            summary[status] = summary.get(status, 0) + 1

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"  OK:    {summary['ok']}")
    print(f"  SKIP:  {summary['skip']}")
    print(f"  FAIL:  {summary['fail']}")

    rows = generate_summary()
    if rows:
        print("\nResults Table:")
        print(f"{'Dataset':<35} {'Model':<10} {'ACC':>7} {'NMI':>7} {'ARI':>7}")
        print("-" * 70)
        for r in rows:
            print(f"{r['dataset']:<35} {r['model']:<10} "
                  f"{r['ACC']:>7.4f} {r['NMI']:>7.4f} {r['ARI']:>7.4f}")


if __name__ == "__main__":
    main()
