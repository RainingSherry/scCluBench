#!/usr/bin/env Rscript
# =============================================================================
# SC3 Clustering for scCluBench
# =============================================================================
# Runs SC3 (Single-Cell Consensus Clustering) on preprocessed data.
#
# Usage:
#   Rscript run_sc3.R <data_h5ad> <n_clusters> <save_dir>
#
# Depends on: SingleCellExperiment, SC3, SC3 is from Bioconductor

suppressPackageStartupMessages({
    library(optparse)
    library(SingleCellExperiment)
    library(SC3)
})

# Parse arguments
option_list <- list(
    make_option(c("-d", "--data"), type = "character",
                help = "Input h5ad file path", metavar = "file"),
    make_option(c("-k", "--n_clusters"), type = "integer", default = 10,
                help = "Number of clusters [default %default]"),
    make_option(c("-s", "--save_dir"), type = "character", default = ".",
                help = "Output directory"),
    make_option(c("-r", "--reduce"), type = "character", default = "both",
                help = "Dimensionality reduction: pca, kmeans, or both [default %default]")
)

opt <- parse_args(OptionParser(option_list = option_list,
                               description = "SC3 clustering for scCluBench"))

cat("SC3 Clustering for scCluBench\n")
cat("================================\n")
cat("Input: ", opt$data, "\n")
cat("K:     ", opt$n_clusters, "\n")
cat("Save:  ", opt$save_dir, "\n\n")

# Load data from Python-processed CSV if available
data_file <- file.path(opt$save_dir, "sc3_input.csv")
if (!file.exists(data_file)) {
    cat("ERROR: sc3_input.csv not found. Run preprocess first.\n")
    quit(status = 1)
}

# Read data
cat("Loading data...\n")
exprs_vals <- as.matrix(read.csv(data_file, row.names = 1))

# Create SingleCellExperiment object
cat("Creating SingleCellExperiment object...\n")
sce <- SingleCellExperiment(
    assays = list(counts = exprs_vals),
    reducedDims = SimpleList()
)

# Log-transform if not already done
if (max(exprs_vals) > 100) {
    assays(sce)$logcounts <- log2(exprs_vals + 1)
} else {
    assays(sce)$logcounts <- exprs_vals
}

# Get true labels if available
labels_file <- file.path(opt$save_dir, "true_labels.csv")
if (file.exists(labels_file)) {
    true_labels <- as.character(read.csv(labels_file, row.names = 1)$x)
    colData(sce)$true_label <- true_labels
    cat("True labels loaded:", length(true_labels), "cells\n")
}

n_cells <- ncol(sce)
cat("Cells:", n_cells, "\n")
cat("Genes:", nrow(sce), "\n\n")

# Run SC3
cat("Running SC3 clustering (k =", opt$n_clusters, ")...\n")

# SC3 parameters
sc3_ks <- opt$n_clusters  # number of clusters

# Run SC3
system.time({
    sce <- sc3(sce, ks = sc3_ks, biology = FALSE, n_cores = 4)
})

# Get predictions
pred_col <- paste0("sc3_", opt$n_clusters, "_clusters")
y_pred <- colData(sce)[[pred_col]]
y_pred <- as.numeric(as.character(y_pred)) - 1  # 0-indexed

cat("\nSC3 Results:\n")
cat("Predicted clusters:", length(unique(y_pred)), "\n")
cat("Cluster sizes:", paste(sort(table(y_pred), decreasing = TRUE), collapse = ", "), "\n")

# Save predictions
pred_df <- data.frame(
    cell = colnames(sce),
    cluster = y_pred
)
write.csv(pred_df, file.path(opt$save_dir, "types_pred.csv"),
          row.names = FALSE, quote = FALSE)

# Compute metrics if true labels available
if (file.exists(labels_file)) {
    source(file.path(opt$save_dir, "..", "..", "..", "..",
                     "..", "..", "evaluation.R"))

    true_labels <- colData(sce)$true_label
    if (!is.null(true_labels)) {
        acc <- SC3::sc3_cluster_distances(
            as.factor(y_pred),
            as.factor(as.numeric(true_labels))
        )$clustering_ac
        nmi <- SC3::sc3_cluster_distances(
            as.factor(y_pred),
            as.factor(as.numeric(true_labels))
        )$nmi
        ari <- SC3::sc3_cluster_distances(
            as.factor(y_pred),
            as.factor(as.numeric(true_labels))
        )$ari

        # Compute with sklearn-style metrics via Python bridge
        cat("\nMetrics:\n")
        cat(sprintf("  ACC: %.4f\n", acc))
        cat(sprintf("  NMI: %.4f\n", nmi))
        cat(sprintf("  ARI: %.4f\n", ari))

        metrics <- list(
            acc = acc,
            nmi = nmi,
            ari = ari
        )
    } else {
        metrics <- list(acc = NA, nmi = NA, ari = NA)
    }
} else {
    metrics <- list(acc = NA, nmi = NA, ari = NA)
}

# Save metrics as JSON
metrics_json <- jsonlite::toJSON(metrics, auto_unbox = TRUE, pretty = TRUE)
writeLines(metrics_json, file.path(opt$save_dir, "metrics.json"))

cat("\nResults saved to:", opt$save_dir, "\n")
cat("Done.\n")
