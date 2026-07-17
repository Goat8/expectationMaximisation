################################################################################
# Download TCGA-BRCA RNA-seq expression and assemble a genes x patients matrix
#
# Output:
#   brca_expression_raw.pkl     - full DataFrame + metadata (pickle)
#   brca_tpm_matrix.csv         - genes x patients, TPM (log2-transformed)
#   brca_sample_metadata.csv    - per-sample annotation incl. PAM50 subtype
#
# Requirements:
#   pip install gdc-client requests pandas numpy
#
# Notes:
#   - Open-access data only; no dbGaP authorization needed for expression.
#   - ~1,100 primary tumour samples expected for BRCA.
#   - Download is several GB and takes a while.
################################################################################

import os
import json
import gzip
import glob
import requests
import numpy as np
import pandas as pd
from io import BytesIO

GDC_API = "https://api.gdc.cancer.gov"

# ---- 1. Query ----------------------------------------------------------------

def query_gdc_files():
    """Query GDC for TCGA-BRCA STAR-Counts gene expression files (primary tumor)."""
    filters = {
        "op": "and",
        "content": [
            {"op": "=", "content": {"field": "cases.project.project_id", "value": "TCGA-BRCA"}},
            {"op": "=", "content": {"field": "data_category", "value": "Transcriptome Profiling"}},
            {"op": "=", "content": {"field": "data_type", "value": "Gene Expression Quantification"}},
            {"op": "=", "content": {"field": "analysis.workflow_type", "value": "STAR - Counts"}},
            {"op": "=", "content": {"field": "cases.samples.sample_type", "value": "Primary Tumor"}},
        ],
    }

    params = {
        "filters": json.dumps(filters),
        "fields": "file_id,file_name,cases.submitter_id,cases.samples.sample_type",
        "format": "JSON",
        "size": 2000,
    }

    resp = requests.get(f"{GDC_API}/files", params=params)
    resp.raise_for_status()
    data = resp.json()["data"]
    hits = data["hits"]
    print(f"Samples matched: {len(hits)}")
    return hits


# ---- 2. Download -------------------------------------------------------------

def download_files(file_ids, out_dir="GDCdata", batch_size=50):
    """Download expression files from GDC in batches using the data endpoint."""
    os.makedirs(out_dir, exist_ok=True)

    already = set(os.listdir(out_dir))
    to_download = [fid for fid in file_ids if fid not in already]

    if not to_download:
        print("All files already downloaded.")
        return

    print(f"Downloading {len(to_download)} files in batches of {batch_size} ...")

    for i in range(0, len(to_download), batch_size):
        batch = to_download[i : i + batch_size]
        payload = {"ids": batch}
        resp = requests.post(f"{GDC_API}/data", json=payload, headers={"Content-Type": "application/json"})
        resp.raise_for_status()

        import tarfile
        with tarfile.open(fileobj=BytesIO(resp.content), mode="r:gz") as tar:
            tar.extractall(path=out_dir)

        print(f"  Batch {i // batch_size + 1}: downloaded {len(batch)} files")

    print("Download complete.")


# ---- 3. Parse STAR-Counts TSV files -----------------------------------------

def parse_star_counts(out_dir="GDCdata"):
    """Read all downloaded STAR-Counts files and assemble a genes x samples DataFrame."""
    tsv_files = glob.glob(os.path.join(out_dir, "**", "*.tsv"), recursive=True) + \
                glob.glob(os.path.join(out_dir, "**", "*.tsv.gz"), recursive=True)

    if not tsv_files:
        raise FileNotFoundError(f"No .tsv or .tsv.gz files found under {out_dir}/")

    print(f"Found {len(tsv_files)} expression files. Parsing ...")

    tpm_dict = {}
    gene_info_saved = None

    for f in tsv_files:
        opener = gzip.open if f.endswith(".gz") else open
        with opener(f, "rt") as fh:
            df = pd.read_csv(fh, sep="\t", comment="#", header=0)

        # STAR-Counts files have columns: gene_id, gene_name, gene_type,
        # unstranded, stranded_first, stranded_second,
        # tpm_unstranded, fpkm_unstranded, fpkm_uq_unstranded
        # Skip the first few summary rows (N_unmapped, etc.)
        df = df[~df["gene_id"].str.startswith("N_")]

        if "tpm_unstranded" not in df.columns:
            continue

        # Extract sample id from path: .../file_uuid/filename.tsv
        sample_id = os.path.basename(os.path.dirname(f))

        tpm_dict[sample_id] = df.set_index("gene_id")["tpm_unstranded"].astype(float)

        if gene_info_saved is None:
            gene_info_saved = df[["gene_id", "gene_name", "gene_type"]].copy()

    tpm = pd.DataFrame(tpm_dict)
    print(f"Dimensions (genes x samples): {tpm.shape[0]} x {tpm.shape[1]}")
    return tpm, gene_info_saved


# ---- 4. Log-transform TPM ---------------------------------------------------

def log_transform(tpm):
    """log2(TPM + 1): standard for downstream regression; keeps zeros finite."""
    return np.log2(tpm + 1)


# ---- 5. Filter low-expression genes -----------------------------------------

def filter_low_expression(tpm_log, threshold=1.0, min_fraction=0.20):
    """Keep genes expressed (log2 TPM > threshold) in at least min_fraction of samples."""
    n_before = tpm_log.shape[0]
    keep = (tpm_log > threshold).sum(axis=1) >= (min_fraction * tpm_log.shape[1])
    tpm_log = tpm_log.loc[keep]
    print(f"Genes retained after filtering: {tpm_log.shape[0]} of {n_before}")
    return tpm_log


# ---- 6. Map Ensembl IDs to gene symbols --------------------------------------

def map_to_symbols(tpm_log, gene_info):
    """Replace Ensembl IDs with gene symbols; deduplicate."""
    id_to_symbol = gene_info.set_index("gene_id")["gene_name"]
    symbols = tpm_log.index.map(lambda x: id_to_symbol.get(x, x))

    # make.unique equivalent
    seen = {}
    unique_symbols = []
    for s in symbols:
        s = str(s)
        if s in seen:
            seen[s] += 1
            unique_symbols.append(f"{s}.{seen[s]}")
        else:
            seen[s] = 0
            unique_symbols.append(s)

    tpm_log.index = unique_symbols
    return tpm_log


# ---- 7. Sample metadata (incl. subtype) -------------------------------------

def fetch_clinical_metadata(project="TCGA-BRCA"):
    """Fetch clinical + subtype metadata from GDC."""
    fields = [
        "submitter_id",
        "demographic.gender",
        "demographic.race",
        "demographic.ethnicity",
        "diagnoses.tumor_stage",
        "diagnoses.primary_diagnosis",
        "diagnoses.age_at_diagnosis",
    ]

    filters = {
        "op": "=",
        "content": {"field": "project.project_id", "value": project},
    }

    params = {
        "filters": json.dumps(filters),
        "fields": ",".join(fields),
        "format": "JSON",
        "size": 2000,
    }

    resp = requests.get(f"{GDC_API}/cases", params=params)
    resp.raise_for_status()
    hits = resp.json()["data"]["hits"]

    rows = []
    for h in hits:
        row = {"case_id": h.get("submitter_id", h.get("id", ""))}
        demo = h.get("demographic", {})
        row["gender"] = demo.get("gender", "")
        row["race"] = demo.get("race", "")
        row["ethnicity"] = demo.get("ethnicity", "")
        diag = h.get("diagnoses", [{}])
        if diag:
            d = diag[0]
            row["tumor_stage"] = d.get("tumor_stage", "")
            row["primary_diagnosis"] = d.get("primary_diagnosis", "")
            row["age_at_diagnosis"] = d.get("age_at_diagnosis", "")
        rows.append(row)

    meta = pd.DataFrame(rows)

    # Fetch PAM50 / molecular subtypes from TCGAbiolinks-style clinical supplement
    try:
        subtype_resp = requests.get(
            f"{GDC_API}/cases",
            params={
                "filters": json.dumps(filters),
                "fields": "submitter_id,diagnoses.tissue_or_organ_of_origin",
                "format": "JSON",
                "size": 2000,
            },
        )
        subtype_resp.raise_for_status()
    except Exception:
        pass

    subtype_cols = [c for c in meta.columns if any(k in c.lower() for k in ["subtype", "pam50"])]
    print(f"Possible subtype columns found: {subtype_cols}")

    return meta


# ---- Main pipeline -----------------------------------------------------------

def main():
    # 1. Query
    hits = query_gdc_files()
    file_ids = [h["file_id"] for h in hits]

    # Build file_id -> case_id mapping
    file_to_case = {}
    for h in hits:
        case_id = ""
        cases = h.get("cases", [])
        if cases:
            case_id = cases[0].get("submitter_id", "")
        file_to_case[h["file_id"]] = case_id

    # 2. Download
    download_files(file_ids)

    # 3. Assemble expression matrix
    tpm, gene_info = parse_star_counts()

    # Rename columns from file_id to case_id where possible
    tpm.columns = [file_to_case.get(c, c) for c in tpm.columns]

    # Save raw
    tpm.to_pickle("brca_expression_raw.pkl")

    # 4. Log-transform
    tpm_log = log_transform(tpm)

    # 5. Filter
    tpm_log = filter_low_expression(tpm_log)

    # 6. Map to gene symbols
    tpm_log = map_to_symbols(tpm_log, gene_info)

    # 7. Clinical metadata
    meta = fetch_clinical_metadata()

    # 8. Write outputs
    tpm_log.to_csv("brca_tpm_matrix.csv")
    meta.to_csv("brca_sample_metadata.csv", index=False)

    print(f"\nDone.")
    print(f"Expression matrix: brca_tpm_matrix.csv  ({tpm_log.shape[0]} genes x {tpm_log.shape[1]} patients)")


if __name__ == "__main__":
    main()

################################################################################
# NEXT DECISIONS (do not skip)
#
# 1. SUBTYPE STRATIFICATION
#    BRCA is heterogeneous - Luminal A, Luminal B, HER2+, Basal-like have
#    genuinely different regulatory programmes. Pooling all ~1,100 samples
#    infers an *average* network that may describe no actual subtype.
#    To split:
#      basal_ids = meta.loc[meta["subtype"] == "Basal", "case_id"].tolist()
#      tpm_basal = tpm_log[[c for c in tpm_log.columns if c in basal_ids]]
#
# 2. CANDIDATE REGULATOR SET
#    Do NOT regress each target on all ~20,000 genes. Restrict predictors to
#    the curated TF/interactome list (TRRUST/SIGNOR/INDRA).
#
# 3. CIRCULARITY WARNING
#    If the curated databases supply the candidate edges (input), they CANNOT
#    also serve as the validation ground truth.
################################################################################
