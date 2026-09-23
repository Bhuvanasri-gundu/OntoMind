"""
merge_and_clean.py
==================
Merges the original PubMed dataset with newly collected papers,
performs deduplication, basic cleaning, and produces a cleaned dataset
ready for quality analysis and relevance filtering.

Usage:
    python src/merge_and_clean.py

Inputs:
    data/raw/mhealth_pubmed_original_491.csv   (original 491 papers)
    data/raw/pubmed_expanded_*.csv              (newly collected papers)

Outputs:
    data/processed/mhealth_merged_dataset.csv
    data/processed/mhealth_final_cleaned_dataset.csv
    reports/summaries/merge_report.txt

Author: Project Team
Date: 2026-08-29
"""

import os
import re
import glob
import datetime
import pandas as pd

# ==============================================================
# PATHS
# ==============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "summaries")

ORIGINAL_FILE = os.path.join(RAW_DIR, "mhealth_pubmed_original_491.csv")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def extract_year(year_str):
    """Extract 4-digit year from year string like '2026 Aug 15'."""
    if pd.isna(year_str):
        return None
    match = re.search(r"\d{4}", str(year_str))
    return match.group(0) if match else None


def clean_text_for_nlp(text):
    """
    Clean text for traditional NLP tasks (TF-IDF, LDA).
    - Lowercase
    - Remove URLs
    - Remove special characters but keep alphanumeric
    - Normalize whitespace
    
    Preserves biomedical terms as much as possible by not
    aggressively removing hyphens or abbreviations.
    """
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)

    # Remove special characters (keep letters, numbers, spaces)
    # Keep hyphens between words for terms like "e-health", "m-health"
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)

    # Normalize multiple hyphens
    text = re.sub(r"-{2,}", " ", text)

    # Remove standalone hyphens
    text = re.sub(r"\s-\s", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_title(title):
    """Normalize title for deduplication comparison."""
    title = str(title).lower().strip()
    # Remove punctuation and extra spaces
    title = re.sub(r"[^a-z0-9\s]", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def main():
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("MERGE AND DEDUPLICATION REPORT")
    report_lines.append(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 60)

    # --------------------------------------------------
    # 1. Load original dataset
    # --------------------------------------------------
    print("Loading original dataset...")
    df_original = pd.read_csv(ORIGINAL_FILE)
    df_original["source"] = "original"
    original_count = len(df_original)
    print(f"  Original papers: {original_count}")
    report_lines.append(f"\nOriginal dataset: {original_count} papers")
    report_lines.append(f"  File: {ORIGINAL_FILE}")
    report_lines.append(f"  Columns: {df_original.columns.tolist()}")

    # --------------------------------------------------
    # 2. Load newly collected data
    # --------------------------------------------------
    print("Loading newly collected data...")
    expanded_files = sorted(glob.glob(os.path.join(RAW_DIR, "pubmed_expanded_*.csv")))

    if not expanded_files:
        print("ERROR: No expanded data files found in data/raw/")
        print("Run src/collect_pubmed_data.py first.")
        return

    # Use the most recent expanded file
    expanded_file = expanded_files[-1]
    df_new = pd.read_csv(expanded_file)
    df_new["source"] = "expanded"
    new_count = len(df_new)
    print(f"  Newly collected papers: {new_count}")
    print(f"  From file: {expanded_file}")
    report_lines.append(f"\nNewly collected dataset: {new_count} papers")
    report_lines.append(f"  File: {expanded_file}")

    # --------------------------------------------------
    # 3. Standardize columns before merge
    # --------------------------------------------------
    print("Standardizing columns...")

    # Original has: pmid, title, abstract, year, mesh_terms
    # New has: pmid, title, abstract, year_raw, mesh_terms, journal, doi, authors, search_query

    # Rename year_raw to year_raw in new, and ensure we have consistent cols
    if "year_raw" in df_new.columns and "year" not in df_new.columns:
        df_new["year"] = df_new["year_raw"]

    # Ensure both have the same core columns
    core_columns = ["pmid", "title", "abstract", "year", "mesh_terms"]

    for col in core_columns:
        if col not in df_original.columns:
            df_original[col] = ""
        if col not in df_new.columns:
            df_new[col] = ""

    # Add missing columns to original with empty values
    for col in ["journal", "doi", "authors", "search_query", "year_raw"]:
        if col not in df_original.columns:
            df_original[col] = ""

    # --------------------------------------------------
    # 4. Merge datasets
    # --------------------------------------------------
    print("Merging datasets...")
    all_columns = list(
        set(df_original.columns.tolist() + df_new.columns.tolist())
    )

    # Ensure both DataFrames have all columns
    for col in all_columns:
        if col not in df_original.columns:
            df_original[col] = ""
        if col not in df_new.columns:
            df_new[col] = ""

    df_merged = pd.concat([df_original, df_new], ignore_index=True)
    merged_count = len(df_merged)
    print(f"  Total after merge: {merged_count}")
    report_lines.append(f"\nTotal after merge (before dedup): {merged_count}")

    # --------------------------------------------------
    # 5. Deduplicate by PMID
    # --------------------------------------------------
    print("Removing duplicates by PMID...")
    df_merged["pmid"] = df_merged["pmid"].astype(str).str.strip()
    before_pmid_dedup = len(df_merged)

    # Keep 'original' source preferentially when duplicates exist
    df_merged = df_merged.sort_values("source", ascending=True)  # 'expanded' before 'original'
    df_merged = df_merged.drop_duplicates(subset=["pmid"], keep="last")  # Keep 'original'

    pmid_dupes_removed = before_pmid_dedup - len(df_merged)
    print(f"  PMID duplicates removed: {pmid_dupes_removed}")
    report_lines.append(f"\nDuplicates removed by PMID: {pmid_dupes_removed}")

    # --------------------------------------------------
    # 6. Additional dedup by normalized title
    # --------------------------------------------------
    print("Checking for title-based duplicates...")
    df_merged["title_normalized"] = df_merged["title"].apply(normalize_title)
    before_title_dedup = len(df_merged)

    # Only remove exact title matches where PMIDs differ
    # Group by normalized title and keep the one with the most information
    df_merged = df_merged.sort_values(
        by=["abstract", "mesh_terms"],
        key=lambda x: x.str.len(),
        ascending=False,
    )
    df_merged = df_merged.drop_duplicates(subset=["title_normalized"], keep="first")
    df_merged = df_merged.drop(columns=["title_normalized"])

    title_dupes_removed = before_title_dedup - len(df_merged)
    print(f"  Title duplicates removed: {title_dupes_removed}")
    report_lines.append(f"Duplicates removed by title: {title_dupes_removed}")

    # --------------------------------------------------
    # 7. Clean year column
    # --------------------------------------------------
    print("Cleaning year column...")
    # Use year_raw if available, otherwise use year
    if "year_raw" in df_merged.columns:
        df_merged["year_original"] = df_merged["year_raw"].fillna(df_merged["year"])
    else:
        df_merged["year_original"] = df_merged["year"]

    df_merged["year"] = df_merged["year_original"].apply(extract_year)

    # --------------------------------------------------
    # 8. Handle missing/empty abstracts and titles
    # --------------------------------------------------
    print("Handling missing data...")
    df_merged["title"] = df_merged["title"].fillna("").astype(str).str.strip()
    df_merged["abstract"] = df_merged["abstract"].fillna("").astype(str).str.strip()
    df_merged["mesh_terms"] = df_merged["mesh_terms"].fillna("").astype(str).str.strip()

    before_missing = len(df_merged)
    # Remove papers without titles or abstracts
    df_clean = df_merged[
        (df_merged["title"] != "") & (df_merged["abstract"] != "")
    ].copy()
    missing_removed = before_missing - len(df_clean)
    print(f"  Papers removed (missing title/abstract): {missing_removed}")
    report_lines.append(f"Papers removed (missing title/abstract): {missing_removed}")

    # Remove very short abstracts (< 100 chars is likely incomplete)
    before_short = len(df_clean)
    df_clean = df_clean[df_clean["abstract"].str.len() >= 100].copy()
    short_removed = before_short - len(df_clean)
    print(f"  Papers removed (abstract < 100 chars): {short_removed}")
    report_lines.append(f"Papers removed (short abstracts < 100 chars): {short_removed}")

    # --------------------------------------------------
    # 9. Create document and cleaned_text columns
    # --------------------------------------------------
    print("Creating document and cleaned_text columns...")

    # document: original text (title + abstract) for transformer models
    df_clean["document"] = df_clean["title"] + ". " + df_clean["abstract"]

    # cleaned_text: cleaned version for traditional NLP (TF-IDF, LDA)
    df_clean["cleaned_text"] = df_clean["document"].apply(clean_text_for_nlp)

    # --------------------------------------------------
    # 10. Normalize whitespace in all text columns
    # --------------------------------------------------
    for col in ["title", "abstract"]:
        df_clean[col] = df_clean[col].apply(
            lambda x: re.sub(r"\s+", " ", str(x)).strip()
        )

    # --------------------------------------------------
    # 11. Reset index
    # --------------------------------------------------
    df_clean = df_clean.reset_index(drop=True)

    # --------------------------------------------------
    # 12. Select and order final columns
    # --------------------------------------------------
    final_columns = [
        "pmid", "title", "abstract", "year", "mesh_terms",
        "journal", "authors", "doi", "document", "cleaned_text", "source"
    ]
    # Only keep columns that exist
    final_columns = [c for c in final_columns if c in df_clean.columns]
    df_clean = df_clean[final_columns]

    # --------------------------------------------------
    # 13. Save outputs
    # --------------------------------------------------
    # Save merged (pre-cleaning) dataset
    merged_path = os.path.join(PROCESSED_DIR, "mhealth_merged_dataset.csv")
    df_merged.to_csv(merged_path, index=False, encoding="utf-8")
    print(f"\nMerged dataset saved: {merged_path}")

    # Save final cleaned dataset
    cleaned_path = os.path.join(PROCESSED_DIR, "mhealth_final_cleaned_dataset.csv")
    df_clean.to_csv(cleaned_path, index=False, encoding="utf-8")
    print(f"Cleaned dataset saved: {cleaned_path}")

    # --------------------------------------------------
    # 14. Generate report
    # --------------------------------------------------
    final_count = len(df_clean)

    report_lines.append(f"\n{'='*60}")
    report_lines.append("FINAL SUMMARY")
    report_lines.append(f"{'='*60}")
    report_lines.append(f"Original dataset:           {original_count} papers")
    report_lines.append(f"Newly collected:            {new_count} papers")
    report_lines.append(f"Total before dedup:         {merged_count} papers")
    report_lines.append(f"PMID duplicates removed:    {pmid_dupes_removed}")
    report_lines.append(f"Title duplicates removed:   {title_dupes_removed}")
    report_lines.append(f"Missing title/abstract:     {missing_removed}")
    report_lines.append(f"Short abstracts removed:    {short_removed}")
    report_lines.append(f"Final cleaned dataset:      {final_count} papers")
    report_lines.append(f"\nColumns: {df_clean.columns.tolist()}")
    report_lines.append(f"\nYear distribution:")
    year_dist = df_clean["year"].value_counts().sort_index()
    for year, count in year_dist.items():
        report_lines.append(f"  {year}: {count}")
    report_lines.append(f"\nPapers with MeSH terms: {(df_clean['mesh_terms'] != '').sum()}")
    report_lines.append(f"Papers without MeSH terms: {(df_clean['mesh_terms'] == '').sum()}")

    report_text = "\n".join(report_lines)

    report_path = os.path.join(REPORTS_DIR, "merge_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Report saved: {report_path}")

    print(f"\n{report_text}")

    return df_clean


if __name__ == "__main__":
    main()
