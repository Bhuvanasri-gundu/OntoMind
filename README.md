# Ontology-Guided Hypothesis Generation Using LLMs and Topic Modeling in mHealth Research

## Project Overview

This project applies **Ontology-Guided Hypothesis Generation** using **Large Language Models (LLMs)** and **Topic Modeling** to the domain of **mHealth (Mobile Health) Research**. The pipeline begins with systematic dataset collection from PubMed, followed by data quality analysis, relevance filtering, and exploratory analysis — all before proceeding to TF-IDF keyword extraction and topic modeling.

## Current Stage

**✅ Dataset Preparation Complete** — The project has completed all data collection, validation, cleaning, relevance filtering, and exploratory analysis steps. The dataset is ready for TF-IDF keyword extraction.

## Project Structure

```
MAJOR PROJECT/
│
├── data/
│   ├── raw/                                    # Raw PubMed downloads
│   │   ├── mhealth_pubmed_original_491.csv     # Original 491 papers
│   │   ├── pubmed_expanded_20260829.csv        # Expanded collection (~2177 papers)
│   │   └── query_log_20260829.csv              # Search queries used
│   └── processed/                              # Cleaned and filtered datasets
│       ├── mhealth_merged_dataset.csv          # Merged (pre-cleaning)
│       ├── mhealth_final_cleaned_dataset.csv   # Cleaned and standardized
│       ├── mhealth_relevance_annotated.csv     # All papers with relevance scores
│       ├── mhealth_relevant_dataset.csv        # ★ Final relevant dataset
│       └── mhealth_low_relevance.csv           # Removed papers (with reasons)
│
├── notebooks/
│   ├── 01_data_quality_analysis.ipynb          # Data quality checks
│   ├── 02_dataset_relevance_analysis.ipynb     # Relevance filtering analysis
│   ├── 03_exploratory_data_analysis.ipynb      # EDA with visualizations
│   └── 04_final_dataset_summary.ipynb          # Pipeline summary & validation
│
├── src/
│   ├── collect_pubmed_data.py                  # PubMed data collection (10 queries)
│   ├── merge_and_clean.py                      # Merge, deduplicate, clean
│   └── relevance_filter.py                     # Multi-signal relevance scoring
│
├── reports/
│   ├── figures/                                # Saved visualizations
│   └── summaries/                              # Text reports
│       ├── merge_report.txt
│       ├── data_quality_report.txt
│       ├── relevance_report.txt
│       ├── manual_validation_sample.csv
│       └── final_dataset_summary.txt
│
├── original_files/                             # Backup of pre-existing files
│
├── requirements.txt                            # Python dependencies
└── README.md                                   # This file
```

## Pipeline Execution Order

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Collect PubMed Data
```bash
python src/collect_pubmed_data.py
```
Collects ~2000+ papers from PubMed using 10 diverse mHealth search queries. Requires internet connection. Output: `data/raw/pubmed_expanded_YYYYMMDD.csv`

### Step 3: Merge and Clean Data
```bash
python src/merge_and_clean.py
```
Merges original + expanded datasets, deduplicates by PMID and title, removes papers with missing/short abstracts, creates `document` and `cleaned_text` columns. Output: `data/processed/mhealth_final_cleaned_dataset.csv`

### Step 4: Relevance Filtering
```bash
python src/relevance_filter.py
```
Applies multi-signal relevance scoring (keyword 40%, MeSH 30%, TF-IDF 30%) to classify papers. Output: `data/processed/mhealth_relevant_dataset.csv`

### Step 5: Run Analysis Notebooks
```bash
jupyter nbconvert --execute notebooks/01_data_quality_analysis.ipynb
jupyter nbconvert --execute notebooks/02_dataset_relevance_analysis.ipynb
jupyter nbconvert --execute notebooks/03_exploratory_data_analysis.ipynb
jupyter nbconvert --execute notebooks/04_final_dataset_summary.ipynb
```

## Dataset Columns

| Column | Description |
|--------|-------------|
| `pmid` | PubMed unique identifier |
| `title` | Original paper title |
| `abstract` | Original paper abstract |
| `year` | Publication year (4-digit) |
| `mesh_terms` | PubMed MeSH subject headings |
| `journal` | Journal/source name |
| `authors` | Author list |
| `doi` | Digital Object Identifier |
| `document` | Title + Abstract (for BERTopic/transformers) |
| `cleaned_text` | Cleaned text (for TF-IDF/LDA) |
| `relevance_score` | Composite relevance score (0-1) |
| `relevance_category` | Highly relevant / Relevant / Possibly relevant |

## Relevance Scoring Methodology

The relevance filter uses three complementary signals:

1. **Keyword Matching (40%)** — Weighted presence of mHealth domain terms in title and abstract
2. **MeSH Term Matching (30%)** — PubMed-assigned subject headings checked against mHealth-relevant terms
3. **TF-IDF Cosine Similarity (30%)** — Semantic similarity to a comprehensive mHealth reference description

Papers scoring < 0.3 are classified as "Low relevance" and excluded from the final dataset.

## Search Queries Used

The expanded collection uses 10 PubMed queries covering:
- `mHealth`, `m-health`
- `mobile health`
- `mobile health application/app`
- `smartphone health`
- `mobile healthcare`
- `digital health` + mobile/smartphone
- `wearable health technology/device`
- `remote patient monitoring` + mobile
- `telemedicine/telehealth` + mobile
- `health monitoring` + app/smartphone

Date range: 2018–2026 | Abstracts required

## Next Steps

The following phases are **not yet started** and should be executed in order:

1. **TF-IDF Keyword Extraction** — Extract important terms from the corpus
2. **Topic Modeling** (LDA / BERTopic) — Discover latent topics
3. **Ontology Mapping** — Map topics to established health ontologies
4. **Hypothesis Generation** — Use LLMs to generate research hypotheses

## Technical Notes

- **Python**: Anaconda Python 3.13 (recommended)
- **Random Seed**: 42 (used throughout for reproducibility)
- **Encoding**: UTF-8 for all files
- **PubMed API**: Uses Biopython Entrez (rate limited to ~3 req/sec)
