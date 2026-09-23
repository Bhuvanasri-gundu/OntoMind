"""
relevance_filter.py
===================
Multi-signal relevance scoring for mHealth research papers.

Uses three complementary signals:
1. Keyword matching (weighted mHealth terms in title/abstract) — 40%
2. MeSH term matching (mHealth-relevant MeSH headings) — 30%
3. TF-IDF cosine similarity to mHealth reference description — 30%

Classification:
- Highly relevant (score >= 0.7)
- Relevant (score >= 0.5)
- Possibly relevant (score >= 0.3)
- Low relevance (score < 0.3)

Usage:
    python src/relevance_filter.py

Inputs:
    data/processed/mhealth_final_cleaned_dataset.csv

Outputs:
    data/processed/mhealth_relevance_annotated.csv
    data/processed/mhealth_relevant_dataset.csv     (filtered: excludes low relevance)
    data/processed/mhealth_low_relevance.csv        (removed papers with reasons)
    reports/summaries/relevance_report.txt

Author: Project Team
Date: 2026-08-29
"""

import os
import re
import datetime
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==============================================================
# PATHS
# ==============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "summaries")

INPUT_FILE = os.path.join(PROCESSED_DIR, "mhealth_final_cleaned_dataset.csv")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ==============================================================
# RELEVANCE CRITERIA
# ==============================================================

# Core mHealth keywords (high weight)
CORE_MHEALTH_KEYWORDS = [
    "mhealth", "m-health", "mobile health", "mobile healthcare",
    "mobile health application", "mobile health app",
    "health app", "health application",
    "smartphone health", "smartphone-based health",
    "mobile phone health", "mobile device health",
    "wearable health", "wearable health device",
    "wearable health technology", "wearable sensor",
    "remote patient monitoring", "remote health monitoring",
    "telehealth", "telemedicine", "teleconsultation",
    "digital health", "digital therapeutics",
    "electronic health", "ehealth", "e-health",
    "connected health", "mobile wellness",
    "health monitoring app", "health tracking",
    "fitness tracker", "activity tracker",
    "patient portal", "mobile clinical",
    "mobile intervention", "mobile-based intervention",
    "sms health", "text message intervention",
    "mobile sensing", "mobile technology health",
]

# Primary mHealth keywords (these strongly indicate mHealth relevance)
PRIMARY_KEYWORDS = [
    "mhealth", "m-health", "mobile health", "mobile healthcare",
    "mobile health app", "mobile health application",
    "smartphone health", "wearable health",
    "health app", "health application",
    "remote patient monitoring", "telehealth",
    "telemedicine", "digital health", "ehealth", "e-health",
]

# MeSH terms that indicate mHealth relevance
RELEVANT_MESH_TERMS = [
    "telemedicine", "mobile applications", "smartphone",
    "cell phone", "wearable electronic devices",
    "remote sensing technology", "monitoring, ambulatory",
    "fitness trackers", "text messaging",
    "mhealth", "mobile health", "digital health",
    "telehealth", "remote consultation",
    "patient monitoring", "health monitoring",
    "mobile app", "mobile device",
    "wearable", "sensor", "accelerometer",
    "remote patient monitoring",
    "electronic health records",
    "patient-generated health data",
]

# Reference mHealth description for TF-IDF similarity
MHEALTH_REFERENCE = """
Mobile health mHealth refers to the use of mobile devices such as smartphones
tablets and wearable sensors for health services and information. mHealth
encompasses mobile health applications health monitoring apps remote patient
monitoring telemedicine telehealth digital health interventions wearable
health technology fitness trackers activity trackers mobile clinical decision
support systems SMS-based health interventions and mobile-based disease
management. Key areas include chronic disease management medication adherence
mental health support physical activity monitoring vital signs tracking
patient engagement health behavior change and mobile health data analytics.
mHealth applications are used in healthcare delivery, public health
surveillance, clinical trials, and health education across diverse
populations including rural and underserved communities.
"""


def compute_keyword_score(text, title):
    """
    Compute keyword-based relevance score.
    
    Higher weight for:
    - Primary keywords appearing in title
    - Multiple keyword matches
    - Core mHealth terminology
    
    Returns score between 0.0 and 1.0
    """
    text_lower = str(text).lower()
    title_lower = str(title).lower()

    # Count primary keyword matches in title (high weight)
    title_matches = 0
    for kw in PRIMARY_KEYWORDS:
        if kw in title_lower:
            title_matches += 1

    # Count all keyword matches in full text
    text_matches = 0
    matched_keywords = []
    for kw in CORE_MHEALTH_KEYWORDS:
        if kw in text_lower:
            text_matches += 1
            matched_keywords.append(kw)

    # Score calculation
    # Title match is very strong signal
    title_score = min(title_matches / 2.0, 1.0)  # Cap at 1.0

    # Text match score (diminishing returns after 5 matches)
    text_score = min(text_matches / 5.0, 1.0)

    # Combined: 60% title, 40% text
    combined = 0.6 * title_score + 0.4 * text_score

    return combined, matched_keywords


def compute_mesh_score(mesh_terms):
    """
    Compute MeSH-based relevance score.
    
    Returns score between 0.0 and 1.0
    If no MeSH terms available, returns neutral 0.5
    """
    if pd.isna(mesh_terms) or str(mesh_terms).strip() == "":
        # No MeSH terms — neutral score (don't penalize)
        return 0.5, []

    mesh_lower = str(mesh_terms).lower()
    matched_mesh = []

    for term in RELEVANT_MESH_TERMS:
        if term.lower() in mesh_lower:
            matched_mesh.append(term)

    if len(matched_mesh) == 0:
        return 0.1, matched_mesh  # Has MeSH but none relevant
    elif len(matched_mesh) == 1:
        return 0.5, matched_mesh
    elif len(matched_mesh) == 2:
        return 0.7, matched_mesh
    else:
        return 1.0, matched_mesh


def compute_tfidf_similarity(documents):
    """
    Compute TF-IDF cosine similarity between each document and
    the mHealth reference description.
    
    Returns array of similarity scores (0.0 to 1.0)
    """
    # Combine reference with all documents
    all_texts = [MHEALTH_REFERENCE] + documents.tolist()

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
    )

    tfidf_matrix = vectorizer.fit_transform(all_texts)

    # Cosine similarity between reference (index 0) and all documents
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])

    return similarities[0]


def classify_relevance(score):
    """Classify relevance based on composite score."""
    if score >= 0.70:
        return "Highly relevant"
    elif score >= 0.50:
        return "Relevant"
    elif score >= 0.30:
        return "Possibly relevant"
    else:
        return "Low relevance"


def generate_reason(row):
    """Generate human-readable reason for the classification."""
    reasons = []

    kw_score = row["keyword_score"]
    mesh_score = row["mesh_score"]
    tfidf_score = row["tfidf_score"]
    category = row["relevance_category"]

    if kw_score >= 0.5:
        reasons.append(f"Strong keyword match ({kw_score:.2f})")
    elif kw_score >= 0.2:
        reasons.append(f"Moderate keyword match ({kw_score:.2f})")
    else:
        reasons.append(f"Weak keyword match ({kw_score:.2f})")

    if mesh_score >= 0.7:
        reasons.append(f"Good MeSH alignment ({mesh_score:.2f})")
    elif mesh_score == 0.5 and row["mesh_terms"] == "":
        reasons.append("No MeSH terms available")
    else:
        reasons.append(f"MeSH score: {mesh_score:.2f}")

    if tfidf_score >= 0.15:
        reasons.append(f"Good semantic similarity ({tfidf_score:.2f})")
    else:
        reasons.append(f"Low semantic similarity ({tfidf_score:.2f})")

    return "; ".join(reasons)


def main():
    """Main relevance filtering pipeline."""
    print("=" * 60)
    print("RELEVANCE FILTERING PIPELINE")
    print(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Load data
    # --------------------------------------------------
    print("\nLoading dataset...")
    df = pd.read_csv(INPUT_FILE)
    total_papers = len(df)
    print(f"  Total papers: {total_papers}")

    # Ensure text columns are strings
    df["title"] = df["title"].fillna("").astype(str)
    df["abstract"] = df["abstract"].fillna("").astype(str)
    df["mesh_terms"] = df["mesh_terms"].fillna("").astype(str)
    df["document"] = df["document"].fillna("").astype(str)
    df["cleaned_text"] = df["cleaned_text"].fillna("").astype(str)

    # --------------------------------------------------
    # 2. Keyword scoring
    # --------------------------------------------------
    print("Computing keyword scores...")
    keyword_results = df.apply(
        lambda row: compute_keyword_score(row["document"], row["title"]),
        axis=1,
    )
    df["keyword_score"] = [r[0] for r in keyword_results]
    df["matched_keywords"] = ["; ".join(r[1][:5]) for r in keyword_results]

    # --------------------------------------------------
    # 3. MeSH scoring
    # --------------------------------------------------
    print("Computing MeSH scores...")
    mesh_results = df["mesh_terms"].apply(compute_mesh_score)
    df["mesh_score"] = [r[0] for r in mesh_results]
    df["matched_mesh"] = ["; ".join(r[1][:5]) for r in mesh_results]

    # --------------------------------------------------
    # 4. TF-IDF similarity scoring
    # --------------------------------------------------
    print("Computing TF-IDF similarity scores...")
    tfidf_scores = compute_tfidf_similarity(df["cleaned_text"])

    # Normalize to 0-1 range
    if tfidf_scores.max() > 0:
        tfidf_normalized = tfidf_scores / tfidf_scores.max()
    else:
        tfidf_normalized = tfidf_scores

    df["tfidf_score"] = tfidf_normalized

    # --------------------------------------------------
    # 5. Composite relevance score
    # --------------------------------------------------
    print("Computing composite relevance scores...")

    # Weighted combination: keyword 40%, MeSH 30%, TF-IDF 30%
    df["relevance_score"] = (
        0.40 * df["keyword_score"]
        + 0.30 * df["mesh_score"]
        + 0.30 * df["tfidf_score"]
    )

    # Classify
    df["relevance_category"] = df["relevance_score"].apply(classify_relevance)

    # Generate reasons
    print("Generating classification reasons...")
    df["relevance_reason"] = df.apply(generate_reason, axis=1)

    # --------------------------------------------------
    # 6. Save annotated dataset
    # --------------------------------------------------
    annotated_path = os.path.join(PROCESSED_DIR, "mhealth_relevance_annotated.csv")
    df.to_csv(annotated_path, index=False, encoding="utf-8")
    print(f"\nAnnotated dataset saved: {annotated_path}")

    # --------------------------------------------------
    # 7. Filter and save relevant papers
    # --------------------------------------------------
    df_relevant = df[df["relevance_category"] != "Low relevance"].copy()
    relevant_path = os.path.join(PROCESSED_DIR, "mhealth_relevant_dataset.csv")
    df_relevant.to_csv(relevant_path, index=False, encoding="utf-8")
    print(f"Relevant dataset saved: {relevant_path} ({len(df_relevant)} papers)")

    # --------------------------------------------------
    # 8. Save low-relevance papers separately (with reasons)
    # --------------------------------------------------
    df_low = df[df["relevance_category"] == "Low relevance"].copy()
    low_rel_path = os.path.join(PROCESSED_DIR, "mhealth_low_relevance.csv")
    df_low.to_csv(low_rel_path, index=False, encoding="utf-8")
    print(f"Low relevance papers saved: {low_rel_path} ({len(df_low)} papers)")

    # --------------------------------------------------
    # 9. Generate report
    # --------------------------------------------------
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("RELEVANCE FILTERING REPORT")
    report_lines.append(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 60)

    cat_counts = df["relevance_category"].value_counts()
    report_lines.append(f"\nTotal papers analyzed: {total_papers}")
    report_lines.append(f"\nRelevance Distribution:")
    for cat in ["Highly relevant", "Relevant", "Possibly relevant", "Low relevance"]:
        count = cat_counts.get(cat, 0)
        pct = 100.0 * count / total_papers if total_papers > 0 else 0
        report_lines.append(f"  {cat}: {count} ({pct:.1f}%)")

    report_lines.append(f"\nFinal relevant dataset: {len(df_relevant)} papers")
    report_lines.append(f"Removed (low relevance): {len(df_low)} papers")

    # Score statistics
    report_lines.append(f"\nRelevance Score Statistics:")
    report_lines.append(f"  Mean:   {df['relevance_score'].mean():.3f}")
    report_lines.append(f"  Median: {df['relevance_score'].median():.3f}")
    report_lines.append(f"  Std:    {df['relevance_score'].std():.3f}")
    report_lines.append(f"  Min:    {df['relevance_score'].min():.3f}")
    report_lines.append(f"  Max:    {df['relevance_score'].max():.3f}")

    # Examples from each category
    report_lines.append(f"\n{'='*60}")
    report_lines.append("EXAMPLES FROM EACH CATEGORY")
    report_lines.append(f"{'='*60}")

    for cat in ["Highly relevant", "Relevant", "Possibly relevant", "Low relevance"]:
        subset = df[df["relevance_category"] == cat]
        if len(subset) > 0:
            sample = subset.sample(min(3, len(subset)), random_state=RANDOM_SEED)
            report_lines.append(f"\n--- {cat} (showing up to 3 examples) ---")
            for _, row in sample.iterrows():
                report_lines.append(f"  PMID: {row['pmid']}")
                report_lines.append(f"  Title: {row['title'][:100]}...")
                report_lines.append(f"  Score: {row['relevance_score']:.3f}")
                report_lines.append(f"  Reason: {row['relevance_reason']}")
                report_lines.append("")

    # Method description
    report_lines.append(f"\n{'='*60}")
    report_lines.append("METHODOLOGY")
    report_lines.append(f"{'='*60}")
    report_lines.append("Relevance scoring uses three complementary signals:")
    report_lines.append("1. Keyword matching (40% weight)")
    report_lines.append("   - Weighted mHealth keyword presence in title and abstract")
    report_lines.append("   - Title matches weighted higher than abstract matches")
    report_lines.append("2. MeSH term matching (30% weight)")
    report_lines.append("   - Checks MeSH headings against mHealth-relevant terms")
    report_lines.append("   - Papers without MeSH terms receive neutral score (0.5)")
    report_lines.append("3. TF-IDF cosine similarity (30% weight)")
    report_lines.append("   - Computes similarity to mHealth reference description")
    report_lines.append("   - Uses bigrams and normalized TF-IDF vectors")
    report_lines.append("")
    report_lines.append("LIMITATIONS:")
    report_lines.append("- Keyword lists may not capture all mHealth subdomains")
    report_lines.append("- MeSH terms are missing for ~50% of papers (neutral scored)")
    report_lines.append("- TF-IDF similarity depends on vocabulary overlap")
    report_lines.append("- Borderline papers (score ~0.3) need manual review")
    report_lines.append("- The system may under-score novel mHealth applications")

    report_text = "\n".join(report_lines)

    report_path = os.path.join(REPORTS_DIR, "relevance_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\nReport saved: {report_path}")

    print(f"\n{report_text}")

    return df


if __name__ == "__main__":
    main()
