"""
collect_pubmed_data.py
======================
Expanded PubMed data collection for mHealth research.

Uses Biopython's Entrez API to search PubMed with multiple mHealth-related
queries. Collects papers across a broad date range (2018-2026) and stores
raw results for downstream merging and cleaning.

Usage:
    python src/collect_pubmed_data.py

Output:
    data/raw/pubmed_expanded_YYYYMMDD.csv

Author: Project Team
Date: 2026-08-29
"""

import os
import sys
import time
import datetime
import pandas as pd
from Bio import Entrez, Medline

# ==============================================================
# CONFIGURATION
# ==============================================================

# NCBI requires an email for Entrez queries
Entrez.email = "bhuvanavijayam19@gmail.com"

# Optional: NCBI API key for higher rate limits (10 req/sec vs 3 req/sec)
# Entrez.api_key = "YOUR_NCBI_API_KEY"

# Maximum papers to retrieve per query
MAX_PER_QUERY = 250

# Batch size for efetch calls
BATCH_SIZE = 100

# Delay between API calls (seconds) — respect NCBI rate limits
API_DELAY = 0.4

# Date range for search
DATE_RANGE = "2018:2026"

# Output directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

# Collection date stamp
COLLECTION_DATE = datetime.datetime.now().strftime("%Y%m%d")

# ==============================================================
# SEARCH QUERIES
# ==============================================================
# Multiple diverse queries to capture the breadth of mHealth research
# while keeping results focused on mobile health

SEARCH_QUERIES = [
    {
        "name": "mhealth_core",
        "query": (
            '("mHealth"[Title/Abstract] OR "m-health"[Title/Abstract]) '
            f'AND {DATE_RANGE}[dp] AND hasabstract[text]'
        ),
    },
    {
        "name": "mobile_health",
        "query": (
            '"mobile health"[Title/Abstract] '
            f'AND {DATE_RANGE}[dp] AND hasabstract[text]'
        ),
    },
    {
        "name": "mobile_health_apps",
        "query": (
            '("mobile health application"[Title/Abstract] OR '
            '"mobile health app"[Title/Abstract]) '
            f'AND {DATE_RANGE}[dp] AND hasabstract[text]'
        ),
    },
    {
        "name": "smartphone_health",
        "query": (
            '("smartphone health"[Title/Abstract] OR '
            '"smartphone-based health"[Title/Abstract] OR '
            '"smartphone health monitoring"[Title/Abstract]) '
            f'AND {DATE_RANGE}[dp] AND hasabstract[text]'
        ),
    },
    {
        "name": "mobile_healthcare",
        "query": (
            '"mobile healthcare"[Title/Abstract] '
            f'AND {DATE_RANGE}[dp] AND hasabstract[text]'
        ),
    },
    {
        "name": "digital_health_mobile",
        "query": (
            '"digital health"[Title/Abstract] AND '
            '(mobile[Title/Abstract] OR smartphone[Title/Abstract] OR app[Title/Abstract]) '
            f'AND {DATE_RANGE}[dp] AND hasabstract[text]'
        ),
    },
    {
        "name": "wearable_health",
        "query": (
            '("wearable health"[Title/Abstract] OR '
            '"wearable health technology"[Title/Abstract] OR '
            '"wearable health device"[Title/Abstract]) '
            f'AND {DATE_RANGE}[dp] AND hasabstract[text]'
        ),
    },
    {
        "name": "remote_patient_monitoring",
        "query": (
            '"remote patient monitoring"[Title/Abstract] '
            'AND (mobile[Title/Abstract] OR app[Title/Abstract] OR smartphone[Title/Abstract]) '
            f'AND {DATE_RANGE}[dp] AND hasabstract[text]'
        ),
    },
    {
        "name": "telemedicine_mobile",
        "query": (
            '("telemedicine"[Title/Abstract] OR "telehealth"[Title/Abstract]) '
            'AND (mobile[Title/Abstract] OR smartphone[Title/Abstract] OR app[Title/Abstract]) '
            f'AND {DATE_RANGE}[dp] AND hasabstract[text]'
        ),
    },
    {
        "name": "health_monitoring_apps",
        "query": (
            '"health monitoring"[Title/Abstract] '
            'AND (app[Title/Abstract] OR application[Title/Abstract] OR smartphone[Title/Abstract]) '
            f'AND {DATE_RANGE}[dp] AND hasabstract[text]'
        ),
    },
]


def search_pubmed(query, max_results):
    """Search PubMed and return list of PMIDs."""
    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        retmax=max_results,
        sort="relevance",
    )
    results = Entrez.read(handle)
    handle.close()
    return results["IdList"], int(results["Count"])


def fetch_papers(pmids, query_name):
    """Fetch paper details for a list of PMIDs using Medline format."""
    records = []

    for start in range(0, len(pmids), BATCH_SIZE):
        batch = pmids[start : start + BATCH_SIZE]

        try:
            handle = Entrez.efetch(
                db="pubmed",
                id=",".join(batch),
                rettype="medline",
                retmode="text",
            )

            for record in Medline.parse(handle):
                pmid = record.get("PMID", "")
                title = record.get("TI", "")
                abstract = record.get("AB", "")

                # Publication date (may contain month info like "2026 Aug")
                year_raw = record.get("DP", "")

                # MeSH terms
                mesh_list = record.get("MH", [])
                mesh_terms = "; ".join(mesh_list) if mesh_list else ""

                # Journal title
                journal = record.get("JT", "")  # Full journal title

                # DOI
                doi_list = record.get("AID", [])
                doi = ""
                for aid in doi_list:
                    if aid.endswith("[doi]"):
                        doi = aid.replace(" [doi]", "")
                        break

                # Authors
                authors_list = record.get("AU", [])
                authors = "; ".join(authors_list) if authors_list else ""

                records.append(
                    {
                        "pmid": pmid,
                        "title": title,
                        "abstract": abstract,
                        "year_raw": year_raw,
                        "mesh_terms": mesh_terms,
                        "journal": journal,
                        "doi": doi,
                        "authors": authors,
                        "search_query": query_name,
                    }
                )

            handle.close()

        except Exception as e:
            print(f"  ERROR fetching batch starting at {start}: {e}")
            # Continue with remaining batches rather than failing entirely

        processed = min(start + BATCH_SIZE, len(pmids))
        print(f"  Fetched {processed}/{len(pmids)} papers")
        time.sleep(API_DELAY)

    return records


def main():
    """Main collection pipeline."""
    print("=" * 60)
    print("PubMed mHealth Data Collection")
    print(f"Date: {COLLECTION_DATE}")
    print(f"Date range: {DATE_RANGE}")
    print(f"Number of queries: {len(SEARCH_QUERIES)}")
    print("=" * 60)

    all_records = []
    query_log = []

    for i, q in enumerate(SEARCH_QUERIES, 1):
        print(f"\n[{i}/{len(SEARCH_QUERIES)}] Query: {q['name']}")
        print(f"  Search: {q['query'][:80]}...")

        try:
            pmids, total_count = search_pubmed(q["query"], MAX_PER_QUERY)
            print(f"  Total available: {total_count}, Retrieving: {len(pmids)}")

            query_log.append(
                {
                    "query_name": q["name"],
                    "query": q["query"],
                    "total_available": total_count,
                    "retrieved": len(pmids),
                }
            )

            if len(pmids) > 0:
                records = fetch_papers(pmids, q["name"])
                all_records.extend(records)
                print(f"  Successfully fetched {len(records)} paper details")
            else:
                print("  No papers found for this query")

        except Exception as e:
            print(f"  ERROR with query '{q['name']}': {e}")
            query_log.append(
                {
                    "query_name": q["name"],
                    "query": q["query"],
                    "total_available": 0,
                    "retrieved": 0,
                    "error": str(e),
                }
            )

        time.sleep(1)  # Extra delay between queries

    # --------------------------------------------------
    # Create DataFrame and deduplicate within new collection
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("POST-PROCESSING")
    print("=" * 60)

    df = pd.DataFrame(all_records)
    print(f"Total records fetched: {len(df)}")

    # Deduplicate by PMID (keep first occurrence, which preserves the
    # query that found it first)
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["pmid"], keep="first")
    after_dedup = len(df)
    print(f"Duplicates removed (cross-query): {before_dedup - after_dedup}")
    print(f"Unique papers: {after_dedup}")

    # --------------------------------------------------
    # Save raw data
    # --------------------------------------------------
    os.makedirs(RAW_DIR, exist_ok=True)

    output_file = os.path.join(RAW_DIR, f"pubmed_expanded_{COLLECTION_DATE}.csv")
    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"\nRaw data saved to: {output_file}")

    # Save query log
    query_log_df = pd.DataFrame(query_log)
    query_log_file = os.path.join(RAW_DIR, f"query_log_{COLLECTION_DATE}.csv")
    query_log_df.to_csv(query_log_file, index=False, encoding="utf-8")
    print(f"Query log saved to: {query_log_file}")

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("COLLECTION SUMMARY")
    print("=" * 60)
    print(f"Collection date: {COLLECTION_DATE}")
    print(f"Total unique papers: {len(df)}")
    print(f"Papers with abstracts: {(df['abstract'].str.strip() != '').sum()}")
    print(f"Papers with MeSH terms: {(df['mesh_terms'].str.strip() != '').sum()}")
    print(f"Year range: {df['year_raw'].unique()[:5]}...")
    print(f"\nPapers per query:")
    print(df["search_query"].value_counts().to_string())
    print("=" * 60)

    return df


if __name__ == "__main__":
    main()
