import os
import ir_datasets
import pandas as pd
from pathlib import Path

# Set output directory
OUTPUT_DIR = Path("project-root/data/raw")
SCRIPT_DIR = Path("project-root/scripts")

# Target dataset
DATASET_ID = "msmarco-passage/dev/small"

def download_and_save_dataset():
    """Download and save msmarco-passage/dev/small dataset."""
    # Load dataset
    print(f"Loading {DATASET_ID} from ir-datasets...")
    dataset = ir_datasets.load(DATASET_ID)

    # File paths
    collection_path = OUTPUT_DIR / "collection.tsv"
    queries_path = OUTPUT_DIR / "queries.dev.small.tsv"
    qrels_path = OUTPUT_DIR / "qrels.dev.tsv"

    # Save collection (passages)
    if not collection_path.exists():
        print(f"Saving passages to {collection_path}...")
        with open(collection_path, "w", encoding="utf-8") as f:
            for doc in dataset.docs_iter():
                f.write(f"{doc.doc_id}\t{doc.text}\n")
        print(f"Saved {dataset.docs_count()} passages.")
    else:
        print(f"{collection_path} already exists, skipping.")

    # Save queries
    if not queries_path.exists():
        print(f"Saving queries to {queries_path}...")
        with open(queries_path, "w", encoding="utf-8") as f:
            for query in dataset.queries_iter():
                f.write(f"{query.query_id}\t{query.text}\n")
        print(f"Saved {dataset.queries_count()} queries.")
    else:
        print(f"{queries_path} already exists, skipping.")

    # Save qrels
    if not qrels_path.exists():
        print(f"Saving qrels to {qrels_path}...")
        with open(qrels_path, "w", encoding="utf-8") as f:
            for qrel in dataset.qrels_iter():
                f.write(f"{qrel.query_id}\t0\t{qrel.doc_id}\t{qrel.relevance}\n")
        print(f"Saved {dataset.qrels_count()} qrels.")
    else:
        print(f"{qrels_path} already exists, skipping.")

def main():
    # Create directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SCRIPT_DIR.mkdir(parents=True, exist_ok=True)

    # Download and save dataset
    download_and_save_dataset()

    # Verify
    print(f"\nDownload complete! Files in {OUTPUT_DIR}:")
    for file in OUTPUT_DIR.iterdir():
        if file.is_file():
            size = file.stat().st_size / (1024 * 1024)  # Size in MB
            print(f"{file.name}: {size:.2f} MB")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)