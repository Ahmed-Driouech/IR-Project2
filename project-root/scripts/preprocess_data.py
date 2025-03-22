#!/usr/bin/env python3
"""
preprocess_data.py

This script processes raw MS MARCO data (collection.tsv and queries.dev.small.tsv)
and creates preprocessed versions for different models:
  - BM25: Uses a Lucene analyzer (with stemming and stopwords removal)
  - BERT/ColBERT/Dense: Uses Hugging Face's "bert-base-uncased" tokenizer
  - T5: Uses Hugging Face's "t5-base" tokenizer

It processes the files in parallel using multiprocessing.
"""

import re
import string
import pandas as pd
import multiprocessing as mp
from pathlib import Path
from functools import partial
from tqdm import tqdm

# Define directories
RAW_DIR = Path("project-root/data/raw")
PROCESSED_DIR = Path("project-root/data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Global variables for tokenizers/analzyers (will be reinitialized in workers)
bm25_analyzer = None
hf_tokenizer = None
t5_tokenizer = None

def init_worker():
    """Initializer for worker processes to set up global tokenizer and analyzer objects."""
    global bm25_analyzer, hf_tokenizer, t5_tokenizer
    from pyserini.analysis import Analyzer, get_lucene_analyzer
    from transformers import AutoTokenizer
    bm25_analyzer = Analyzer(get_lucene_analyzer(stemmer="porter", stopwords=True))
    hf_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    t5_tokenizer = AutoTokenizer.from_pretrained("t5-base")

# Text cleaning functions
def clean_text_for_bm25(text: str) -> str:
    """Lowercase, remove punctuation, and extra spaces for BM25."""
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return re.sub(r'\s+', ' ', text).strip()

def clean_text_for_hf(text: str) -> str:
    """Lowercase and strip extra spaces for HF-based models."""
    return text.lower().strip()

def process_chunk(chunk, output_prefix):
    """Process a pandas DataFrame chunk and return a list of processed records."""
    processed_records = []
    for doc_id, text in chunk.itertuples(index=False):
        try:
            if not isinstance(text, str) or not text.strip():
                continue
            original_text = text.strip()
            
            # BM25 processing
            bm25_clean = clean_text_for_bm25(original_text)
            bm25_tokens = list(bm25_analyzer.analyze(bm25_clean))
            bm25_processed = " ".join(bm25_tokens)
            
            # HF-based processing for BERT/ColBERT/Dense
            hf_clean = clean_text_for_hf(original_text)
            hf_tokens = hf_tokenizer.tokenize(hf_clean)
            hf_processed = " ".join(hf_tokens)
            
            # T5 processing
            t5_tokens = t5_tokenizer.tokenize(hf_clean)
            t5_processed = " ".join(t5_tokens)
            
            processed_records.append({
                "id": doc_id,
                "original_text": original_text,
                "bm25": bm25_processed,
                "bert": hf_processed,
                "colbert": hf_processed,  # Same as BERT for now
                "dense": hf_processed,    # Same as BERT for now
                "t5": t5_processed
            })
        except Exception as e:
            print(f"Error processing id {doc_id}: {e}")
            continue
    return processed_records

def count_lines(file_path: Path) -> int:
    """Count the number of lines in a file."""
    with file_path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)

def process_file(file_path: Path, output_prefix: str, chunksize=100):
    """
    Process a raw TSV file (format: id<TAB>text) in chunks using multiprocessing.
    Saves processed data as TSV.
    """
    output_file = PROCESSED_DIR / f"{output_prefix}_preprocessed.tsv"
    
    if not file_path.exists():
        print(f"Error: {file_path} not found. Run download_data.py first.")
        return

    total_lines = count_lines(file_path)
    total_chunks = (total_lines // chunksize) + 1

    print(f"Processing {file_path.name} with {mp.cpu_count()} CPU cores...")
    
    reader = pd.read_csv(file_path, sep="\t", names=["id", "text"], dtype=str, chunksize=chunksize)
    
    # Use partial to fix output_prefix argument for process_chunk.
    process_func = partial(process_chunk, output_prefix=output_prefix)
    
    pool = mp.Pool(mp.cpu_count(), initializer=init_worker)
    results = []
    
    # Use imap_unordered to process chunks and update progress bar.
    for chunk_result in tqdm(pool.imap_unordered(process_func, reader), total=total_chunks, desc=f"Processing {output_prefix}"):
        results.extend(chunk_result)
    
    pool.close()
    pool.join()
    
    # Save the results.
    df = pd.DataFrame(results)
    df.to_csv(output_file, sep="\t", index=False)
    print(f"Saved preprocessed data to {output_file}")

def preprocess_collection():
    """Process collection.tsv."""
    process_file(RAW_DIR / "collection.tsv", "collection")

def preprocess_queries():
    """Process queries.dev.small.tsv."""
    process_file(RAW_DIR / "queries.dev.small.tsv", "queries")

def preprocess_qrels():
    """Process qrels.dev.tsv by copying it with basic validation."""
    qrels_raw = RAW_DIR / "qrels.dev.tsv"
    qrels_processed = PROCESSED_DIR / "qrels_preprocessed.tsv"
    if not qrels_raw.exists():
        print(f"Error: {qrels_raw} not found. Run download_data.py first.")
        return
    try:
        df = pd.read_csv(qrels_raw, sep="\t", header=None, dtype=str)
        if df.shape[1] != 4:
            print("Warning: Expected 4 columns in qrels file, got", df.shape[1])
        df.to_csv(qrels_processed, sep="\t", header=False, index=False)
        print(f"Saved processed qrels to {qrels_processed}")
    except Exception as e:
        print(f"Error processing qrels: {e}")

def main():
    preprocess_collection()
    preprocess_queries()
    preprocess_qrels()

if __name__ == "__main__":
    main()
