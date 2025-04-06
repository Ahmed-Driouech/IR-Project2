import os
import json
import random
import subprocess
import numpy as np
import pandas as pd
import argparse
import logging
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import torch
import ir_datasets
import gensim.downloader as gensim_downloader
from pyserini.search.lucene import LuceneSearcher
import lightgbm as lgb
from transformers import BertTokenizer, BertForSequenceClassification
import pytrec_eval
import optuna
import matplotlib.pyplot as plt
import itertools

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    logger.info("Using GPU")
    torch.cuda.manual_seed_all(42)

# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="IR Experiment Pipeline with Full Collection and Table Output")
    parser.add_argument("--project_dir", type=str, default=os.path.expanduser("~/Separate_IR_Project"), help="Project directory")
    parser.add_argument("--use_lotte", action="store_true", help="Include LOTTE dataset")
    return parser.parse_args()

# Download required NLTK data
nltk.download('wordnet')
nltk.download('punkt')

# Load Word2Vec model
word2vec_model = gensim_downloader.load("glove-wiki-gigaword-100")
logger.info("GPU Available: %s", torch.cuda.is_available())

# Set global project directory
args = parse_args()
project_dir = Path(args.project_dir)
project_dir.mkdir(parents=True, exist_ok=True)
os.chdir(project_dir)
logger.info("Current directory: %s", os.getcwd())

OUTPUT_DIR = project_dir / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Data Preprocessing Functions ---
def preprocess_dataset(dataset_id: str) -> None:
    dataset_name = dataset_id.replace('/', '_')
    logger.info("Loading %s...", dataset_id)
    dataset = ir_datasets.load(dataset_id)

    collection_path = OUTPUT_DIR / f"{dataset_name}_collection.tsv"
    queries_path = OUTPUT_DIR / f"{dataset_name}_queries.tsv"
    qrels_path = OUTPUT_DIR / f"{dataset_name}_qrels.tsv"

    if not queries_path.exists():
        with open(queries_path, "w", encoding="utf-8") as f:
            for q in dataset.queries_iter():
                f.write(f"{q.query_id}\t{q.text}\n")
        logger.info("Saved all queries for %s", dataset_id)

    if not qrels_path.exists():
        qrels_count = 0
        with open(qrels_path, "w", encoding="utf-8") as f:
            for qrel in dataset.qrels_iter():
                f.write(f"{qrel.query_id}\t0\t{qrel.doc_id}\t{qrel.relevance}\n")
                qrels_count += 1
        logger.info("Saved %d qrels for %s", qrels_count, dataset_id)

    if not collection_path.exists():
        doc_count = 0
        # max_docs = 100000  # Limit to 100k documents
        with open(collection_path, "w", encoding="utf-8") as f:
            for doc in dataset.docs_iter():
                # if doc_count >= max_docs:
                    # break
                f.write(f"{doc.doc_id}\t{doc.text}\n")
                doc_count += 1
def index_corpus(dataset_id: str) -> None:
    dataset_name = dataset_id.replace('/', '_')
    input_path = OUTPUT_DIR / f"{dataset_name}_collection.tsv"
    index_dir = project_dir / "data" / "processed" / f"{dataset_name}_index"
    jsonl_path = project_dir / "data" / "processed" / f"{dataset_name}_jsonl"
    jsonl_path.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f_in, open(jsonl_path / "docs.jsonl", "w", encoding="utf-8") as f_out:
        for line in f_in:
            doc_id, text = line.strip().split("\t", 1)
            doc = {"id": doc_id, "contents": text}
            f_out.write(json.dumps(doc) + "\n")

    subprocess.run([
        "python", "-m", "pyserini.index.lucene",
        "-collection", "JsonCollection",
        "-input", str(jsonl_path),
        "-index", str(index_dir),
        "-generator", "DefaultLuceneDocumentGenerator",
        "-threads", "2",
        "-storePositions", "-storeDocvectors", "-storeRaw"
    ], check=True)
    logger.info("Indexing complete for dataset %s", dataset_id)

# --- Data Loading ---
def load_data(dataset_id: str):
    dataset_name = dataset_id.replace('/', '_')
    queries_path = OUTPUT_DIR / f"{dataset_name}_queries.tsv"
    qrels_path = OUTPUT_DIR / f"{dataset_name}_qrels.tsv"

    logger.info("Loading %s", dataset_id)
    logger.info("  Queries file size: %d bytes", os.path.getsize(queries_path))
    logger.info("  Qrels file size: %d bytes", os.path.getsize(qrels_path))

    # Load queries into memory (6,980 is manageable, ~1-2MB)
    queries = pd.read_csv(queries_path, sep="\t", header=None, names=["qid", "query"])
    query_dict = {str(q.qid): q.query for q in queries.itertuples()}

    # Load qrels into memory (7,437 is small, ~1MB)
    qrels = pd.read_csv(qrels_path, sep="\t", header=None, names=["qid", "iter", "doc_id", "rel"])
    qrels_dict = {}
    for qid, group in qrels.groupby("qid"):
        qid_str = str(qid)
        qrels_dict[qid_str] = {str(r.doc_id): int(r.rel) for r in group.itertuples()}

    # Don’t load docs; rely on Lucene index
    logger.info("  Number of queries loaded: %d", len(queries))
    logger.info("  Number of qrels loaded: %d", len(qrels))
    logger.info("  Documents will be accessed via Lucene index")

    return query_dict, qrels_dict  # No doc_dict
# --- Query Expansion ---
class QueryExpander:
    def __init__(self):
        self.word2vec_model = word2vec_model
        self.ps = PorterStemmer()

    def expand_query_wordnet(self, query: str) -> str:
        expanded = []
        tokens = word_tokenize(query)
        for word in tokens:
            synsets = wordnet.synsets(word)
            if synsets:
                synonyms = [lemma.name() for syn in synsets for lemma in syn.lemmas()]
                expanded.extend(synonyms[:2])
        return " ".join(set(tokens + expanded))

    def expand_query_word2vec(self, query: str) -> str:
        expanded = []
        tokens = word_tokenize(query)
        for word in tokens:
            if word in self.word2vec_model:
                similar = self.word2vec_model.most_similar(word, topn=2)
                expanded.extend([w for w, _ in similar])
        return " ".join(set(tokens + expanded))

    def expand_query_pseudo_relevance(self, query: str, vectorizer: TfidfVectorizer, index_dir: str) -> str:
        searcher = LuceneSearcher(str(index_dir))
        hits = searcher.search(query, k=10)
        docs = [hit.raw for hit in hits]
        tfidf_matrix = vectorizer.transform(docs)
        top_term_indices = np.argsort(tfidf_matrix.mean(axis=0).A1)[-3:]
        top_terms = vectorizer.get_feature_names_out()[top_term_indices]
        return " ".join(set(word_tokenize(query) + list(top_terms)))

    def expand_porter_stemmer(self, query: str) -> str:
        tokens = word_tokenize(query)
        return " ".join(self.ps.stem(word) for word in tokens)

    def expand_query(self, query: str, method: str, vectorizer=None, index_dir=None) -> str:
        if method == 'wordnet':
            return self.expand_query_wordnet(query)
        elif method == 'word2vec':
            return self.expand_query_word2vec(query)
        elif method == 'pseudo_relevance' and vectorizer and index_dir:
            return self.expand_query_pseudo_relevance(query, vectorizer, index_dir)
        elif method == 'porter_stemmer':
            return self.expand_porter_stemmer(query)
        elif method == 'combined' and vectorizer and index_dir:
            q = self.expand_query_wordnet(query)
            q = self.expand_query_word2vec(q)
            q = self.expand_query_pseudo_relevance(q, vectorizer, index_dir)
            return self.expand_porter_stemmer(q)
        return query

# --- BM25 Retriever ---
class BM25Retriever:
    def __init__(self, index_dir: str, k1: float = 0.9, b: float = 0.4):
        self.searcher = LuceneSearcher(index_dir)
        self.searcher.set_bm25(k1=k1, b=b)

    def retrieve(self, query: str, top_k: int = 20):
        hits = self.searcher.search(query, k=top_k)
        return [(hit.docid, hit.score) for hit in hits]

# --- LambdaMART Ranker ---
class LambdaMARTRanker:
    def __init__(self):
        self.model = None

    def extract_features(self, queries: dict, bm25_retriever: BM25Retriever, qrels: dict, batch_size=100):
        features = []
        labels = []
        groups = []
        query_list = list(queries.items())
        for i in range(0, len(query_list), batch_size):
            batch = query_list[i:i + batch_size]
            for qid, query in batch:
                bm25_results = bm25_retriever.retrieve(query, top_k=20)
                count = 0
                for doc_id, bm25_score in bm25_results:
                    # Fetch doc text from Lucene on-demand
                    doc_text = bm25_retriever.searcher.doc(doc_id).contents()
                    features.append([bm25_score, len(word_tokenize(query)), len(word_tokenize(doc_text))])
                    labels.append(1 if doc_id in qrels.get(qid, {}) else 0)
                    count += 1
                groups.append(count)
        return np.array(features), np.array(labels), groups

    def train(self, queries: dict, qrels: dict, bm25_retriever: BM25Retriever, params=None, num_boost_round: int = 100):
        X, y, groups = self.extract_features(queries, bm25_retriever, qrels)
        train_data = lgb.Dataset(X, label=y, group=groups)
        if params is None:
            params = {
                'objective': 'lambdarank',
                'metric': 'ndcg',
                'ndcg_at': [10],
                'learning_rate': 0.1,
                'num_leaves': 31,
                'min_data_in_leaf': 20,
            }
        self.model = lgb.train(params, train_data, num_boost_round=num_boost_round)
        logger.info("LambdaMART training complete.")

    def rerank(self, query: str, bm25_retriever: BM25Retriever):
        bm25_results = bm25_retriever.retrieve(query, top_k=20)
        X = []
        doc_ids = []
        for doc_id, bm25_score in bm25_results:
            doc_text = bm25_retriever.searcher.doc(doc_id).contents()
            X.append([bm25_score, len(word_tokenize(query)), len(word_tokenize(doc_text))])
            doc_ids.append(doc_id)
        if self.model:
            scores = self.model.predict(np.array(X))
            return sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)
        return bm25_results

# --- BERT Reranker ---
class BERTReranker:
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self, queries: dict, qrels: dict, bm25_retriever: BM25Retriever, epochs: int = 1, batch_size: int = 2):
        from torch.utils.data import DataLoader, Dataset

        class RankingDataset(Dataset):
            def __init__(self, queries, qrels, bm25_retriever, tokenizer):
                self.data = []
                for qid, query in queries.items():
                    bm25_results = bm25_retriever.retrieve(query, top_k=5)  # Reduced from 20
                    for doc_id, _ in bm25_results:
                        doc_text = bm25_retriever.searcher.doc(doc_id).contents()
                        label = 1 if doc_id in qrels.get(qid, {}) else 0
                        self.data.append((query, doc_text, label))
                self.tokenizer = tokenizer

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                query, doc, label = self.data[idx]
                encoding = self.tokenizer(query, doc, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
                return {k: v.squeeze(0) for k, v in encoding.items()}, torch.tensor(label, dtype=torch.float)

        dataset = RankingDataset(queries, qrels, bm25_retriever, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        self.model.train()
        for epoch in range(epochs):
            for batch in dataloader:
                inputs, labels = batch
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                labels = labels.to(self.device)
                outputs = self.model(**inputs).logits.squeeze(-1)
                loss = torch.nn.BCEWithLogitsLoss()(outputs, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()  # Free GPU memory
            logger.info("Epoch %d/%d completed for BERT training", epoch + 1, epochs)

    def rerank(self, query: str, bm25_retriever: BM25Retriever, top_k: int = 20):
        bm25_results = bm25_retriever.retrieve(query, top_k=top_k)
        self.model.eval()
        doc_ids = []
        doc_texts = []
        for doc_id, _ in bm25_results:
            doc_ids.append(doc_id)
            doc_texts.append(bm25_retriever.searcher.doc(doc_id).contents())
        encoding = self.tokenizer([query] * len(doc_texts), doc_texts, return_tensors='pt', max_length=512, truncation=True, padding=True)
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        with torch.no_grad():
            logits = self.model(**encoding).logits.squeeze(-1)
        scores = logits.cpu().numpy()
        torch.cuda.empty_cache()
        return sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)

# --- Evaluation Metrics ---
def evaluate_run(qrels: dict, run: dict):
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'map'})
    results = evaluator.evaluate(run)
    map_score = np.mean([r['map'] for r in results.values()])
    return map_score

# --- Hyperparameter Tuning Functions ---
def tune_bm25_parameters(index_dir, queries, qrels, sample_size=100):
    sample_queries = dict(random.sample(list(queries.items()), min(sample_size, len(queries))))
    sample_qrels = {qid: qrels[qid] for qid in sample_queries if qid in qrels}
    logger.info("Tuning BM25 with %d sampled queries", len(sample_queries))
    k1_values = [0.7, 0.9, 1.1]
    b_values = [0.3, 0.4, 0.5]
    best_score = -float('inf')
    best_params = None
    for k1, b in itertools.product(k1_values, b_values):
        bm25 = BM25Retriever(str(index_dir), k1=k1, b=b)
        run = {qid: dict(bm25.retrieve(query)) for qid, query in sample_queries.items()}
        map_score = evaluate_run(sample_qrels, run)
        logger.info(f"BM25 Params: k1={k1}, b={b} -> MAP: {map_score}")
        if map_score > best_score:
            best_score = map_score
            best_params = (k1, b)
    logger.info("Best BM25 params: %s", best_params)
    return best_params

def tune_lambda_mart(lambdamart, bm25, queries, docs, qrels, sample_size=100):
    sample_queries = dict(random.sample(list(queries.items()), min(sample_size, len(queries))))
    sample_qrels = {qid: qrels[qid] for qid in sample_queries if qid in qrels}
    
    def objective(trial):
        params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_at': [10],
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 16, 64),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 50),
        }
        num_boost_round = trial.suggest_int('num_boost_round', 50, 200)
        lambdamart.train(sample_queries, docs, sample_qrels, bm25, params=params, num_boost_round=num_boost_round)
        run = {qid: dict(lambdamart.rerank(query, docs, bm25)) for qid, query in sample_queries.items()}
        map_score = evaluate_run(sample_qrels, run)
        return map_score
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=5)
    logger.info("Best LambdaMART params: %s", study.best_params)
    
    # Save tuning progress plot
    results_dir = project_dir / "results"
    results_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot(range(len(study.trials)), [t.value for t in study.trials], marker='o')
    ax.set_xlabel("Trial")
    ax.set_ylabel("MAP")
    ax.set_title("LambdaMART Hyperparameter Tuning Progress")
    fig.savefig(results_dir / "lambdamart_tuning_progress.png")
    plt.close(fig)
    
    return study.best_params

# --- Main Experiment Pipeline ---
def run_experiment():
    datasets = ["msmarco-passage/dev/small"]
    if args.use_lotte:
        datasets.append("lotte/lifestyle/dev/search")
    
    methods = ['none', 'wordnet', 'word2vec', 'pseudo_relevance', 'porter_stemmer', 'combined']
    table_data = {method: {'BM25_MSMARCO': 0.0, 'BM25_LOTTE': 0.0, 'BERT_MSMARCO': 0.0, 'BERT_LOTTE': 0.0, 
                          'LambdaMART_MSMARCO': 0.0, 'LambdaMART_LOTTE': 0.0} for method in methods}

    for dataset_id in datasets:
        preprocess_dataset(dataset_id)
        index_corpus(dataset_id)
        queries, docs, qrels = load_data(dataset_id)

        logger.info("Running experiment with %d queries, %d documents, and %d qrels", 
                    len(queries), len(docs), len(qrels))
        vectorizer = TfidfVectorizer().fit(list(docs.values()))
        index_dir = project_dir / "data" / "processed" / f"{dataset_id.replace('/', '_')}_index"

        # Tune and train BM25
        best_k1, best_b = tune_bm25_parameters(index_dir, queries, qrels)
        bm25 = BM25Retriever(str(index_dir), k1=best_k1, b=best_b)

        # Tune and train LambdaMART
        lambdamart = LambdaMARTRanker()
        best_lambda_params = tune_lambda_mart(lambdamart, bm25, queries, docs, qrels)
        lambdamart.train(queries, docs, qrels, bm25, params={
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_at': [10],
            'learning_rate': best_lambda_params['learning_rate'],
            'num_leaves': best_lambda_params['num_leaves'],
            'min_data_in_leaf': best_lambda_params['min_data_in_leaf']
        }, num_boost_round=best_lambda_params['num_boost_round'])

        # Train BERT
        bert = BERTReranker()
        bert.train(queries, docs, qrels, bm25)

        expander = QueryExpander()
        for method in methods:
            run_bm25 = {}
            run_lambdamart = {}
            run_bert = {}
            for qid, query in queries.items():
                expanded_query = query if method == 'none' else expander.expand_query(query, method, vectorizer, str(index_dir))
                run_bm25[qid] = {doc_id: float(score) for doc_id, score in bm25.retrieve(expanded_query)}
                run_lambdamart[qid] = {doc_id: float(score) for doc_id, score in lambdamart.rerank(expanded_query, docs, bm25)}
                run_bert[qid] = {doc_id: float(score) for doc_id, score in bert.rerank(expanded_query, docs, bm25)}

            
            bm25_map = evaluate_run(qrels, run_bm25)
            lambdamart_map = evaluate_run(qrels, run_lambdamart)
            bert_map = evaluate_run(qrels, run_bert)

            dataset_short = "MSMARCO" if "msmarco" in dataset_id else "LOTTE"
            table_data[method][f"BM25_{dataset_short}"] = bm25_map
            table_data[method][f"BERT_{dataset_short}"] = bert_map
            table_data[method][f"LambdaMART_{dataset_short}"] = lambdamart_map

    # Create and save table
    columns = ["BM25_MSMARCO"]
    if args.use_lotte:
        columns.append("BM25_LOTTE")
    columns.extend(["BERT_MSMARCO", "BERT_LOTTE", "LambdaMART_MSMARCO", "LambdaMART_LOTTE"])
    df = pd.DataFrame(table_data).T[columns]
    df.index.name = "Query Expansion"

    # Fill missing LOTTE columns with 0.0 if not used
    if not args.use_lotte:
        df["BERT_LOTTE"] = 0.0
        df["LambdaMART_LOTTE"] = 0.0

    results_dir = project_dir / "results"
    results_dir.mkdir(exist_ok=True)
    df.to_csv(results_dir / "results_table.csv")
    logger.info("Results Table:\n%s", df.to_string())
    logger.info("Table saved to %s", results_dir / "results_table.csv")

if __name__ == "__main__":
    run_experiment()