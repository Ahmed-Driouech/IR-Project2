import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import vstack
from tqdm import tqdm
from typing import Dict, List
from pathlib import Path
import time
import random
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
import gensim.downloader

nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')

class QueryExpander:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = nltk.WordNetLemmatizer()
        print("Loading Word2Vec model...")
        self.word2vec = gensim.downloader.load('word2vec-google-news-300')
        print("Word2Vec model loaded!")
        self.stemmer = nltk.PorterStemmer()

    def get_wordnet_pos(self, tag):
        tag_dict = {
            "JJ": wn.ADJ,
            "NN": wn.NOUN,
            "VB": wn.VERB,
            "RB": wn.ADV
        }
        return tag_dict.get(tag[:2], wn.NOUN)

    def wordnet_query_expansion(self, query: str) -> str:
        tokens = word_tokenize(query.lower())
        pos_tags = nltk.pos_tag(tokens)

        expanded_terms = set(tokens)

        for token, pos in pos_tags:
            if token in self.stop_words or len(token) <= 2:
                continue

            wn_pos = self.get_wordnet_pos(pos)
            lemma = self.lemmatizer.lemmatize(token, pos=wn_pos)
            expanded_terms.add(lemma)

            synsets = wn.synsets(token, pos=wn_pos)

            for synset in synsets[:2]:
                for lemma_synset in synset.lemmas():
                    expanded_terms.add(lemma_synset.name().lower())

        expanded_query = ' '.join(expanded_terms)
        return expanded_query

    def word2vec_query_expansion(self, query: str) -> str:
        tokens = word_tokenize(query.lower())
        expanded_terms = set(tokens)

        for token in tokens:
            if token in self.stop_words or len(token) <= 2:
                continue

            try:
                similar_words = self.word2vec.most_similar(token, topn=2)
                expanded_terms.update(word for word, score in similar_words)
            except KeyError:
                continue

        expanded_query = ' '.join(expanded_terms)
        return expanded_query

    def pseudo_relevance_query_expansion(self, query: str, collection: Dict[str, str],
                                         vectorizer: TfidfVectorizer, top_k: int = 3,
                                         num_expansions: int = 2) -> str:

        query_vec = vectorizer.transform([query])
        doc_texts = list(collection.values())
        doc_vecs = vectorizer.transform(doc_texts)

        similarities = cosine_similarity(query_vec, doc_vecs)[0]
        top_k_indices = np.argsort(similarities)[-top_k:]
        top_k_texts = [doc_texts[idx] for idx in top_k_indices]

        top_k_vecs = vectorizer.transform(top_k_texts).toarray()
        mean_top_k = np.mean(top_k_vecs, axis=0)
        original_vec = query_vec.toarray()[0]
        combined_vec = 0.7 * mean_top_k + 0.3 * original_vec
        feature_names = vectorizer.get_feature_names_out()

        top_indices = np.argsort(combined_vec)[-num_expansions:]
        query_terms = query.lower().split()
        expansion_terms = [feature_names[idx] for idx in top_indices
                           if feature_names[idx] not in query_terms]


        return query + ' ' + ' '.join(expansion_terms)

    def porter_stemmer_query_expansion(self, query: str) -> str:
        tokens = word_tokenize(query.lower())
        return ' '.join([self.stemmer.stem(token) for token in tokens if token.isalpha()])

    def expand_query(self, query: str, method: str, collection: Dict[str, str] = None,
                     vectorizer: TfidfVectorizer = None) -> str:
        if method == 'wordnet':
            return self.wordnet_query_expansion(query)
        elif method == 'word2vec':
            return self.word2vec_query_expansion(query)
        elif method == 'pseudo_relevance' and collection is not None and vectorizer is not None:
            return self.pseudo_relevance_query_expansion(query, collection, vectorizer)
        elif method == 'porter_stemmer':
            return self.porter_stemmer_query_expansion(query)
        elif method == 'combined' and collection is not None and vectorizer is not None:
            q = self.wordnet_query_expansion(query)
            q = self.word2vec_query_expansion(q)
            return self.porter_stemmer_query_expansion(q)
        return query


class BM25Evaluator:
    def __init__(self, data_dir: str = "../data/raw", num_queries: int = None):
        self.data_dir = Path(data_dir)
        self.collection: Dict[str, str] = {}
        self.queries: Dict[str, str] = {}
        self.qrels: Dict[str, Dict[str, int]] = {}
        self.num_queries = num_queries
        self.vectorizer = None
        self.doc_vectors = None
        self.doc_ids = []
        self.query_expansion = QueryExpander()

    def load_data(self):
        print("Loading collection...")
        with open(self.data_dir / "collection.tsv", 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                doc_id, text = line.strip().split('\t')
                self.collection[doc_id] = text
                self.doc_ids.append(doc_id)

        print("\nLoading queries...")
        all_queries = {}
        with open(self.data_dir / "queries.dev.small.tsv", 'r', encoding='utf-8') as f:
            for line in f:
                query_id, text = line.strip().split('\t')
                all_queries[query_id] = text

        if self.num_queries and self.num_queries < len(all_queries):
            print(f"Sampling {self.num_queries} queries randomly from {len(all_queries)} total queries")
            query_ids = random.sample(list(all_queries.keys()), self.num_queries)
            self.queries = {qid: all_queries[qid] for qid in query_ids}
        else:
            self.queries = all_queries

        print("Loading qrels...")
        self.qrels = {}
        with open(self.data_dir / "qrels.dev.tsv", 'r', encoding='utf-8') as f:
            for line in f:
                query_id, _, doc_id, relevance = line.strip().split('\t')
                if query_id in self.queries:
                    if query_id not in self.qrels:
                        self.qrels[query_id] = {}
                    self.qrels[query_id][doc_id] = int(relevance)

    def build_index(self):
        print("\nBuilding TF-IDF index...")

        # Initialize vectorizer with BM25-like parameters
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            token_pattern=r'\S+',
            use_idf=True,
            norm='l2',
            smooth_idf=True
        )

        batch_size = 100000
        all_vectors = []

        for i in tqdm(range(0, len(self.doc_ids), batch_size)):
            batch_docs = [self.collection[doc_id] for doc_id in self.doc_ids[i:i + batch_size]]

            if i == 0:
                batch_vectors = self.vectorizer.fit_transform(batch_docs)
            else:
                batch_vectors = self.vectorizer.transform(batch_docs)

            all_vectors.append(batch_vectors)

        # Combine all document vectors
        print("Combining document vectors...")
        self.doc_vectors = vstack(all_vectors)
        print(f"Index built with {self.doc_vectors.shape[1]} features")

    def search(self, query: str, k: int = 1000) -> List[str]:
        query_vector = self.vectorizer.transform([query])

        batch_size = 100000
        all_scores = []

        for i in range(0, self.doc_vectors.shape[0], batch_size):
            doc_vectors_batch = self.doc_vectors[i:i + batch_size]

            # Calculate cosine similarity for this batch
            batch_scores = (doc_vectors_batch @ query_vector.T).toarray().flatten()
            all_scores.extend(batch_scores)

        # Get top k documents
        top_k_idx = np.argsort(all_scores)[::-1][:k]
        return [self.doc_ids[idx] for idx in top_k_idx]

    def evaluate(self, method):
        if self.vectorizer is None:
            self.build_index()

        print("\nProcessing queries...")
        ap_scores = []
        rr_scores = []
        p10_scores = []

        start_time = time.time()
        processed_queries = 0

        for query_id, query_text in tqdm(self.queries.items()):
            query_text = self.query_expansion.expand_query(query_text, method, self.collection, self.vectorizer)


            # Get top documents for this query
            pred_doc_ids = self.search(query_text, k=1000)

            # Get relevant documents for this query
            relevant_docs = set(doc_id for doc_id, rel in self.qrels[query_id].items() if rel > 0)

            # Calculate MAP
            num_relevant = 0
            precision_sum = 0
            for rank, doc_id in enumerate(pred_doc_ids, 1):
                if doc_id in relevant_docs:
                    num_relevant += 1
                    precision_sum += num_relevant / rank
            ap = precision_sum / len(relevant_docs) if relevant_docs else 0
            ap_scores.append(ap)

            # Calculate MRR
            for rank, doc_id in enumerate(pred_doc_ids, 1):
                if doc_id in relevant_docs:
                    rr_scores.append(1.0 / rank)
                    break
            else:
                rr_scores.append(0.0)

            # Calculate P@10
            top_10_relevant = sum(1 for doc_id in pred_doc_ids[:10] if doc_id in relevant_docs)
            p10_scores.append(top_10_relevant / 10)

            processed_queries += 1
            if processed_queries % 100 == 0:
                elapsed_time = time.time() - start_time
                print(f"\nProcessed {processed_queries} queries in {elapsed_time:.2f} seconds")
                print(f"Current MAP: {np.mean(ap_scores):.6f} MRR:{np.mean(rr_scores):.6f} P@10:{np.mean(p10_scores):.6f}")

        map_score = np.mean(ap_scores)
        mrr_score = np.mean(rr_scores)
        p10_score = np.mean(p10_scores)

        # Print timing information
        total_time = time.time() - start_time
        print(f"\nEvaluation completed in {total_time:.2f} seconds")
        print(f"Average time per query: {total_time / len(self.queries):.3f} seconds")

        return {
            "MAP": map_score,
            "MRR": mrr_score,
            "P@10": p10_score
        }


def main():
    evaluator = BM25Evaluator(num_queries=500)

    print("Starting evaluation...")
    evaluator.load_data()
    metrics_none = evaluator.evaluate("none")
    metrics_wn = evaluator.evaluate("wordnet")
    metrics_wv = evaluator.evaluate("word2vec")
    metrics_ps = evaluator.evaluate("porter_stemmer")
    metrics_combined = evaluator.evaluate("combined")
    print("evaluation finished!")

    with open("./results/bm25_summary_results.txt", "w") as f:
        f.write("\n=== SUMMARY ===\n")
        f.write(f"{'':20} {'MAP':>10} {'MRR':>10} {'P@10':>10}\n")
        f.write(f"{'none':20} {metrics_none['MAP']:10.6f} {metrics_none['MRR']:10.6f} {metrics_none['P@10']:10.6f}\n")
        f.write(f"{'WordNet':20} {metrics_wn['MAP']:10.6f} {metrics_wn['MRR']:10.6f} {metrics_wn['P@10']:10.6f}\n")
        f.write(f"{'Word2Vec':20} {metrics_wv['MAP']:10.6f} {metrics_wv['MRR']:10.6f} {metrics_wv['P@10']:10.6f}\n")
        f.write(
            f"{'porter_stemmer':20} {metrics_ps['MAP']:10.6f} {metrics_ps['MRR']:10.6f} {metrics_ps['P@10']:10.6f}\n")
        f.write(
            f"{'combined':20} {metrics_combined['MAP']:10.6f} {metrics_combined['MRR']:10.6f} {metrics_combined['P@10']:10.6f}\n")


if __name__ == "__main__":
    main()