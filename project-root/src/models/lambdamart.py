import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import lightgbm as lgb
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Download NLTK resources
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

class QueryExpander:
    def __init__(self, collection_texts: list):
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        # Train Word2Vec on full collection
        sentences = [self._preprocess_text(text) for text in collection_texts]
        self.w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=5, workers=4)

    def _preprocess_text(self, text: str) -> list:
        tokens = word_tokenize(text.lower())
        return [self.lemmatizer.lemmatize(token) for token in tokens 
                if token.isalpha() and token not in self.stop_words]

    def expand_query_wordnet(self, query: str, num_expansions: int = 2) -> str:
        tokens = self._preprocess_text(query)
        expanded_terms = set(tokens)
        for token in tokens:
            synonyms = set()
            for syn in wordnet.synsets(token)[:2]:  # Limit to top 2 synsets
                for lemma in syn.lemmas()[:num_expansions]:
                    synonym = lemma.name().lower()
                    if synonym != token and synonym not in synonyms:
                        synonyms.add(synonym)
                if len(synonyms) >= num_expansions:
                    break
            expanded_terms.update(synonyms)
        return ' '.join(expanded_terms)

    def expand_query_word2vec(self, query: str, num_expansions: int = 2, threshold: float = 0.7) -> str:
        tokens = self._preprocess_text(query)
        expanded_terms = set(tokens)
        for token in tokens:
            try:
                similar = self.w2v_model.wv.most_similar(token, topn=num_expansions * 2)
                for word, score in similar:
                    if score > threshold:
                        expanded_terms.add(word)
                    if len(expanded_terms) >= len(tokens) + num_expansions:
                        break
            except KeyError:
                pass
        return ' '.join(expanded_terms)

    def expand_query_pseudo_relevance(self, query: str, collection: pd.DataFrame, 
                                     vectorizer: TfidfVectorizer, top_k: int = 3, 
                                     num_expansions: int = 2) -> str:
        query_vec = vectorizer.transform([query])
        doc_vecs = vectorizer.transform(collection['text'])
        similarities = cosine_similarity(query_vec, doc_vecs)[0]
        top_k_indices = np.argsort(similarities)[-top_k:]
        top_k_docs = collection.iloc[top_k_indices]
        top_k_vecs = vectorizer.transform(top_k_docs['text']).toarray()
        mean_top_k = np.mean(top_k_vecs, axis=0)
        original_vec = query_vec.toarray()[0]
        combined_vec = 0.7 * mean_top_k + 0.3 * original_vec  # Rocchio-like weighting
        feature_names = vectorizer.get_feature_names_out()
        top_indices = np.argsort(combined_vec)[-num_expansions:]
        expansion_terms = [feature_names[idx] for idx in top_indices 
                          if feature_names[idx] not in query.lower().split()]
        return query + ' ' + ' '.join(expansion_terms)

    def expand_porter_stemmer(self, text: str) -> str:
        tokens = word_tokenize(text.lower())
        return ' '.join([self.stemmer.stem(token) for token in tokens if token.isalpha()])

    def expand_query(self, query: str, method: str, collection: pd.DataFrame = None, 
                     vectorizer: TfidfVectorizer = None) -> str:
        if method == 'wordnet':
            return self.expand_query_wordnet(query)
        elif method == 'word2vec':
            return self.expand_query_word2vec(query)
        elif method == 'pseudo_relevance' and collection is not None and vectorizer is not None:
            return self.expand_query_pseudo_relevance(query, collection, vectorizer)
        elif method == 'porter_stemmer':
            return self.expand_porter_stemmer(query)
        elif method == 'combined' and collection is not None and vectorizer is not None:
            q = self.expand_query_wordnet(query)
            q = self.expand_query_word2vec(q)
            q = self.expand_query_pseudo_relevance(q, collection, vectorizer)
            return self.expand_porter_stemmer(q)
        return query

class LambdaMART:
    def __init__(self, data_dir: Path, n_queries: int = 100, query_expansion_method: str = None):
        self.data_dir = data_dir
        self.n_queries = n_queries
        self.method = query_expansion_method
        self.model = None
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.query_expander = None

    def load_data(self):
        print("Loading data...")
        collection = pd.read_csv(self.data_dir / "collection.tsv", sep='\t', 
                                names=['doc_id', 'text'])
        queries = pd.read_csv(self.data_dir / "queries.dev.small.tsv", sep='\t', 
                            names=['query_id', 'text'])
        qrels = pd.read_csv(self.data_dir / "qrels.dev.tsv", sep='\t', 
                        names=['query_id', 'unused', 'doc_id', 'relevance'])
        
        # Check and limit to n_queries
        total_queries = len(queries['query_id'].unique())
        if self.n_queries > total_queries:
            print(f"Warning: Requested {self.n_queries} queries, but only {total_queries} available. Using {total_queries}.")
            self.n_queries = total_queries
        
        selected_query_ids = queries['query_id'].unique()[:self.n_queries]
        queries = queries[queries['query_id'].isin(selected_query_ids)]
        qrels = qrels[qrels['query_id'].isin(selected_query_ids)]
        
        print(f"Loaded {len(collection):,} documents, {len(queries):,} queries, {len(qrels):,} relevance judgments")
        
        # Initialize query_expander only if method is not None
        if self.method is not None:
            print(collection['text'].tolist())
            self.query_expander = QueryExpander(collection['text'].tolist())
        return collection, queries, qrels
    def create_features(self, collection, queries, qrels):
        print(f"Creating features with method: {self.method or 'none'}...")
        collection = collection[collection['doc_id'].isin(qrels['doc_id'])].reset_index(drop=True)
        doc_vectors = self.vectorizer.fit_transform(collection['text'])
        print("Expanding queries...")
        processed_queries = {}
        for _, row in queries.iterrows():
            qid, text = row['query_id'], row['text']
            processed_queries[qid] = (self.query_expander.expand_query(text, method=self.method, 
                                                                      collection=collection, 
                                                                      vectorizer=self.vectorizer)
                                     if self.method else text)

        query_vectors = self.vectorizer.transform([processed_queries[q] for q in queries['query_id']])
        training_data = []

        for i, qid in enumerate(queries['query_id']):
            print(f"Processing query {qid} ... {i} / {len(queries['query_id'])}")
            query_text = processed_queries[qid]
            q_vec = query_vectors[i].toarray()[0]
            relevant_docs = qrels[qrels['query_id'] == qid]

            # Positive examples
            for _, rel in relevant_docs.iterrows():
                idx = collection[collection['doc_id'] == rel['doc_id']].index[0]
                d_vec = doc_vectors[idx].toarray()[0]
                doc_text = collection.iloc[idx]['text']
                features = self._compute_features(query_text, q_vec, doc_text, d_vec)
                features.update({'query_id': qid, 'doc_id': rel['doc_id'], 'relevance': rel['relevance']})
                training_data.append(features)

            # Hard negatives
            irrelevant = collection[~collection['doc_id'].isin(relevant_docs['doc_id'])]
            if len(irrelevant) > 0:
                ir_vecs = self.vectorizer.transform(irrelevant['text'])
                sims = cosine_similarity(q_vec.reshape(1, -1), ir_vecs)[0]
                top_neg_indices = np.argsort(sims)[-min(5, len(irrelevant)):]
                for idx in top_neg_indices:
                    d_vec = ir_vecs[idx].toarray()[0]
                    doc_text = irrelevant.iloc[idx]['text']
                    features = self._compute_features(query_text, q_vec, doc_text, d_vec)
                    features.update({'query_id': qid, 'doc_id': irrelevant.iloc[idx]['doc_id'], 'relevance': 0})
                    training_data.append(features)

        training_df = pd.DataFrame(training_data)
        print(f"Created {len(training_df):,} training examples")
        return training_df

    def _compute_features(self, query_text, q_vec, doc_text, d_vec):
        cosine_sim = cosine_similarity(q_vec.reshape(1, -1), d_vec.reshape(1, -1))[0][0]
        query_tokens = self.query_expander._preprocess_text(query_text) if self.method else query_text.lower().split()
        doc_tokens = self.query_expander._preprocess_text(doc_text) if self.method else doc_text.lower().split()
        query_terms = set(query_tokens)
        doc_terms = set(doc_tokens)
        term_overlap = len(query_terms & doc_terms)
        query_coverage = term_overlap / len(query_terms) if query_terms else 0
        w2v_sim = self._w2v_similarity(query_tokens, doc_tokens) if self.method else 0.0
        return {'cosine_sim': cosine_sim, 'term_overlap': term_overlap, 
                'query_coverage': query_coverage, 'w2v_sim': w2v_sim}

    def _w2v_similarity(self, tokens1, tokens2):
        valid_tokens1 = [t for t in tokens1 if t in self.query_expander.w2v_model.wv]
        valid_tokens2 = [t for t in tokens2 if t in self.query_expander.w2v_model.wv]
        if not valid_tokens1 or not valid_tokens2:
            return 0.0
        return self.query_expander.w2v_model.wv.n_similarity(valid_tokens1, valid_tokens2)

    def train(self, training_data):
        print("Training LambdaMART model...")
        X = training_data[['cosine_sim', 'term_overlap', 'query_coverage', 'w2v_sim']]
        y = training_data['relevance']
        groups = training_data['query_id']

        gss = GroupShuffleSplit(test_size=0.2, random_state=42)
        train_idx, val_idx = next(gss.split(X, y, groups))
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        groups_train = training_data.iloc[train_idx].groupby('query_id').size().to_numpy()
        groups_val = training_data.iloc[val_idx].groupby('query_id').size().to_numpy()

        train_data = lgb.Dataset(X_train, y_train, group=groups_train)
        val_data = lgb.Dataset(X_val, y_val, group=groups_val, reference=train_data)

        params = {
            'objective': 'lambdarank',
            'metric': ['ndcg', 'map'],
            'boosting_type': 'gbdt',
            'num_leaves': 15,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'verbose': -1
        }
        self.model = lgb.train(params, train_data, num_boost_round=100, 
                              valid_sets=[val_data], callbacks=[lgb.early_stopping(10)])

    def evaluate(self, test_data):
        print("Evaluating model...")
        X_test = test_data[['cosine_sim', 'term_overlap', 'query_coverage', 'w2v_sim']]
        test_data['pred_score'] = self.model.predict(X_test)
        return {
            'MAP': self._calculate_map(test_data),
            'MRR': self._calculate_mrr(test_data),
            'nDCG@10': self._calculate_ndcg(test_data, k=10)
        }

    def _calculate_map(self, df):
        ap_scores = []
        for qid in df['query_id'].unique():
            q_df = df[df['query_id'] == qid].sort_values('pred_score', ascending=False)
            rel_docs = q_df[q_df['relevance'] > 0]['doc_id'].tolist()
            if not rel_docs:
                continue
            prec_at_k = []
            rel_count = 0
            for i, (_, row) in enumerate(q_df.iterrows()):
                if row['relevance'] > 0:
                    rel_count += 1
                    prec_at_k.append(rel_count / (i + 1))
            ap_scores.append(sum(prec_at_k) / len(rel_docs))
        return sum(ap_scores) / len(ap_scores) if ap_scores else 0

    def _calculate_mrr(self, df):
        rr_scores = []
        for qid in df['query_id'].unique():
            q_df = df[df['query_id'] == qid].sort_values('pred_score', ascending=False)
            for i, (_, row) in enumerate(q_df.iterrows()):
                if row['relevance'] > 0:
                    rr_scores.append(1 / (i + 1))
                    break
            else:
                rr_scores.append(0)
        return sum(rr_scores) / len(rr_scores) if rr_scores else 0

    def _calculate_ndcg(self, df, k=10):
        ndcg_scores = []
        for qid in df['query_id'].unique():
            q_df = df[df['query_id'] == qid]
            pred = q_df.sort_values('pred_score', ascending=False).head(k)
            ideal = q_df.sort_values('relevance', ascending=False).head(k)
            dcg = sum((2 ** row['relevance'] - 1) / np.log2(i + 2) for i, (_, row) in enumerate(pred.iterrows()))
            idcg = sum((2 ** row['relevance'] - 1) / np.log2(i + 2) for i, (_, row) in enumerate(ideal.iterrows()))
            ndcg_scores.append(dcg / idcg if idcg > 0 else 0)
        return sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0

    def predict(self, query_text, collection, top_k=5):
        if self.method:
            expanded_query = self.query_expander.expand_query(query_text, self.method, 
                                                             collection=collection, vectorizer=self.vectorizer)
            print(f"Method: {self.method}, Original: {query_text}, Expanded: {expanded_query}")
            query_text = expanded_query
        q_vec = self.vectorizer.transform([query_text]).toarray()[0]
        doc_vecs = self.vectorizer.transform(collection['text'])
        results = []
        for i, (_, doc) in enumerate(collection.iterrows()):
            d_vec = doc_vecs[i].toarray()[0]
            features = self._compute_features(query_text, q_vec, doc['text'], d_vec)
            results.append({'doc_id': doc['doc_id'], 'text': doc['text'], **features})
        results_df = pd.DataFrame(results)
        X_pred = results_df[['cosine_sim', 'term_overlap', 'query_coverage', 'w2v_sim']]
        results_df['score'] = self.model.predict(X_pred)
        return results_df.sort_values('score', ascending=False).head(top_k)

    def plot_feature_importance(self, name):
        importance = self.model.feature_importance(importance_type='gain')
        names = self.model.feature_name()
        plt.figure(figsize=(6, 4))
        plt.barh(names, importance)
        plt.xlabel("Feature Importance (Gain)")
        plt.title(f"Feature Importance — {name}")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f"{name}_feature_importance.png")
        plt.close()

def main():
    data_dir = Path("project-root/data/raw")
    methods = [None, 'wordnet', 'word2vec', 'pseudo_relevance', 'porter_stemmer', 'combined']
    results = {}
    queries = pd.read_csv(data_dir / "queries.dev.small.tsv", sep='\t', 
                            names=['query_id', 'text'])
    # Load data once
    base = LambdaMART(data_dir, n_queries=len(queries), query_expansion_method=None)
    collection, queries, qrels = base.load_data()
    collection = collection[collection['doc_id'].isin(qrels['doc_id'])].reset_index(drop=True)

    for method in methods:
        name = method or 'none'
        print(f"\n===== EXPANSION: {name.upper()} =====")
        lm = LambdaMART(data_dir, n_queries=len(queries), query_expansion_method=method)
        if method is not None:
            lm.query_expander = QueryExpander(collection['text'].tolist())


        train_df = lm.create_features(collection, queries, qrels)
        train_df, test_df = train_test_split(train_df, test_size=0.2, random_state=42)
        lm.train(train_df)

        metrics = lm.evaluate(test_df)
        results[name] = metrics
        print(f"Metrics → MAP: {metrics['MAP']:.4f}, MRR: {metrics['MRR']:.4f}, nDCG@10: {metrics['nDCG@10']:.4f}")
        lm.plot_feature_importance(name)

        example = queries.iloc[0]['text']
        print(f"\nExample query ({name}): {example}")
        preds = lm.predict(example, collection, top_k=5)
        print(preds[['doc_id', 'score', 'text']].to_string(index=False))

    print("\n=== SUMMARY ===")
    print(pd.DataFrame(results).T)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)