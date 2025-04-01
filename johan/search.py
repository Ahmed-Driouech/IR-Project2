import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as gensim_downloader
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher
import joblib
import warnings
import os
import datetime



class QueryExpander:
    def __init__(self, collection_texts: list):
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        # Train Word2Vec on full collection
        # print('Pre-processing...')
        # sentences = [self._preprocess_text(text) for text in collection_texts]
        # print('Training...')
        # self.w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=5, workers=4)
        print("Loading Word2Vec model...")
        self.word2vec_model = gensim_downloader.load("glove-wiki-gigaword-100")

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
        topn=3
        words = query.split()
        expanded_words = words.copy()
    
        for word in words:
            try:
                similar_words = [w for w, _ in self.word2vec_model.most_similar(word, topn=topn) 
                                if w.lower() != word.lower()]
                expanded_words.extend(similar_words)
            except KeyError:
                continue
    
        return ' '.join(expanded_words)

    def expand_query_pseudo_relevance(self, doc_vecs, query: str, collection: pd.DataFrame, 
                                     vectorizer: TfidfVectorizer, top_k: int = 3, 
                                     num_expansions: int = 2) -> str:
        query_vec = vectorizer.transform([query])
        # doc_vecs = vectorizer.transform(collection['text'])
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
                     vectorizer: TfidfVectorizer = None, doc_vecs = None) -> str:
        if method == 'wordnet':
            return self.expand_query_wordnet(query)
        elif method == 'word2vec':
            return self.expand_query_word2vec(query)
        elif method == 'pseudo_relevance' and collection is not None and vectorizer is not None:
            return self.expand_query_pseudo_relevance(doc_vecs, query, collection, vectorizer)
        elif method == 'porter_stemmer':
            return self.expand_porter_stemmer(query)
        elif method == 'combined' and collection is not None and vectorizer is not None:
            q = self.expand_query_wordnet(query)
            q = self.expand_query_word2vec(q)
            q = self.expand_query_pseudo_relevance(doc_vecs, q, collection, vectorizer)
            return self.expand_porter_stemmer(q)
        return query

def evaluate(collection, queries, qrels, method, vect, qe, doc_vecs):
    dfs = []
    for i in range(len(queries)):
        if i % 100 == 0: 
            now = datetime.datetime.now()
            print('[', now.strftime("%B"), ' ', now.strftime("%d"), ', ', now.strftime('%H:%M:%S'), '] ', 'Query: ', i)
        query = queries['text'][i]
        if method is not None:
            query = qe.expand_query(query, method, collection, vect, doc_vecs)
        query_id = queries['query_id'][i]
        results = searcher.search(query, k=100)
        res = pd.DataFrame(results)
        res = res.transpose()
        res.columns = ['doc_id','rank','pred_score']
        res['query_id'] = query_id
        res['relevance'] = 0       
        for doc_id in qrels[qrels['query_id'] == query_id]['doc_id'].to_list():
            res.loc[res['doc_id'] == doc_id, 'relevance'] = 1
        dfs.append(res)
    query_results = pd.concat(dfs, ignore_index=True) 
    print('Calculating MAP, MRR and nDCG@10...')
    return {
        'MAP': _calculate_map(query_results),
        'MRR': _calculate_mrr(query_results),
        'nDCG@10': _calculate_ndcg(query_results, k=10)
    }

def _calculate_map(df):
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

def _calculate_mrr(df):
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

def _calculate_ndcg(df, k=10):
    ndcg_scores = []
    for qid in df['query_id'].unique():
        q_df = df[df['query_id'] == qid]
        pred = q_df.sort_values('pred_score', ascending=False).head(k)
        ideal = q_df.sort_values('relevance', ascending=False).head(k)
        dcg = sum((2 ** row['relevance'] - 1) / np.log2(i + 2) for i, (_, row) in enumerate(pred.iterrows()))
        idcg = sum((2 ** row['relevance'] - 1) / np.log2(i + 2) for i, (_, row) in enumerate(ideal.iterrows()))
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0)
    return sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0

warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__=='__main__':
    with Run().context(RunConfig(nranks=1, experiment="msmarcoindex")):
        config = ColBERTConfig(
            root="experiments",
        )
        searcher = Searcher(index="msmarcoindex.nbits_2", config=config)

methods = [None, 'wordnet', 'word2vec', "pseudo_relevance", 'porter_stemmer', 'combined']


data_dir = './project-root/data/raw/'
print('Loading tsv files...')
collection = pd.read_csv(data_dir + "collection.tsv", sep='\t', 
                                names=['doc_id', 'text'])
queries = pd.read_csv(data_dir + "queries.dev.small.tsv", sep='\t', 
                    names=['query_id', 'text'])
qrels = pd.read_csv(data_dir + "qrels.dev.tsv", sep='\t', 
                names=['query_id', 'unused', 'doc_id', 'relevance'])
vectorizer = joblib.load('trained/vectorizer.pkl')
file = open("trained/doc_vecs.pickle",'rb') 
doc_vecs = pickle.load(file)
query_expander = QueryExpander(None)
for method in methods:
    name = method or 'none'
    print(f"\n===== EXPANSION: {name.upper()} =====")

    metrics = evaluate(collection, queries, qrels, method, vectorizer, query_expander, doc_vecs)
    print(f"Metrics → MAP: {metrics['MAP']:.4f}, MRR: {metrics['MRR']:.4f}, nDCG@10: {metrics['nDCG@10']:.4f}")