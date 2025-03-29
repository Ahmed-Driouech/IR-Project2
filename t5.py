import numpy as np
import pandas as pd
import torch
from transformers import T5EncoderModel, T5Tokenizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import os
import gc
import nltk
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
import gensim.downloader as gensim_downloader
import argparse
import time

def calculate_ndcg(relevance_scores, k=10):
    ndcg_values = []
    for relevance in relevance_scores:
        processed_relevance = []
        for rel in relevance:
            if isinstance(rel, np.ndarray):
                rel_value = rel.item() if rel.size == 1 else rel.any()
            else:
                rel_value = rel
            processed_relevance.append(int(rel_value))

        relevance_k = processed_relevance[:k]
        if sum(relevance_k) == 0:
            ndcg_values.append(0.0)
            continue
        dcg = 0.0
        for i, rel in enumerate(relevance_k):
            dcg += rel / np.log2(i + 2)
        ideal_relevance = sorted(processed_relevance, reverse=True)[:k]
        idcg = 0.0
        for i, rel in enumerate(ideal_relevance):
            idcg += rel / np.log2(i + 2)
        if idcg == 0:
            ndcg_values.append(0.0)
        else:
            ndcg_values.append(dcg / idcg)
    return np.mean(ndcg_values)
    
def calculate_mrr(relevance_scores):
    reciprocal_ranks = []
    for scores in relevance_scores:
        found = False
        for i, score in enumerate(scores):
            if isinstance(score, np.ndarray):
                score_value = score.item() if score.size == 1 else score.any()
            else:
                score_value = score

            if score_value == 1:
                reciprocal_ranks.append(1.0 / (i + 1))
                found = True
                break
        if not found:
            reciprocal_ranks.append(0.0)
    return np.mean(reciprocal_ranks)

def calculate_map(relevance_scores):
    average_precisions = []

    for relevance in relevance_scores:
        processed_relevance = []
        for rel in relevance:
            if isinstance(rel, np.ndarray):
                rel_value = rel.item() if rel.size == 1 else rel.any()
            else:
                rel_value = rel
            processed_relevance.append(int(rel_value))

        if sum(processed_relevance) == 0:  
            average_precisions.append(0.0)
            continue
        precisions = []
        relevant_count = 0

        for i, rel in enumerate(processed_relevance):
            if rel == 1:
                relevant_count += 1
                precisions.append(relevant_count / (i + 1))

        if precisions:
            average_precisions.append(sum(precisions) / sum(processed_relevance))
        else:
            average_precisions.append(0.0)

    return np.mean(average_precisions)

def encode_text(model, tokenizer, texts, batch_size=4, max_length=256, device="cpu"):
    
    all_embeddings = []
    if not isinstance(texts, list):
        texts = [texts]

    texts = [str(text) for text in texts]

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]

        inputs = tokenizer(
            batch_texts, 
            padding=True, 
            truncation=True, 
            max_length=max_length, 
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

            embeddings = outputs.last_hidden_state.mean(dim=1)
            all_embeddings.append(embeddings.cpu().numpy())

        del inputs, outputs
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return np.vstack(all_embeddings)


def expand_with_wordnet(query, max_synonyms=3):

    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)

    words = query.split()
    expanded_words = words.copy()

    for word in words:
        synsets = wordnet.synsets(word)
        synonyms = set()

        for synset in synsets[:2]:  
            for lemma in synset.lemmas()[:2]:  
                synonym = lemma.name().replace('_', ' ')
                if synonym != word and synonym not in synonyms:
                    synonyms.add(synonym)
                    if len(synonyms) >= max_synonyms:
                        break
            if len(synonyms) >= max_synonyms:
                break

        expanded_words.extend(list(synonyms))

    return ' '.join(expanded_words)

def expand_with_word2vec(query, word2vec_model, topn=3):
    words = query.split()
    expanded_words = words.copy()

    for word in words:
        try:
            similar_words = [w for w, _ in word2vec_model.most_similar(word, topn=topn) 
                            if w.lower() != word.lower()]
            expanded_words.extend(similar_words)
        except KeyError:
            continue

    return ' '.join(expanded_words)

def expand_with_porter_stemmer(query):
    stemmer = PorterStemmer()
    words = query.split()
    stemmed_words = [stemmer.stem(word) for word in words]

    expanded_words = words + [stem for stem in stemmed_words if stem not in words]

    return ' '.join(expanded_words)

def expand_with_pseudo_relevance(query, passages, model, tokenizer, top_k=3, terms_to_add=5, device="cpu"):

    query_embedding = encode_text(model, tokenizer, query, device=device)
    passage_embeddings = encode_text(model, tokenizer, passages, batch_size=4, device=device)
    similarity_scores = cosine_similarity(query_embedding, passage_embeddings)[0]

    top_indices = np.argsort(similarity_scores)[-top_k:]
    top_passages = [passages[i] for i in top_indices]

    query_terms = set(query.lower().split())
    term_counts = {}

    for passage in top_passages:
        for term in passage.lower().split():
            if term not in query_terms and len(term) > 3:  
                term_counts[term] = term_counts.get(term, 0) + 1

    
    expansion_terms = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)[:terms_to_add]
    expansion_terms = [term for term, _ in expansion_terms]

    expanded_query = query + " " + " ".join(expansion_terms)

    return expanded_query

def combined_expansion(query, passages, model, tokenizer, word2vec_model, device="cpu"):

    wordnet_expanded = expand_with_wordnet(query)
    word2vec_expanded = expand_with_word2vec(query, word2vec_model)
    stemmed_expanded = expand_with_porter_stemmer(query)

    sample_passages = passages[:min(100, len(passages))]
    pseudo_rel_expanded = expand_with_pseudo_relevance(
        query, sample_passages, model, tokenizer, device=device
    )

    all_terms = set()
    all_terms.update(wordnet_expanded.split())
    all_terms.update(word2vec_expanded.split())
    all_terms.update(stemmed_expanded.split())
    all_terms.update(pseudo_rel_expanded.split())

    return " ".join(all_terms)

def evaluate_with_expansion(
    validation_path, 
    model_name="t5-small", 
    expansion_method="none",
    limit=100, 
    batch_size=4, 
    max_length=256
):

    print(f"Loading validation data from {validation_path}")
    df = pd.read_parquet(validation_path)
 
    word2vec_model = None

    unique_queries = df['query'].unique()[:limit]
    print(f"Evaluating on {len(unique_queries)} unique queries")
 
    print(f"Loading T5 model: {model_name}")
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5EncoderModel.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    model.eval()


    if expansion_method in ["word2vec", "combined"]:
        print("Loading Word2Vec model...")
        word2vec_model = gensim_downloader.load("glove-wiki-gigaword-100")

  
    all_relevance_scores = []

    print(f"Using expansion method: {expansion_method}")

    for query in tqdm(unique_queries):
        query_df = df[df['query'] == query]
        
        for _, row in query_df.iterrows():
            try:
                passages = row['passages']['passage_text']
                relevances = row['passages']['is_selected']

                if not isinstance(passages, (list, np.ndarray)):
                    passages = [passages]
                if not isinstance(relevances, (list, np.ndarray)):
                    relevances = [relevances]

                if isinstance(passages, np.ndarray):
                    passages = passages.tolist()

                if len(passages) == 0:
                    continue
            except Exception as e:
                print(f"Error processing row: {e}")
                print(f"Row structure: {row.keys()}")
                if 'passages' in row:
                    print(f"Passages structure: {row['passages'].keys()}")
                continue

    
            if expansion_method == "wordnet":
                expanded_query = expand_with_wordnet(query)
            elif expansion_method == "word2vec":
                expanded_query = expand_with_word2vec(query, word2vec_model)
            elif expansion_method == "porter_stemmer":
                expanded_query = expand_with_porter_stemmer(query)
            elif expansion_method == "pseudo_relevance":
                expanded_query = expand_with_pseudo_relevance(
                    query, passages, model, tokenizer, device=device
                )
            elif expansion_method == "combined":
                expanded_query = combined_expansion(
                    query, passages, model, tokenizer, word2vec_model, device=device
                )
            else: 
                expanded_query = query

            
            query_embedding = encode_text(
                model, tokenizer, expanded_query, batch_size=1, max_length=max_length, device=device
            )

            passage_embeddings = encode_text(
                model, tokenizer, passages, batch_size=batch_size, max_length=max_length, device=device
            )

            similarity_scores = cosine_similarity(query_embedding, passage_embeddings)[0]

            passage_scores = list(zip(similarity_scores, relevances))
            passage_scores.sort(key=lambda x: x[0], reverse=True)

            ranked_relevances = [rel for _, rel in passage_scores]
            

            all_relevance_scores.append(ranked_relevances)

            gc.collect()

    print("Calculating evaluation metrics...")
    mrr = calculate_mrr(all_relevance_scores)
    map_score = calculate_map(all_relevance_scores)
    ndcg_at_10 = calculate_ndcg(all_relevance_scores, k=10)  

    return {
        "MRR": mrr,
        "MAP": map_score,
        "nDCG@10": ndcg_at_10  
    }

def run_all_expansion_methods(validation_path, model_name="t5-small", limit=100):

    methods = [
        "none", 
        "wordnet", 
        "word2vec", 
        "pseudo_relevance", 
        "porter_stemmer", 
        "combined"
    ]

    results = {}

    print("=== SUMMARY ===")
    print(f"{'Method':20} {'MAP':10} {'MRR':10} {'nDCG@10':10}")  
    print("-" * 60)  

    for method in methods:
        print(f"\nEvaluating with {method} expansion...")
        start_time = time.time()

        method_results = evaluate_with_expansion(
            validation_path=validation_path,
            model_name=model_name,
            expansion_method=method,
            limit=limit
        )

        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.2f} seconds")

        results[method] = method_results

    
        print(f"{method:20} {method_results['MAP']:10.6f} {method_results['MRR']:10.6f} "
              f"{method_results['nDCG@10']:10.6f}")  

    print("\n=== SUMMARY ===")
    print(f"{'':20} {'MAP':10} {'MRR':10} {'nDCG@10':10}")  
    for method in methods:
        print(f"{method:20} {results[method]['MAP']:10.6f} {results[method]['MRR']:10.6f} "
              f"{results[method]['nDCG@10']:10.6f}")  

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate T5 with query expansion on MS MARCO")
    parser.add_argument("--validation_path", type=str, default="msmarco_data/validation-00000-of-00001.parquet",
                        help="Path to validation file")
    parser.add_argument("--model_name", type=str, default="t5-small",
                        help="T5 model name (e.g. t5-small)")
    parser.add_argument("--limit", type=int, default=10,
                        help="Number of queries to evaluate")
    parser.add_argument("--method", type=str, default="all",
                        choices=["none", "wordnet", "word2vec", "pseudo_relevance", "porter_stemmer", "combined", "all"],
                        help="Expansion method to use")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug output")

    args = parser.parse_args()

   
    if not os.path.exists(args.validation_path):
        print(f"Error: Validation file not found at {args.validation_path}")
        print("Please download the MS MARCO dataset first.")
        exit(1)

    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("Downloading WordNet corpus...")
        nltk.download('wordnet', quiet=True)

    if args.method == "all":
        run_all_expansion_methods(
            validation_path=args.validation_path,
            model_name=args.model_name,
            limit=args.limit
        )
    else:
        results = evaluate_with_expansion(
            validation_path=args.validation_path,
            model_name=args.model_name,
            expansion_method=args.method,
            limit=args.limit,
        )

        print(f"\nResults for {args.method} expansion:")
        print(f"MAP: {results['MAP']:.6f}")
        print(f"MRR: {results['MRR']:.6f}")
        print(f"nDCG@10: {results['nDCG@10']:.6f}") 