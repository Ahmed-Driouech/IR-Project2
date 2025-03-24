from evaluate_bm25 import QueryExpander, BM25Evaluator
from sklearn.feature_extraction.text import TfidfVectorizer


def main():
    test_collection = {
        "doc1": "This is a test document about information retrieval",
        "doc2": "Another document discussing search engines and ranking",
        "doc3": "Relevance feedback can improve search results",
        "doc4": "Query expansion is a technique to improve recall"
    }

    vectorizer = TfidfVectorizer(min_df=2, max_df=0.85)
    vectorizer.fit(list(test_collection.values()))

    query_expansion = QueryExpander()
    test_query = "information retrieval"

    try:
        expanded_query = query_expansion.pseudo_relevance_query_expansion(
            test_query,
            test_collection,
            vectorizer
        )
        print(f"Original query: {test_query}")
        print(f"Expanded query: {expanded_query}")
        print("Pseudo-relevance expansion test succeeded!")
    except Exception as e:
        print(f"Error during pseudo-relevance expansion: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()