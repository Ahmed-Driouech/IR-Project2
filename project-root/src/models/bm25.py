from pyserini.search import SimpleSearcher

class BM25Retriever:
    def __init__(self, index_dir="loading_data/processed/indexes/msmarco-passage", k1=0.9, b=0.4):
        self.searcher = SimpleSearcher(index_dir)
        self.searcher.set_bm25(k1=k1, b=b)

    def retrieve(self, query, top_k=100):
        hits = self.searcher.search(query, k=top_k)
        return [(hit.docid, hit.score) for hit in hits]
