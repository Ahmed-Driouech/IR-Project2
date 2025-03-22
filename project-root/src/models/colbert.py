from colbert.infra import Run, RunConfig
from colbert import Indexer, Searcher
from colbert.data import Queries
import os

class ColBERTRetriever:
    def __init__(self, model_name="colbert-ir/colbertv2.0", index_dir="loading_data/processed/colbert_index"):
        self.model_name = model_name
        self.index_dir = index_dir
        self.checkpoint = model_name
        self.indexed = False

    def index(self, passages):
        # Passages: list of (docid, text)
        if not os.path.exists(self.index_dir):
            with Run().context(RunConfig(nranks=1)):  # Single GPU/CPU for simplicity
                indexer = Indexer(checkpoint=self.checkpoint)
                indexer.index(name="msmarco", collection=[text for _, text in passages], overwrite=True)
            self.indexed = True

    def retrieve(self, query, top_k=100):
        if not self.indexed:
            raise RuntimeError("Index not built. Call index() first.")
        with Run().context(RunConfig(nranks=1)):
            searcher = Searcher(index=self.index_dir, checkpoint=self.checkpoint)
            results = searcher.search(query, k=top_k)
            # Returns (docids, scores, ranks); we want (docid, score)
            return list(zip(results[0], results[1]))
