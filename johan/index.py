from colbert.data import Queries, Collection
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer, Searcher
import os

collection = os.path.join('./project-root/data/raw/collection.tsv')
collection = Collection(path=collection)
checkpoint = './colbertv2.0' 

if __name__=='__main__':
    with Run().context(RunConfig(nranks=1, experiment="msmarcoindex", avoid_fork_if_possible = True)): 
        config = ColBERTConfig(
            nbits=2,
            root="experiments",
            kmeans_niters = 2,
        )
        indexer = Indexer(checkpoint=checkpoint, config=config)
        indexer.index(name="msmarcoindex.nbits_2", collection=collection, overwrite=True)