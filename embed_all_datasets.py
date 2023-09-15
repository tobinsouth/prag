import time
from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
# from beir.retrieval.search.dense import DenseRetrievalExactSearch
from MPCDenseRetrievalExactSearch import MPCDenseRetrievalExactSearch

import logging
import pathlib, os, numpy as np
import random, pickle


datasets = ['trec-covid','nfcorpus', 'fiqa', 'arguana', 'quora', 'scidocs',  'msmarco']
# datasets = ['msmarco']


from LocalDenseRetrievalExactSearch import DenseRetrievalExactSearch

embedding_model = models.SentenceBERT("msmarco-distilbert-base-v3", device="cuda:2")
from sentence_transformers import SentenceTransformer
embedding_model.q_model = SentenceTransformer("msmarco-distilbert-base-v3", device="cuda:2") # This is just to force the pytorch device for speed reasons
model = DenseRetrievalExactSearch(embedding_model, batch_size=256, corpus_chunk_size=512*2**2)
model_name = 'mdb3'


for dataset in datasets:
    print("Loading dataset: {}".format(dataset))
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
    data_path = util.download_and_unzip(url, out_dir)
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    print("Corpus size: {} on {} queries".format(len(corpus), len(queries)))

    # It's good to save these for reproducibility
    pickle.dump([corpus, qrels, queries], open(f"datasets/corpus_{dataset}_full_{model_name}.pkl", "wb"))


    print("Beginning embedding")
    model.preembed_queries(queries, save_path=f"datasets/query_embeddings_{dataset}_full_{model_name}.pt")
    model.preemebed_corpus(corpus, save_path=f"datasets/corpus_embeddings_{dataset}_full_{model_name}.pt")

