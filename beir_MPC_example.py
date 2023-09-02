from time import time
from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
# from beir.retrieval.search.dense import DenseRetrievalExactSearch
from LocalDenseRetrievalExactSearch import DenseRetrievalExactSearch
from MPCDenseRetrievalExactSearch import MPCDenseRetrievalExactSearch


import logging
import pathlib, os, numpy as np
import random, pickle
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

# Setup
dataset = "trec-covid"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

# We're going to trim down the dataset for testing.

def reduce_corpus_size(corpus, qrels, queries, sample_size: int = 500):
    sample_size = 500
    queries = {k:v for k,v in queries.items() if np.random.rand() < 0.05}
    corpus_ids, query_ids = list(corpus), list(queries)
    qrels = {k:v for k,v in qrels.items() if k in query_ids}
    corpus_set = set()
    for query_id in qrels:
        corpus_set.update(list(qrels[query_id].keys()))
    corpus_new = {corpus_id: corpus[corpus_id] for corpus_id in corpus_set}

    remaining_corpus = list(set(corpus_ids) - corpus_set)

    for corpus_id in random.sample(remaining_corpus, sample_size):
        corpus_new[corpus_id] = corpus[corpus_id]
    corpus = corpus_new
    return corpus, qrels, queries

def benchmark_retriever(retriever, corpus, queries, qrels):
    start_time = time()
    results = retriever.retrieve(corpus, queries)
    end_time = time()
    print("Time taken to retrieve: {:.2f} seconds".format(end_time - start_time))
    #### Evaluate your retrieval using NDCG@k, MAP@K ...
    logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")
    recall_cap = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="r_cap")
    hole = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="hole")
    print("Performance of DenseRetrievalExactSearch: {recall}, {precision}, {ndcg}, {map}, {mrr}, {recall_cap}, {hole}".format(recall=recall, precision=precision, ndcg=ndcg, map=_map, mrr=mrr, recall_cap=recall_cap, hole=hole))
    return recall, precision


corpus, qrels, queries = reduce_corpus_size(corpus, qrels, queries, 500)
print("Corpus size: {} on {} queries".format(len(corpus), len(queries)))
# It's good to save these for reproducibility
pickle.dump([corpus, qrels, queries], open("corpus.pkl", "wb"))
corpus, qrels, queries = pickle.load(open("corpus.pkl", "rb"))


#### Dense Retrieval using SBERT (Sentence-BERT) ####
embedding_model = models.SentenceBERT("msmarco-distilbert-base-tas-b")
from sentence_transformers import SentenceTransformer
embedding_model.q_model = SentenceTransformer("msmarco-distilbert-base-tas-b", device="mps") # This is just to force the pytorch device for speed reasons

model = DenseRetrievalExactSearch(embedding_model, batch_size=256, corpus_chunk_size=512*9999)

# model.preemebed_corpus(corpus, save_path="corpus_embeddings.pt")
# model.preembed_queries(queries, save_path="query_embeddings.pt")
model.load_preembeddings("corpus_embeddings.pt", "query_embeddings.pt")

# Now we benchmark normal dense retrieval
retriever = EvaluateRetrieval(model, score_function="cos_sim")
benchmark_retriever(retriever, corpus, queries, qrels)


# Now we benchmark MPC dense retrieval
model = MPCDenseRetrievalExactSearch(embedding_model, batch_size=256, corpus_chunk_size=512*9999)
model.load_preembeddings("corpus_embeddings.pt", "query_embeddings.pt")

retriever = EvaluateRetrieval(model, score_function="cos_sim")
benchmark_retriever(retriever, corpus, queries, qrels)

retriever = EvaluateRetrieval(model, score_function="mpc_opt")
benchmark_retriever(retriever, corpus, queries, qrels)

retriever = EvaluateRetrieval(model, score_function="mpc_naive")
benchmark_retriever(retriever, corpus, queries, qrels)

retriever = EvaluateRetrieval(model, score_function="dot")
benchmark_retriever(retriever, corpus, queries, qrels)

retriever = EvaluateRetrieval(model, score_function="mpc_dot")
benchmark_retriever(retriever, corpus, queries, qrels)
