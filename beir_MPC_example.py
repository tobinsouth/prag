import time
from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch
from MPCDenseRetrievalExactSearch import MPCDenseRetrievalExactSearch

import logging
import pathlib, os, numpy as np
import random, pickle
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

def benchmark_retriever(retriever, corpus, queries, qrels):
    start_time = time.time()
    results = retriever.retrieve(corpus, queries)
    end_time = time.time()
    try:
        # print("Time taken to retrieve: {:.2f} seconds".format(end_time - start_time))
        #### Evaluate your retrieval using NDCG@k, MAP@K ...
        # logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
        ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
        mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")
        recall_cap = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="r_cap")
        hole = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="hole")
        # print("Performance of DenseRetrievalExactSearch: {recall}, {precision}, {ndcg}, {map}, {mrr}, {recall_cap}, {hole}".format(recall=recall, precision=precision, ndcg=ndcg, map=_map, mrr=mrr, recall_cap=recall_cap, hole=hole))
        print("Time taken: {:.2f} Recall@1: {}, Recall@5: {}".format(end_time-start_time, recall['Recall@1'], recall.get('Recall@5', np.NaN)))
    except:
        print("Time taken: {:.2f}".format(end_time-start_time))
        return end_time-start_time, None, None, None, None, None, None, results
    return end_time-start_time, recall, precision, ndcg, mrr, recall_cap, hole, results

def setup(reduce_corpus_size: bool = True, sample_size: int = 500, proportion: float = 0.1):
    """This is a good one-time function to run to start embedding the corpus and queries and then save them to file for later use."""
    from LocalDenseRetrievalExactSearch import DenseRetrievalExactSearch
    # Setup
    dataset = "fiqa"
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
    data_path = util.download_and_unzip(url, out_dir)
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    # We're going to trim down the dataset for testing.

    def reduce_corpus_size(corpus, qrels, queries, sample_size: int = 500, proportion: float = 0.1):
        sample_size = 500
        queries = {k:v for k,v in queries.items() if np.random.rand() < proportion}
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
    
    if reduce_corpus_size:
        corpus, qrels, queries = reduce_corpus_size(corpus, qrels, queries, sample_size, proportion)
    print("Corpus size: {} on {} queries".format(len(corpus), len(queries)))
    # It's good to save these for reproducibility
    pickle.dump([corpus, qrels, queries], open("datasets/corpus_fiqa.pkl", "wb"))

    #### Dense Retrieval using SBERT (Sentence-BERT) ####
    print("Beginning embedding")
    embedding_model = models.SentenceBERT("msmarco-distilbert-base-v3", device="cuda")
    # from sentence_transformers import SentenceTransformer
    # embedding_model.q_model = SentenceTransformer("msmarco-distilbert-base-v3", device="cuda") # This is just to force the pytorch device for speed reasons

    model = DenseRetrievalExactSearch(embedding_model, batch_size=256, corpus_chunk_size=512*2**6, k_values=[1,3,5,50])

    model.preembed_queries(queries, save_path="datasets/query_embeddings_fiqa.pt")
    model.preemebed_corpus(corpus, save_path="datasets/corpus_embeddings_fiqa.pt")

    model.load_preembeddings("datasets/corpus_embeddings_fiqa.pt", "datasets/query_embeddings_fiqa.pt")


    print("Finished embedding, testing out retrieval")
    # Now we benchmark normal dense retrieval
    retriever = EvaluateRetrieval(model, score_function="dot_score")
    results = retriever.retrieve(corpus, queries)

    # timetaken, recall, precision, ndcg, mrr, recall_cap, hole, results = benchmark_retriever(retriever, corpus, queries, qrels)

    top_k = 10
    query_id = 1824

    # query_id, ranking_scores = random.choice(list(results.items()))
    ranking_scores = results[str(query_id)]
    scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
    print("Query : %s\n" % queries[str(query_id)])

    for rank in range(top_k):
        doc_id = scores_sorted[rank][0]
        # Format: Rank x: ID [Title] Body
        print("Rank %d: %s [%s] - %s\n" % (rank+1, doc_id, corpus[doc_id].get("title"), corpus[doc_id].get("text")))

# setup(reduce_corpus_size=False, sample_size=500, proportion=0.1)

corpus, qrels, queries = pickle.load(open("datasets/corpus_fiqa.pkl", "rb"))

# Now we benchmark MPC dense retrieval
print("Building BEIR model and loading pre-embeddings.")
model = MPCDenseRetrievalExactSearch(None, corpus_chunk_size=512*6)

# Load in premade embeddings
model.load_preembeddings("datasets/corpus_embeddings_fiqa.pt", "datasets/query_embeddings_fiqa.pt")
print("Loaded embeddings")

# model._search(corpus, queries, top_k=5, score_function="dot_score");
# model._search_mulit_threaded(corpus, queries, top_k=5, score_function="cos_sim");


# print("Running basic retrieval")
retriever = EvaluateRetrieval(model, score_function="dot_score",  k_values=[1,3,5,10])
timetaken, recall, *the_rest =benchmark_retriever(retriever, corpus, queries, qrels)
pickle.dump([timetaken, recall, *the_rest], open("beir_results_dot_score.pkl", "wb"))
print(timetaken, recall)

# print("Running MPC distance with basic top-k")
# retriever = EvaluateRetrieval(model, score_function="mpc_dot_vanilla_topk",  k_values=[1,3,5, 10])
# timetaken, recall, *the_rest =benchmark_retriever(retriever, corpus, queries, qrels)
# pickle.dump([timetaken, recall, *the_rest], open("beir_results_dot_score.pkl", "wb"))


print("Running MPC distance with MPC top-k")
retriever = EvaluateRetrieval(model, score_function="mpc_dot_topk",  k_values=[1,3,5])
timetaken, recall, *the_rest =benchmark_retriever(retriever, corpus, queries, qrels)
print(timetaken, recall)
pickle.dump([timetaken, recall, *the_rest], open("beir_results_mpc_dot_topk.pkl", "wb"))

# print("Now we loop through and benchmark everything, saving the results to beir_results.pkl")
# results = {}
# for score_function in ["cos_sim", "dot_score", "mpc_dot_vanilla_topk", "mpc_cos_vanilla_topk", "mpc_cos2_vanilla_topk", "mpc_eucld_vanilla_topk", "mpc_dot_topk", "mpc_cos_topk", "mpc_cos2_topk", "mpc_eucld_topk"]:
#     retriever = EvaluateRetrieval(model, score_function="cos_sim",  k_values=[1,3,5,10])
#     timetaken, recall, *the_rest = benchmark_retriever(retriever, corpus, queries, qrels)
#     results[score_function] = [timetaken, recall]
    
#     pickle.dump(results, open("beir_results.pkl", "wb"))




from beir.retrieval.search.dense import DenseRetrievalExactSearch

