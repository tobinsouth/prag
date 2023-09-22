from MPCDenseRetrievalExactSearch import MPCDenseRetrievalExactSearch
import pickle
import numpy as np, torch
import time


load_string = 'fiqa'
corpus, qrels, queries = pickle.load(open(f"datasets/corpus_{load_string}.pkl", "rb"))
model = MPCDenseRetrievalExactSearch(None, corpus_chunk_size=512*6)
model.load_preembeddings(f"datasets/corpus_embeddings_{load_string}.pt", f"datasets/query_embeddings_{load_string}.pt")

corpus_embeddings = model.corpus_embeddings
query_embeddings = model.query_embeddings

from beir.retrieval.evaluation import EvaluateRetrieval
retriever = EvaluateRetrieval(model, score_function="dot_score",  k_values=[1,3,5,10,100])
results_basic = retriever.retrieve(corpus, queries)
ndcg, _map, recall, precision = retriever.evaluate(qrels, results_basic, retriever.k_values)



# timetaken, recall, *the_rest =benchmark_retriever(retriever, corpus, queries, qrels)
# pickle.dump([timetaken, recall, *the_rest], open("beir_results_dot_score.pkl", "wb"))
# print(timetaken, recall)


from ptmodels import IVFRetrievalModel

nprobe = 50
db_size = corpus_embeddings.shape[0]
c = 5
distance_func = 'dot_score'
modelIVF = IVFRetrievalModel(nlist=int(c*np.sqrt(db_size)), nprobe=nprobe, distance_func=distance_func)
modelIVF.train(corpus_embeddings)


start_time = time.time()
query_ids = list(queries.keys())
corpus_ids = list(corpus.keys())
results_IVF = {}
for query_vector, query_id in zip(query_embeddings, query_ids):
    top_k_indices, top_k_values = modelIVF.query(query_vector, 100)
    results_IVF[query_id] = {corpus_ids[idx]: float(val) for idx, val in zip(top_k_indices, top_k_values)}
end_time = time.time()
timetaken = end_time-start_time

ndcg, _map, recall, precision = retriever.evaluate(qrels, results_IVF, retriever.k_values)




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
