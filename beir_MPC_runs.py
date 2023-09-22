from beir.retrieval.evaluation import EvaluateRetrieval
from MPCDenseRetrievalExactSearch import MPCDenseRetrievalExactSearch
import pickle, numpy as np, time, torch
from tqdm import tqdm
import pandas as pd

dataset_name = "fiqa_full_mdb3"

corpus, qrels, queries = pickle.load(open(f"datasets/corpus_{dataset_name}.pkl", "rb"))

# Now we benchmark MPC dense retrieval
print("Building BEIR model and loading pre-embeddings.")
beir_model = MPCDenseRetrievalExactSearch(None, corpus_chunk_size=512*9999)

# Load in premade embeddings
beir_model.load_preembeddings(f"datasets/corpus_embeddings_{dataset_name}.pt", f"datasets/query_embeddings_{dataset_name}.pt")
print("Loaded embeddings")

retriever = EvaluateRetrieval(beir_model, score_function="dot_score",  k_values=[1,3,5,10])
results_dot = retriever.retrieve(corpus, queries)
ndcg, _map, recall, precision = retriever.evaluate(qrels, results_dot, retriever.k_values)
print("dot_score", recall)

retriever = EvaluateRetrieval(beir_model, score_function="cos_sim",  k_values=[1,3,5,10])
results_cos = retriever.retrieve(corpus, queries)
ndcg, _map, recall, precision = retriever.evaluate(qrels, results_cos, retriever.k_values)
print("cos_sim", recall)

retriever = EvaluateRetrieval(beir_model, score_function="mpc_cos_topk",  k_values=[1,3,5,10])
results_mpc_cos_topk = retriever.retrieve(corpus, queries)
ndcg, _map, recall, precision = retriever.evaluate(qrels, results_mpc_cos_topk, retriever.k_values)
print("cos_sim", recall)


# beir_model.train_model(corpus, "ivf_topk")
from ptmodels import IVFRetrievalModel
nlist = int(5*np.sqrt(len(corpus)))
ret_model = IVFRetrievalModel(nlist=nlist, nprobe=100, distance_func='cos_sim')
ret_model.train(beir_model.corpus_embeddings)
beir_model.IVF_model = ret_model
retriever = EvaluateRetrieval(beir_model, score_function="ivf_topk",  k_values=[1,3,5,10])
results = retriever.retrieve(corpus, queries)
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
print("ivf_topk", recall)













# Timing

from ptmodels import IVFRetrievalModel
nlist = int(5*np.sqrt(len(beir_model.corpus_embeddings)))
ret_model = IVFRetrievalModel(nlist=nlist, nprobe=1, distance_func='cos_sim')
ret_model.train(beir_model.corpus_embeddings)
top_k_indices, top_k_values = ret_model.query(beir_model.query_embeddings[0], 500)

# Ground truth

recall_fun = lambda top_k_indices, true_top_k_indices: len(set(top_k_indices).intersection(set(true_top_k_indices)))/len(true_top_k_indices)

nprobe_to_recall = []
for qid in tqdm(range(len(beir_model.query_embeddings))): #
    cos_scores = torch.nn.functional.cosine_similarity(beir_model.query_embeddings[qid], beir_model.corpus_embeddings)
    top_k_values_true, top_k_indices_true = torch.topk(cos_scores, 100, largest=True, sorted=True)
    for nprobe in range(1,200):
        ret_model.nprobe = nprobe
        top_k_indices, top_k_values = ret_model.query(beir_model.query_embeddings[0], 100)
        value_recall = recall_fun(top_k_indices, top_k_indices_true.tolist())
        nprobe_to_recall.append([nprobe, value_recall, qid])
    nprobe_to_recall_df = pd.DataFrame(nprobe_to_recall, columns=["nprobe", "recall", "qid"])
    nprobe_to_recall_df.to_csv(f"nprobe_to_recall_{dataset_name}.csv")
# nprobe_to_recall_df.groupby('nprobe')['recall'].mean().plot()



# from mpc_functions_stable import handle_binary
from mpcmodels import MPCIVFRetrievalModel
nlist = int(10*np.sqrt(len(beir_model.corpus_embeddings)))
ret_model = MPCIVFRetrievalModel(nlist=nlist, nprobe=50, distance_func='dot_prod')
ret_model.train(beir_model.corpus_embeddings)
ret_model.encrypt()
top_k_indices_mpc, top_k_values_mpc = ret_model.query(beir_model.query_embeddings[0], 100)


beir_model.mpc_dot_topk(beir_model.query_embeddings[0], beir_model.corpus_embeddings, 100)







# Get the key result, for a fix dataset and nlist size, how many nprobe do we need to get 90% accuracy?


