from mpcmodels import MPCIVFRetrievalModel
from ptmodels import IVFRetrievalModel
from tqdm import tqdm
import torch, numpy as np, time, pandas as pd
import crypten
crypten.init()


corpus_embeddings = torch.load("datasets/corpus_embeddings_quora_full_mdb3.pt").to('cpu')
query_embeddings = torch.load("datasets/query_embeddings_quora_full_mdb3.pt").to('cpu')

recall_fun = lambda a,b: len(set(a).intersection(set(b)))/len(set(a))


n = 10000
top_k = 500
nprobe = 100
sub_corpus_embeddings = corpus_embeddings[:n]
nlist = int(5*np.sqrt(len(sub_corpus_embeddings)))

# IVF
ret_model = IVFRetrievalModel(nlist=nlist, nprobe=nprobe, distance_func='cos_sim')
ret_model.train(sub_corpus_embeddings)

# MPC
ret_model_mpc = MPCIVFRetrievalModel(nlist=nlist, nprobe=nprobe, distance_func='cos_sim')
ret_model_mpc.train(sub_corpus_embeddings)
ret_model_mpc.encrypt()

for qid in tqdm(range(10)):
    # Real
    cos_sim_score = torch.nn.functional.cosine_similarity(query_embeddings[qid], corpus_embeddings)
    top_k_values_true, top_k_indices_true = torch.topk(cos_sim_score, 500, largest=True, sorted=True)
    # IVF
    top_k_indices, top_k_values = ret_model.query(query_embeddings[qid], top_k)
    # MPC
    top_k_indices_mpc, top_k_values_mpc = ret_model_mpc.query(query_embeddings[qid], top_k)
    # Loop over k
    recall_by_k_nprobe = []
    for k in range(1, top_k):
        recall_by_k = recall_fun(top_k_indices_mpc[:k].tolist(), top_k_indices[:k])
        correct_recall = recall_fun(top_k_indices[:k], top_k_indices_true[:k].tolist())
        correct_mpc_recall = recall_fun(top_k_indices_mpc[:k].tolist(), top_k_indices_true[:k].tolist())
        recall_by_k_nprobe.append([k, nprobe, recall_by_k, correct_recall, correct_mpc_recall])
        
import pandas as pd
recall_by_k_nprobe_df = pd.DataFrame(recall_by_k_nprobe, columns=["k", "nprobe", "recall", "correct_recall", "correct_mpc_recall"])
recall_by_k_nprobe_df.to_csv("results/ivf_recall_by_k_nprobe.csv")
recall_by_k_nprobe = pd.read_csv("results/ivf_recall_by_k_nprobe.csv")
