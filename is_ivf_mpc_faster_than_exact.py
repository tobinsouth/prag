import pickle, time, torch, numpy as np, pandas as pd
from mpc_functions_stable import mpc_distance_top_k_with_distance_func, cosine_similarity_mpc_opt
from mpcmodels import MPCIVFRetrievalModel
from tqdm import tqdm

import crypten
crypten.init()

datasets/corpus_embeddings_quora_full_mdb3.pt

corpus_embeddings = torch.load("datasets/corpus_embeddings_quora_full_mdb3.pt").to('cpu')
query_embeddings = torch.load("datasets/query_embeddings_quora_full_mdb3.pt").to('cpu')


query_embedding = query_embeddings[0]
k =100


time_results = []
# for n in tqdm(range(80000,150000,5000)): # range(5000, 10000, 1000)
n = 10**10
    sub_corpus_embeddings = corpus_embeddings[:n]

    start = time.time()
    distance_and_top_k_func = mpc_distance_top_k_with_distance_func(cosine_similarity_mpc_opt)
    crypten_binary = distance_and_top_k_func(query_embedding, sub_corpus_embeddings.t(), k)
    top_k_idx, top_k_values = pickle.loads(crypten_binary[0])
    top_k_values.cpu().tolist(), top_k_idx.cpu().tolist()
    end = time.time()
    time_results.append([n, end-start, "MPC Exact Search"])
    print("MPC Exact Search", n, end-start)

    nlist = int(10*np.sqrt(len(sub_corpus_embeddings)))
    ret_model = MPCIVFRetrievalModel(nlist=nlist, nprobe=20, distance_func='dot_prod')
    ret_model.train(sub_corpus_embeddings)
    ret_model.encrypt()

    start = time.time()
    top_k_indices, top_k_values = ret_model.query(query_embedding, k)
    end = time.time()
    time_results.append([n, end-start, "MPC IVF Search"])
    print("MPC IVF Search", n, end-start)

    time_results_df = pd.DataFrame(time_results, columns=["n", "time", "type"])

    time_results_df.to_csv("results/is_ivf_mpc_faster_than_exact.csv")

import matplotlib.pyplot as plt, seaborn as sns
sns.scatterplot(data=time_results_df, x="n", y="time", hue="type")
plt.ylabel("Time taken for MPC Exact Search vs MPC IVF Search")
plt.xlabel("Database size")

    

time_results_df_k10 = time_results_df.copy()


import matplotlib.pyplot as plt, seaborn as sns
sns.scatterplot(data=time_results_df_k10, x="n", y="time", hue="type")
plt.ylabel("Time taken for MPC Exact Search vs MPC IVF Search")
plt.xlabel("Database size")




