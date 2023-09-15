from MPCDenseRetrievalExactSearch import MPCDenseRetrievalExactSearch
import pickle
import numpy as np, torch


load_string = 'fiqa'
corpus, qrels, queries = pickle.load(open(f"datasets/corpus_{load_string}.pkl", "rb"))
model = MPCDenseRetrievalExactSearch(None, corpus_chunk_size=512*6)
model.load_preembeddings(f"datasets/corpus_embeddings_{load_string}.pt", f"datasets/query_embeddings_{load_string}.pt")

corpus_embeddings = model.corpus_embeddings
query_embeddings = model.query_embeddings




from ptmodels import IVFRetrievalModel



nprobe = 50
db_size = corpus_embeddings.shape[0]
c = 5
distance_func = 'dot_score'
model = IVFRetrievalModel(nlist=int(c*np.sqrt(db_size)), nprobe=nprobe, distance_func=distance_func)
model.train(corpus_embeddings)
(top_k_indices, top_k_values), timetaken = model.query(query_embeddings, 20)
