from LocalDenseRetrievalExactSearch import DenseRetrievalExactSearch
from  beir.util import cos_sim, dot_score
import logging
from mpc_functions_stable import *
import torch, pickle
import numpy as np
logger = logging.getLogger(__name__)


class MPCDenseRetrievalExactSearch(DenseRetrievalExactSearch):
    """
        This is a class that mimics the DenseRetrievalExactSearch class, but it uses the MPC model to calculate the similarity measures. Using this class will allow us to (1) interoperate with the BEIR EvaluateRetrieval API and (2) directly compare with traditional dense retrieval methods.
    """
    def __init__(self, model, batch_size: int = 128, corpus_chunk_size: int = 50000, **kwargs):
        super().__init__(model, batch_size, corpus_chunk_size, **kwargs) # This is just constructing the parent
        self.score_functions.update({'mpc_dot_vanilla_topk':self.mpc_dot_vanilla_topk, 'mpc_cos_vanilla_topk':self.mpc_cos_vanilla_topk, 'mpc_cos2_vanilla_topk':self.mpc_cos2_vanilla_topk, 'mpc_eucld_vanilla_topk':self.mpc_eucld_vanilla_topk, 'mpc_dot_topk':self.mpc_dot_topk, 'mpc_cos_topk':self.mpc_cos_topk, 'mpc_cos2_topk':self.mpc_cos2_topk, 'mpc_eucld_topk':self.mpc_eucld_topk, 'ivf_topk':self.ivf_topk, 'ivf_topk_mpc':self.ivf_topk_mpc})
        self.IVF_model = lambda x: print("You need to train the model first.")

    def mpc_dot_vanilla_topk(self, query_embedding, sub_corpus_embeddings, k):
        dot_score_res = handle_binary(dot_score_mpc, mpc=True)(query_embedding, sub_corpus_embeddings.t()).unsqueeze(0)
        return self.topk_vanilla(dot_score_res, k)
    
    def mpc_cos_vanilla_topk(self, query_embedding, sub_corpus_embeddings, k):
        cos_score_res = handle_binary(cosine_similarity_mpc_opt, mpc=True)(query_embedding, sub_corpus_embeddings.t()).unsqueeze(0)
        return self.topk_vanilla(cos_score_res, k)
    
    def mpc_cos2_vanilla_topk(self, query_embedding, sub_corpus_embeddings, k):
        cos_score_res = handle_binary(cosine_similarity_mpc_opt2, mpc=True)(query_embedding, sub_corpus_embeddings.t()).unsqueeze(0)
        return self.topk_vanilla(cos_score_res, k)
    
    def mpc_eucld_vanilla_topk(self, query_embedding, sub_corpus_embeddings, k):
        eucld_score_res = handle_binary(euclidean_mpc, mpc=True)(query_embedding, sub_corpus_embeddings.t()).sqrt().unsqueeze(0)
        return self.topk_vanilla(eucld_score_res, k)

    def mpc_dot_topk(self, query_embedding, sub_corpus_embeddings, k):
        distance_and_top_k_func = mpc_distance_top_k_with_distance_func(dot_score_mpc)
        crypten_binary = distance_and_top_k_func(query_embedding, sub_corpus_embeddings.t(), k)
        top_k_idx, top_k_values = pickle.loads(crypten_binary[0])
        return top_k_values.cpu().tolist(), top_k_idx.cpu().tolist()
    
    def mpc_cos_topk(self, query_embedding, sub_corpus_embeddings, k):
        distance_and_top_k_func = mpc_distance_top_k_with_distance_func(cosine_similarity_mpc_opt)
        crypten_binary = distance_and_top_k_func(query_embedding, sub_corpus_embeddings.t(), k)
        top_k_idx, top_k_values = pickle.loads(crypten_binary[0])
        return top_k_values.cpu().tolist(), top_k_idx.cpu().tolist()
    
    def mpc_cos2_topk(self, query_embedding, sub_corpus_embeddings, k):
        distance_and_top_k_func = mpc_distance_top_k_with_distance_func(cosine_similarity_mpc_opt2)
        crypten_binary = distance_and_top_k_func(query_embedding, sub_corpus_embeddings.t(), k)
        top_k_idx, top_k_values = pickle.loads(crypten_binary[0])
        return top_k_values.cpu().tolist(), top_k_idx.cpu().tolist()
    
    def mpc_eucld_topk(self, query_embedding, sub_corpus_embeddings, k):
        distance_and_top_k_func = mpc_distance_top_k_with_distance_func(euclidean_mpc)
        crypten_binary = distance_and_top_k_func(query_embedding, sub_corpus_embeddings.t(), k)
        top_k_idx, top_k_values = pickle.loads(crypten_binary[0])
        return top_k_values.cpu().tolist(), top_k_idx.cpu().tolist()
    
    def ivf_topk(self, query_embedding, sub_corpus_embeddings, k):
        top_k_indices, top_k_values = self.IVF_model.query(query_embedding, k)
        return top_k_values, top_k_indices        
    
    def ivf_topk_mpc(self, query_embedding, sub_corpus_embeddings, k):
        top_k_indices, top_k_values = self.IVF_model.query(query_embedding, k)
        return top_k_values, top_k_indices
    

    def train_model(self, corpus_embeddings, model_type):
        print("Training model on subcorpus of size %d" % len(corpus_embeddings))
        if model_type == 'ivf_topk':
            from ptmodels import IVFRetrievalModel
            nlist = int(5*np.sqrt(len(corpus_embeddings)))
            ret_model = IVFRetrievalModel(nlist=nlist, nprobe=100, distance_func='cos_sim')
            ret_model.train(corpus_embeddings)
            self.IVF_model = ret_model
            return 
        
        elif model_type == 'ivf_topk_mpc':
            from mpcmodels import MPCIVFRetrievalModel
            nlist = int(5*np.sqrt(len(corpus_embeddings)))
            ret_model = MPCIVFRetrievalModel(nlist=nlist, nprobe=100, distance_func='cos_sim')
            ret_model.train(corpus_embeddings)
            self.IVF_model = ret_model
            self.IVF_model.ret_model.encrypt()
            return
        
        print('Invalid model type.')

        