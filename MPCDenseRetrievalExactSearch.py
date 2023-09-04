from LocalDenseRetrievalExactSearch import DenseRetrievalExactSearch
from  beir.util import cos_sim, dot_score
import logging
from mpc_functions import cosine_similarity_mpc_naive, cosine_similarity_mpc_opt, preprocess_cosine_similarity_mpc_opt, dot_score_mpc
import torch, pickle

logger = logging.getLogger(__name__)


class MPCDenseRetrievalExactSearch(DenseRetrievalExactSearch):
    """
        This is a class that mimics the DenseRetrievalExactSearch class, but it uses the MPC model to calculate the similarity measures. Using this class will allow us to (1) interoperate with the BEIR EvaluateRetrieval API and (2) directly compare with traditional dense retrieval methods.
    """
    def __init__(self, model, batch_size: int = 128, corpus_chunk_size: int = 50000, **kwargs):
        super().__init__(model, batch_size, corpus_chunk_size, **kwargs) # This is just constructing the parent
        self.score_functions = {'cos_sim': cos_sim, 'dot': dot_score, 'mpc_opt': self.mpc_opt, 'mpc_naive': self.mpc_naive, 'mpc_dot': self.mpc_dot}

    def mpc_naive(self, query_embeddings, sub_corpus_embeddings):
        # NOTE: This is a naive implementation of MPC. It does not use the precomputed magnitudes of the vectors, and instead computes them on the fly. 

        # query_vector = torch.tensor(query_vector, dtype=torch.float32).t()
        # database_vectors = torch.tensor(database_vectors, dtype=torch.float32)

        cosine_sim_binary = cosine_similarity_mpc_naive(query_embeddings, sub_corpus_embeddings.t())
        cosine_sim = pickle.loads(cosine_sim_binary[0]) # remove [0] for single threaded

        # # Include for debugging
        # default_cos_sim = cos_sim(query_embeddings, sub_corpus_embeddings)
        # print("All close: ", torch.allclose(default_cos_sim, cosine_sim, atol=0.02))

        return cosine_sim



    def mpc_opt(self, query_embeddings, sub_corpus_embeddings):
        
        query_vector, database_vectors, qv_mag_recip, db_mag_recip = preprocess_cosine_similarity_mpc_opt([query_embeddings, sub_corpus_embeddings.t()])

        cosine_sim_binary = cosine_similarity_mpc_opt(query_vector, database_vectors, qv_mag_recip, db_mag_recip)
        cosine_sim = pickle.loads(cosine_sim_binary[0]) # remove [0] for single threaded

        return cosine_sim
    
    def mpc_dot(self, query_embeddings, sub_corpus_embeddings):
        dot_score_binary = dot_score_mpc(query_embeddings, sub_corpus_embeddings.t())
        dot_score_res = pickle.loads(dot_score_binary[0]) # remove [0] for single threaded
        return dot_score_res