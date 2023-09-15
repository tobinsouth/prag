import logging
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
import pandas as pd
from collections import defaultdict
import faiss
import numpy as np
import random
import crypten
import pickle
import crypten.mpc as mpc
import crypten.communicator as comm
from ptmodels import IVFRetrievalModel, load_preembeddings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class MPCIVFRetrievalModel:
    
    def __init__(self, distance_func: str = 'cos_sim', **kwargs):
        self.distance_func = distance_func
        self.model_name = 'mpcivf_retrieval_model'
        
        self.nlist = kwargs.get('nlist', 100)  # Number of clusters
        self.nprobe = kwargs.get('nprobe', 10)  # Number of clusters to search

    '''
    This function needs to run after training is done.
    It will encrypt the centroids and DB and store them in the model.
    '''
    def encrypt(self):
        self.encrypted_centroids = crypten.cryptensor(self.centroids, ptype=crypten.mpc.arithmetic)
        self.encrypted_database = crypten.cryptensor(self.database, ptype=crypten.mpc.arithmetic)
        self.encrypted_clusters_ids, self.encrypted_clusters_distances = self._encrypt_clusters()

    def _encrypt_clusters(self):
        invlists = self.index.invlists
        clusters_id_list = []
        clusters_distance_list = []

        max_size = 0
        for i in range(self.nlist):
            list_size = invlists.list_size(i)
            max_size = max(max_size, list_size)  # Find the maximum list size for padding
            self.max_cluster_size = max_size

        for i in range(self.nlist):
            list_size = invlists.list_size(i)
            ids_for_cluster = [invlists.get_single_id(i, j) for j in range(list_size)]
            cluster_data = self.database[ids_for_cluster]

            # Padding the list to ensure all tensors have the same shape
            ids_for_cluster.extend([-1] * (max_size - list_size))
            # cluster_data.extend([-1] * (max_size - list_size))
            # cluster_data = np.pad(cluster_data, ((0, (max_size - list_size)), (0, 0)), mode='constant', constant_values=0) # TODO: make sure adding zeroes doesn't mess it up.. prob depends which dist metric we use
            cluster_data = np.pad(cluster_data, ((0, (max_size - list_size)), (0, 0)), mode='constant', constant_values=23423) # TODO: make sure adding zeroes doesn't mess it up.. prob depends which dist metric we use
         
            clusters_id_list.append(torch.Tensor(ids_for_cluster))
            clusters_distance_list.append(torch.Tensor(cluster_data).flatten())
            logger.debug(f"Cluster {i} has {list_size} items")

        # Stacking the list of tensors to form a matrix
        clusters_ids = torch.stack(clusters_id_list)
        clusters_distances = torch.stack(clusters_distance_list)
        encrypted_clusters_ids = crypten.cryptensor(clusters_ids, ptype=crypten.mpc.arithmetic)
        encrypted_clusters_distances = crypten.cryptensor(clusters_distances, ptype=crypten.mpc.arithmetic)

        return encrypted_clusters_ids, encrypted_clusters_distances
    
    # We assume a trusted owner performs this
    # In practice, we should have fed learning k-means or MPC-k-means for this    
    def train(self, database: torch.Tensor):
        self.database = database.cpu().numpy()
        
        dimension = self.database.shape[1]
        
        if (self.distance_func in ['cos_sim', 'dot_prod']):
            # Since we are using cosine similarity, we normalize the data
            faiss.normalize_L2(self.database)
            # Define the index
            quantizer = faiss.IndexFlatIP(dimension)  # This is for clustering
            self.index = faiss.IndexIVFFlat(quantizer, dimension, self.nlist, faiss.METRIC_INNER_PRODUCT)
        else:
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, self.nlist, faiss.METRIC_L2)
        
        self.index.nprobe = self.nprobe

        # Train the index
        self.index.train(self.database)
        self.index.add(self.database)
        
        # Storing centroids for direct querying
        self.centroids = torch.Tensor(faiss.rev_swig_ptr(quantizer.get_xb(), self.nlist * dimension).reshape(self.nlist, dimension))
    
    def _compute_distance(self, query: torch.Tensor, database: torch.Tensor):
        if (self.distance_func in ['cos_sim', 'dot_prod']):
            # Compute distances to centroids
            distances = np.dot(query, database.T)
        else:
            # Compute distances to centroids (L2)
            distances = np.linalg.norm(query - database, axis=1)
        return distances
    
    def encrypted_topk(self, encrypted_tensor, k, one_hot=0, second_dim=0):
        # Decrypt the tensor
        plaintext_tensor = encrypted_tensor.get_plain_text()

        # Compute the top k indices
        _, topk_indices = (-plaintext_tensor).topk(k)
        topk_indices = topk_indices.flatten()

        # If one_hot is not zero, transform each index into a one-hot vector
        if one_hot != 0:
            one_hot_tensors = []
            for idx in topk_indices:
                if second_dim != 0:
                    one_hot_vector = torch.zeros(one_hot, second_dim)
                else:
                    one_hot_vector = torch.zeros(one_hot)
                one_hot_vector[idx] = 1
                one_hot_tensors.append(one_hot_vector.flatten())
            
            # Stack the one-hot vectors
            topk_tensor = torch.stack(one_hot_tensors, dim=0)
        else:
            topk_tensor = torch.tensor(topk_indices)

        # Re-encrypt the indices/vector and return
        enc_topk = crypten.cryptensor(topk_tensor, ptype=crypten.mpc.arithmetic)

        return enc_topk

    def query(self, query: torch.Tensor, top_k: int=10, database=None):
        if database is None:
            # Assume database is encrypted as well
            encrypted_database = self.encrypted_database
        else:
            if self.distance_func == 'cos_sim':
                # Normalize vectors (for cosine similarity)
                faiss.normalize_L2(database)
            encrypted_database = crypten.cryptensor(database.cpu(), ptype=crypten.mpc.arithmetic)

        query = query.reshape(1, -1).cpu().numpy()
        if self.distance_func == 'cos_sim':
            # Normalize vectors (for cosine similarity)
            faiss.normalize_L2(query)
        
        encrypted_query = crypten.cryptensor(query, ptype=crypten.mpc.arithmetic)
        
        # Compute encrypted distances to centroids
        if self.distance_func in ['cos_sim', 'dot_prod']:
            encrypted_distances_to_centroids = encrypted_query.matmul(self.encrypted_centroids.t())
        else:
            # L2 distance can be trickier with encrypted data
            diff = encrypted_query - self.encrypted_centroids
            encrypted_distances_to_centroids = (diff * diff).sum(1).sqrt()
        
        encrypted_top_centroid_indices = self.encrypted_topk(-encrypted_distances_to_centroids, self.nprobe, one_hot=self.nlist)

        ## TODO: remove temp
        top_centroid_indices = encrypted_top_centroid_indices.get_plain_text().numpy()
        self._top_centroid_indices = top_centroid_indices
        ##

        # Obliviously reduce the dataset size to (n_probes, cluster_size).
        # This has O(N) comm between the servers (can be improved ..), but after which the search space is greatly reduced
        # and we can run comparisons

        # TODO: do this over unencrypted if you want honest-maj
        enc_distance_candidates_matrix = encrypted_top_centroid_indices.matmul(self.encrypted_clusters_distances).reshape(self.nprobe*self.max_cluster_size, self.database.shape[1])
        indices_candidates_matrix = encrypted_top_centroid_indices.matmul(self.encrypted_clusters_ids).flatten().get_plain_text() # TODO: this is temp, can do this more efficiently. For now
        
        # Compute encrypted distances to candidates
        if self.distance_func in ['cos_sim', 'dot_prod']:
            encrypted_distances_to_candidates = encrypted_query.matmul(enc_distance_candidates_matrix.t())
        else:
            # L2 distance can be trickier with encrypted data
            diff = encrypted_query - enc_distance_candidates_matrix
            encrypted_distances_to_candidates = (diff * diff).sum(1).sqrt()
            
        enc_top_k_indices = self.encrypted_topk(-encrypted_distances_to_candidates, top_k)
        
        top_k_indices_local = enc_top_k_indices.get_plain_text()

        # TODO: map back to global indices obliviously
        top_k_indices = indices_candidates_matrix[top_k_indices_local.int()]

        # TODO: do I need the distances even?
        return top_k_indices, None
    
    def query_with_faiss(self, query: torch.Tensor, top_k: int=10, database=None):
        database = self.database if database is None else database.cpu().numpy()
        query_np = query.cpu().numpy().reshape(1, -1)

        if self.distance_func == 'cos_sim':
            # Normalize vectors (for cosine similarity)
            faiss.normalize_L2(query_np)
            faiss.normalize_L2(database)

        # Using the stock FAISS search function to retrieve the closest neighbors
        distances, indices = self.index.search(query_np, top_k)
        return indices[0], distances[0]
    
    def sanity_check(self, query: torch.Tensor, top_k: int = 10):
        # Retrieve results using the original query function
        query_indices, query_values = self.query(query, top_k)

        # Retrieve results using the FAISS query function
        faiss_indices, faiss_values = self.query_with_faiss(query, top_k)

        # Print results for visual verification
        print("Original query function results:")
        print("Indices:", query_indices)
        print("Values:", query_values)

        print("\nFAISS query function results:")
        print("Indices:", faiss_indices)
        print("Values:", faiss_values)

        # Calculate recall/precision between the two methods
        common_indices = set(query_indices).intersection(set(faiss_indices))
        recall = len(common_indices) / top_k
        precision = len(common_indices) / len(query_indices)

        print("\nRecall:", recall)
        print("Precision:", precision)
        
        return recall, precision

@mpc.run_multiprocess(world_size=2)
# Test to ensure MPC == Plaintext. Assumes both models were pre-trained and use the same seed for reproducibility
def mpc_query_vs_plaintext(query_tensor: torch.Tensor, model: MPCIVFRetrievalModel, modelpt: IVFRetrievalModel) -> bytes:
    model.encrypt()
    top_k_indices, top_k_values = model.query(query_tensor, top_k=10)
    top_k_indices, top_k_values = modelpt.query(query_tensor, top_k=10)

    # Check centroids are the same
    centroids1 = model._top_centroid_indices
    centroids2 = modelpt._top_centroid_indices
    print(f"Centroids are equal? --> {np.array_equal(centroids1.sort(), centroids2.sort())}")


@mpc.run_multiprocess(world_size=2)
def mpc_approx_query(query_tensor: torch.Tensor, model: MPCIVFRetrievalModel) -> bytes:
    model.encrypt()
    top_k_indices, top_k_values = model.query(query_tensor, top_k=10)

    # # secret-share A, B
    # A_enc = crypten.cryptensor(A, ptype=crypten.mpc.arithmetic)
    # B_enc = crypten.cryptensor(B, ptype=crypten.mpc.arithmetic)
    # logger.debug(f"A={A_enc.shape}")
    # logger.debug(f"B={B_enc.shape}")
    # # Compute the dot product of A and B
    # dot_product = A_enc.matmul(B_enc)

    # # Compute the magnitudes of A and B
    # mag_A_enc = (A_enc * A_enc).sum(dim=1, keepdim=True).sqrt()
    # mag_B_enc = (B_enc * B_enc).sum(dim=0, keepdim=True).sqrt()
    # logger.debug(f"dot={dot_product.shape}")
    # logger.debug(f"A_magrec={mag_A_enc.shape}")
    # logger.debug(f"B_magrec={mag_B_enc.shape}")

    # # Compute the cosine similarity
    # cosine_sim = (dot_product / (mag_A_enc * mag_B_enc)).get_plain_text()
    
    # cosine_sim_binary = pickle.dumps(cosine_sim)
    # return cosine_sim_binary

def test_pt_vs_mpc():
    query_tensor, database_tensor = load_preembeddings('datasets/corpus_embeddings_large.pt', 'datasets/query_embeddings_large.pt')
    query_tensor = query_tensor[0].flatten()
    database_tensor = database_tensor[0:1000,:]
    # database_tensor = database_tensor[0:10000,:]
    # database_tensor = database_tensor[0:100000,:]
    
    modelpt = IVFRetrievalModel(nlist=50, nprobe=10)
    modelpt.train(database_tensor)
    # top_k_indices, top_k_values = modelpt.query(query_tensor, top_k=100)

    model = MPCIVFRetrievalModel(nlist=50, nprobe=10)
    model.train(database_tensor)

    # Move to MPC-land
    mpc_query_vs_plaintext(query_tensor, model, modelpt)
    
if __name__ == "__main__":
    test_pt_vs_mpc()