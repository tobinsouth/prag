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
    
    def query(self, query: torch.Tensor, top_k: int=10, database=None):
        if database is None:
            # Assume database is encrypted as well
            encrypted_database = self.encrypted_database
        else:
            if self.distance_func == 'cos_sim':
                # Normalize vectors (for cosine similarity)
                faiss.normalize_L2(database)
            encrypted_database = crypten.cryptensor(database.cpu(), ptype=crypten.mpc.arithmetic)

        if self.distance_func == 'cos_sim':
            # Normalize vectors (for cosine similarity)
            faiss.normalize_L2(query)
        
        encrypted_query = crypten.cryptensor(query.cpu(), ptype=crypten.mpc.arithmetic).reshape(1, -1)
        
        # Compute encrypted distances to centroids
        if self.distance_func in ['cos_sim', 'dot_prod']:
            encrypted_distances_to_centroids = encrypted_query.mm(self.encrypted_centroids.t())
        else:
            # L2 distance can be trickier with encrypted data
            diff = encrypted_query - self.encrypted_centroids
            encrypted_distances_to_centroids = (diff * diff).sum(1).sqrt()
        
        


        
        # TODO: continue here..
        # Get closest centroids
        if (self.distance_func in ['cos_sim', 'dot_prod']):
            top_centroid_indices = np.argpartition(-distances_to_centroids, self.nprobe)[:, :self.nprobe]
        else:
            top_centroid_indices = np.argpartition(distances_to_centroids.reshape(1, -1), self.nprobe)[:, :self.nprobe]

        top_k_indices = []
        top_k_values = []
        invlists = self.index.invlists
        for q_idx, centroid_indices in enumerate(top_centroid_indices):
            all_indices = []
            all_distances = []

            for c_idx in centroid_indices:
                c_idx = int(c_idx)
                list_size = invlists.list_size(c_idx)
                # Retrieve the IDs of the points assigned to the cluster
                # ids_for_cluster = invlists.get_ids(c_idx)
                ids_for_cluster = [invlists.get_single_id(c_idx, i) for i in range(list_size)]
                logger.debug(f"Cluster {c_idx} has {list_size} items")
                
                # Calculate the distances from query to all points in this cluster
                cluster_data = database[ids_for_cluster]
                distances = self._compute_distance(query_np[q_idx], cluster_data)

                # Store results for this cluster
                all_indices.extend(ids_for_cluster)
                all_distances.extend(distances)

            # Now, select top_k results across all clusters searched
            if (self.distance_func in ['cos_sim', 'dot_prod']):
                selected_indices = np.argpartition(-np.array(all_distances), top_k)[:top_k]
            else:
                selected_indices = np.argpartition(np.array(all_distances), top_k)[:top_k]

            top_k_indices_cluster = [all_indices[i] for i in selected_indices]
            top_k_distances_cluster = [all_distances[i] for i in selected_indices]

            top_k_indices.extend(top_k_indices_cluster)
            top_k_values.extend(top_k_distances_cluster)

        return top_k_indices, top_k_values
    
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