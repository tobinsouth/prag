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
        self.debug = kwargs.get('debug', True)

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
            cluster_data = np.pad(cluster_data, ((0, (max_size - list_size)), (0, 0)), mode='constant', constant_values=0) # TODO: make sure adding zeroes doesn't mess it up.. prob depends which dist metric we use
         
            # cluster_data = np.pad(cluster_data, ((0, (max_size - list_size)), (0, 0)), mode='constant', constant_values=-23423) # TODO: make sure adding zeroes doesn't mess it up.. prob depends which dist metric we use
            # Compute how many rows you need to pad
            
            
            # padding_rows = max_size - cluster_data.shape[0]

            # # If padding_rows is less than or equal to 0, no padding is needed
            # if padding_rows > 0:
            #     # Create the pad array
            #     pad_array = np.repeat(cluster_data[0, :][np.newaxis, :], padding_rows, axis=0)
            #     # Vertical stack the cluster_data and pad_array
            #     cluster_data = np.vstack((cluster_data, pad_array))            
         
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
        self.database = database
        db_cpu = database.cpu().numpy() # NOTE: .cpu() messes crypten's encryption
        
        dimension = self.database.shape[1]
        
        if (self.distance_func in ['cos_sim', 'dot_prod']):
            # Since we are using cosine similarity, we normalize the data
            self.database = F.normalize(self.database, p=2, dim=1)
            # Define the index
            quantizer = faiss.IndexFlatIP(dimension)  # This is for clustering
            self.index = faiss.IndexIVFFlat(quantizer, dimension, self.nlist, faiss.METRIC_INNER_PRODUCT)
        else:
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, self.nlist, faiss.METRIC_L2)
        
        self.index.nprobe = self.nprobe

        # Train the index
        self.index.train(db_cpu)
        self.index.add(db_cpu)
        
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
        print(f"Distances are = {plaintext_tensor}")
        
        # _, topk_indices = plaintext_tensor.topk(k) # TODO: this returns the biggest ones. Figure out if this what I want - sanity says no, logic says yes
        topk_indices = plaintext_tensor.sort().indices[0][:k] # TODO: this returns the smallest
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
                # faiss.normalize_L2(database) # TODO: don't normalize for now
                self.database = F.normalize(database, p=2, dim=1)
            encrypted_database = crypten.cryptensor(database, ptype=crypten.mpc.arithmetic)
        
        ## TODO: remove debug
        print(f"db reconsutrcted = {encrypted_database.get_plain_text()}")

        # TODO: debug, disable for now..
        query = query.reshape(1, -1)
        # query = query.reshape(1, -1).cpu().numpy()
        if self.distance_func == 'cos_sim':
            # Normalize vectors (for cosine similarity)
            # faiss.normalize_L2(query) # TODO: debug, disable for now..
            query = F.normalize(query, p=2, dim=1)
        
        encrypted_query = crypten.cryptensor(query, ptype=crypten.mpc.arithmetic)
        
        # Compute encrypted distances to centroids
        if self.distance_func in ['cos_sim', 'dot_prod']:
            encrypted_distances_to_centroids = encrypted_query.matmul(self.encrypted_centroids.t())

            ## TODO: remove DEBUG
            ptdist = self.encrypted_centroids.get_plain_text()
            print(f"distances in the clear are = {torch.mm(query.reshape(1, -1),torch.Tensor(ptdist).t())}")
            print(f"distances after encryption = {encrypted_distances_to_centroids.get_plain_text()}")
            print(f"query={query}, ptcentroids={ptdist}")
            print(f"reconstructed query={encrypted_query.get_plain_text()}, ptdist={ptdist}")
            print("okay")
            
        else:
            # L2 distance can be trickier with encrypted data
            diff = encrypted_query - self.encrypted_centroids
            encrypted_distances_to_centroids = (diff * diff).sum(1).sqrt()
        
        encrypted_top_centroid_indices = self.encrypted_topk(-encrypted_distances_to_centroids, self.nprobe, one_hot=self.nlist)

        if (self.debug):
            top_centroid_indices = encrypted_top_centroid_indices.get_plain_text().numpy()
            self._top_centroid_indices = top_centroid_indices

        # Obliviously reduce the dataset size to (n_probes, cluster_size).
        # This has O(N) comm between the servers (can be improved ..), but after which the search space is greatly reduced
        # and we can run comparisons

        # TODO: do this over unencrypted if you want honest-maj
        enc_distance_candidates_matrix = encrypted_top_centroid_indices.matmul(self.encrypted_clusters_distances).reshape(self.nprobe*self.max_cluster_size, self.database.shape[1])
        indices_candidates_matrix = encrypted_top_centroid_indices.matmul(self.encrypted_clusters_ids).flatten().get_plain_text() # TODO: this is temp, can do this more efficiently. For now

        if (self.debug):
            self._last_candidates = indices_candidates_matrix
            t1 = encrypted_top_centroid_indices.get_plain_text()
            t2 = self.encrypted_clusters_ids.get_plain_text()
            t3 = torch.mm(t1, t2)
            print(f"result shape - {t3.shape}")


            indices = self.encrypted_topk(-encrypted_distances_to_centroids, self.nprobe).get_plain_text()
            idx_list_count = 0
            for idx in indices:
                ii = int(idx)
                idx_list = [self.index.invlists.get_single_id(ii, j) for j in range(self.index.invlists.list_size( ii ))]
                idx_list_count += len(idx_list)
                for jj in range(t3.shape[0]):
                    if t3[jj][0] == idx_list[0]:
                        t3jj = t3[jj][t3[jj] != -1].int()
                        print("-=-=-")
                        print(f"Cluster {idx}={torch.all(t3jj == torch.Tensor(idx_list))}")
                        # print(f"Cluster {idx}={t3[jj].int()}, and originally={idx_list}")
            print(f"idx_list_count={idx_list_count}")

            # print(f"Cluster 26={t2[26].int()}, and originally={}")
            print("Ok done")
        
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
    top_k_indicespt, top_k_values = modelpt.query(query_tensor, top_k=10)

    # Check centroids are the same
    centroids1 = model._top_centroid_indices
    cent1 = []
    for row in range(centroids1.shape[0]):
        cent1.append(centroids1[row].argmax())
    centroids1 = np.array([cent1])
    centroids2 = modelpt._top_centroid_indices
    # print(f"Centroids are equal? --> {np.array_equal(centroids1.sort(), centroids2.sort())}")
    print(f"centroids1 are:{np.sort(centroids1)}")
    print(f"centroids2 are:{np.sort(centroids2)}")

    last_candidates1 = model._last_candidates
    last_candidates2 = modelpt._last_candidates

    print(f"last_candidates1={last_candidates1[last_candidates1 != -1].sort()}, last_candidates2={sorted(last_candidates2)}")
    print(f"last_candidates1.shape={last_candidates1[last_candidates1 != -1].shape}, last_candidates2={len(last_candidates2)}")
    

    print(f"top k from encrypted model: {np.sort(top_k_indices)}")
    print(f"top k from plaintext model: {np.sort(top_k_indicespt)}")
    print("Done")
    


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
    
from crypten.config import cfg
def set_precision(bits):
    cfg.encoder.precision_bits = bits

if __name__ == "__main__":
    #initialize crypten
    crypten.init()
    #Disables OpenMP threads -- needed by @mpc.run_multiprocess which uses fork
    torch.set_num_threads(1)
    # set_precision(30)
    test_pt_vs_mpc()