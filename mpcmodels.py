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
import time
from mpc_functions_stable import top_k_mpc_tobin, mpc_distance_and_topk, cosine_similarity_mpc_opt, _top_k_mpc_tobin, _top_k_mpc_tobin_one_hot_padding

logger = logging.getLogger("prag.mpcmodels")
# logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)

# turn logging off
logging.disable(logging.CRITICAL)

class MPCIVFRetrievalModel:
    
    def __init__(self, distance_func: str = 'cos_sim', **kwargs):
        self.distance_func = distance_func
        self.model_name = 'mpcivf_retrieval_model'
        
        self.nlist = kwargs.get('nlist', 100)  # Number of clusters
        self.nprobe = kwargs.get('nprobe', 10)  # Number of clusters to search
        self.debug = kwargs.get('debug', False)
        self.is_honest_majority = kwargs.get('is_honest_majority', True)
        self.plaintext_topk = True

        self.tobin_encrypt = True
        self.use_old_bins = False

    '''
    This function needs to run after training is done.
    It will encrypt the centroids and DB and store them in the model.
    '''
    def encrypt(self):
        self.encrypted_centroids = crypten.cryptensor(self.centroids, ptype=crypten.mpc.arithmetic)
        self.encrypted_database = crypten.cryptensor(self.database, ptype=crypten.mpc.arithmetic)
        if self.tobin_encrypt:
            if self.debug: print("Using leaky approach")
            self._encrypt_clusters_tobin()
        else:
            if self.use_old_bins:
                if self.debug: print("Using old bins")
                self.encrypted_clusters_ids, self.encrypted_clusters_distances = self._encrypt_clusters_old()
            else:
                self.encrypted_clusters_ids, self.encrypted_clusters_distances = self._encrypt_clusters()

    def _encrypt_clusters(self):
        invlists = self.index.invlists
        clusters_id_list = []
        clusters_distance_list = []

        iqr_max = np.sort([invlists.list_size(i) for i in range(self.nlist)])[int(self.nlist*0.75)]
        self.max_cluster_size = iqr_max

        for i in range(self.nlist):
            list_size = invlists.list_size(i)
            ids_for_cluster = [invlists.get_single_id(i, j) for j in range(list_size)]
            cluster_data = self.database[ids_for_cluster]

            # Padding the list to ensure all tensors have the same shape
            if len(ids_for_cluster) < iqr_max:
                ids_for_cluster.extend([-1] * (iqr_max - list_size))
                if (self.distance_func in ['cos_sim', 'dot_prod']):
                    cluster_data = np.pad(cluster_data, ((0, (iqr_max - list_size)), (0, 0)), mode='constant', constant_values=0)
                else:
                    # TODO: Check it works for euc as well
                    cluster_data = np.pad(cluster_data, ((0, (iqr_max - list_size)), (0, 0)), mode='constant', constant_values=242343)

            else:
                cluster_data = cluster_data[:iqr_max]
                ids_for_cluster = ids_for_cluster[:iqr_max]
         
            clusters_id_list.append(torch.Tensor(ids_for_cluster))
            clusters_distance_list.append(torch.Tensor(cluster_data).flatten())
            if self.debug: print(f"Cluster {i} has {len(ids_for_cluster)} items")

        # Stacking the list of tensors to form a matrix
        clusters_ids = torch.stack(clusters_id_list)
        clusters_distances = torch.stack(clusters_distance_list)

        if not self.is_honest_majority:
            encrypted_clusters_ids = crypten.cryptensor(clusters_ids, ptype=crypten.mpc.arithmetic)
            encrypted_clusters_distances = crypten.cryptensor(clusters_distances, ptype=crypten.mpc.arithmetic)
        else:
            encrypted_clusters_ids = clusters_ids
            encrypted_clusters_distances = clusters_distances

        return encrypted_clusters_ids, encrypted_clusters_distances
    

    def _encrypt_clusters_old(self):
        invlists = self.index.invlists
        clusters_id_list = []
        clusters_distance_list = []

        iqr_max = np.sort([invlists.list_size(i) for i in range(self.nlist)])[-1]
        self.max_cluster_size = iqr_max

        for i in range(self.nlist):
            list_size = invlists.list_size(i)
            ids_for_cluster = [invlists.get_single_id(i, j) for j in range(list_size)]
            cluster_data = self.database[ids_for_cluster]

            # Padding the list to ensure all tensors have the same shape
            ids_for_cluster.extend([-1] * (iqr_max - list_size))
            if (self.distance_func in ['cos_sim', 'dot_prod']):
                cluster_data = np.pad(cluster_data, ((0, (iqr_max - list_size)), (0, 0)), mode='constant', constant_values=0)
            else:
                # TODO: Check it works for euc as well
                cluster_data = np.pad(cluster_data, ((0, (iqr_max - list_size)), (0, 0)), mode='constant', constant_values=242343)

            clusters_id_list.append(torch.Tensor(ids_for_cluster))
            clusters_distance_list.append(torch.Tensor(cluster_data).flatten())
            if self.debug: print(f"Cluster {i} has {len(ids_for_cluster)} items")

        # Stacking the list of tensors to form a matrix
        clusters_ids = torch.stack(clusters_id_list)
        clusters_distances = torch.stack(clusters_distance_list)

        if not self.is_honest_majority:
            encrypted_clusters_ids = crypten.cryptensor(clusters_ids, ptype=crypten.mpc.arithmetic)
            encrypted_clusters_distances = crypten.cryptensor(clusters_distances, ptype=crypten.mpc.arithmetic)
        else:
            encrypted_clusters_ids = clusters_ids
            encrypted_clusters_distances = clusters_distances

        return encrypted_clusters_ids, encrypted_clusters_distances
    
    def _encrypt_clusters_tobin(self):
        invlists = self.index.invlists
        self.cluster_dict = {}
        self.encrypted_clusters_ids = {}

        for i in range(self.nlist):
            list_size = invlists.list_size(i)
            ids_for_cluster = [invlists.get_single_id(i, j) for j in range(list_size)]
            cluster_data = self.database[ids_for_cluster]
            self.cluster_dict[i] =  crypten.cryptensor(cluster_data, ptype=crypten.mpc.arithmetic)
            self.encrypted_clusters_ids[i] = crypten.cryptensor(ids_for_cluster, ptype=crypten.mpc.arithmetic) 

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
    
    # TODO: build a better approx. algorithm
    def encrypted_topk(self, encrypted_tensor, k, one_hot=0, second_dim=0):
        if self.debug: print(encrypted_tensor.shape)
        if not self.plaintext_topk:
            if (one_hot != 0):
                enc_topk = _top_k_mpc_tobin_one_hot_padding(encrypted_tensor, k, one_hot)
            else:
                enc_topk = _top_k_mpc_tobin(encrypted_tensor, k)
        else:
            # Decrypt the tensor
            plaintext_tensor = encrypted_tensor.get_plain_text()

            
            # _, topk_indices = plaintext_tensor.topk(k) # TODO: this returns the biggest ones. Figure out if this what I want - sanity says no, logic says yes
            topk_indices = plaintext_tensor.sort().indices[:k] # TODO: this returns the smallest
            topk_indices = topk_indices.flatten()

            if self.debug: print('one hotting with', one_hot)
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
                topk_tensor = topk_indices # torch.tensor(topk_indices) (Tobin: it should already be a tensor)

            # Re-encrypt the indices/vector and return
            enc_topk = crypten.cryptensor(topk_tensor, ptype=crypten.mpc.arithmetic)
        return enc_topk

    def query(self, query: torch.Tensor, top_k: int=10):
        # query = query.reshape(1, -1) # this was causing errors
        
        if self.distance_func == 'cos_sim':
            # Normalize vectors (for cosine similarity)
            query = F.normalize(query, p=2, dim=0)
        encrypted_query = crypten.cryptensor(query, ptype=crypten.mpc.arithmetic)
        
        # Compute encrypted distances to centroids
        if self.distance_func in ['cos_sim', 'dot_prod']:
            encrypted_distances_to_centroids = encrypted_query.matmul(self.encrypted_centroids.t())

            if self.debug:
                ptdist = self.encrypted_centroids.get_plain_text()
                logger.debug(f"distances in the clear are = {torch.mm(query.reshape(1, -1),torch.Tensor(ptdist).t())}")
                logger.debug(f"distances after encryption = {encrypted_distances_to_centroids.get_plain_text()}")
                logger.debug(f"query={query}, ptcentroids={ptdist}")
                logger.debug(f"reconstructed query={encrypted_query.get_plain_text()}, ptdist={ptdist}")
            
        else:
            # L2 distance can be trickier with encrypted data
            diff = encrypted_query - self.encrypted_centroids
            encrypted_distances_to_centroids = (diff * diff).sum(1).sqrt()
        
        if self.tobin_encrypt:
            if self.debug:  print("Running Tobin's leaky approach")
            encrypted_top_centroid_indices = self.encrypted_topk(-encrypted_distances_to_centroids, self.nprobe, one_hot=self.nlist)
            encrypted_top_centroid_indices = encrypted_top_centroid_indices.sum(dim=0) # Remove order

            top_centroid_indices = encrypted_top_centroid_indices.get_plain_text()
            top_centroid_indices = torch.where(top_centroid_indices)[0]

            best_clusters = [self.cluster_dict[i.item()] for i in top_centroid_indices]
            enc_distance_candidates_matrix = crypten.cat(best_clusters, dim=0)

            cluster_id_map = [self.encrypted_clusters_ids[i.item()] for i in top_centroid_indices]
            cluster_id_map = crypten.cat(cluster_id_map, dim=0)


        else:
            if self.debug: print("Running full oblivious lookup")
            # Obliviously reduce the dataset size to (n_probes, cluster_size).
            # This has O(N) comm between the servers (can be improved ..), but after which the search space is greatly reduced
            # and we can run comparisons

            ## These are the only two operations that take O(N) communication. By using honest majority and partial-muls/sum-of-products
            ## We are able to turn these into local operations. Since Crypten doesn't support Shamir SS, we simulate this by keeping
            ## the encrypted_clusters_distances and encrypted_clusters_ids as scalars
            ## Another point of optimization (which we don't implement), that is relevant to the DISHONEST majority setting
            ## is doing these two matrix multiplications in parallel

            encrypted_top_centroid_indices = self.encrypted_topk(-encrypted_distances_to_centroids, self.nprobe, one_hot=self.nlist)
        
            if self.debug: print(encrypted_top_centroid_indices.shape, self.encrypted_clusters_distances.shape)
            enc_distance_candidates_matrix = encrypted_top_centroid_indices.matmul(self.encrypted_clusters_distances)
            if self.debug: print(enc_distance_candidates_matrix.shape)
            enc_distance_candidates_matrix = enc_distance_candidates_matrix.reshape(self.nprobe*self.max_cluster_size, self.database.shape[1])
            if self.debug: print(enc_distance_candidates_matrix.shape)
            enc_indices_candidates_matrix = encrypted_top_centroid_indices.matmul(self.encrypted_clusters_ids).flatten()
        

        if (self.debug):
            self._last_candidates = enc_indices_candidates_matrix.get_plain_text()
            t1 = encrypted_top_centroid_indices.get_plain_text()
            if hasattr(self.encrypted_clusters_ids, 'get_plain_text'):
                t2 = self.encrypted_clusters_ids.get_plain_text()
            else:
                t2 = self.encrypted_clusters_ids
                
            t3 = torch.mm(t1.float(), t2)
            logger.debug(f"result shape - {t3.shape}")

            indices = self.encrypted_topk(-encrypted_distances_to_centroids, self.nprobe).get_plain_text()
            idx_list_count = 0
            for idx in indices:
                ii = int(idx)
                idx_list = [self.index.invlists.get_single_id(ii, j) for j in range(self.index.invlists.list_size( ii ))]
                idx_list_count += len(idx_list)
                for jj in range(t3.shape[0]):
                    if t3[jj][0] == idx_list[0]:
                        t3jj = t3[jj][t3[jj] != -1].int()
                        logger.debug("-=-=-")
                        logger.debug(f"Cluster {idx}={torch.all(t3jj == torch.Tensor(idx_list))}")
                        # logger.debug(f"Cluster {idx}={t3[jj].int()}, and originally={idx_list}")
            logger.debug(f"idx_list_count={idx_list_count}")
        
        # Compute encrypted distances to candidates
        if self.distance_func in ['cos_sim', 'dot_prod']:
            encrypted_distances_to_candidates = encrypted_query.matmul(enc_distance_candidates_matrix.t())
        else:
            # L2 distance can be trickier with encrypted data
            diff = encrypted_query - enc_distance_candidates_matrix
            encrypted_distances_to_candidates = (diff * diff).sum(1).sqrt()
            
        enc_top_k_indices = self.encrypted_topk(-encrypted_distances_to_candidates, top_k)
        
        top_k_indices_local = enc_top_k_indices.get_plain_text()

        if self.tobin_encrypt:
            #  use cluster_id_map to map back to top_k_indices_local. This is not oblivious same as below, but could be with some rearrangement.
            cluster_id_map_dec = cluster_id_map.get_plain_text()
            top_k_indices = [int(cluster_id_map_dec[int(i.item())].item()) for i in top_k_indices_local]
        else:
            # TODO: map back to global indices obliviously
            indices_candidates_matrix = enc_indices_candidates_matrix.get_plain_text()
            top_k_indices = indices_candidates_matrix[top_k_indices_local.long()].long()

        # TODO: do I need the distances even? Tobin: yes, for BEIR batching, Tobin later: no we get away without it now.
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
def mpc_benchmark_knn(query_tensor: torch.Tensor, database: torch.Tensor, model: MPCIVFRetrievalModel, k: int=10) -> None:
    model.encrypt()

    start_time = time.time()
    top_k_indices, top_k_values = model.query(query_tensor, top_k=k)
    end_time = time.time()
    approx_knn_time = end_time - start_time

    start_time = time.time()
    distances = cosine_similarity_mpc_opt(query_tensor, database.t())
    top_k_exact = _top_k_mpc_tobin(distances, k)
    end_time = time.time()
    exact_knn_time = end_time - start_time

    # exact_knn_time = 0
    crypten.print(f"Parameters: data size = {database.shape}, k={k}")
    crypten.print(f"exact_knn_time = {exact_knn_time}, approx_knn_time = {approx_knn_time}")

def benchmark_knn(k: int=10, N: int=0, nprobe: int=10, nlist: int=50):
    query_tensor, database_tensor = load_preembeddings('datasets/corpus_embeddings_large.pt', 'datasets/query_embeddings_large.pt')
    query_tensor = query_tensor[0].flatten()
    if N > 0:
        database_tensor = database_tensor[0:N,:]
    model = MPCIVFRetrievalModel(nlist=nlist, nprobe=nprobe)
    model.train(database_tensor)

    # Move to MPC-land
    mpc_benchmark_knn(query_tensor, database_tensor, model, k)

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
    
    ## Helps to test correctness
    # test_pt_vs_mpc()

    ## Benchmark
    benchmark_knn(N=1000, nlist=50)
    benchmark_knn(N=10000, nlist=250)
    benchmark_knn(N=100000, nlist=250)
    benchmark_knn(N=100000, nlist=1500)
    benchmark_knn(N=100000, nlist=1500, k=100)



