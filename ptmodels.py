import logging
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
import pandas as pd
from collections import defaultdict
import faiss
import numpy as np
import random

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class SimpleRetrievalModel:
    
    def __init__(self, distance_func: str = 'cos_sim', **kwargs):
        self.distance_func = distance_func
        self.model_name = 'simple_retrieval_model'

    def train(self, database: torch.Tensor):
        self.database = database

    def _compute_scores(self, A, B):
        """Compute distance or similarity scores between two sets of vectors A and B."""
        if self.distance_func == 'cos_sim':
            A_norm = F.normalize(A, p=2, dim=1)
            B_norm = F.normalize(B, p=2, dim=0)
            scores = torch.mm(A_norm, B_norm.unsqueeze(0).t()).squeeze()
            print(f"Computed scores for model={self.model_name} have shape: {scores.shape}")
        elif self.distance_func == 'dot_prod':
            scores = torch.mm(A, B.unsqueeze(0).t()).squeeze()
        elif self.distance_func == 'euclidean':
            scores = torch.norm(A - B, p=2, dim=1)
        else:
            logger.error(f"Unsupported distance function: {self.distance_func}")
            return None
        
        return scores

    def query(self, query: torch.Tensor, top_k: int=10, database=None):
        database = database if database is not None else self.database
        scores = self._compute_scores(database, query)

        # Negate scores for similarity measures
        if self.distance_func in ['cos_sim', 'dot_prod']:
            scores = -scores

        # Get top_k results
        top_k_values, top_k_indices = torch.topk(scores, k=top_k, largest=True, sorted=True)

        # Return negated top_k values for similarity measures to get back the original values
        if self.distance_func in ['cos_sim', 'dot_prod']:
            top_k_values = -top_k_values
        
        return top_k_indices, top_k_values

def fit_clusters(data: np.ndarray, n_clusters: int, distance_func: str = 'cos_sim', seed: int = 0):
    """
    Cluster the given database using the specified distance function.
    
    Args:
    - database (np.ndarray): Data to be clustered.
    - n_clusters (int): Number of clusters.
    - distance_func (str): Distance function ('cos_sim' or 'euclidean').

    Returns:
    - clusters (np.ndarray): Cluster centers.
    - labels (np.ndarray): Labels for each entry in the database.
    """
    if seed == 0:
        seed = random.randint(0, 100000000)

    if distance_func == 'cos_sim':
        # Convert data to float32 (required by faiss)
        data = data.cpu().numpy().astype('float32')
        
        # Normalize vectors (for cosine similarity)
        faiss.normalize_L2(data)

        # Initialize k-means with cosine similarity
        kmeans = faiss.Kmeans(data.shape[1], min(n_clusters, len(data)), niter=20, verbose=False, seed=seed)
        kmeans.train(data)

        # Assign points to clusters
        _, labels = kmeans.index.search(data, 1)
        labels = labels.flatten().tolist()
        clusters = kmeans.centroids
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed).fit(data)  # reproducible
        clusters = kmeans.cluster_centers_
        labels = kmeans.labels_
        
    return clusters, labels

class KMeansRetrievalModel(SimpleRetrievalModel):

    def __init__(self, n_clusters: int, distance_func: str = 'cos_sim', **kwargs):
        super().__init__(distance_func=distance_func, **kwargs)
        self.n_clusters = n_clusters
        self.clusters = None
        self.clusters_map = {}
        self.model_name = 'kmeans_retrieval_model'

    def train(self, database: torch.Tensor):
        self.database = database        
        clusters, labels = fit_clusters(database, self.n_clusters, self.distance_func)
        
        self.clusters = torch.tensor(clusters)
        self.clusters_map = {i: [] for i in range(self.n_clusters)}
        for i, label in enumerate(labels):
            self.clusters_map[label].append(i)

    def query(self, query: torch.Tensor, top_k: int=10, n_clusters_to_search: int=10):
        scores = self._compute_scores(self.clusters, query)

        # Negate scores for similarity measures
        if self.distance_func in ['cos_sim', 'dot_prod']:
            scores = -scores

        # Fetch the top clusters
        _, top_clusters = torch.topk(scores, k=n_clusters_to_search)

        # Retrieve database items from the top clusters
        candidates_indices = [self.clusters_map[cluster.item()] for cluster in top_clusters]
        candidates_indices = sorted([idx for sublist in candidates_indices for idx in sublist])
        candidates = self.database[candidates_indices]

        print("Candidates indices:", candidates_indices)
        print(f"Candidates database first entry: {candidates[0]}, shape: {candidates.shape}")
        print("Database first entry:", self.database[0])
        self.last_candidates = candidates

        # Query using the candidates
        local_indices, top_k_values = super().query(query, top_k=top_k, database=candidates)

        # Map local indices back to the original database
        mapped_indices = torch.tensor([candidates_indices[idx] for idx in local_indices])

        return mapped_indices, top_k_values

class KMeansRecursiveRetrievalModel(SimpleRetrievalModel):

    def __init__(self, n_clusters: int, m: int, depth: int, distance_func: str = 'cos_sim', **kwargs):
        super().__init__(distance_func=distance_func, **kwargs)
        self.n_clusters = n_clusters
        self.m = m
        self.depth = depth
        self.all_clusters = []
        self.remainder = None
        self.remainder_indices = None

    def _compute_scores(self, A, B):
        """Compute distance or similarity scores between two sets of vectors A and B."""
        if self.distance_func == 'cos_sim':
            A_norm = F.normalize(A, p=2, dim=1)
            B_norm = F.normalize(B, p=2, dim=0)
            scores = torch.mm(A_norm, B_norm.unsqueeze(0).t()).squeeze()
        elif self.distance_func == 'dot_prod':
            scores = torch.mm(A, B.unsqueeze(0).t()).squeeze()
        elif self.distance_func == 'euclidean':
            scores = torch.norm(A - B, p=2, dim=1)
        else:
            logger.error(f"Unsupported distance function: {self.distance_func}")
            return None
        
        return scores

    def bucketize_with_kmeans(self, data, local_indices):
        # kmeans = KMeans(n_clusters=min(self.n_clusters, len(data))).fit(data.cpu().numpy())
        kmeans = KMeans(n_clusters=min(self.n_clusters, len(data)), random_state=42).fit(data.cpu().numpy()) # reproducible
        clusters = torch.tensor(kmeans.cluster_centers_)
        clusters_map = {i: [] for i in range(len(clusters))}
        for i, label in enumerate(kmeans.labels_):
            clusters_map[label].append(local_indices[i])
        return clusters, clusters_map
    
    def bucketize_with_kmeans_cos(self, data, local_indices):
        # Convert data to float32 (required by faiss)
        data = data.cpu().numpy().astype('float32')
        
        # Normalize vectors (for cosine similarity)
        faiss.normalize_L2(data)

        # Initialize k-means with cosine similarity
        kmeans = faiss.Kmeans(data.shape[1], min(self.n_clusters, len(data)), niter=20, verbose=False)
        kmeans.train(data)

        # Assign points to clusters
        _, labels = kmeans.index.search(data, 1)
        labels = labels.ravel()

        clusters = torch.tensor(kmeans.centroids)

        clusters_map = {i: [] for i in range(len(clusters))}
        for i, label in enumerate(labels):
            clusters_map[label].append(local_indices[i])

        return clusters, clusters_map

    def train(self, database: torch.Tensor):
        self.database = database
        new_db = database
        to_split_indices = list(range(len(database)))
        
        for d in range(self.depth):
            if self.distance_func == 'cos_sim':
                logger.debug("Bucketizing with cos")
                clusters, clusters_map = self.bucketize_with_kmeans_cos(new_db, to_split_indices)
            else:
                logger.debug(f"Bucketizing with {self.distance_func}")
                clusters, clusters_map = self.bucketize_with_kmeans(new_db, to_split_indices)
            
            self.all_clusters.append((clusters, clusters_map))
            to_split_indices = []

            for i, indices in clusters_map.items():
                if len(indices) > self.m:
                    # logger.debug(f"Depth {d+1}, Splitting cluster {i} with {len(indices)} items")
                    # Keep m items in the current cluster and take the rest out for further splitting
                    retained_indices = indices[:self.m]
                    to_split_indices.extend(indices[self.m:])
                    clusters_map[i] = retained_indices
                    # logger.debug(f"Depth {d+1}, Retained {retained_indices} items and splitting {to_split_indices} items")
            
            new_db = self.database[to_split_indices]
            if not to_split_indices:
                break
        
        # Handling the remainder
        if to_split_indices:
            self.remainder = self.database[to_split_indices]
            self.remainder_indices = to_split_indices
            logger.debug(f"Moved {len(self.remainder)} items to the remainder database")


    def query(self, query: torch.Tensor, top_k: int=10, n_clusters_to_search: int=10):
        all_candidates_indices = []
        total_clusters_searched = 0
        total_sim_computed = 0

        for clusters, clusters_map in self.all_clusters:
            scores = self._compute_scores(clusters, query)
            total_sim_computed += len(clusters)
            if self.distance_func in ['cos_sim', 'dot_prod']:
                scores = -scores
            _, top_clusters = torch.topk(scores, k=n_clusters_to_search)
            total_clusters_searched += len(top_clusters)
            candidates_indices = [clusters_map[cluster.item()] for cluster in top_clusters]
            all_candidates_indices.extend(sorted([idx for sublist in candidates_indices for idx in sublist]))

        # Querying the remainder if it exists
        if self.remainder is not None:
            scores_remainder = self._compute_scores(self.remainder, query)
            total_sim_computed += len(self.remainder)
            _, remainder_top_k_indices = torch.topk(scores_remainder, k=top_k)
            all_candidates_indices.extend(remainder_top_k_indices.tolist())

        # Compute a final top_k over the reduced candidate set
        candidates = self.database[all_candidates_indices]
        local_indices, top_k_values = super().query(query, top_k=top_k, database=candidates)
        mapped_indices = torch.tensor([all_candidates_indices[idx] for idx in local_indices])

        logger.debug(f"Total clusters searched: {total_clusters_searched}")
        logger.debug(f"Total similarities computed: {total_sim_computed}")
        logger.debug(f"Total size of shortlist: {len(all_candidates_indices)}")

        return mapped_indices, top_k_values

class KMeansRetrievalModel2D(SimpleRetrievalModel):

    def __init__(self, n_clusters: int, m: int, distance_func: str = 'cos_sim', seed: int = 0, **kwargs):
        super().__init__(distance_func=distance_func, **kwargs)
        self.n_clusters = n_clusters
        self.m = m
        self.clusters = None
        self.clusters_map = {}
        self.model_name = 'kmeans_retrieval_model2d'
        self.remainder_indices = []

    def train(self, database: torch.Tensor):
        self.database = database
        clusters, labels = fit_clusters(database, self.n_clusters, self.distance_func)
        
        self.clusters = torch.tensor(clusters)
        self.clusters_map = {i: [] for i in range(self.n_clusters)}
        for i, label in enumerate(labels):
            if len(self.clusters_map[label]) < self.m:
                self.clusters_map[label].append(i)
            else:
                self.remainder_indices.append(i)

        logger.debug(f"Moved {len(self.remainder_indices)} items to the remainder database")

    def query(self, query: torch.Tensor, top_k: int=10, n_clusters_to_search: int=10):
        scores = self._compute_scores(self.clusters, query)

        # Negate scores for similarity measures
        if self.distance_func in ['cos_sim', 'dot_prod']:
            scores = -scores

        # Fetch the top clusters
        _, top_clusters = torch.topk(scores, k=n_clusters_to_search)

        # Retrieve database items from the top clusters
        candidates_indices = [self.clusters_map[cluster.item()] for cluster in top_clusters]
        candidates_indices.extend([self.remainder_indices])
        candidates_indices = sorted([idx for sublist in candidates_indices for idx in sublist])
        candidates = self.database[candidates_indices] 

        # logger.debug("Candidates indices:", candidates_indices)
        # logger.debug(f"Candidates database first entry: {candidates[0]}, shape: {candidates.shape}")
        # logger.debug("Database first entry:", self.database[0])
        logger.debug(f"Number of candidates after clustering: {len(candidates_indices)}")
        self.last_candidates = candidates

        # Query using the candidates
        local_indices, top_k_values = super().query(query, top_k=top_k, database=candidates)

        # Map local indices back to the original database
        mapped_indices = torch.tensor([candidates_indices[idx] for idx in local_indices])

        return mapped_indices, top_k_values

class LSHRetrievalModel(SimpleRetrievalModel):

    def __init__(self, num_tables: int = 32, hash_size: int = 10, distance_func: str = 'cos_sim', num_probes: int = 2, **kwargs):
        self.num_tables = num_tables
        self.hash_size = hash_size
        self.distance_func = distance_func
        self.model_name = 'lsh_retrieval_model'
        self.hash_tables = []
        self.num_probes = num_probes

    def train(self, database: torch.Tensor):
        self.database = database
        self._build_lsh_tables(database)

    def _build_lsh_tables(self, data: torch.Tensor):
        self.hash_tables = []
        for _ in range(self.num_tables):
            random_planes = torch.randn((self.hash_size, data.shape[1]))
            hashes = (torch.mm(data, random_planes.t()) > 0).int().numpy()
            table = {}
            for idx, h in enumerate(hashes):
                h = tuple(h)
                if h not in table:
                    table[h] = []
                table[h].append(idx)

            # Debug: Print min and max row size for each table
            row_sizes = [len(row) for row in table.values()]
            logger.debug(f"Table {_}: Min row size = {min(row_sizes)}, Max row size = {max(row_sizes)}")

            self.hash_tables.append((random_planes, table))

    def _generate_neighbors(self, hash_val):
        neighbors = []
        for i in range(len(hash_val)):
            if len(neighbors) >= (self.num_probes - 1):  # Limit the number of probes
                break
            flip = list(hash_val)
            flip[i] = 1 - flip[i]
            neighbors.append(tuple(flip))
        return neighbors

    def _lsh_query(self, query: torch.Tensor):
        candidates = set()
        for planes, table in self.hash_tables:
            h = tuple((torch.mm(query.unsqueeze(0), planes.t()) > 0).int().numpy()[0])
            if h in table:
                candidates.update(table[h])
            for neighbor in self._generate_neighbors(h): 
                if neighbor in table:
                    candidates.update(table[neighbor])
        return list(candidates)

    def query(self, query: torch.Tensor, top_k: int = 10, database=None):
        database = database if database is not None else self.database
        
        # Only compute scores for LSH candidates
        candidate_indices = self._lsh_query(query)
        logger.debug(f"Number of candidates after LSH filtering: {len(candidate_indices)}")
        candidates = database[candidate_indices]
        
        if len(candidate_indices) == 0:  # No candidates found by LSH
            return torch.tensor([]), torch.tensor([])
        
        scores = self._compute_scores(candidates, query)

        # Negate scores for similarity measures
        if self.distance_func in ['cos_sim', 'dot_prod']:
            scores = -scores

        # Get top_k results
        top_k_values, top_k_relative_indices = torch.topk(scores, k=min(top_k, len(candidate_indices)), largest=True, sorted=True)
        top_k_indices = torch.tensor([candidate_indices[idx] for idx in top_k_relative_indices])

        # Return negated top_k values for similarity measures to get back the original values
        if self.distance_func in ['cos_sim', 'dot_prod']:
            top_k_values = -top_k_values
        
        return top_k_indices, top_k_values
    

class KLSHRetrievalModel(SimpleRetrievalModel):

    def __init__(self, num_tables: int = 32, hash_size: int = 10, distance_func: str = 'cos_sim', num_probes: int = 2, **kwargs):
        self.num_tables = num_tables
        self.hash_size = hash_size
        self.distance_func = distance_func
        self.model_name = 'klsh_retrieval_model'
        self.hash_tables = []
        self.num_probes = num_probes

    def train(self, database: torch.Tensor):
        self.database = database
        self._build_klsh_tables(database)

    def _build_klsh_tables(self, data: torch.Tensor):
        self.hash_tables = []
        for _ in range(self.num_tables):
            # kmeans = KMeans(n_clusters=self.hash_size, random_state=0).fit(data.numpy())
            kmeans = KMeans(n_clusters=self.hash_size).fit(data.numpy())
            centroids = torch.tensor(kmeans.cluster_centers_)
            hashes = (torch.mm(data, centroids.t()) > 0).int().numpy()
            table = {}
            for idx, h in enumerate(hashes):
                h = tuple(h)
                if h not in table:
                    table[h] = []
                table[h].append(idx)

            # Debug: Print min and max row size for each table
            row_sizes = [len(row) for row in table.values()]
            logger.debug(f"Table {_}: Min row size = {min(row_sizes)}, Max row size = {max(row_sizes)}")

            self.hash_tables.append((centroids, table))

    def _generate_neighbors(self, hash_val):
        neighbors = []
        for i in range(len(hash_val)):
            if len(neighbors) >= (self.num_probes - 1):  # Limit the number of probes
                break
            flip = list(hash_val)
            flip[i] = 1 - flip[i]
            neighbors.append(tuple(flip))
        return neighbors

    def _lsh_query(self, query: torch.Tensor):
        candidates = set()
        for planes, table in self.hash_tables:
            h = tuple((torch.mm(query.unsqueeze(0), planes.t()) > 0).int().numpy()[0])
            if h in table:
                candidates.update(table[h])
            for neighbor in self._generate_neighbors(h): 
                if neighbor in table:
                    candidates.update(table[neighbor])
        return list(candidates)

    def query(self, query: torch.Tensor, top_k: int = 10, database=None):
        database = database if database is not None else self.database
        
        # Only compute scores for LSH candidates
        candidate_indices = self._lsh_query(query)
        logger.debug(f"Number of candidates after LSH filtering: {len(candidate_indices)}")
        candidates = database[candidate_indices]
        
        if len(candidate_indices) == 0:  # No candidates found by LSH
            return torch.tensor([]), torch.tensor([])
        
        scores = self._compute_scores(candidates, query)

        # Negate scores for similarity measures
        if self.distance_func in ['cos_sim', 'dot_prod']:
            scores = -scores

        # Get top_k results
        top_k_values, top_k_relative_indices = torch.topk(scores, k=min(top_k, len(candidate_indices)), largest=True, sorted=True)
        top_k_indices = torch.tensor([candidate_indices[idx] for idx in top_k_relative_indices])

        # Return negated top_k values for similarity measures to get back the original values
        if self.distance_func in ['cos_sim', 'dot_prod']:
            top_k_values = -top_k_values
        
        return top_k_indices, top_k_values

def benchmark(query: torch.Tensor, database: torch.Tensor=None, model=None, top_k: int=10, train_params={}, query_params={}):

    # If database tensor doesn't exist, initialize it at random
    if database is None:
        database = torch.randn(100, 50)
    
    # Initialize and train the SimpleRetrievalModel
    simple_model = SimpleRetrievalModel()
    simple_model.train(database)
    
    # Get the top_k results as ground truth
    gt_indices, _ = simple_model.query(query, top_k=top_k)
    gt_set = set(sorted(gt_indices.tolist()))
    print("Ground Truth Indices:", gt_set)
    
    # If another model is provided, initialize, train, and run it
    if model:
        model_instance = model(**train_params)
        model_instance.train(database)
        pred_indices, _ = model_instance.query(query, top_k=top_k, **query_params)
        pred_set = set(sorted(pred_indices.tolist()))
        print("Predicted Indices:", pred_set)
    else:
        print("No secondary model provided!")
        return
    
    # Update to report clusters
    total_count = 0
    total_items = 0
    if model in [KMeansRetrievalModel, KMeansRetrievalModel2D]:
        cluster_counts = {i: 0 for i in range(model_instance.n_clusters)}
        for idx in gt_indices:
            for cluster, indices in model_instance.clusters_map.items():
                if idx.item() in indices:
                    cluster_counts[cluster] += 1
                    total_count += 1
                    break

        for cluster, count in cluster_counts.items():
            logger.debug(f'cluster {cluster}={count} top_k items')

        if hasattr(model_instance, 'remainder_indices') and model_instance.remainder_indices is not None:
            curr_matches = len(gt_set.intersection(set(model_instance.remainder_indices)))
            total_count += curr_matches
            logger.debug(f"Remainder={len(model_instance.remainder_indices)} items and top_k={curr_matches} items")
        logger.debug(f'Total top k found in all clusters: {total_count}')
    elif model == KMeansRecursiveRetrievalModel:
        ## TODO: remove temp - sanity check
        # rebuilt_db_indices = []
        
        for depth, (clusters, clusters_map) in enumerate(model_instance.all_clusters):
            for cluster_id, items in clusters_map.items():
                # rebuilt_db_indices.extend(items)
                curr_matches = len(gt_set.intersection(set(items)))
                assert len(set(items)) == len(items) # TODO: remove this, sanity check
                total_count += curr_matches
                logger.debug(f"Depth {depth + 1}, Cluster {cluster_id}={len(items)} items and top_k={curr_matches} items")
                total_items += len(items)
                # logger.debug(f"Depth {depth}, Cluster {cluster_id}={len(items)} items")
        if model_instance.remainder_indices is not None:
            curr_matches = len(gt_set.intersection(set(model_instance.remainder_indices)))
            total_count += curr_matches
            logger.debug(f"Remainder={len(model_instance.remainder)} items and top_k={curr_matches} items")
            total_items += len(model_instance.remainder_indices)
            # rebuilt_db_indices.extend(model_instance.remainder_indices)
        # logger.debug(f'rebuilt db length={len(rebuilt_db_indices)}')
        # rebuilt_db_indices = sorted(rebuilt_db_indices)
        # for i in range(len(rebuilt_db_indices)):
            # print(f"i={i}, rebuilt_db_indices[i]={rebuilt_db_indices[i]}")
            # assert(i == rebuilt_db_indices[i])
        logger.debug(f'Total items={total_items} and top k found in all clusters (recursive): {total_count} out of {len(gt_set)}')
    
    # Calculate metrics
    intersection = gt_set.intersection(pred_set)
    
    accuracy = len(intersection) / top_k
    recall = len(intersection) / len(gt_set)
    precision = len(intersection) / len(pred_set)
    
    # Avoid division by zero
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    # Print metrics
    print(f"\nMetrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1:.4f}")


def test_clusters_map_equality(model: KMeansRetrievalModel, database: torch.Tensor):
    # Flatten clusters_map to get all indices
    all_indices = [idx for cluster_indices in model.clusters_map.values() for idx in cluster_indices]
    
    # Convert to set to ensure no duplicates
    unique_indices = set(all_indices)
    
    # Check length to ensure no missing entries
    if len(unique_indices) != len(database):
        print("Error: Some entries from the original database are missing in clusters_map.")
        return
    
    # Retrieve entries using indices and compare with original database
    retrieved_db = database[torch.tensor(list(unique_indices))]
    if not torch.all(torch.eq(retrieved_db, database)):
        print("Error: Retrieved database entries from clusters_map do not match the original database.")
    else:
        print("Success: Iterating over clusters_map gives the exact same results as the original database without duplicates.")

def load_preembeddings(corpus_path: str, queries_path: str) -> None:
    corpus_embeddings = torch.load(corpus_path, map_location=torch.device('cpu')).to('cpu')
    query_embeddings = torch.load(queries_path, map_location=torch.device('cpu')).to('cpu')
    return query_embeddings, corpus_embeddings

# # Example Usage
# query_tensor = torch.randn(50)
# # torch.manual_seed(42)  # Ensures reproducibility
# database_tensor = torch.randn(1000, 50)
# print(f"Benchmarking on random data")
# benchmark(query=query_tensor, database=database_tensor, model=KMeansRetrievalModel, top_k=50, train_params={'n_clusters': 100, 'm': 10, 'depth': 2}, query_params={'n_clusters_to_search': 10})

# # Read data and check
# print(f"Benchmarking on real data (small)")
# query_tensor = pd.read_csv('datasets/query_vector.csv').values
# database_tensor = pd.read_csv('datasets/D.csv').values

# # Convert the data to PyTorch tensors
# query_tensor = torch.tensor(query_tensor, dtype=torch.float32).flatten()
# database_tensor = torch.tensor(database_tensor, dtype=torch.float32).t()
# print(f"Query tensor shape: {query_tensor.shape}")
# print(f"Database tensor shape: {database_tensor.shape}")
# benchmark(query=query_tensor, database=database_tensor, model=KMeansRetrievalModel, top_k=10, train_params={'n_clusters': 9, 'm': 10, 'depth': 2}, query_params={'n_clusters_to_search': 3})

print(f"Benchmarking on real data (large)")
query_tensor, database_tensor = load_preembeddings('datasets/corpus_embeddings_large.pt', 'datasets/query_embeddings_large.pt')
query_tensor = query_tensor[0].flatten()
# database_tensor = database_tensor[0:1000,:]
database_tensor = database_tensor[0:100000,:]
# database_tensor = database_tensor[50000:150000,:]
# database_tensor = database_tensor[10000:20000,:]
print(f"Query tensor shape: {query_tensor.shape}")
print(f"Database tensor shape: {database_tensor.shape}")
# benchmark(query=query_tensor, database=database_tensor, model=KMeansRetrievalModel, top_k=100, train_params={'n_clusters': 16, 'm': 100, 'depth': 3}, query_params={'n_clusters_to_search': 4})
## benchmark(query=query_tensor, database=database_tensor, model=KMeansRetrievalModel, top_k=100, train_params={'n_clusters': 25}, query_params={'n_clusters_to_search': 5})
# benchmark(query=query_tensor, database=database_tensor, model=KMeansRecursiveRetrievalModel, top_k=100, train_params={'n_clusters': 25, 'm': 1000, 'depth': 3}, query_params={'n_clusters_to_search': 5}) # this returns the same as a single depth model
# benchmark(query=query_tensor, database=database_tensor, model=KMeansRecursiveRetrievalModel, top_k=100, train_params={'n_clusters': 9, 'm': 100, 'depth': 5}, query_params={'n_clusters_to_search': 3})
# benchmark(query=query_tensor, database=database_tensor, model=KMeansRecursiveRetrievalModel, top_k=100, train_params={'n_clusters': 9, 'm': 10000, 'depth': 5}, query_params={'n_clusters_to_search': 2})
# benchmark(query=query_tensor, database=database_tensor, model=KMeansRecursiveRetrievalModel, top_k=100, train_params={'n_clusters': 9, 'm': 100, 'depth': 5, 'distance_func':'euclidean'}, query_params={'n_clusters_to_search': 3})
# benchmark(query=query_tensor, database=database_tensor, model=KMeansRecursiveRetrievalModel, top_k=25, train_params={'n_clusters': 50, 'm': 5, 'depth': 5}, query_params={'n_clusters_to_search': 3})
# benchmark(query=query_tensor, database=database_tensor, model=KMeansRecursiveRetrievalModel, top_k=10, train_params={'n_clusters': 5, 'm': 10, 'depth': 3, 'distance':'euclidean'}, query_params={'n_clusters_to_search': 5})

# benchmark(query=query_tensor, database=database_tensor, model=LSHRetrievalModel, top_k=10, train_params={'num_probes': 3, 'num_tables': 32, 'hash_size': 12}, query_params={})
# benchmark(query=query_tensor, database=database_tensor, model=KLSHRetrievalModel, top_k=10, train_params={'num_probes': 5, 'num_tables': 10, 'hash_size': 16}, query_params={})

# benchmark(query=query_tensor, database=database_tensor, model=KMeansRetrievalModel2D, top_k=10, train_params={'n_clusters': 9, 'm': 100}, query_params={'n_clusters_to_search': 2})
benchmark(query=query_tensor, database=database_tensor, model=KMeansRetrievalModel2D, top_k=100, train_params={'n_clusters': 9, 'm': 10000}, query_params={'n_clusters_to_search': 2})

## Conclusions
# TODO: recall or precision may be calc wrong? They are always the same
# 1. Searching a random DB for sqrt(clusters), which is basically like searching sqrt(N), means we get roughly 40%-50% recall/precision. 
# 2. Searching a real DB means less clusters are populated (because of the sparsity of the data), which means we generally want to use less clusters in k_means, so the chances we hit 
# populated cluster is high. For example over a db of size (5000 samples, 784 dim), we get much better results if we use n_clusters: 16 and n_clusters_to_search: 4, compared to say n_clusters: 100 and n_clusters_to_search: 10.
# Both have a similar query time (actually, the less the better), but the former has much better recall/precision. Ironically, training is faster that way too, so we 'win'
# Next thing to realize is that we still want avoid searching empty clusters, so we should use more k-means initiations to avoid that. We may also want to multi-probe.
# benchmark(query=query_tensor, database=database_tensor, model=KMeansRetrievalModel, top_k=100, train_params={'n_clusters': 16}, query_params={'n_clusters_to_search': 4})
# BTW: setting n_clusters_to_search:3 and n_clusters to 9 boosts us generally to 97%! (4,2) to 99% - but clearly this will not be the same for every database
## END Conclusions


# Test KMeans Coverage - this now breaks with new kmeans version
# model = KMeansRetrievalModel(n_clusters=10)
# database_tensor = torch.randn(100, 50)
# model.train(database_tensor)
# test_clusters_map_equality(model, database_tensor)

# # Example Usage:
# model = KMeansRetrievalModel(n_clusters=10, distance_func="dot_prod")
# database = torch.randn(100, 50)
# model.train(database)
# query_vec = torch.randn(50)
# indices, values = model.query(query_vec, top_k=5)
# print(indices, values)
