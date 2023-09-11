import logging
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
import pandas as pd
from collections import defaultdict

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

class KMeansRetrievalModel(SimpleRetrievalModel):

    def __init__(self, n_clusters: int, distance_func: str = 'cos_sim', **kwargs):
        super().__init__(distance_func=distance_func, **kwargs)
        self.n_clusters = n_clusters
        self.clusters = None
        self.clusters_map = {}
        self.model_name = 'kmeans_retrieval_model'

    def train(self, database: torch.Tensor):
        self.database = database
        
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42).fit(database.cpu().numpy()) # reproducibile
        # kmeans = KMeans(n_clusters=self.n_clusters).fit(database.cpu().numpy())
        
        self.clusters = torch.tensor(kmeans.cluster_centers_)
        self.clusters_map = {i: [] for i in range(self.n_clusters)}
        for i, label in enumerate(kmeans.labels_):
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

    def bucketize_with_kmeans(self, data):
        # kmeans = KMeans(n_clusters=min(self.n_clusters, len(data))).fit(data.cpu().numpy())
        kmeans = KMeans(n_clusters=min(self.n_clusters, len(data)), random_state=42).fit(data.cpu().numpy()) # reproducible
        clusters = torch.tensor(kmeans.cluster_centers_)
        clusters_map = {i: [] for i in range(len(clusters))}
        for i, label in enumerate(kmeans.labels_):
            clusters_map[label].append(i)
        return clusters, clusters_map

    def train(self, database: torch.Tensor):
        self.database = database
        data_to_split = database
        data_indices_to_split = list(range(len(database)))

        for d in range(self.depth - 1):
            clusters, clusters_map = self.bucketize_with_kmeans(data_to_split)
            self.all_clusters.append((clusters, clusters_map))
            next_data_to_split = []
            next_data_indices_to_split = []

            for i, indices in clusters_map.items():
                if len(indices) > self.m:
                    logger.debug(f"Depth {d+1}, Splitting cluster {i} with {len(indices)} items")
                    
                    # Keep m items in the current cluster and take the rest out for further splitting
                    retained_indices = indices[:self.m]
                    to_split_indices = indices[self.m:]

                    clusters_map[i] = retained_indices
                    next_data_to_split.extend(data_to_split[to_split_indices].tolist())
                    next_data_indices_to_split.extend([data_indices_to_split[index] for index in to_split_indices])

            if not next_data_to_split:
                break

            data_to_split = torch.tensor(next_data_to_split)
            data_indices_to_split = next_data_indices_to_split

        # Handling the remainder
        remaining_indices = []
        last_clusters_map = self.all_clusters[-1][1]
        for cluster_id, indices in last_clusters_map.items():
            if len(indices) > self.m:
                remaining_indices.extend(indices[self.m:])
                last_clusters_map[cluster_id] = indices[:self.m]

        if remaining_indices:
            self.remainder = database[remaining_indices]
            self.all_clusters.append(self.bucketize_with_kmeans(self.remainder))
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
            _, top_clusters_remainder = torch.topk(scores_remainder, k=n_clusters_to_search)
            total_clusters_searched += len(top_clusters_remainder)
            all_candidates_indices.extend(top_clusters_remainder.tolist())

        candidates = self.database[all_candidates_indices]
        local_indices, top_k_values = super().query(query, top_k=top_k, database=candidates)
        mapped_indices = torch.tensor([all_candidates_indices[idx] for idx in local_indices])

        logger.debug(f"Total clusters searched: {total_clusters_searched}")
        logger.debug(f"Total similarities computed: {total_sim_computed}")

        return mapped_indices, top_k_values

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
    if model == KMeansRetrievalModel:
        cluster_counts = {i: 0 for i in range(model_instance.n_clusters)}
        for idx in gt_indices:
            for cluster, indices in model_instance.clusters_map.items():
                if idx.item() in indices:
                    cluster_counts[cluster] += 1
                    total_count += 1
                    break

        for cluster, count in cluster_counts.items():
            logger.debug(f'cluster {cluster}={count} top_k items')
        logger.debug(f'Total top k found in all clusters: {total_count}')
    elif model == KMeansRecursiveRetrievalModel:
        for depth, (clusters, clusters_map) in enumerate(model_instance.all_clusters):
            for cluster_id, items in clusters_map.items():
                curr_matches = len(gt_set.intersection(set(items)))
                total_count += curr_matches
                logger.debug(f"Depth {depth}, Cluster {cluster_id}={len(items)} items and top_k={curr_matches} items")
                # logger.debug(f"Depth {depth}, Cluster {cluster_id}={len(items)} items")
        if model_instance.remainder is not None:
            curr_matches = len(gt_set.intersection(set(model_instance.remainder.tolist())))
            total_count += curr_matches
            logger.debug(f"Remainder={len(model_instance.remainder)} items and top_k={curr_matches} items")
        logger.debug(f'Total top k found in all clusters (recursive): {total_count}')
    
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
database_tensor = database_tensor[0:10000,:]
print(f"Query tensor shape: {query_tensor.shape}")
print(f"Database tensor shape: {database_tensor.shape}")
# benchmark(query=query_tensor, database=database_tensor, model=KMeansRetrievalModel, top_k=100, train_params={'n_clusters': 16, 'm': 100, 'depth': 3}, query_params={'n_clusters_to_search': 4})
benchmark(query=query_tensor, database=database_tensor, model=KMeansRetrievalModel, top_k=100, train_params={'n_clusters': 25}, query_params={'n_clusters_to_search': 5})
# benchmark(query=query_tensor, database=database_tensor, model=KMeansRecursiveRetrievalModel, top_k=100, train_params={'n_clusters': 25, 'm': 1000, 'depth': 3}, query_params={'n_clusters_to_search': 5}) # this returns the same as a single depth model
benchmark(query=query_tensor, database=database_tensor, model=KMeansRecursiveRetrievalModel, top_k=100, train_params={'n_clusters': 25, 'm': 100, 'depth': 3}, query_params={'n_clusters_to_search': 5})


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
