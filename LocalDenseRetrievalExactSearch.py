from beir.retrieval.search import BaseSearch
from  beir.util import cos_sim, dot_score
import logging
import torch
from typing import Dict
import heapq

logger = logging.getLogger(__name__)

class DenseRetrievalExactSearch(BaseSearch):
    """
    This is a modified version of `beir.retrieval.search.dense.DenseRetrievalExactSearch` for use with testing distance metric functions. The original DenseRetrievalExactSearch class would do embeddings of the entire corpus on every call of the key `search` method. To save on time (since we fix our embedding model), we rewrite the class to do the embedding of the corpus only once, and then use the embeddings to calculate the similarity measures. 
    """
    
    def __init__(self, model, batch_size: int = 128, corpus_chunk_size: int = 50000, **kwargs):
        #model is class that provides encode_corpus() and encode_queries()
        self.model = model
        self.batch_size = batch_size
        self.score_functions = {'cos_sim': self.default_cos_topk, 'dot_score':self.default_dot_topk}
        self.corpus_chunk_size = corpus_chunk_size
        self.show_progress_bar = kwargs.get("show_progress_bar", True)
        self.convert_to_tensor = kwargs.get("convert_to_tensor", True)
        self.results = {}
        self.corpus_embeddings = None
        self.query_embeddings = None

    def preemebed_corpus(self, corpus: Dict[str, Dict[str, str]], save_path: str = None) -> None:
              
        corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")), reverse=True)
        corpus = [corpus[cid] for cid in corpus_ids]

        print("Encoding Corpus in batches... Warning: This might take a while!")

        itr = range(0, len(corpus), self.corpus_chunk_size)
        
        for batch_num, corpus_start_idx in enumerate(itr):
            print("Encoding Batch {}/{}...".format(batch_num+1, len(itr)))
            corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(corpus))

            # Encode chunk of corpus    
            sub_corpus_embeddings = self.model.encode_corpus(
                corpus[corpus_start_idx:corpus_end_idx],
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar, 
                convert_to_tensor = self.convert_to_tensor
                )
            # Add embeddings to the large matrix
            if batch_num == 0:
                corpus_embeddings = sub_corpus_embeddings
            else:
                corpus_embeddings = torch.cat((corpus_embeddings, sub_corpus_embeddings), dim=0)

        self.corpus_embeddings = corpus_embeddings.to('cpu')

        # Save embeddings to file
        if save_path:
            torch.save(corpus_embeddings, save_path)


    def preembed_queries(self, queries: Dict[str, str], save_path: str = None) -> None:
        queries = [queries[qid] for qid in queries]
        query_embeddings = self.model.encode_queries(
            queries, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_tensor=self.convert_to_tensor).to('cpu')
        self.query_embeddings = query_embeddings
        if save_path:
            torch.save(query_embeddings, save_path)

    def load_preembeddings(self, corpus_path: str, queries_path: str) -> None:
        self.corpus_embeddings = torch.load(corpus_path).to('cpu')
        self.query_embeddings = torch.load(queries_path).to('cpu')

    def search(self, 
               corpus: Dict[str, Dict[str, str]], 
               queries: Dict[str, str], 
               top_k: int, 
               score_function: str,
               return_sorted: bool = False, 
               **kwargs) -> Dict[str, Dict[str, float]]:
        """
        This function is a reworked version of the original `search` function from `beir.retrieval.search.dense.DenseRetrievalExactSearch`. The original function would do the embedding of the corpus on every call of the `search` function. This function instead uses the pre-embedded corpus and queries to calculate the similarity measures.

        Further, this function now does each query in sequence rather than as a bunch to make it easier to compare the MPC functions (top-k) with others.
        """

        print("Running")
        if score_function not in self.score_functions:
            raise ValueError("score function: {} must be either (cos_sim) for cosine similarity or (dot) for dot product".format(score_function))
            
        if self.query_embeddings is not None:
            query_embeddings = self.query_embeddings
        else:
             raise RuntimeError('You have not pre-embedded the queries. Please run preembed_queries() or load_preembeddings()  first or use the original BEIR code.')
         
        if self.corpus_embeddings is not None:
            corpus_embeddings = self.corpus_embeddings
        else:
            raise RuntimeError('You have not pre-embedded the corpus. Please run preembed_corpus() or load_preembeddings() first or use the original BEIR code.')

        query_ids = list(queries.keys())
        self.results = {qid: {} for qid in query_ids}

        print("Sorting Corpus by document length (Longest first)...")
        corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")), reverse=True)
        corpus = [corpus[cid] for cid in corpus_ids]

        itr = range(0, len(corpus), self.corpus_chunk_size)

        result_heaps = {qid: [] for qid in query_ids}  # Keep only the top-k docs for each query
        for batch_num, corpus_start_idx in enumerate(itr):
            print("Encoding Batch {}/{}...".format(batch_num+1, len(itr)))
            corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(corpus))

            # Get chunk of corpus embeddings
            sub_corpus_embeddings = corpus_embeddings[corpus_start_idx:corpus_end_idx]

            for query_itr, query_embedding in enumerate(query_embeddings):

                top_k_values, top_k_idx  = self.score_functions[score_function](query_embedding, sub_corpus_embeddings, top_k+1)

                query_id = query_ids[query_itr]                  
                for sub_corpus_id, score in zip(top_k_idx, top_k_values):
                    corpus_id = corpus_ids[corpus_start_idx+sub_corpus_id]
                    if corpus_id != query_id:
                        if len(result_heaps[query_id]) < top_k:
                            # Push item on the heap
                            heapq.heappush(result_heaps[query_id], (score, corpus_id))
                        else:
                            # If item is larger than the smallest in the heap, push it on the heap then pop the smallest element
                            heapq.heappushpop(result_heaps[query_id], (score, corpus_id))

        for qid in result_heaps:
            for score, corpus_id in result_heaps[qid]:
                self.results[qid][corpus_id] = score
        
        return self.results 
    
    def topk_vanilla(self, distance_vector, k):
        # Get top-k values
        top_k_values, top_k_idx = torch.topk(distance_vector, k=k, dim=1, largest=True, sorted=True)
        top_k_values = top_k_values.squeeze(0).cpu().tolist()
        top_k_idx = top_k_idx.squeeze(0).cpu().tolist()
        return top_k_values, top_k_idx

    def default_cos_topk(self, query_embedding, sub_corpus_embeddings, k):
        # Compute similarites using either cosine-similarity or dot product
        cos_scores = cos_sim(query_embedding, sub_corpus_embeddings)
        cos_scores[torch.isnan(cos_scores)] = -1
        return self.topk_vanilla(cos_scores, k)
    
    def default_dot_topk(self, query_embedding, sub_corpus_embeddings, k):
        dot_scores = dot_score(query_embedding, sub_corpus_embeddings)
        return self.topk_vanilla(dot_scores, k)
    
    def default_euclidean_topk(self, query_embedding, sub_corpus_embeddings, k):
        euclidean_distance = torch.norm(query_embedding - sub_corpus_embeddings, dim=1)
        return self.topk_vanilla(euclidean_distance, k)
