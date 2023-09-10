import crypten
import torch
import pickle
import crypten.mpc as mpc
# import crypten.common.functions.maximum
import crypten.communicator as comm
from crypten.config import cfg
import pandas as pd
from torch.nn.functional import cosine_similarity
import time
import numpy as np
import math
from crypten.common.functions.maximum import _one_hot_to_index


#initialize crypten
crypten.init()
#Disables OpenMP threads -- needed by @mpc.run_multiprocess which uses fork
torch.set_num_threads(1)



def cosine_similarity_mpc_naive(A: torch.Tensor, B: torch.Tensor) -> bytes | None:
    """
    Computes the cosine similarity between two tensors A and B using crypten. The magnitude of A and B are computed in-function.
    """
    # secret-share A, B
    A_enc = crypten.cryptensor(A, ptype=crypten.mpc.arithmetic)
    B_enc = crypten.cryptensor(B, ptype=crypten.mpc.arithmetic)

    dot_product = A_enc.matmul(B_enc)

    # Compute the magnitudes of A and B
    mag_A_enc = (A_enc * A_enc).sum(dim=1, keepdim=True).sqrt()
    mag_B_enc = (B_enc * B_enc).sum(dim=0, keepdim=True).sqrt()

    # Compute the cosine similarity
    cosine_sim = (dot_product / (mag_A_enc * mag_B_enc)).get_plain_text()
    
    cosine_sim_binary = pickle.dumps(cosine_sim)
    return cosine_sim_binary

def preprocess_cosine_similarity_mpc_opt(A: torch.Tensor, B: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
    """A helper function before calling cosine_similarity_mpc_pass_in_mags. Precomputes the magnitudes of A and B, and their reciprocals."""
    A_mag_recip = torch.sqrt(torch.sum(A * A, dim=1, keepdim=True))**(-1)
    B_mag_recip = torch.sqrt(torch.sum(B * B, dim=0, keepdim=True))**(-1)
    return [A_mag_recip, B_mag_recip]

def cosine_similarity_mpc_pass_in_mags(A: torch.Tensor, B: torch.Tensor, A_mag_recip: torch.Tensor, B_mag_recip: torch.Tensor) -> bytes | None:
    """
    Computes the cosine similarity between two tensors A and B using crypten. The magnitude of A and B are pre-computed and passed in (different from cosine_similarity_mpc_naive).

    A_mag_recip and B_mag_recip are the reciprocals of the magnitudes of A and B, stored in single element tensors.
    """

    # secret-share A, B
    A_enc = crypten.cryptensor(A, ptype=crypten.mpc.arithmetic)
    B_enc = crypten.cryptensor(B, ptype=crypten.mpc.arithmetic)

    A_mag_recip_enc = crypten.cryptensor(A_mag_recip, ptype=crypten.mpc.arithmetic)
    B_mag_recip_enc = crypten.cryptensor(B_mag_recip, ptype=crypten.mpc.arithmetic)

    # Compute the dot product of A and B

    dot_product = A_enc.matmul(B_enc)
    cosine_sim = (dot_product * (A_mag_recip_enc * B_mag_recip_enc))

    cosine_sim_binary = pickle.dumps(cosine_sim.get_plain_text())
    return cosine_sim_binary


def cosine_similarity_mpc_opt(A: torch.Tensor, B: torch.Tensor) -> bytes | None:
    """
    Pre-normalizes for efficiency.
    This is secure because the user owns A and can do this locally, and the parties can jointly pre-process norm(B)
    which is a one-time MPC operation
    """

    A_normed = A / torch.sqrt(torch.sum(A * A, dim=1, keepdim=True))
    B_normed = B / torch.sqrt(torch.sum(B * B, dim=1, keepdim=True))

    # secret-share A_normed, B_normed
    A_normed_enc = crypten.cryptensor(A_normed, ptype=crypten.mpc.arithmetic)
    B_normed_enc = crypten.cryptensor(B_normed, ptype=crypten.mpc.arithmetic)

    # Compute the dot product of A and B
    cosine_sim = A_normed_enc.matmul(B_normed_enc)

    cosine_sim_binary = pickle.dumps(cosine_sim.get_plain_text())
    return cosine_sim_binary

def cosine_similarity_mpc_opt2(A: torch.Tensor, B: torch.Tensor) -> bytes | None:
    """
    The magnitude of A and B are calculated in MPC.
    """
    # secret-share A, B
    A_enc = crypten.cryptensor(A, ptype=crypten.mpc.arithmetic)
    B_enc = crypten.cryptensor(B, ptype=crypten.mpc.arithmetic)

    A_mag_recip = torch.sqrt(torch.sum(A * A, dim=1, keepdim=True))**(-1)
    B_mag_recip = torch.sqrt(torch.sum(B * B, dim=0, keepdim=True))**(-1)
    A_mag_recip_enc = crypten.cryptensor(A_mag_recip, ptype=crypten.mpc.arithmetic)
    B_mag_recip_enc = crypten.cryptensor(B_mag_recip, ptype=crypten.mpc.arithmetic)

    # Compute the dot product of A and B
    dot_product = A_enc.matmul(B_enc)
    cosine_sim = (dot_product * (A_mag_recip_enc * B_mag_recip_enc)).get_plain_text()
    
    cosine_sim_binary = pickle.dumps(cosine_sim)
    return cosine_sim_binary

def dot_score_mpc(A: torch.Tensor, B: torch.Tensor) -> bytes:
    """
    Computes the dot-product dot_prod(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = dot_prod(a[i], b[j])
    """
    # secret-share A, B
    A_enc = crypten.cryptensor(A, ptype=crypten.mpc.arithmetic)
    B_enc = crypten.cryptensor(B, ptype=crypten.mpc.arithmetic)

    # Compute the dot product of A and B
    dot_product = A_enc.matmul(B_enc)

    dot_score_binary = pickle.dumps(dot_product.get_plain_text())
    return dot_score_binary

def euclidean_mpc(A: torch.Tensor, B: torch.Tensor) -> bytes:
    """
    Computes the euclidean distance between two tensors A and B using crypten.
    :return: The euclidean distance without having square rooted it.

    Expalantion of missing sqrt: crypten  
    """
    
    # secret-share A, B
    A_enc = crypten.cryptensor(A, ptype=crypten.mpc.arithmetic)
    B_enc = crypten.cryptensor(B.t(), ptype=crypten.mpc.arithmetic)
    
    # Compute the differences
    diff_matrix = A_enc - B_enc

    # Get the euclidean norm
    # euclidean_distance = diff_matrix.norm(dim=1)
    euclidean_distance = diff_matrix.square().sum(dim=1)

    euclidean_distance_binary = pickle.dumps(euclidean_distance.get_plain_text())
    return euclidean_distance_binary


# Now we deal with top-k & max calculations

def argmax_mpc_tobin(v: torch.Tensor) -> bytes | None:
    v_enc = crypten.cryptensor(v, ptype=crypten.mpc.arithmetic)
    res = v_enc.argmax(one_hot=False)
    return pickle.dumps(res.get_plain_text())

def top_k_mpc_tobin(v: torch.Tensor, k: int) -> bytes | None:
    v_enc = crypten.cryptensor(v, ptype=crypten.mpc.arithmetic)
    top_k = []
    for i in range(k):
        max_one_hot = v_enc.argmax(one_hot=True)
        top_k.append(_one_hot_to_index(max_one_hot, dim=0, keepdim=False).unsqueeze(0))
        v_enc = v_enc - 100*max_one_hot # This will top the current max from being the top (could be done better)
    top_k = crypten.cat(top_k)
    return pickle.dumps(top_k.get_plain_text())

def tobin_top_k_mpc_return_embedding_vectors(v: torch.Tensor, k: int, B: torch.Tensor) -> bytes:
    v_enc = crypten.cryptensor(v, ptype=crypten.mpc.arithmetic)
    B_enc = crypten.cryptensor(B, ptype=crypten.mpc.arithmetic)

    # First we get the argmax as a one hot vector and multiply it by B to get the embedding vector out
    max_one_hot = v_enc.argmax(one_hot=True)
    max_vector = B_enc @ max_one_hot # This gives us a (1, embedding_dim) vector (that we can return)
    max_vector = max_vector.unsqueeze(0) # Reshape for concatenation
    if k==1:
        return pickle.dumps(max_vector.get_plain_text())
    
    max_vectors = [max_vector]
    for i in range(1,k):
        v_enc = v_enc - 100*max_one_hot # This will top the current max from being the top (could be done better)
        # Now we repeat and concat
        max_one_hot = v_enc.argmax(one_hot=True)
        new_max_vector = B_enc @ max_one_hot
        max_vectors.append(new_max_vector.unsqueeze(0))
    max_vectors = crypten.cat(max_vectors)
    return pickle.dumps(max_vectors.get_plain_text())


# We need to wrap all these functions in MPC wrappers for real use. 

from functools import wraps
def dectorate_mpc(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        return func(*args, **kwargs)
    return mpc.run_multiprocess(world_size=2)(wrapped)

# We now want to make a wrapper that combines the distance calculation with the top-k calculation and wraps in. It will take in a top_k function and a distance function and return a new function that does both after being wrapped in dectorate_mpc.

def mpc_distance_top_k(distance_func, top_k_func):
    """
    Creates a function that calculates distances between tensors A and B using `distance_func` and
    then finds the top-k indices using `top_k_func`. The returned function is wrapped for MPC.
    
    Args:
    - distance_func: function to compute the distance between tensors
    - top_k_func: function to compute the top-k indices/values
    
    Returns:
    A function that first computes distance and then gets the top-k results.
    """

    @dectorate_mpc
    def mpc_distance_and_top_k(A: torch.Tensor, B: torch.Tensor, k: int) -> bytes:
        distance_results = distance_func(A, B)
        top_k_results = top_k_func(distance_results, k)
        return top_k_results

    return mpc_distance_and_top_k
