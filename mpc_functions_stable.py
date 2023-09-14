import crypten
import torch
import pickle
import crypten.mpc as mpc
import time
from crypten.common.functions.maximum import _one_hot_to_index
from crypten.mpc.mpc import MPCTensor


#initialize crypten
crypten.init()
#Disables OpenMP threads -- needed by @mpc.run_multiprocess which uses fork
torch.set_num_threads(1)


def cosine_similarity_mpc_opt(A: torch.Tensor, B: torch.Tensor) -> MPCTensor:
    """
    Pre-normalizes for efficiency.
    This is secure because the user owns A and can do this locally, and the parties can jointly pre-process norm(B)
    which is a one-time MPC operation
    """

    A_normed = A / torch.norm(A)
    B_normed = B / torch.norm(B)

    # secret-share A_normed, B_normed
    A_normed_enc = crypten.cryptensor(A_normed, ptype=crypten.mpc.arithmetic)
    B_normed_enc = crypten.cryptensor(B_normed, ptype=crypten.mpc.arithmetic)

    # Compute the dot product of A and B
    cosine_sim = A_normed_enc.matmul(B_normed_enc)

    return cosine_sim

def cosine_similarity_mpc_opt2(A: torch.Tensor, B: torch.Tensor) -> MPCTensor:
    """
    The magnitude of A and B are calculated in MPC.
    """
    # secret-share A, B
    A_enc = crypten.cryptensor(A, ptype=crypten.mpc.arithmetic)
    B_enc = crypten.cryptensor(B, ptype=crypten.mpc.arithmetic)

    A_mag_recip = A.norm()**(-1)
    B_mag_recip = B.norm()**(-1)
    A_mag_recip_enc = crypten.cryptensor(A_mag_recip, ptype=crypten.mpc.arithmetic)
    B_mag_recip_enc = crypten.cryptensor(B_mag_recip, ptype=crypten.mpc.arithmetic)

    # Compute the dot product of A and B
    dot_product = A_enc.matmul(B_enc)
    cosine_sim = (dot_product * (A_mag_recip_enc * B_mag_recip_enc))
    return cosine_sim

def dot_score_mpc(A: torch.Tensor, B: torch.Tensor) -> MPCTensor:
    """
    Computes the dot-product dot_prod(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = dot_prod(a[i], b[j])
    """
    # secret-share A, B
    A_enc = crypten.cryptensor(A, ptype=crypten.mpc.arithmetic)
    B_enc = crypten.cryptensor(B, ptype=crypten.mpc.arithmetic)

    # Compute the dot product of A and B
    dot_product = A_enc.matmul(B_enc)
    return dot_product

def euclidean_mpc(A: torch.Tensor, B: torch.Tensor) -> MPCTensor:
    """
    Computes the euclidean distance between two tensors A and B using crypten.
    :return: The euclidean distance without having square rooted it.

    Explanation of missing sqrt: crypten  
    """
    
    # secret-share A, B
    A_enc = crypten.cryptensor(A, ptype=crypten.mpc.arithmetic)
    B_enc = crypten.cryptensor(B.t(), ptype=crypten.mpc.arithmetic)
    
    # Compute the differences
    diff_matrix = A_enc - B_enc

    # Get the euclidean norm
    # euclidean_distance = diff_matrix.norm(dim=1)
    euclidean_distance = diff_matrix.square().sum(dim=1)

    return euclidean_distance


# Now we deal with top-k & max calculations

def _argmax_mpc(v_enc: MPCTensor) -> MPCTensor:
    return v_enc.argmax(one_hot=False)

def argmax_mpc_tobin(v: torch.Tensor) -> bytes:
    v_enc = crypten.cryptensor(v, ptype=crypten.mpc.arithmetic)
    return _argmax_mpc(v_enc)

def _top_k_mpc_tobin(v_enc: MPCTensor, k: int) -> MPCTensor:
    top_k = []
    for i in range(k):
        max_one_hot = v_enc.argmax(one_hot=True)
        top_k.append(_one_hot_to_index(max_one_hot, dim=0, keepdim=False).unsqueeze(0))
        v_enc = v_enc - 100*max_one_hot # This will top the current max from being the top (could be done better)
    top_k = crypten.cat(top_k)
    return top_k

def top_k_mpc_tobin(v: torch.Tensor, k: int) -> bytes:
    v_enc = crypten.cryptensor(v, ptype=crypten.mpc.arithmetic)
    return _top_k_mpc_tobin(v_enc, k)

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
    """This is a simple wrapped function that will return the exact same function but wrapped in mpc.run_multiprocess(world_size=2). ` @wraps(func)` serves to keep function metadata.
    """
    @wraps(func)
    def wrapped_to_multiprocess(*args, **kwargs):
        return func(*args, **kwargs)
    return mpc.run_multiprocess(world_size=2)(wrapped_to_multiprocess)


# When we use a function on it's own we don't want it to return the pickle, so when we're testing solo functions, we wrap them so they return binary to us for processing.

def handle_binary(func, mpc=False):
    """This function takes in any crypten function that returns an MPCTensor and returns a *function* that handles the decryption and returns the result as normal torch tensor. This is done in two stages, first by call pickling the result of get_plain_text() on the MPCTensor (done inside MPC), and then by unpickling the result."""

    def decrypt_result(func, mpc):
        """ This function will handle the final crypten step of decrypting the result and returning it as binary. """
        @wraps(func)
        def wrapped_to_pickle(*args, **kwargs):
            return pickle.dumps(func(*args, **kwargs).get_plain_text())
        return dectorate_mpc(wrapped_to_pickle) if mpc else wrapped_to_pickle

    pickled_function = decrypt_result(func, mpc)

    if mpc:
        @wraps(func)
        def wrapped_to_unpickle(*args, **kwargs):
            return pickle.loads(pickled_function(*args, **kwargs)[0])
    else:
        @wraps(func)
        def wrapped_to_unpickle(*args, **kwargs):
            return pickle.loads(pickled_function(*args, **kwargs))

    return wrapped_to_unpickle


# We now want to make a wrapper that combines the distance calculation with the top-k calculation and wraps in. It will take in a top_k function and a distance function and return a new function that does both after being wrapped in dectorate_mpc.

def mpc_distance_and_argmax(distance_func, argmax_func, mpc=False):
    def joint_func(A: torch.Tensor, B: torch.Tensor) -> MPCTensor:
        distance_results = distance_func(A, B)
        argmax_results = argmax_func(distance_results)
        return argmax_results 
    return handle_binary(joint_func, mpc=mpc)

def mpc_distance_and_topk(distance_func, topk_func, mpc=False):
    def joint_func(A: torch.Tensor, B: torch.Tensor, k: int) -> MPCTensor:
        distance_results = distance_func(A, B)
        topk_results = topk_func(distance_results.squeeze(), k)
        return topk_results 
    return handle_binary(joint_func, mpc=mpc)

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

# Work in progress - we need to return the distances with the top-k for batching

def top_k_mpc_with_distance(v_enc: MPCTensor, k: int) -> MPCTensor:
    top_k_max, top_k = [], []
    for i in range(k):
        max_result, max_one_hot = v_enc.max(dim=0,one_hot=True)
        top_k.append(_one_hot_to_index(max_one_hot, dim=0, keepdim=False).unsqueeze(0))
        top_k_max.append(max_result.unsqueeze(0))
        v_enc = v_enc - 100*max_one_hot # This will top the current max from being the top (could be done better)
    top_k = crypten.cat(top_k)
    top_k_max = crypten.cat(top_k_max)
    return top_k, top_k_max

def mpc_distance_top_k_with_distance_func(distance_func, top_k_func=top_k_mpc_with_distance):
    """
    An extention of `mpc_distance_top_k()` that returns the distance values as well as the indices. It also handles the binary out of the box

    A function that first computes distance and then gets the top-k results *as well as the distance values*
    """

    @dectorate_mpc
    def mpc_distance_and_top_k(A: torch.Tensor, B: torch.Tensor, k: int) -> bytes:
        distance_results = distance_func(A, B)
        top_k, top_k_max  = top_k_func(distance_results.squeeze(0), k)
        return pickle.dumps((top_k.get_plain_text(), top_k_max.get_plain_text())) 

    return mpc_distance_and_top_k