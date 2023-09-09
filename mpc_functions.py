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
import logging

logger = logging.getLogger("prag.mpc_functions")
logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.INFO)

# TODO: currently does not work if DIM1[0] is not 1. Something about the mag computation. Most likely because multiplying mag_A (a scalar) times mag_B (a vector) does not work if the shapes are different? Need different code most likely.
# TODO: naive MPC accuracy doesn't work anymore? unless very small vectors
# TODO: understand how playing with diff dimensions operates in terms of performance. Not sure why embedding dimen is sometimes relevant and sometimes isn't..
BENCHMARK_DIM1 = (1, 1000) # query tokens, embedding dimensionality
BENCHMARK_DIM2 = (1000, 5000) # embedding dimensionality, number of samples

#initialize crypten
crypten.init()
#Disables OpenMP threads -- needed by @mpc.run_multiprocess which uses fork
torch.set_num_threads(1)

def relative_error(y_pred, y_true):
    return torch.mean(torch.abs(y_pred - y_true) / torch.abs(y_true))

def preprocess_cosine_similarity_mpc_naive(args) -> [torch.Tensor, torch.Tensor]:
    if (args != []):
        query_vector = args[0]
        database_vectors = args[1]
    else:
        query_vector = torch.randn(*BENCHMARK_DIM1)
        database_vectors = torch.randn(*BENCHMARK_DIM2)
    return [query_vector, database_vectors]

@mpc.run_multiprocess(world_size=2)
def cosine_similarity_mpc_naive(A: torch.Tensor, B: torch.Tensor) -> bytes | None:
    """
    Computes the cosine similarity between two tensors A and B using crypten. The magnitude of A and B are computed in-function.
    """
    # secret-share A, B
    A_enc = crypten.cryptensor(A, ptype=crypten.mpc.arithmetic)
    B_enc = crypten.cryptensor(B, ptype=crypten.mpc.arithmetic)
    logger.debug(f"A={A_enc.shape}")
    logger.debug(f"B={B_enc.shape}")
    # Compute the dot product of A and B
    dot_product = A_enc.matmul(B_enc)

    # Compute the magnitudes of A and B
    mag_A_enc = (A_enc * A_enc).sum(dim=1, keepdim=True).sqrt()
    mag_B_enc = (B_enc * B_enc).sum(dim=0, keepdim=True).sqrt()
    logger.debug(f"dot={dot_product.shape}")
    logger.debug(f"A_magrec={mag_A_enc.shape}")
    logger.debug(f"B_magrec={mag_B_enc.shape}")

    # Compute the cosine similarity
    cosine_sim = (dot_product / (mag_A_enc * mag_B_enc)).get_plain_text()
    
    cosine_sim_binary = pickle.dumps(cosine_sim)
    return cosine_sim_binary

@mpc.run_multiprocess(world_size=2)
def cosine_similarity_mpc_opt(A: torch.Tensor, B: torch.Tensor) -> bytes | None:
    """
    Computes the cosine similarity between two tensors A and B using crypten. Pre-normalizes for efficiency
    This is secure because the user owns A and can do this locally, and the parties can jointly pre-process norm(B)
    which is a one-time MPC operation
    """

    A_normed = A / torch.sqrt(torch.sum(A * A, dim=1, keepdim=True))
    B_normed = B / torch.sqrt(torch.sum(B * B, dim=1, keepdim=True))

    # secret-share A_normed, B_normed
    A_normed_enc = crypten.cryptensor(A_normed, ptype=crypten.mpc.arithmetic)
    B_normed_enc = crypten.cryptensor(B_normed, ptype=crypten.mpc.arithmetic)
    logger.debug(f"A={A_normed_enc.shape}")
    logger.debug(f"B={B_normed_enc.shape}")

    # Compute the dot product of A and B
    cosine_sim = A_normed_enc.matmul(B_normed_enc)
    logger.debug(f"dot={cosine_sim.shape}")

    cosine_sim_binary = pickle.dumps(cosine_sim.get_plain_text())
    return cosine_sim_binary

# This version is supposed to be slower, but has similar performance and much better out-of-the-box accuracy
@mpc.run_multiprocess(world_size=2)
def cosine_similarity_mpc_opt2(A: torch.Tensor, B: torch.Tensor) -> bytes | None:
    """
    Computes the cosine similarity between two tensors A and B using crypten. The magnitude of A and B are pre-computed and passed in (different from cosine_similarity_mpc_naive).

    A_mag_recip and B_mag_recip are the reciprocals of the magnitudes of A and B, stored in single element tensors.
    """
    # secret-share A, B
    A_enc = crypten.cryptensor(A, ptype=crypten.mpc.arithmetic)
    B_enc = crypten.cryptensor(B, ptype=crypten.mpc.arithmetic)
    logger.debug(f"A={A_enc.shape}")
    logger.debug(f"B={B_enc.shape}")

    A_mag_recip = torch.sqrt(torch.sum(A * A, dim=1, keepdim=True))**(-1)
    B_mag_recip = torch.sqrt(torch.sum(B * B, dim=0, keepdim=True))**(-1)
    A_mag_recip_enc = crypten.cryptensor(A_mag_recip, ptype=crypten.mpc.arithmetic)
    B_mag_recip_enc = crypten.cryptensor(B_mag_recip, ptype=crypten.mpc.arithmetic)

    # Compute the dot product of A and B
    dot_product = A_enc.matmul(B_enc)
    logger.debug(f"dot={dot_product.shape}")
    logger.debug(f"A_magrec={A_mag_recip_enc.shape}")
    logger.debug(f"B_magrec={B_mag_recip_enc.shape}")
    # Compute the cosine similarity
    cosine_sim = (dot_product * (A_mag_recip_enc * B_mag_recip_enc)).get_plain_text()
    
    cosine_sim_binary = pickle.dumps(cosine_sim)
    return cosine_sim_binary
    
@mpc.run_multiprocess(world_size=2)
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

# This is much slower, like 5 times slower for k=10
@mpc.run_multiprocess(world_size=2)
def approx_top_k_mpc_naive(v: torch.Tensor, k: int) -> bytes | None:
    # Utilizes the vanilla crypten max() function to compute the approx top_k
    v_enc = crypten.cryptensor(v, ptype=crypten.mpc.arithmetic)
    N = v.size(0)
    bucket_size = int(N/k) # TODO: don't ignore the remainder (i.e, if N/k isn't exact)
    top_k_list = []
    for i in range(k):
        top_k_list.append(v_enc[i * bucket_size: (i+1) * bucket_size].max().unsqueeze(0))
    top_k = crypten.cat(top_k_list)

    return pickle.dumps(top_k.get_plain_text())

@mpc.run_multiprocess(world_size=2)
def approx_top_k_mpc(v: torch.Tensor, k: int) -> bytes | None:
    v_enc = crypten.cryptensor(v, ptype=crypten.mpc.arithmetic)
    top_k = _approx_top_k_log_reduction(v_enc, k)
    return pickle.dumps(top_k.get_plain_text())

@mpc.run_multiprocess(world_size=2)
def approx_top_k_multiround_mpc(v: torch.Tensor, k: int, permutations: int) -> bytes | None:
    vs_shuffled_list = []
    for i in range(permutations):
        vs_shuffled_list.append(v[torch.randperm(v.size(0))])
    vs_shuffled = torch.stack(vs_shuffled_list)
    vs_enc = crypten.cryptensor(vs_shuffled, ptype=crypten.mpc.arithmetic)
    
    # For now, easier to return max and see if it's even worth the effort
    # top_k, _ = vs_enc.max(dim=1)
    top_k = _approx_top_k_log_reduction(vs_enc, k, dim=1)
    
    # top_k = _approx_top_k_log_reduction(v_enc, k)
    return pickle.dumps(top_k.get_plain_text())

'''
Gets a tensor of size (n, 1) and returns a vector of size (m, n/m, 1)
'''
def stack_em_vectors(v: torch.Tensor, m: int):
    n = v.size(0)
    # Calculate group size
    group_size = n // m

    # Split the tensor
    splitted_tensors = torch.split(v, group_size)


    # Handle edge case
    if n % m != 0:
        last_tensor = splitted_tensors[-1]
        padding = torch.zeros(group_size - last_tensor.shape[0], 1)
        splitted_tensors = splitted_tensors[:-1] + (torch.cat([last_tensor, padding]),)

    # Stack them up
    result_tensor = torch.stack(splitted_tensors)
    return result_tensor

@mpc.run_multiprocess(world_size=2)
def max_mpc(v: torch.Tensor) -> bytes | None:
    v_enc = crypten.cryptensor(v, ptype=crypten.mpc.arithmetic)
    return pickle.dumps(v_enc.max().get_plain_text()) # TODO: crypten max to debug?

@mpc.run_multiprocess(world_size=2)
def max_mpc2(v: torch.Tensor) -> bytes | None:
    v_enc = crypten.cryptensor(v, ptype=crypten.mpc.arithmetic)
    res = _max_helper_log_reduction(v_enc)
    return pickle.dumps(res.get_plain_text())

@mpc.run_multiprocess(world_size=2)
def batch_max_mpc(v: torch.Tensor, bucket_size: int) -> bytes | None:
    stack = stack_em_vectors(v, 5)
    vs_enc = crypten.cryptensor(stack, ptype=crypten.mpc.arithmetic)
    
    ## Normal log reduction
    # res = _max_helper_log_reduction(vs_enc, dim=1)

    ## pairwise
    with cfg.temp_override({"functions.max_method": "pairwise"}):
        res, _ = vs_enc.max(dim=1)

    return pickle.dumps(res.get_plain_text())

# This max-of-max algorithm reduces the search space to N/bucket_size in each iteration, until limit is reached and then it computes the max in one go
# This is a plaintext algorithm to help ensure correctness. The MPC algorithm is similar
# TODO: formal analysis of the complexity, especially in the number of total comparisons and the number of maximum single-round comparisons
def bucketize_and_max(v: torch.Tensor, bucket_size: int, limit: int):
    N = v.size(0)
    curr_v = v.clone()
    depth = 0
    while curr_v.size(0) > limit:
        n_buckets = int(N/bucket_size)
        logger.debug(f"depth={depth}, n_buckets={n_buckets}")
        buckets = [curr_v[i * bucket_size: (i+1) * bucket_size] for i in range(n_buckets)]
        stack = torch.stack(buckets)
        curr_v, _ = stack.max(dim=1)
        depth += 1
        N = n_buckets

    return curr_v.max()

def test_stack_em_vectors():
    n = 20
    v = torch.arange(n).view(n, 1)
    stack = stack_em_vectors(v, 5)
    logger.debug(stack)
    logger.debug(stack.shape)

def cosine_sim_naive_test():
    logger.info("Running test: cosine_sim_naive_test")
    # Load the data from CSV files
    query_vector = pd.read_csv('datasets/query_vector.csv').values
    database_vectors = pd.read_csv('datasets/D.csv').values

    # Convert the data to PyTorch tensors
    query_vector = torch.tensor(query_vector, dtype=torch.float32).t()
    database_vectors = torch.tensor(database_vectors, dtype=torch.float32)

    logger.debug(f"query vector shape = {query_vector.shape}")
    logger.debug(f"database shape = {database_vectors.shape}")

    # Compute the cosine similarity between the query vector and all database vectors
    cosine_similarities = cosine_similarity(query_vector, database_vectors.t())

    logger.debug(f"cosine similarities torch native={cosine_similarities}")
    logger.debug(f"cosine similarities torch native shape={cosine_similarities.shape}")

    cosine_similarities2_binary = cosine_similarity_mpc_naive(query_vector, database_vectors)
    cosine_similarities2 = pickle.loads(cosine_similarities2_binary[0]) # remove [0] for single threaded
    logger.debug(cosine_similarities2)

    if torch.allclose(cosine_similarities, cosine_similarities2, atol=0.02):
        logger.info("They are approx equal")
    else:
        logger.info("They are NOT approx equal")

    if torch.equal(cosine_similarities, cosine_similarities2):
        logger.info("They are EXACTLY equal")
    else:
        logger.info("They are NOT EXACTLY equal")

    err = relative_error(cosine_similarities2, cosine_similarities)
    logger.info(f"Average precision loss: {err*100}%")

def cosine_sim_opt_test(just_dot_it=True):
    # Load the data from CSV files
    query_vector = pd.read_csv('datasets/query_vector.csv').values
    database_vectors = pd.read_csv('datasets/D.csv').values

    # Convert the data to PyTorch tensors
    query_vector = torch.tensor(query_vector, dtype=torch.float32).t()
    database_vectors = torch.tensor(database_vectors, dtype=torch.float32)

    logger.debug(query_vector.shape)
    logger.debug(database_vectors.shape)
    
    # Compute the cosine similarity between the query vector and all database vectors
    cosine_similarities = cosine_similarity(query_vector, database_vectors.t())

    logger.debug(cosine_similarities)
    logger.debug(cosine_similarities.shape)

    if (just_dot_it):
        logger.info("Running test: cosine_sim_opt_test")
        cosine_similarities2_binary = cosine_similarity_mpc_opt(query_vector, database_vectors)
    else:
        logger.info("Running test: cosine_sim_opt2_test")
        cosine_similarities2_binary = cosine_similarity_mpc_opt2(query_vector, database_vectors)

    cosine_similarities2 = pickle.loads(cosine_similarities2_binary[0]) # remove [0] for single threaded
    logger.debug(cosine_similarities2)

    if torch.allclose(cosine_similarities, cosine_similarities2, atol=0.02):
        logger.info("They are approx equal")
    else:
        logger.info("They are NOT approx equal")

    if torch.equal(cosine_similarities, cosine_similarities2):
        logger.info("They are EXACTLY equal")
    else:
        logger.info("They are NOT EXACTLY equal")

    err = relative_error(cosine_similarities2, cosine_similarities)
    logger.info(f"(Optimized) Average precision loss: {err*100}%")

# def benchmark_crypten_max(num_trials=3, size=1024):
def benchmark_crypten_max(num_trials=2, size=2**10):
    # Goal is to get a better sense of how fast crypten's max functions work
    times = {
        'log_reduction': np.zeros(num_trials),
        # 'double_log_reduction': np.zeros(num_trials),
        # 'accelerated_cascade': np.zeros(num_trials),
        # 'local_log_reduction': np.zeros(num_trials),
        # 'recursive_pairwise': np.zeros(num_trials),
        # 'pairwise': np.zeros(num_trials),
        'batch_max': np.zeros(num_trials),
        'top_k_v1': np.zeros(num_trials),
        'top_k_v2': np.zeros(num_trials),
    }
    results_mpc = {
        'log_reduction': [],
        # 'double_log_reduction': [],
        # 'accelerated_cascade': [],
        # 'local_log_reduction': [],
        # 'recursive_pairwise': [],
        # 'pairwise': [],
        'batch_max': [],
        # 'top_k_v1': [],
        # 'top_k_v2': [],
    }
    errs = {
        'log_reduction': [],
        # 'double_log_reduction': [],
        # 'accelerated_cascade': [],
        # 'local_log_reduction': [],
        # 'recursive_pairwise': [],
        # 'pairwise': [],
        'batch_max': [],
        'top_k_v1': [],
        'top_k_v2': [],
    }

    for i in range(num_trials):
        v = torch.randn(size, 1)
        # v_enc = crypten.cryptensor(v, ptype=crypten.mpc.arithmetic)
        vmax = v.max()
        for k in results_mpc.keys():
            start_time = time.time()
            if (k == 'local_log_reduction'):
                res_binary = max_mpc2(v)
            elif (k == 'recursive_pairwise'):
                res_binary = bucketize_and_max_mpc(v, 2**4, 2**4, is_pairwise=False)
                # res_binary = bucketize_and_max_mpc(v, 32, int(size**(1/2))+1)
                # res_binary = bucketize_and_max_mpc(v, 32, int(size**(1/4))+1)
            elif (k == 'batch_max'):
                batch_max_mpc(v, 2**5)
            elif (k == 'top_k_v1'):
                res_binary = approx_top_k_mpc(v, 10)
            elif (k == 'top_k_v2'):
                res_binary = approx_top_k_multiround_mpc(v, 10, 10)
            else:
                cfg.functions.max_method = k
                res_binary = max_mpc(v)
            
            end_time = time.time()
            res = pickle.loads(res_binary[0])
            
            times[k][i] = end_time - start_time
            results_mpc[k].append(res)
            errs[k].append(relative_error(res, vmax))

    for k in results_mpc.keys():
        avg_time = np.mean(times[k])
        logger.info(f"Results for method={k}")
        logger.info(f"Times={times[k]}")
        logger.info(f"Average running time: {avg_time} seconds")

        avg_error = np.mean(errs[k])
        logger.info(f"Errors={errs[k]}")
        logger.info(f"Average precision loss: {avg_error*100}%")

def run_benchmark(func, preprocess_func, args, num_trials=5):
    logger.info(f"Starting benchmark for func: {func}")
    ## Benchmark time
    # record the time it takes to run the function
    times = np.zeros(num_trials)
    results_mpc = []
    results = []
    errs = []

    for i in range(num_trials):
        new_args = preprocess_func(args)

        start_time = time.time()
        res_binary = func(*new_args)
        end_time = time.time()
        times[i] = end_time - start_time

        # res_mpc = pickle.loads(res_binary) # removed [0]
        res_mpc = pickle.loads(res_binary[0])
        # print(f"res_mpc shape = {res_mpc.shape}")
        results_mpc.append(res_mpc)
        res = cosine_similarity(new_args[0], new_args[1].t())
        # print(f"res shape = {res.shape}")
        results.append(res)

        if torch.allclose(res, res_mpc, atol=0.02):
            logger.info("They are approx equal")
        else:
            logger.info("They are NOT approx equal")

        errs.append(relative_error(res_mpc, res))

    avg_time = np.mean(times)
    logger.info(f"Times={times}")
    logger.info(f"Average running time: {avg_time} seconds")

    avg_error = np.mean(errs)
    logger.info(f"Errors={errs}")
    logger.info(f"Average precision loss: {avg_error*100}%")

def weird_crypten_tests():
    v = torch.randn(10, 1)
    v_enc = crypten.cryptensor(v, ptype=crypten.mpc.arithmetic)
    print(v_enc._tensor.data.dtype)


## TODO: TEMP
def _compute_pairwise_comparisons_for_steps(input_tensor, dim, steps):
    """
    Helper function that does pairwise comparisons by splitting input
    tensor for `steps` number of steps along dimension `dim`.
    """
    enc_tensor_reduced = input_tensor.clone()
    for _ in range(steps):
        m = enc_tensor_reduced.size(dim)
        x, y, remainder = enc_tensor_reduced.split([m // 2, m // 2, m % 2], dim=dim)
        logger.debug(f"x={x.shape}, y={y.shape}, remainder={remainder.shape}, cfg={cfg.functions.max_method}")
        # NOTE: this is not N^2 because it's a simple all-to-all comparison
        res = x >= y
        pairwise_max = crypten.where(res, x, y)
        enc_tensor_reduced = crypten.cat([pairwise_max, remainder], dim=dim)
    return enc_tensor_reduced


def _max_helper_log_reduction(enc_tensor, dim=None):
    """Returns max along dim `dim` using the log_reduction algorithm"""
    if enc_tensor.dim() == 0:
        return enc_tensor
    input, dim_used = enc_tensor, dim
    if dim is None:
        dim_used = 0
        input = enc_tensor.flatten()
    n = input.size(dim_used)  # number of items in the dimension
    steps = int(math.log(n))
    enc_tensor_reduced = _compute_pairwise_comparisons_for_steps(input, dim_used, steps)

    # compute max over the resulting reduced tensor with n^2 algorithm
    # note that the resulting one-hot vector we get here finds maxes only
    # over the reduced vector in enc_tensor_reduced, so we won't use it
    logger.debug(f"running pairwise with size={enc_tensor_reduced.shape}")
    with cfg.temp_override({"functions.max_method": "pairwise"}):
        enc_max_vec, enc_one_hot_reduced = enc_tensor_reduced.max(dim=dim_used)
    return enc_max_vec

def _approx_top_k_log_reduction(enc_tensor, k, dim=None):
    """Returns max along dim `dim` using the log_reduction algorithm"""
    if enc_tensor.dim() == 0:
        return enc_tensor
    input, dim_used = enc_tensor, dim
    if dim is None:
        dim_used = 0
        input = enc_tensor.flatten()
    n = input.size(dim_used)  # number of items in the dimension
    steps = int(math.log2(n/k)) # TODO: handle edge cases..
    logger.debug(f"There are {n} documents and we want to get {k} of which. So we will run {steps} steps to reduce the set to: {n/2**steps}.")  
    enc_tensor_reduced = _compute_pairwise_comparisons_for_steps(input, dim_used, steps)
    return enc_tensor_reduced

if __name__ == "__main__":
    # test_stack_em_vectors()
    # bucketize_max_test()
    benchmark_crypten_max()

    # cosine_sim_naive_test()
    # cosine_sim_opt_test()
    # cosine_sim_opt_test(just_dot_it=False)

    # run_benchmark(cosine_similarity_mpc_naive, preprocess_cosine_similarity_mpc_naive, [], num_trials=5)
    # run_benchmark(cosine_similarity_mpc_opt, preprocess_cosine_similarity_mpc_naive, [], num_trials=5)
    # run_benchmark(cosine_similarity_mpc_opt2, preprocess_cosine_similarity_mpc_naive, [], num_trials=5)
