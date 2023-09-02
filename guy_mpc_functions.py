import crypten
import torch
import pickle
import crypten.mpc as mpc
import crypten.communicator as comm
import pandas as pd
from torch.nn.functional import cosine_similarity
import time
import numpy as np

import logging
# logging.basicConfig(level=logging.DEBUG)
IS_BENCHMARK = False
# TODO: currently does not work if DIM1[0] is not 1. Something about the mag computation. Most likely because multiplying mag_A (a scalar) times mag_B (a vector) does not work if the shapes are different? Need different code most likely.
# TODO: naive MPC accuracy doesn't work anymore? unless very small vectors
BENCHMARK_DIM1 = (1, 100) # query tokens, embedding dimensionality
BENCHMARK_DIM2 = (100, 10) # embedding dimensionality, number of samples

#initialize crypten
crypten.init()
#Disables OpenMP threads -- needed by @mpc.run_multiprocess which uses fork
torch.set_num_threads(1)

def relative_error(y_pred, y_true):
    return torch.mean(torch.abs(y_pred - y_true) / torch.abs(y_true))


def preprocess_cosine_similarity_mpc_naive(args):
    if (args != []):
        query_vector = args[0]
        database_vectors = args[1]
    else:
        query_vector = torch.randn(*BENCHMARK_DIM1)
        database_vectors = torch.randn(*BENCHMARK_DIM2)
    return [query_vector, database_vectors]

# @mpc.run_multiprocess(world_size=2)
def cosine_similarity_mpc_naive(A, B):
    # secret-share A, B
    A_enc = crypten.cryptensor(A, ptype=crypten.mpc.arithmetic)
    B_enc = crypten.cryptensor(B, ptype=crypten.mpc.arithmetic)
    print(f"A={A_enc.shape}")
    print(f"B={B_enc.shape}")
    # Compute the dot product of A and B
    dot_product = A_enc.matmul(B_enc)

    # Compute the magnitudes of A and B
    mag_A_enc = (A_enc * A_enc).sum(dim=1, keepdim=True).sqrt()
    mag_B_enc = (B_enc * B_enc).sum(dim=0, keepdim=True).sqrt()
    print(f"dot={dot_product.shape}")
    print(f"A_magrec={mag_A_enc.shape}")
    print(f"B_magrec={mag_B_enc.shape}")

    # Compute the cosine similarity
    cosine_sim = (dot_product / (mag_A_enc * mag_B_enc)).get_plain_text()
    
    # NOTE: from testing, this isn't very significant but slightly, maybe, improves ..
    if (not IS_BENCHMARK):
        cosine_sim_binary = pickle.dumps(cosine_sim)
        return cosine_sim_binary
    else:
        return None

def preprocess_cosine_similarity_mpc_opt(args):
    args = preprocess_cosine_similarity_mpc_naive(args)
    A = args[0]
    B = args[1]
    A_mag_recip = torch.sqrt(torch.sum(A * A, dim=1, keepdim=True))**(-1)
    B_mag_recip = torch.sqrt(torch.sum(B * B, dim=0, keepdim=True))**(-1)

    return [A, B, A_mag_recip, B_mag_recip]

# @mpc.run_multiprocess(world_size=2)
def cosine_similarity_mpc_opt(A, B, A_mag_recip, B_mag_recip):
    # secret-share A, B
    A_enc = crypten.cryptensor(A, ptype=crypten.mpc.arithmetic)
    B_enc = crypten.cryptensor(B, ptype=crypten.mpc.arithmetic)
    print(f"A={A_enc.shape}")
    print(f"B={B_enc.shape}")

    A_mag_recip_enc = crypten.cryptensor(A_mag_recip, ptype=crypten.mpc.arithmetic)
    B_mag_recip_enc = crypten.cryptensor(B_mag_recip, ptype=crypten.mpc.arithmetic)

    # Compute the dot product of A and B
    dot_product = A_enc.matmul(B_enc)
    print(f"dot={dot_product.shape}")
    print(f"A_magrec={A_mag_recip_enc.shape}")
    print(f"B_magrec={B_mag_recip_enc.shape}")
    # Compute the cosine similarity
    cosine_sim = (dot_product * (A_mag_recip_enc * B_mag_recip_enc)).get_plain_text()
    
    # NOTE: from testing, this isn't very significant but slightly, maybe, improves ..
    if (not IS_BENCHMARK):
        cosine_sim_binary = pickle.dumps(cosine_sim)
        return cosine_sim_binary
    else:
        return None

def cosine_sim_naive_test():
    # Load the data from CSV files
    query_vector = pd.read_csv('query_vector.csv').values
    database_vectors = pd.read_csv('D.csv').values

    # Convert the data to PyTorch tensors
    query_vector = torch.tensor(query_vector, dtype=torch.float32).t()
    database_vectors = torch.tensor(database_vectors, dtype=torch.float32)

    print(f"query vector shape = {query_vector.shape}")
    print(f"database shape = {database_vectors.shape}")

    # Compute the cosine similarity between the query vector and all database vectors
    cosine_similarities = cosine_similarity(query_vector, database_vectors.t())

    print(f"cosine similarities torch native={cosine_similarities}")
    print(f"cosine similarities torch native shape={cosine_similarities.shape}")

    cosine_similarities2_binary = cosine_similarity_mpc_naive(query_vector, database_vectors)
    cosine_similarities2 = pickle.loads(cosine_similarities2_binary) # removed [0]
    print(cosine_similarities2)

    if torch.allclose(cosine_similarities, cosine_similarities2, atol=0.02):
        print("They are approx equal")
    else:
        print("They are NOT approx equal")

    if torch.equal(cosine_similarities, cosine_similarities2):
        print("They are EXACTLY equal")
    else:
        print("They are NOT EXACTLY equal")

    err = relative_error(cosine_similarities2, cosine_similarities)
    print(f"Average precision loss: {err*100}%")

def cosine_sim_opt_test():
    # Load the data from CSV files
    query_vector = pd.read_csv('query_vector.csv').values
    database_vectors = pd.read_csv('D.csv').values

    # Convert the data to PyTorch tensors
    query_vector = torch.tensor(query_vector, dtype=torch.float32).t()
    database_vectors = torch.tensor(database_vectors, dtype=torch.float32)

    print(query_vector.shape)
    print(database_vectors.shape)
    query_vector, database_vectors, qv_mag_recip, db_mag_recip = preprocess_cosine_similarity_mpc_opt([query_vector, database_vectors])
    
    # Compute the cosine similarity between the query vector and all database vectors
    cosine_similarities = cosine_similarity(query_vector, database_vectors.t())

    print(cosine_similarities)
    print(cosine_similarities.shape)

    cosine_similarities2_binary = cosine_similarity_mpc_opt(query_vector, database_vectors, qv_mag_recip, db_mag_recip)
    cosine_similarities2 = pickle.loads(cosine_similarities2_binary) # [0] removed
    print(cosine_similarities2)

    if torch.allclose(cosine_similarities, cosine_similarities2, atol=0.02):
        print("They are approx equal")
    else:
        print("They are NOT approx equal")

    if torch.equal(cosine_similarities, cosine_similarities2):
        print("They are EXACTLY equal")
    else:
        print("They are NOT EXACTLY equal")

    err = relative_error(cosine_similarities2, cosine_similarities)
    print(f"(Optimized) Average precision loss: {err*100}%")

def run_benchmark(func, preprocess_func, args, num_trials=5):
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

        res_mpc = pickle.loads(res_binary[0])
        # print(f"res_mpc shape = {res_mpc.shape}")
        results_mpc.append(res_mpc)
        res = cosine_similarity(new_args[0], new_args[1].t())
        # print(f"res shape = {res.shape}")
        results.append(res)

        if torch.allclose(res, res_mpc, atol=0.02):
            print("They are approx equal")
        else:
            print("They are NOT approx equal")

        errs.append(relative_error(res_mpc, res))

    avg_time = np.mean(times)
    print(f"Times={times}")
    print(f"Average running time: {avg_time} seconds")

    avg_error = np.mean(errs)
    print(f"Errors={errs}")
    print(f"Average precision loss: {avg_error*100}%")
    

# run_benchmark(cosine_similarity_mpc_naive, preprocess_cosine_similarity_mpc_naive, [], num_trials=1)
# run_benchmark(cosine_similarity_mpc_opt, preprocess_cosine_similarity_mpc_opt, [], num_trials=1)

if __name__ == "__main__":
    cosine_sim_naive_test()
    cosine_sim_opt_test()