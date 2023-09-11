import torch, numpy as np, pandas as pd
import time
import signal

from tqdm import tqdm
import crypten
crypten.init()

from mpc_functions_stable import *

# torch.random.manual_seed(2023)

# 1. First we do a simple run through to test everything is working
query_vector = torch.rand(1,768) * 2 - 1
database_vectors = torch.rand(768, 10000) * 2 - 1

def relative_error(y_pred, y_true):
    return torch.mean(torch.abs(y_pred - y_true) / torch.abs(y_true))

def timeout_handler(signum, frame):
    raise TimeoutError("Function timed out")

def timethis(func, *args):
    start_time = time.time()
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)
        res =  func(*args)
        end_time = time.time()
    except TimeoutError:
        print("Timed out!")
        return [None], np.NaN
    signal.alarm(0) # Disable alarm
    return res, end_time - start_time

distance_result_df_list = []
db_sizes = range(500,500,1000)
embedding_dims = [256, 768, 1024, 2048, 4096, 8192]
embedding_dim = 768
for db_size in tqdm(db_sizes):
    # Make a random vector and database and scale to -1 to 1
    query_vector = torch.rand(1,embedding_dim) * 2 - 1
    database_vectors = torch.rand(embedding_dim, db_size) * 2 - 1

    # Dot score results
    real_dot, default_time = timethis(lambda a,b: a @ b, query_vector, database_vectors)
    dot_score_res_binary_mpc, timetaken = timethis(dectorate_mpc(dot_score_mpc), query_vector, database_vectors)
    dot_score_res_mpc = pickle.loads(dot_score_res_binary_mpc[0])
    error = relative_error(real_dot, dot_score_res_mpc).item()

    distance_result_df_list.append([db_size, embedding_dim, 'dot_score', error, timetaken, default_time])

    # Cosine similarity results
    real_cos_sim, default_time = timethis(lambda a,b: a @ b / (torch.norm(a) * torch.norm(b)), query_vector, database_vectors)
    cosine_sim_binary_mpc, timetaken = timethis(dectorate_mpc(cosine_similarity_mpc_opt), query_vector, database_vectors)
    cosine_sim_mpc = pickle.loads(cosine_sim_binary_mpc[0])
    error = relative_error(real_cos_sim, cosine_sim_mpc).item()

    distance_result_df_list.append([db_size, embedding_dim, 'cosine_similarity_mpc_opt', error, timetaken, default_time])

    cosine_sim_binary_mpc, timetaken = timethis(dectorate_mpc(cosine_similarity_mpc_opt2), query_vector, database_vectors)
    cosine_sim_mpc = pickle.loads(cosine_sim_binary_mpc[0])
    error = relative_error(real_cos_sim, cosine_sim_mpc).item()

    distance_result_df_list.append([db_size, embedding_dim, 'cosine_similarity_mpc_opt2', error, timetaken, default_time])


    A_mag_recip, B_mag_recip = preprocess_cosine_similarity_mpc_opt(query_vector, database_vectors)
    cosine_sim_binary_mpc, timetaken = timethis(dectorate_mpc(cosine_similarity_mpc_pass_in_mags), query_vector, database_vectors, A_mag_recip, B_mag_recip)
    cosine_sim_mpc = pickle.loads(cosine_sim_binary_mpc[0])
    error = relative_error(real_cos_sim, cosine_sim_mpc).item()

    distance_result_df_list.append([db_size, embedding_dim, 'cosine_similarity_mpc_pass_in_mags', error, timetaken, default_time])


    #  Euclidean distance check
    real_euclidian, default_time = timethis(lambda a,b: torch.norm(a - b.t(), dim=1), query_vector, database_vectors)

    euclidian_res_binary_mpc, timetaken = timethis(dectorate_mpc(euclidean_mpc), query_vector, database_vectors)
    euclidian_res_mpc = pickle.loads(euclidian_res_binary_mpc[0]).sqrt()
    error = relative_error(real_euclidian, euclidian_res_mpc).item()

    distance_result_df_list.append([db_size, embedding_dim, 'euclidean_mpc', error, timetaken, default_time])

distance_result_df = pd.DataFrame(distance_result_df_list, columns=['db_size', 'embedding_dim', 'function', 'error', 'time', 'default_time'])


# Measuring top-k accuracy & speed
simple_argmax_df_list = []
for db_size in tqdm(db_sizes):
    for distribution in ['normal', 'uniform']:
        if distribution == 'normal':
            cos_sim = torch.tensor(np.random.normal(0,0.05, size=(embedding_dim))) # Similar to random distance between points
        elif distribution == 'uniform':
            cos_sim = torch.tensor(np.random.uniform(-1,1, size=(embedding_dim)))

        real_argmax, default_time = timethis(lambda a: torch.argmax(a), cos_sim)
        argmax_res_binary_mpc, timetaken = timethis(dectorate_mpc(argmax_mpc_tobin), cos_sim)
        argmax_res_mpc = pickle.loads(argmax_res_binary_mpc[0])

        simple_argmax_df_list.append([db_size, embedding_dim, 'argmax', int(real_argmax==argmax_res_mpc), timetaken, default_time, distribution])

simple_argmax_df = pd.DataFrame(simple_argmax_df_list, columns=['db_size', 'embedding_dim', 'function', 'matches', 'time', 'default_time', 'distribution'])


def top_k_f1(real_top_k, mpc_top_k):
    """
    real_top_k: torch.Tensor of shape (k, )
    mpc_top_k: torch.Tensor of shape (k, )
    """
    precision = len(set(real_top_k.tolist()).intersection(set(mpc_top_k.tolist()))) / len(mpc_top_k)
    recall = len(set(real_top_k.tolist()).intersection(set(mpc_top_k.tolist()))) / len(real_top_k)
    return 2 * (precision * recall) / (precision + recall)

top_k_recall_df_list = []
for db_size in tqdm(db_sizes):
    for k in [1, 5, 10]:
        for distribution in ['normal', 'uniform']:
            if distribution == 'normal':
                cos_sim = torch.tensor(np.random.normal(0,0.05, size=(embedding_dim))) # Similar to random distance between points
            elif distribution == 'uniform':
                cos_sim = torch.tensor(np.random.uniform(-1,1, size=(embedding_dim)))

            real_top_k, default_time = timethis(lambda a: torch.topk(a, k).indices, cos_sim)
            
            top_k_res_binary_mpc, timetaken = timethis(dectorate_mpc(top_k_mpc_tobin), cos_sim, k)
            top_k_res_mpc = pickle.loads(top_k_res_binary_mpc[0])

            top_k_recall_df_list.append([db_size, embedding_dim, 'top_k', top_k_f1(real_top_k, top_k_res_mpc), timetaken, default_time, distribution, k])

            # Put guyz code here

top_k_recall_df = pd.DataFrame(top_k_recall_df_list, columns=['db_size', 'embedding_dim', 'function', 'matches', 'time', 'default_time', 'distribution', 'k'])



mpc_distance_top_k
dot_score_mpc
cosine_similarity_mpc_opt
euclidean_mpc
top_k_mpc_tobin

# scaling_times = pd.DataFrame(scaling_times, columns=['cos_sim', 'mpc_naive', 'mpc_opt', 'dot', 'mpc_dot'])
# scaling_times['size'] = sizes

# # Let's plot speed results
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# ax.plot(scaling_times['size'], scaling_times['cos_sim'], label='Cosine Similarity')
# ax.plot(scaling_times['size'], scaling_times['mpc_naive'], label='MPC Cosine Similarity (Naive)')
# ax.plot(scaling_times['size'], scaling_times['mpc_opt'], label='MPC Cosine Similarity (Optimised)')
# ax.plot(scaling_times['size'], scaling_times['dot'], label='Dot Product')
# ax.plot(scaling_times['size'], scaling_times['mpc_dot'], label='MPC Dot Product')
# ax.set_xlabel('Size of Database')
# ax.set_ylabel('Time Taken (s)')
# # legend
# ax.legend(loc='upper left')

# # Now we plot accuracy
# fig, ax = plt.subplots()
# # ax.plot(scaling_times['size'], MSE_mpc_cos_naive, label='MPC Cosine Similarity (Naive)')
# ax.plot(scaling_times['size'], MSE_mpc_cos_opt, label='MPC Cosine Similarity (Optimised)')
# # ax.plot(scaling_times['size'], MSE_mpc_dot, label='MPC Dot Product')
# ax.set_xlabel('Size of Database')
# ax.set_ylabel('MSE')
# ax.legend(loc='upper left')
