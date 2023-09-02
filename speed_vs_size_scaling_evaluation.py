from MPCDenseRetrievalExactSearch import MPCDenseRetrievalExactSearch # We use this class for it's standard API interface to the distance measures through both MPC and non-MPC

import torch, numpy as np, pandas as pd
import time
from tqdm import tqdm

model = MPCDenseRetrievalExactSearch(None)

torch.random.manual_seed(2023)

# 1. First we do a simple run through to test everything is working
query_vector = torch.rand(1,768) * 2 - 1
database_vectors = torch.rand(1000, 768) * 2 - 1

cos_sim_result = model.score_functions['cos_sim'](query_vector, database_vectors)
mpc_cos_sim_result = model.score_functions['mpc_naive'](query_vector, database_vectors)
mpc_opt_cos_sim_result = model.score_functions['mpc_opt'](query_vector, database_vectors)

dot_score_result = model.score_functions['dot'](query_vector, database_vectors)
mpc_dot_score_result = model.score_functions['mpc_dot'](query_vector, database_vectors)

MSE_mpc_cos_opt = torch.mean((mpc_opt_cos_sim_result - cos_sim_result)**2).item()
MSE_mpc_cos_naive = torch.mean((mpc_cos_sim_result - cos_sim_result)**2).item()
MSE_mpc_dot = torch.mean((mpc_dot_score_result - dot_score_result)**2).item()

print( ("Accuracy scores against baseline. Does everything look right?\n"
        "MSE of MPC Cosine Similarity (Naive): {} \n"
        "MSE of MPC Cosine Similarity (Optimised): {} \n"
        "MSE of MPC Dot Product: {} \n").format(MSE_mpc_cos_naive, MSE_mpc_cos_opt, MSE_mpc_dot)
        )

# 2. Now we do a scaling test to see how the time taken to compute the distance measures scales with the size of the vectors
def time_score_function(score_function, query_vector, database_vectors):
    start_time = time.time()
    res =  model.score_functions[score_function](query_vector, database_vectors)
    end_time = time.time()
    return res, end_time - start_time

scaling_times = []
MSE_mpc_cos_opt, MSE_mpc_cos_naive, MSE_mpc_dot = [], [], []
sizes = range(1000,10**5, 10000)
for size in tqdm(sizes):
    # Make a random vector and database and scale to -1 to 1
    query_vector = torch.rand(1,768) * 2 - 1
    database_vectors = torch.rand(size, 768) * 2 - 1
    times, results = [], {}
    for score_function in ['cos_sim', 'mpc_naive', 'mpc_opt', 'dot', 'mpc_dot']:
        res, time_taken = time_score_function(score_function, query_vector, database_vectors)
        times.append(time_taken)
        results[score_function] = res

    scaling_times.append(times)
    MSE_mpc_cos_opt.append(torch.mean((results['mpc_opt'] - results['cos_sim'])**2).item())
    MSE_mpc_cos_naive.append(torch.mean((results['mpc_naive'] - results['cos_sim'])**2).item())
    MSE_mpc_dot.append(torch.mean((results['mpc_dot'] - results['dot'])**2).item())


scaling_times = pd.DataFrame(scaling_times, columns=['cos_sim', 'mpc_naive', 'mpc_opt', 'dot', 'mpc_dot'])
scaling_times['size'] = sizes

# Let's plot speed results
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(scaling_times['size'], scaling_times['cos_sim'], label='Cosine Similarity')
ax.plot(scaling_times['size'], scaling_times['mpc_naive'], label='MPC Cosine Similarity (Naive)')
ax.plot(scaling_times['size'], scaling_times['mpc_opt'], label='MPC Cosine Similarity (Optimised)')
ax.plot(scaling_times['size'], scaling_times['dot'], label='Dot Product')
ax.plot(scaling_times['size'], scaling_times['mpc_dot'], label='MPC Dot Product')
ax.set_xlabel('Size of Database')
ax.set_ylabel('Time Taken (s)')
# legend
ax.legend(loc='upper left')

# Now we plot accuracy
fig, ax = plt.subplots()
# ax.plot(scaling_times['size'], MSE_mpc_cos_naive, label='MPC Cosine Similarity (Naive)')
ax.plot(scaling_times['size'], MSE_mpc_cos_opt, label='MPC Cosine Similarity (Optimised)')
# ax.plot(scaling_times['size'], MSE_mpc_dot, label='MPC Dot Product')
ax.set_xlabel('Size of Database')
ax.set_ylabel('MSE')
ax.legend(loc='upper left')
