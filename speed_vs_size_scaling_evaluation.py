from MPCDenseRetrievalExactSearch import MPCDenseRetrievalExactSearch # We use this class for it's standard API interface to the distance measures through both MPC and non-MPC

import torch, numpy as np

model = MPCDenseRetrievalExactSearch(None)


torch.random.manual_seed(2023)

# Make a random vector and database and scale to -1 to 1
query_vector = torch.rand(10,768) * 2 - 1
database_vectors = torch.rand(1000, 768) * 2 - 1

cos_sim_result = model.score_functions['cos_sim'](query_vector, database_vectors)
mpc_cos_sim_result = model.score_functions['mpc_naive'](query_vector, database_vectors)
mpc_opt_cos_sim_result = model.score_functions['mpc_opt'](query_vector, database_vectors)

dot_score_result = model.score_functions['dot'](query_vector, database_vectors)
mpc_dot_score_result = model.score_functions['mpc_dot'](query_vector, database_vectors)


MSE_mpc_cos_opt = torch.mean((mpc_opt_cos_sim_result - cos_sim_result)**2).item()
MSE_mpc_cos_naive = torch.mean((mpc_cos_sim_result - cos_sim_result)**2).item()
MSE_mpc_dot = torch.mean((mpc_dot_score_result - dot_score_result)**2).item()


# As a proportion of the range
MSE_mpc_naive / (torch.max(query_vector) - torch.min(query_vector))



