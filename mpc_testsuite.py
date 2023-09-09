import torch, numpy as np, pandas as pd
import time, pickle
from tqdm import tqdm


query_vector = torch.rand(1,768) * 2 - 1
database_vectors = torch.rand(768, 1000) * 2 - 1

def relative_error(y_pred, y_true):
    return torch.mean(torch.abs(y_pred - y_true) / torch.abs(y_true))

def timethis(func, *args):
    start_time = time.time()
    res =  func(*args)
    end_time = time.time()
    return res, end_time - start_time


## Distance function checks

# Test the most basic, dot similarity
real_dot = query_vector @ database_vectors
dot_score_res_binary = dot_score_mpc(query_vector, database_vectors)
dot_score_res = pickle.loads(dot_score_res_binary)

dot_score_res_binary_mpc, timetaken = timethis(dectorate_mpc(dot_score_mpc), query_vector, database_vectors)
dot_score_res_mpc = pickle.loads(dot_score_res_binary_mpc[0])

print(f"dot_score — error {relative_error(real_dot, dot_score_res):.2e} — mpc matches: {torch.allclose(dot_score_res, dot_score_res_mpc, atol=0.001)} — time: {timetaken:.4f}" )


# Cosine similarity checks
real_cos_sim =query_vector @ database_vectors / (torch.norm(query_vector) * torch.norm(database_vectors))

cosine_sim_binary = cosine_similarity_mpc_opt(query_vector, database_vectors)
cosine_sim = pickle.loads(cosine_sim_binary)

cosine_sim_binary_mpc, timetaken = timethis(dectorate_mpc(cosine_similarity_mpc_opt), query_vector, database_vectors)
cosine_sim_mpc = pickle.loads(cosine_sim_binary_mpc[0])

print(f"cosine_similarity_mpc_opt — error {relative_error(real_cos_sim, cosine_sim):.2e} — mpc matches: {torch.allclose(cosine_sim, cosine_sim_mpc, atol=0.001)} — time: {timetaken:.4f}" )


cosine_sim_binary = cosine_similarity_mpc_opt2(query_vector, database_vectors)
cosine_sim = pickle.loads(cosine_sim_binary)

cosine_sim_binary_mpc, timetaken = timethis(dectorate_mpc(cosine_similarity_mpc_opt2), query_vector, database_vectors)
cosine_sim_mpc = pickle.loads(cosine_sim_binary_mpc[0])

print(f"cosine_similarity_mpc_opt2 — error {relative_error(real_cos_sim, cosine_sim):.2e} — mpc matches: {torch.allclose(cosine_sim, cosine_sim_mpc, atol=0.001)} — time: {timetaken:.4f}" )


A_mag_recip, B_mag_recip = preprocess_cosine_similarity_mpc_opt(query_vector, database_vectors)
cosine_sim_binary = cosine_similarity_mpc_pass_in_mags(query_vector, database_vectors, A_mag_recip, B_mag_recip)
cosine_sim = pickle.loads(cosine_sim_binary)

cosine_sim_binary_mpc, timetaken = timethis(dectorate_mpc(cosine_similarity_mpc_pass_in_mags), query_vector, database_vectors, A_mag_recip, B_mag_recip)
cosine_sim_mpc = pickle.loads(cosine_sim_binary_mpc[0])

print(f"cosine_similarity_mpc_pass_in_mags — error {relative_error(real_cos_sim, cosine_sim):.2e} — mpc matches: {torch.allclose(cosine_sim, cosine_sim_mpc, atol=0.001)} — time: {timetaken:.4f}" )


#  Euclidean distance check
real_euclidian= torch.norm(query_vector - database_vectors.t(), dim=1)

euclidian_res_binary = euclidean_mpc(query_vector, database_vectors)
euclidian_res = pickle.loads(euclidian_res_binary).sqrt()

euclidian_res_binary_mpc, timetaken = timethis(dectorate_mpc(euclidean_mpc), query_vector, database_vectors)
euclidian_res_mpc = pickle.loads(euclidian_res_binary_mpc[0]).sqrt()

print(f"euclidean_mpc — error {relative_error(real_euclidian, euclidian_res):.2e} — mpc matches: {torch.allclose(euclidian_res, euclidian_res_mpc, atol=0.001)} — time: {timetaken:.4f}" )




# Now we check with top-k and max

real_cos_sim = (query_vector @ database_vectors / (torch.norm(query_vector) * torch.norm(database_vectors)) ).squeeze(0)

real_argmax = torch.argmax(real_cos_sim)


tobin_argmax_binary = argmax_mpc_tobin(real_cos_sim)
tobin_argmax = pickle.loads(tobin_argmax_binary)
assert tobin_argmax == real_argmax

tobin_argmax_mpc_binary, timetaken = timethis(dectorate_mpc(argmax_mpc_tobin), real_cos_sim)
tobin_argmax_mpc = pickle.loads(tobin_argmax_mpc_binary[0])
assert tobin_argmax == tobin_argmax_mpc

print(f"argmax_mpc_tobin — correct if no assert — time: {timetaken:.4f}" )


#  Guyz code
for k in [1, 5, 10, 20]:
    real_top_k = torch.sort(real_cos_sim, descending=True).values[:k]

    approx_top_k_binary = approx_top_k_mpc(real_cos_sim, k)
    approx_top_k = pickle.loads(approx_top_k_binary)
    # try:
    #     assert torch.equal(torch.tensor(approx_top_k), real_top_k)
    # except AssertionError:
    #     print("guyz approx_top_k_mpc failed to correctly get max")

    approx_top_k_mpc_binary, timetaken = timethis(dectorate_mpc(approx_top_k_mpc), real_cos_sim, k)
    approx_top_k_mpc_res = pickle.loads(approx_top_k_mpc_binary[0])
    assert torch.equal(approx_top_k, approx_top_k_mpc_res)

    print(f"guyz approx_top_k_mpc — run on k={k} — time: {timetaken:.4f}" )


# for k in [1, 5, 10, 20]:
    real_top_k = torch.argsort(real_cos_sim, descending=True)[:k]

    tobin_top_k_binary = top_k_mpc_tobin(real_cos_sim, k)
    tobin_top_k = pickle.loads(tobin_top_k_binary)
    # assert torch.equal(tobin_top_k, real_top_k)

    tobin_top_k_mpc_binary, timetaken = timethis(dectorate_mpc(top_k_mpc_tobin), real_cos_sim, k)
    tobin_top_k_mpc = pickle.loads(tobin_top_k_mpc_binary[0])
    assert torch.equal(tobin_top_k, tobin_top_k_mpc)

    print(f"top_k_mpc_tobin — correct for k={k} — time: {timetaken:.4f}" )


# for k in [1, 5, 10, 20]:
    real_top_k_embedding_vectors = database_vectors[:, torch.argsort(real_cos_sim, descending=True)[:k]].t()

    top_k_embedding_vectors_binary = tobin_top_k_mpc_return_embedding_vectors(real_cos_sim, k, database_vectors)
    top_k_embedding_vectors = pickle.loads(top_k_embedding_vectors_binary)

    top_k_embedding_vectors_mpc_binary, timetaken = timethis(dectorate_mpc(tobin_top_k_mpc_return_embedding_vectors), real_cos_sim, k, database_vectors)
    top_k_embedding_vectors_mpc = pickle.loads(top_k_embedding_vectors_mpc_binary[0])

    print(f"tobin top-k return embeddings — error {relative_error(top_k_embedding_vectors, real_top_k_embedding_vectors):.2e} — mpc matches: {torch.allclose(top_k_embedding_vectors, top_k_embedding_vectors_mpc, atol=0.001)} — time: {timetaken:.4f} — run on k={k}" )




# from crypten.common.functions.maximum import _one_hot_to_index
# v_enc = crypten.cryptensor(real_cos_sim, ptype=crypten.mpc.arithmetic)

# top_k = []
# for i in range(k):
#     max_one_hot = v_enc.argmax(one_hot=True)
#     top_k.append(_one_hot_to_index(max_one_hot, dim=0, keepdim=False).unsqueeze(0))
#     v_enc = v_enc - 100*max_one_hot # This will top the current max from being the top (could be done better)
# top_k = crypten.cat(top_k) # not sure of the security model without keeping this in crypten 
# return pickle.dumps(top_k)