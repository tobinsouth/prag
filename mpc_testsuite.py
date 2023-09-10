import torch, numpy as np, pandas as pd
import time, pickle
from tqdm import tqdm

from mpc_functions_stable import *

query_vector = torch.rand(1,768) * 2 - 1
database_vectors = torch.rand(768, 1000) * 2 - 1

def relative_error(y_pred, y_true):
    return torch.mean(torch.abs(y_pred - y_true) / torch.abs(y_true))

def timethis(func, *args):
    start_time = time.time()
    res =  func(*args)
    end_time = time.time()
    return res, end_time - start_time

def top_k_f1(real_top_k, mpc_top_k):
    """
    real_top_k: torch.Tensor of shape (k, )
    mpc_top_k: torch.Tensor of shape (k, )
    """
    precision = len(set(real_top_k.tolist()).intersection(set(mpc_top_k.tolist()))) / len(mpc_top_k)
    recall = len(set(real_top_k.tolist()).intersection(set(mpc_top_k.tolist()))) / len(real_top_k)
    return 2 * (precision * recall) / (precision + recall)


## Distance function checks

# Test the most basic, dot similarity
real_dot = query_vector @ database_vectors
dot_score_res = handle_binary(dot_score_mpc)(query_vector, database_vectors)

dot_score_res_mpc, timetaken = timethis(handle_binary(dot_score_mpc, mpc=True), query_vector, database_vectors)

print(f"dot_score — error {relative_error(real_dot, dot_score_res):.2e} — mpc matches: {torch.allclose(dot_score_res, dot_score_res_mpc, atol=0.001)} — time: {timetaken:.4f}" )


# Cosine similarity checks
real_cos_sim =query_vector @ database_vectors / (torch.norm(query_vector) * torch.norm(database_vectors))

cosine_sim = handle_binary(cosine_similarity_mpc_opt)(query_vector, database_vectors)
cosine_sim_mpc, timetaken = timethis(handle_binary(cosine_similarity_mpc_opt, mpc=True), query_vector, database_vectors)

print(f"cosine_similarity_mpc_opt — error {relative_error(real_cos_sim, cosine_sim):.2e} — mpc matches: {torch.allclose(cosine_sim, cosine_sim_mpc, atol=0.001)} — time: {timetaken:.4f}" )


cosine_sim2 = handle_binary(cosine_similarity_mpc_opt2)(query_vector, database_vectors)
cosine_sim_mpc2, timetaken = timethis(handle_binary(cosine_similarity_mpc_opt2, mpc=True), query_vector, database_vectors)

print(f"cosine_similarity_mpc_opt2 — error {relative_error(real_cos_sim, cosine_sim2):.2e} — mpc matches: {torch.allclose(cosine_sim2, cosine_sim_mpc2, atol=0.001)} — time: {timetaken:.4f}" )


A_mag_recip, B_mag_recip = preprocess_cosine_similarity_mpc_opt(query_vector, database_vectors)
cosine_sim3 = handle_binary(cosine_similarity_mpc_pass_in_mags)(query_vector, database_vectors, A_mag_recip, B_mag_recip)

cosine_sim_mpc3, timetaken = timethis(handle_binary(cosine_similarity_mpc_pass_in_mags, mpc=True), query_vector, database_vectors, A_mag_recip, B_mag_recip)

print(f"cosine_similarity_mpc_pass_in_mags — error {relative_error(real_cos_sim, cosine_sim):.2e} — mpc matches: {torch.allclose(cosine_sim, cosine_sim_mpc, atol=0.001)} — time: {timetaken:.4f}" )


#  Euclidean distance check
real_euclidian= torch.norm(query_vector - database_vectors.t(), dim=1)

euclidian_res = handle_binary(euclidean_mpc)(query_vector, database_vectors).sqrt()

euclidian_res_mpc, timetaken = timethis(handle_binary(euclidean_mpc, mpc=True), query_vector, database_vectors)
euclidian_res_mpc = euclidian_res_mpc.sqrt()

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
# for k in [1, 5, 10, 20]:
    # real_top_k = torch.sort(real_cos_sim, descending=True).values[:k]

    # approx_top_k_binary = approx_top_k_mpc(real_cos_sim, k)
    # approx_top_k = pickle.loads(approx_top_k_binary)
    # # try:
    # #     assert torch.equal(torch.tensor(approx_top_k), real_top_k)
    # # except AssertionError:
    # #     print("guyz approx_top_k_mpc failed to correctly get max")

    # approx_top_k_mpc_binary, timetaken = timethis(dectorate_mpc(approx_top_k_mpc), real_cos_sim, k)
    # approx_top_k_mpc_res = pickle.loads(approx_top_k_mpc_binary[0])
    # assert torch.equal(approx_top_k, approx_top_k_mpc_res)

    # print(f"guyz approx_top_k_mpc — run on k={k} — time: {timetaken:.4f}" )


for k in [1, 5, 10]:
    real_top_k = torch.argsort(real_cos_sim, descending=True)[:k]

    tobin_top_k_binary = top_k_mpc_tobin(real_cos_sim, k)
    tobin_top_k = pickle.loads(tobin_top_k_binary)

    tobin_top_k_mpc_binary, timetaken = timethis(dectorate_mpc(top_k_mpc_tobin), real_cos_sim, k)
    tobin_top_k_mpc = pickle.loads(tobin_top_k_mpc_binary[0])

    print(f"top_k_mpc_tobin — for k={k} — f1: {top_k_f1(tobin_top_k, tobin_top_k_mpc)}— time: {timetaken:.4f}" )


for k in [1, 5, 10]:
    real_top_k_embedding_vectors = database_vectors[:, torch.argsort(real_cos_sim, descending=True)[:k]].t()

    top_k_embedding_vectors_binary = tobin_top_k_mpc_return_embedding_vectors(real_cos_sim, k, database_vectors)
    top_k_embedding_vectors = pickle.loads(top_k_embedding_vectors_binary)

    top_k_embedding_vectors_mpc_binary, timetaken = timethis(dectorate_mpc(tobin_top_k_mpc_return_embedding_vectors), real_cos_sim, k, database_vectors)
    top_k_embedding_vectors_mpc = pickle.loads(top_k_embedding_vectors_mpc_binary[0])

    print(f"tobin top-k return embeddings — error {relative_error(top_k_embedding_vectors, real_top_k_embedding_vectors):.2e} — mpc matches: {torch.allclose(top_k_embedding_vectors, top_k_embedding_vectors_mpc, atol=0.001)} — time: {timetaken:.4f} — run on k={k}" )


