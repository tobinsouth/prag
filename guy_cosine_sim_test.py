import torch
import pandas as pd
from torch.nn.functional import cosine_similarity

def cosine_similarity_raw(A, B):
    # Compute the dot product of A and B
    dot_product = torch.matmul(A, B.t())
    # Compute the magnitudes of A and B
    mag_A = torch.sqrt(torch.sum(A * A, dim=1, keepdim=True))
    mag_B = torch.sqrt(torch.sum(B * B, dim=1, keepdim=True))
    # Compute the cosine similarity
    cosine_sim = dot_product / (mag_A * mag_B.t())
    return cosine_sim

# Load the data from CSV files
query_vector = pd.read_csv('query_vector.csv').values
database_vectors = pd.read_csv('D.csv').values

# Convert the data to PyTorch tensors
query_vector = torch.tensor(query_vector, dtype=torch.float32).t()
database_vectors = torch.tensor(database_vectors, dtype=torch.float32)

print(query_vector.shape)
print(database_vectors.shape)

# Compute the cosine similarity between the query vector and all database vectors
cosine_similarities = cosine_similarity(query_vector, database_vectors.t())

print(cosine_similarities)
print(cosine_similarities.shape)

cosine_similarities2 = cosine_similarity_raw(query_vector, database_vectors.t())
print(cosine_similarities2)

if torch.allclose(cosine_similarities, cosine_similarities2, atol=0.02):
    print("They are approx equal")
else:
    print("They are NOT approx equal")

if torch.equal(cosine_similarities, cosine_similarities2):
    print("They are EXACTLY equal")
else:
    print("They are NOT EXACTLY equal")