# pip install chromadb langchain
import os, numpy as np

os.environ['OPENAI_API_KEY'] = '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'

from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

# Load the document, split it into chunks, embed each chunk and load it into the vector store.
raw_documents = TextLoader('state_of_the_union.txt').load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
embedding_model = OpenAIEmbeddings()

# Get database
db = Chroma.from_documents(documents, embedding_model)

# Take the raw matrix from the database
db_as_dict = db.get(include=["embeddings", "metadatas", "documents"])

query = "What did the president say about Ketanji Brown Jackson"
query_vector = np.array(embedding_model.embed_query(query))


# 1. Now manually do cosine similarity search without secret sharing
D = np.array(db_as_dict['embeddings']).T

# write D to file
np.savetxt("D.csv", D, delimiter=",")
# read back in
D = np.loadtxt("D.csv", delimiter=",")

np.savetxt("query_vector.csv", query_vector, delimiter=",")

cosine_sim = np.dot(query_vector, D) / (np.linalg.norm(query_vector) * np.linalg.norm(D))
euclidean_dist = np.linalg.norm(D.T - query_vector, axis=1)

db_as_dict['documents'][np.argmax(cosine_sim)]
db_as_dict['documents'][np.argmin(euclidean_dist)]


# 2. Add in secret sharing with in finite field
from secret_sharing import *

share1, share2 = secret_sharing(float_to_int(query_vector))
D_field = float_to_int(D)

cosine_sim_1 = np.dot(share1, D_field) / (np.linalg.norm(share1) * np.linalg.norm(D_field))
euclidean_dist_1 = np.linalg.norm(D_field.T - share1, axis=1)

cosine_sim_2 = np.dot(share2, D_field) / (np.linalg.norm(share2) * np.linalg.norm(D_field))
euclidean_dist_2 = np.linalg.norm(D_field.T - share2, axis=1)

reconstructed_int_vector = reconstruct(cosine_sim_1, cosine_sim_2)
reconstructed_float_vector = int_to_float(reconstructed_int_vector)










share1, share2 = secret_sharing(float_to_int(query_vector))
share1_vector, share2_vector = int_to_float(share1), int_to_float(share2)

cosine_sim_1 = np.dot(share1_vector, D) / (np.linalg.norm(share1_vector) * np.linalg.norm(D))
euclidean_dist_1 = np.linalg.norm(D.T - share1_vector, axis=1)

cosine_sim_2 = np.dot(share2_vector, D) / (np.linalg.norm(share2_vector) * np.linalg.norm(D))
euclidean_dist_2 = np.linalg.norm(D.T - share2_vector, axis=1)

reconstructed_int_vector = reconstruct(float_to_int(cosine_sim_1), float_to_int(cosine_sim_2))
reconstructed_float_vector = int_to_float(reconstructed_int_vector)















# 1. Simple query, no secret sharing using langchain interface
docs = db.similarity_search(query, k=1) # without self embedding
docs = db.similarity_search_by_vector(list(query_vector), k=1) # with self embedding

# Get the full response vector
search_results = db.similarity_search_by_vector_with_relevance_scores(query_vector, k = len(db_as_dict['ids']))
similarity_vector = np.array([result[1] for result in search_results])
search_results[np.argmin(similarity_vector)][0]



# 2. Now we secret share using chromadb interface
share1, share2 = secret_sharing(float_to_int(query_vector))

share1_vector, share2_vector = int_to_float(share1), int_to_float(share2)


search_results = db.similarity_search_by_vector_with_relevance_scores(list(share1_vector), k = len(db_as_dict['ids']))
similarity_vector1 = np.array([result[1] for result in search_results])

search_results = db.similarity_search_by_vector_with_relevance_scores(list(share2_vector), k = len(db_as_dict['ids']))
similarity_vector2 = np.array([result[1] for result in search_results])


reconstructed_similarity_int_vector = reconstruct(float_to_int(similarity_vector1), float_to_int(similarity_vector2))

reconstructed_similarity_vector = int_to_float(reconstructed_similarity_int_vector)

search_results[np.argmin(reconstructed_similarity_vector)]







search_results = db.similarity_search_by_vector_with_relevance_scores(list(share1_vector), k = len(db_as_dict['ids']))


 db.similarity_search_by_vector??