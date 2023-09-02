This is a building blocks repo for now to combine some guyz code and my code.

# Private Retrieval Augmented Generation (PRAG)
PRAG provides a solution to augment Large Language Models (LLM) with the ability to securely query external data sources, ensuring context-specific accuracy without compromising data privacy. While LLMs like GPT demonstrate significant potential in generalizing tasks through prompt engineering, their static view and occasional 'hallucinations' due to lack of specific context can be limiting. PRAG bridges this gap by introducing a unique retrieval phase, allowing LLMs to leverage private data in inference without exposing sensitive information. This repository offers a first-of-its-kind approach, focusing on both the retrieval and inference phases, incorporating secure solutions like Community Transformers, MPCFormer, and PUMA.

## Setup
The two key requirements you will need for this are `crypten` for MPC computations and `beir` for information retrieval benchmarking.

### Structure
The actual MPC functions using crypten (currently only cos and dot product) are present in `mpc_functions.py`. These can be imported, but require some light wrapping to unpickle the outputs.

`LocalDenseRetrievalExactSearch.py` provides an IR wrapper (adapted from the `beir` package) that can be used to embedding datasets and queries using various transformer models and then perform k nearest search over the embeddings. This alteration of the class allows for embeddings to be pre-generated and stored to file (which will help us speed things up a lot!). Note that the default beir search process uses chunked retrieval, which speeds things up a lot on real datasets!

`MPCDenseRetrievalExactSearch.py` extends the DenseRetrievalExactSearch (DRES) class to allow for MPC computations. This class also acts as a standard wrapper for the MPC functions in `mpc_functions.py`.

### Running
A good place to start is the `beir_MPC_example.py`. The setup function (not executed by default) will allow you to create a set of embeddings from a standard dataset (e.g. MS MARCO) and store them to file. You can also embed a sample of the dataset for testing purposes. This file will guide you through using the MPC and non-MPC functions to perform a k-nearest search over the dataset and benchmark.

If you're interested in accuracy or speed on synthetic data, you can run `speed_vs_size_scaling_evaluation.py`. This will generate a random dataset and query vector and perform a k-nearest search using both MPC and non-MPC functions. The results are printed to the console.


## TODO

- [ ] Check why MPC cos_sim naive has extremely high error.
- [ ] Update the `benchmark_retriever()` to return the values we really care about.
- [ ] Setup benchmarking script that cross benchmarks on time and accuracy for (distance measures) x (MPC vs non-MPC) x (standard datasets) x (embedding models). This will have an astronomical runtime but will be good for the paper.
- [ ] Integrate with langchain.
- [ ] Lock down more details in the actual MPC stuff.
- [ ] Add a test suite and CI.

## Notes

Files marked for deletion:

* `D.csv` (we have been example vectors now)
* `query_vector.csv` 
* `guy_cosine_sim_test.py` (once guy is done with it)
* `langchain.py` (or more likely this will be edited to interface with the MPC functions properly)
