import torch, numpy as np, pandas as pd
import time
import signal
import seaborn as sns
from tqdm import tqdm
import argparse

from mpc_functions_stable import *

torch.random.manual_seed(2023)

# def relative_error(y_pred, y_true):
#     return torch.mean(torch.abs(y_pred - y_true) / torch.abs(y_true))

def relative_error(y_pred, y_true):
    """Measure MSE"""
    return torch.mean((y_pred - y_true)**2)

def timeout_handler(signum, frame):
    raise TimeoutError("Function timed out")

def timethis(func, *args):
    start_time = time.time()
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(450)
        res =  func(*args)
        end_time = time.time()
    except TimeoutError:
        print("Timed out!")
        return [None], np.NaN
    signal.alarm(0) # Disable alarm
    return res, end_time - start_time


def top_k_f1(real_top_k, mpc_top_k):
    """
    real_top_k: torch.Tensor of shape (k, )
    mpc_top_k: torch.Tensor of shape (k, )
    """
    real_top_k = real_top_k.tolist() if len(real_top_k.shape) > 0 else [real_top_k.item()]
    mpc_top_k = mpc_top_k.tolist() if len(mpc_top_k.shape) > 0 else [mpc_top_k.item()]
    precision = len(set(real_top_k).intersection(set(mpc_top_k))) / len(mpc_top_k)
    recall = len(set(real_top_k).intersection(set(mpc_top_k))) / len(real_top_k)
    return 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

def mrr(real_rank, mpc_rank):
    mrr = 0
    length = len(real_rank) if len(real_rank.shape) > 0 else 1
    for i, item in enumerate(mpc_rank):
        if item in real_rank:
            mrr += 1 / (i + 1)
    return mrr / length



def distance_experiements():
    distance_result_df_list = []
    db_sizes = list(range(500,7000, 500))
    embedding_dims = [256, 768, 1024, 2048, 4096, 8192]
    # embedding_dim = 768
    np.random.shuffle(db_sizes)
    for db_size in tqdm(db_sizes):
        for embedding_dim in embedding_dims:
            for distribution in ['normal', 'uniform', 'binary', 'normal_large']:
                if distribution == 'normal':
                    query_vector =torch.tensor(np.random.normal(0,0.05, size=(1, embedding_dim)))
                    database_vectors = torch.tensor(np.random.normal(0,0.05, size=(embedding_dim, db_size)))
                elif distribution == 'uniform':
                    query_vector = torch.rand(1,embedding_dim) * 2 - 1
                    database_vectors = torch.rand(embedding_dim, db_size) * 2 - 1   
                elif distribution == 'binary':
                    query_vector = torch.randint(0,2, size=(1, embedding_dim), dtype=torch.float32)
                    database_vectors = torch.randint(0,2, size=(embedding_dim, db_size), dtype=torch.float32)
                elif distribution == 'normal_large':
                    query_vector =torch.tensor(np.random.normal(0,1, size=(1, embedding_dim)))
                    database_vectors = torch.tensor(np.random.normal(0,1, size=(embedding_dim, db_size)))

                # Dot score results
                real_dot, default_time = timethis(lambda a,b: a @ b, query_vector, database_vectors)
                dot_score_res_mpc, timetaken = timethis(handle_binary(dot_score_mpc), query_vector, database_vectors)
                error = relative_error(real_dot, dot_score_res_mpc).item()

                distance_result_df_list.append([db_size, embedding_dim, 'dot_score', error, timetaken, default_time, distribution])

                # Cosine similarity results
                real_cos_sim, default_time = timethis(lambda a,b: a @ b / (torch.norm(a) * torch.norm(b)), query_vector, database_vectors)
                cosine_sim_mpc, timetaken = timethis(handle_binary(cosine_similarity_mpc_opt), query_vector, database_vectors)
                error = relative_error(real_cos_sim, cosine_sim_mpc).item()

                distance_result_df_list.append([db_size, embedding_dim, 'cosine_similarity_mpc_opt', error, timetaken, default_time, distribution])

                cosine_sim_mpc, timetaken = timethis(handle_binary(cosine_similarity_mpc_opt2), query_vector, database_vectors)
                error = relative_error(real_cos_sim, cosine_sim_mpc).item()

                distance_result_df_list.append([db_size, embedding_dim, 'cosine_similarity_mpc_opt2', error, timetaken, default_time, distribution])

                #  Euclidean distance check
                real_euclidian, default_time = timethis(lambda a,b: torch.norm(a - b.t(), dim=1), query_vector, database_vectors)

                euclidian_res_mpc, timetaken = timethis(handle_binary(euclidean_mpc), query_vector, database_vectors)
                error = relative_error(real_euclidian, euclidian_res_mpc.sqrt()).item()

                distance_result_df_list.append([db_size, embedding_dim, 'euclidean_mpc', error, timetaken, default_time, distribution])

        distance_result_df = pd.DataFrame(distance_result_df_list, columns=['db_size', 'embedding_dim', 'function', 'error', 'time', 'default_time', 'distribution'])
        distance_result_df.to_csv('results/distance_result_df.csv', index=False)


def argmax_experiments():
    simple_argmax_df_list = []
    db_sizes = list(range(1000,10**5, 1000)) +list(range(10**5, 10**7, 10**5))+list(range(10**7, 10**8, 10**7//2))
    np.random.shuffle(db_sizes)
    for db_size in tqdm(db_sizes):
        for distribution in ['normal', 'uniform']:
            if distribution == 'normal':
                distance_vector = torch.tensor(np.random.normal(0,0.05, size=(db_size))) # Similar to random distance between points
            elif distribution == 'uniform':
                distance_vector = torch.tensor(np.random.uniform(-1,1, size=(db_size)))

            real_argmax, default_time = timethis(lambda a: torch.argmax(a), distance_vector)
            argmax_res_mpc, timetaken = timethis(handle_binary(argmax_mpc_tobin), distance_vector)

            simple_argmax_df_list.append([db_size, 'argmax', int(real_argmax==argmax_res_mpc), timetaken, default_time, distribution])

        simple_argmax_df = pd.DataFrame(simple_argmax_df_list, columns=['db_size', 'function', 'matches', 'time', 'default_time', 'distribution'])
        simple_argmax_df.to_csv('results/simple_argmax_df.csv', index=False)
    # sns.scatterplot(data=simple_argmax_df, x='db_size', y='time', hue='distribution')

    simple_argmax_df['time_increase'] = simple_argmax_df['time'] / simple_argmax_df['default_time']
    sns.scatterplot(data=simple_argmax_df, x='db_size', y='time_increase', hue='distribution')


def top_k_experiments():
    top_k_recall_df_list = []
    db_sizes = list(range(1000,10**7, 1000)) + list(range(10**7, 10**8, 10**7//2))
    np.random.shuffle(db_sizes)
    for db_size in tqdm(db_sizes):
        for k in [1, 5, 10, 20, 50, 75, 100]:
            for distribution in ['normal', 'uniform']:
                if distribution == 'normal':
                    cos_sim = torch.tensor(np.random.normal(0,0.05, size=(db_size))) # Similar to random distance between points
                elif distribution == 'uniform':
                    cos_sim = torch.tensor(np.random.uniform(-1,1, size=(db_size)))

                real_top_k, default_time = timethis(lambda a: torch.topk(a, k).indices, cos_sim)
                try:
                    top_k_res_mpc, timetaken = timethis(handle_binary(top_k_mpc_tobin), cos_sim, k)

                    top_k_recall_df_list.append([db_size, 'top_k', top_k_f1(real_top_k, top_k_res_mpc),  mrr(real_top_k, top_k_res_mpc), timetaken, default_time, distribution, k])
                except (AttributeError, ValueError) as e:
                    print(e, f" Error for --  k:{k}, dbsize:{db_size}, distribution:{distribution}")

                # Put guyz code here

        top_k_recall_df = pd.DataFrame(top_k_recall_df_list, columns=['db_size', 'function', 'f1', 'mrr', 'time', 'default_time', 'distribution', 'k'])
        top_k_recall_df.to_csv('results/top_k_recall_df.csv', index=False)


def end_to_end_experiments():
    e2e_result_df_list = []
    # db_sizes = list(range(1000,10**5, 1000)) +list(range(10**5, 10**7, 10**6//2))+list(range(10**7, 10**8, 10**7//2))
    db_sizes = list(range(5000,100001, 5000))
    np.random.shuffle(db_sizes)
    # db_sizes = range(100000, 500000,10000)
    embedding_dims = [768, 256, 1024, 2048, 4096]
    # embedding_dim = 768
    for embedding_dim in embedding_dims:
        for db_size in tqdm(db_sizes):
            for k in [1, 5, 10, 20, 50]:
                for distribution in ['normal', 'uniform', 'normal_large']: # 'binary', 
                    if distribution == 'normal':
                        query_vector =torch.tensor(np.random.normal(0,0.05, size=(1, embedding_dim)))
                        database_vectors = torch.tensor(np.random.normal(0,0.05, size=(embedding_dim, db_size)))
                    elif distribution == 'uniform':
                        query_vector = torch.rand(1,embedding_dim) * 2 - 1
                        database_vectors = torch.rand(embedding_dim, db_size) * 2 - 1   
                    elif distribution == 'binary':
                        query_vector = torch.randint(0,2, size=(1, embedding_dim), dtype=torch.float32)
                        database_vectors = torch.randint(0,2, size=(embedding_dim, db_size), dtype=torch.float32)
                    elif distribution == 'normal_large':
                        query_vector =torch.tensor(np.random.normal(0,1, size=(1, embedding_dim)))
                        database_vectors = torch.tensor(np.random.normal(0,1, size=(embedding_dim, db_size)))


                    try:
                        true_vals, true_idx = torch.topk(query_vector @ database_vectors, k)
                        true_vals, true_idx = true_vals.squeeze(0), true_idx.squeeze(0)
                        distance_and_top_k_func = mpc_distance_top_k_with_distance_func(dot_score_mpc)
                        crypten_binary, timetaken = timethis(distance_and_top_k_func, query_vector, database_vectors, k)
                        crypten_binary = distance_and_top_k_func(query_vector, database_vectors, k)
                        top_k_idx, top_k_values = pickle.loads(crypten_binary[0])
                        top_k_values, top_k_idx = top_k_values.cpu(), top_k_idx.cpu()

                        f1_error = top_k_f1(true_idx, top_k_idx)
                        mrr_error = mrr(true_idx, top_k_idx)
                        max_vals_mse = relative_error(true_vals, top_k_values).item()
                        e2e_result_df_list.append([db_size, embedding_dim, 'dot_score', f1_error, mrr_error, max_vals_mse, timetaken, distribution, k])
                    except Exception as e:
                        print(e, f" Error for --  k:{k}, dbsize:{db_size}, embedding_dim:{embedding_dim}, distribution:{distribution}")

                    try:
                        true_vals, true_idx = torch.topk(torch.cosine_similarity(query_vector, database_vectors.t()), k)
                        true_vals, true_idx = true_vals.squeeze(0), true_idx.squeeze(0)
                        distance_and_top_k_func = mpc_distance_top_k_with_distance_func(cosine_similarity_mpc_opt)
                        crypten_binary, timetaken = timethis(distance_and_top_k_func, query_vector, database_vectors, k)
                        top_k_idx, top_k_values = pickle.loads(crypten_binary[0])
                        top_k_values, top_k_idx = top_k_values.cpu(), top_k_idx.cpu()

                        f1_error = top_k_f1(true_idx, top_k_idx)
                        mrr_error = mrr(true_idx, top_k_idx)
                        max_vals_mse = relative_error(true_vals, top_k_values).item()
                        e2e_result_df_list.append([db_size, embedding_dim, 'cos_sim', f1_error, mrr_error, max_vals_mse, timetaken, distribution, k])
                    except Exception as e:
                        print(e, f" Error for --  k:{k}, dbsize:{db_size}, embedding_dim:{embedding_dim}, distribution:{distribution}")

                    e2e_result_df = pd.DataFrame(e2e_result_df_list, columns=['db_size', 'embedding_dim', 'function', 'f1', 'mrr', 'max_vals_mse', 'time', 'distribution', 'k'])
                    e2e_result_df.to_csv('results/e2e_result_df.csv', index=False)


def IVF_experiments():
    from ptmodels import IVFRetrievalModel

    # # if distribution == 'normal':
    # embedding_dim = 768
    # db_size = 10000
    # k =100
    # c = 10
    # nprobe = 50
    # distance_func = 'cos_sim'

    db_sizes = list(range(50000,1000001, 50000))
    np.random.shuffle(db_sizes)

    IVF_experiments_df_list = []
    for embedding_dim in [768]: # [256, 768, 1024, 2048, 4096, 8192]:
        for db_size in tqdm(db_sizes): 
            for distribution in ['normal', 'uniform']: # , 'binary', 'normal_large'
                if distribution == 'normal':
                    query_vector =torch.tensor(np.random.normal(0,0.05, size=(1, embedding_dim)), dtype=torch.float32)
                    database_vectors = torch.tensor(np.random.normal(0,0.05, size=(db_size, embedding_dim)), dtype=torch.float32)
                elif distribution == 'uniform':
                    query_vector = torch.rand(1,embedding_dim, dtype=torch.float32) * 2 - 1
                    database_vectors = torch.rand(db_size, embedding_dim, dtype=torch.float32) * 2 - 1   
                elif distribution == 'binary':
                    query_vector = torch.randint(0,2, size=(1, embedding_dim), dtype=torch.float32)
                    database_vectors = torch.randint(0,2, size=(embedding_dim, db_size), dtype=torch.float32)
                elif distribution == 'normal_large':
                    query_vector =torch.tensor(np.random.normal(0,1, size=(1, embedding_dim)), dtype=torch.float32)
                    database_vectors = torch.tensor(np.random.normal(0,1, size=(embedding_dim, db_size)), dtype=torch.float32)

                for distance_func in ['cos_sim', 'dot_prod']:
                    for c in [1,2,3,5,10,12,15,20]:
                        for nprobe in [10,20,40,60,80,100,120,140]:
                            model = IVFRetrievalModel(nlist=int(c*np.sqrt(db_size)), nprobe=nprobe, distance_func=distance_func)
                            _, modeltrain_time = timethis(model.train, database_vectors)

                            for k in [1,3,5,10,20,35,50,75,100]: # + list(range(50, 500, 50)) + list(range(500, min(db_size, 2000), 100)):     
                                # model.query(query_vector, k)
                                try:
                                    (top_k_indices, top_k_values), timetaken = timethis(model.query, query_vector, k)
                                    top_k_indices, top_k_values = torch.tensor(top_k_indices), torch.tensor(top_k_values)
                        
                                    (true_vals, true_idx), default_time = timethis(lambda a,b,c: torch.topk(torch.cosine_similarity(a,b), c), query_vector, database_vectors, k)

                                    f1_error = top_k_f1(true_idx, top_k_indices)
                                    mrr_error = mrr(true_idx, top_k_indices)

                                    IVF_experiments_df_list.append([db_size, embedding_dim, 'IVF', f1_error, mrr_error, timetaken, default_time, distribution, k, modeltrain_time, c, nprobe, distance_func])
                                except (ValueError, AttributeError) as e:
                                    print(e, f"for --  k:{k}, c: {c}, dbsize:{db_size}, nlist:{int(c*np.sqrt(db_size))}, nprobe:{nprobe}")
                                    IVF_experiments_df_list.append([db_size, embedding_dim, 'IVF', np.NaN, np.NaN, np.NaN, np.NaN, distribution, k, modeltrain_time, c, nprobe, distance_func])

                    IVF_experiments_df = pd.DataFrame(IVF_experiments_df_list, columns=['db_size', 'embedding_dim', 'function', 'f1', 'mrr', 'time', 'default_time', 'distribution', 'k', 'modeltrain_time', 'c', 'nprobe', 'distance_func'])
                    IVF_experiments_df.to_csv('results/IVF_experiments_df.csv', index=False)

                    print("Finished all c, nprob, & k for func: {} - DB:{} - Embd:{} - Dist:{}".format(distance_func, db_size, embedding_dim, distribution))


def IVF_MPC_experiments():
    from mpcmodels import MPCIVFRetrievalModel
    import torch, numpy as np, pickle
    import crypten.mpc as mpc
    import crypten

    #initialize crypten
    crypten.init()
    #Disables OpenMP threads -- needed by @mpc.run_multiprocess which uses fork
    torch.set_num_threads(1)
    # set_precision(30)


    # embedding_dim = 768
    # db_size = 10000
    # k =100
    # c = 10
    # nprobe = 50
    # distance_func = 'cos_sim'

    # query_vector =torch.tensor(np.random.normal(0,0.05, size=(1, embedding_dim)), dtype=torch.float32)
    # database_vectors = torch.tensor(np.random.normal(0,0.05, size=(db_size, embedding_dim)), dtype=torch.float32)
    
    # model = MPCIVFRetrievalModel(nlist=50, nprobe=10)
    # model.train(database_vectors)

    # db_sizes = list(range(5000,100001, 5000))
    # np.random.shuffle(db_sizes)

    db_sizes = list(range(10**5,10**7,10**5)) + list(range(10**7, 10**8+1, 10**7))
    # np.random.shuffle(db_sizes)

    # db_sizes = list(range(5000,100001, 5000))
    # np.random.shuffle(db_sizes)

    IVF_MPC_experiments_df_list = []
    for embedding_dim in [768]: # [256, 768, 1024, 2048, 4096, 8192]:
        for db_size in tqdm(db_sizes):
            for distribution in ['normal']: # , 'binary', 'normal_large'
                if distribution == 'normal':
                    query_vector =torch.tensor(np.random.normal(0,0.05, size=(1, embedding_dim)), dtype=torch.float32)
                    database_vectors = torch.tensor(np.random.normal(0,0.05, size=(db_size, embedding_dim)), dtype=torch.float32)
                elif distribution == 'uniform':
                    query_vector = torch.rand(1,embedding_dim, dtype=torch.float32) * 2 - 1
                    database_vectors = torch.rand(db_size, embedding_dim, dtype=torch.float32) * 2 - 1   
                elif distribution == 'binary':
                    query_vector = torch.randint(0,2, size=(1, embedding_dim), dtype=torch.float32)
                    database_vectors = torch.randint(0,2, size=(embedding_dim, db_size), dtype=torch.float32)
                elif distribution == 'normal_large':
                    query_vector =torch.tensor(np.random.normal(0,1, size=(1, embedding_dim)), dtype=torch.float32)
                    database_vectors = torch.tensor(np.random.normal(0,1, size=(embedding_dim, db_size)), dtype=torch.float32)

                for distance_func in ['cos_sim', 'dot_prod']: # , 'dot_prod'

                    for c in [10]:
                        for nprobe in [10,50,100]:
                            model = MPCIVFRetrievalModel(nlist=int(c*np.sqrt(db_size)), nprobe=nprobe, distance_func=distance_func)
                            _, modeltrain_time = timethis(model.train, database_vectors)

                            for k in [50, 100]: # + list(range(50, 500, 50)) + list(range(500, min(db_size, 2000), 100)):     
                                # model.query(query_vector, k)
                                try:
                                    @mpc.run_multiprocess(world_size=2)
                                    def get_top_k(query_vector, k):
                                        model.encrypt()
                                        top_k, _ = model.query(query_vector, k)
                                        return pickle.dumps(top_k)
                                    
                                    top_k_binary, timetaken = timethis(get_top_k, query_vector, k)
                                    if top_k_binary:
                                        top_k_indices = pickle.loads(top_k_binary[0])
                                    else: 
                                        raise ValueError("No top k returned (timeout)")
                        
                                    (true_vals, true_idx), default_time = timethis(lambda a,b,c: torch.topk(torch.cosine_similarity(a,b), c), query_vector, database_vectors, k)

                                    f1_error = top_k_f1(true_idx, top_k_indices)
                                    mrr_error = mrr(true_idx, top_k_indices)

                                    IVF_MPC_experiments_df_list.append([db_size, embedding_dim, 'IVF', f1_error, mrr_error, timetaken, default_time, distribution, k, modeltrain_time, c, nprobe, distance_func])
                                except ValueError as e:
                                    print(e, f"for --  k:{k}, c: {c}, dbsize:{db_size}, nlist:{int(c*np.sqrt(db_size))}, nprobe:{nprobe}")
                                    IVF_MPC_experiments_df_list.append([db_size, embedding_dim, 'IVF', np.NaN, np.NaN, np.NaN, np.NaN, distribution, k, modeltrain_time, c, nprobe, distance_func])

                    IVF_MPC_experiments_df = pd.DataFrame(IVF_MPC_experiments_df_list, columns=['db_size', 'embedding_dim', 'function', 'f1', 'mrr', 'time', 'default_time', 'distribution', 'k', 'modeltrain_time', 'c', 'nprobe', 'distance_func'])
                    IVF_MPC_experiments_df.to_csv('results/IVF_MPC_experiments_df.csv', index=False)

                    print("Finished all c, nprob, & k for func: {} - DB:{} - Embd:{} - Dist:{}".format(distance_func, db_size, embedding_dim, distribution))
   
    

if __name__ == '__main__':

    # Take in args to choose what to run
    parser = argparse.ArgumentParser(description='Run speed vs size scaling experiments')
    parser.add_argument('--distance', action='store_true', help='Run distance experiments')
    parser.add_argument('--argmax', action='store_true', help='Run argmax experiments')
    parser.add_argument('--top_k', action='store_true', help='Run top_k experiments')
    parser.add_argument('--e2e', action='store_true', help='End to end')
    parser.add_argument('--IVF', action='store_true', help='IVF experiments')
    parser.add_argument('--IVF_MPC', action='store_true', help='IVF-MPC experiments')


    args = parser.parse_args()

    if args.distance:
        print("Running distance experiments")
        distance_experiements()
    if args.argmax:
        print("Running argmax experiments")
        argmax_experiments()
    if args.top_k:
        print("Running top_k experiments")
        top_k_experiments()
    if args.e2e:
        print("Running end-to-end experiments")
        end_to_end_experiments()
    if args.IVF:
        print("Running IVF experiments")
        IVF_experiments()
    if args.IVF_MPC:
        print("Running IVF-MPC experiments")
        IVF_MPC_experiments()
    
   
def visualize():
    """This will never be run on the command line but helps with analysis"""

    import seaborn as sns, pandas as pd, numpy as np
    import matplotlib.pyplot as plt

    distance_result_df = pd.read_csv('results/distance_result_df.csv')
    simple_argmax_df = pd.read_csv('results/simple_argmax_df.csv')
    top_k_recall_df = pd.read_csv('results/top_k_recall_df_old.csv')
    e2e_result_df = pd.read_csv('results/e2e_result_df.csv')

    def distance():
        # groupby function and get mean and CIs
        distance_result_df.groupby('function')['error'].mean()
        distance_result_df.groupby('function')['error'].std()
        distance_result_df.groupby('function')['error'].apply(len)  
        distance_result_df.groupby('function')['error'].apply(lambda x: 1.96 * x.std() / np.sqrt(len(x)))

        distance_result_df['error'].mean()

        distance_result_df.groupby('distribution')['error'].mean()

        # distance_result_df = distance_result_df[distance_result_df['function'] != 'cosine_similarity_mpc_pass_in_mags']

        # distance_result_df.groupby('db_size')['error'].mean()

        # distance_result_df['error_clean'] = distance_result_df['error'].apply(lambda x: x if x < 0.05 else np.NaN)
        distance_result_df['elements'] = distance_result_df['db_size'] * distance_result_df['embedding_dim']
        sns.scatterplot(data=distance_result_df, x='elements', y='time', hue='function')

        sns.scatterplot(data=distance_result_df, x='elements', y='error', hue='function')
        plt.yscale('log')

        distance_result_df.groupby(['function', 'distribution'])['error'].mean()

        distance_result_df['time_increase'] = distance_result_df['time'] / distance_result_df['default_time']
        sns.scatterplot(data=distance_result_df, x='elements', y='time_increase', hue='function')
        sns.scatterplot(data=distance_result_df, x='elements', y='error_clean', hue='function')
        plt.yscale('log')


    def top_k():
        top_k_recall_df.groupby('k')['f1'].mean()
        top_k_recall_df.groupby('distribution')['f1'].mean()

        top_k_recall_df[top_k_recall_df['distribution'] == 'normal'].groupby('k')['f1'].mean()

        top_k_recall_df.groupby('db_size')['time'].mean()

        top_k_recall_df['time_increase'] = top_k_recall_df['time'] / top_k_recall_df['default_time']
        sns.scatterplot(data=top_k_recall_df, x='db_size', y='time', hue='k')
        sns.scatterplot(data=top_k_recall_df, x='k', y='time', hue='distribution')
        sns.scatterplot(data=top_k_recall_df, x='db_size', y='time_increase', hue='k')
        top_k_recall_df.groupby('distribution')['time'].mean()


    def e2e():
        e2e_result_df['elements'] = e2e_result_df['db_size'] * e2e_result_df['embedding_dim']
        sns.scatterplot(data=e2e_result_df, x='db_size', y='time', hue='k')
        sns.scatterplot(data=e2e_result_df, x='db_size', y='time', hue='function')
        sns.scatterplot(data=e2e_result_df, x='db_size', y='f1', hue='function')


        e2e_result_df.groupby('k')['f1'].mean()
        e2e_result_df.groupby('distribution')['f1'].mean()
        e2e_result_df.groupby('function')['f1'].mean()

        e2e_result_df[e2e_result_df['function'] == 'cos_sim'].groupby('db_size')['f1'].mean()
        e2e_result_df[e2e_result_df['function'] == 'dot_score'].groupby('k')['f1'].mean()

        sns.scatterplot(data=e2e_result_df[e2e_result_df['function'] == 'dot_score'], x='db_size', y='f1', hue='k')



        correct_e2e = e2e_result_df[(e2e_result_df['distribution'] != 'binary') & (e2e_result_df['function'] == 'dot_score') & (e2e_result_df['db_size'] < 200000 ) & (e2e_result_df['embedding_dim'] == 256)]

        
        sns.scatterplot(data=correct_e2e, x='db_size', y='time', hue='k')

        # Get F1 means by k and get CIs
        kf1_means = correct_e2e.groupby('k')['f1'].mean()
        kf1_sd = correct_e2e.groupby('k')['f1'].std()
        kf1_nsample = correct_e2e.groupby('k').apply(len)
        kf1_ci = 1.96 * kf1_sd / kf1_nsample.apply(np.sqrt)

        # Make into latex table
        kf1_means = kf1_means.apply(lambda x: f'{x:.3f}')
        kf1_ci = kf1_ci.apply(lambda x: f'{x:.3f}')
        kf1_nsample = kf1_nsample.apply(lambda x: f'{x}')
        kf1_table = pd.DataFrame({'k': kf1_means.index, 'f1': kf1_means.values, 'ci': kf1_ci.values, 'nsample': kf1_nsample.values})
        print(kf1_table.to_latex(index=False))

        # Make a bar chart of the f1 scores by k in correct_e2e with error bars (set the yaxis between 0.95 and 1)
        sns.barplot(data=correct_e2e, x='k', y='f1', ci='sd')
        plt.ylim(0.95, 1)

        fig, ax = plt.subplots(1,1, figsize=(3.5*2, 2.625*2))
        sns.scatterplot(data=correct_e2e, x='db_size', y='time', hue='k', ax=ax)
        # legend title to topk
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[:], labels=labels[:], title='Top K', loc='upper left')
        ax.set_xlabel('Database Size')
        ax.set_ylabel('MPC Retrieval Time (s)')
        plt.show()


    def ivf():
        ivf_experiments_df = pd.read_csv('results/IVF_experiments_df.csv')
        # IVF_experiments_df = pd.read_csv('results/IVF_large_experiments_df.csv')
        # IVF_small_experiments_df = pd.read_csv('results/IVF_small_experiments_df.csv')

        # ivf_experiments_df = pd.concat([IVF_experiments_df, IVF_small_experiments_df, ivf_experiments_df])

        sub_df = ivf_experiments_df.copy()
        sub_df = sub_df[(sub_df['c']==3) & (sub_df['nprobe']==10)]

        sub_df.plot.scatter(x='db_size', y='time')


        ivf_experiments_df.groupby('distribution')['time'].mean()
        ivf_experiments_df.groupby('distribution')['default_time'].mean()
        ivf_experiments_df.groupby('distribution')['f1'].mean()

        ivf_experiments_df.groupby('distance_func')['time'].mean()
        ivf_experiments_df.groupby('distance_func')['f1'].mean()


        ivf_experiments_df['sqrtdbxprobe'] = np.sqrt(ivf_experiments_df['db_size']) * ivf_experiments_df['nprobe']
        ivf_experiments_df['csqrtdbxprobe'] = ivf_experiments_df['c'] * ivf_experiments_df['sqrtdbxprobe']
        ivf_experiments_df['sqrtdbxprobec'] =  ivf_experiments_df['sqrtdbxprobe'] / ivf_experiments_df['c']


        ivf_experiments_df['nlist'] = ivf_experiments_df['c'] * np.sqrt(ivf_experiments_df['db_size'])
        ivf_experiments_df['bucketsize'] = ivf_experiments_df['db_size'] / ivf_experiments_df['nlist']
        ivf_experiments_df['bucketxnprobe'] = ivf_experiments_df['bucketsize'] * ivf_experiments_df['nprobe']

        #  nprobe * sqrt(db) / c 

        sub_df = ivf_experiments_df[(ivf_experiments_df['distribution'] == 'normal') & (ivf_experiments_df['distance_func'] == 'dot_prod')]


        # Tuning choices
        sub_df.groupby('c')['time'].mean()
        sub_df.groupby('c')['f1'].mean()

        sub_df.groupby('nprobe')['time'].mean()
        sub_df.groupby('nprobe')['f1'].mean()

        sub_df.groupby('k')['time'].mean()
        sub_df.groupby('k')['f1'].mean()

        sub_df.groupby('db_size')['time'].mean()
        sub_df.groupby('db_size')['modeltrain_time'].mean()

        # plot 1: speed nprobe * sqrt(db) / c 
        sub_df.plot.scatter(x='bucketxnprobe', y='time', c='k', colormap='viridis')
        plt.xlabel('Bucket Size $\\times$ nprobe')
        plt.ylabel("IVF Retrieval Time (s)")

        # plot 2: accuracy bucketsize * nprobe / dbsize
        # sqrt(db) / c * nprobe / db = nprobe / sqrt(db) * c
        sub_df['p_probe'] = sub_df['nprobe'] / sub_df['nlist']
        sub_df = sub_df[sub_df['k'] > 5]
        sub_df.plot.scatter(x='p_probe', y='f1', c='c', colormap='viridis')
        plt.xlabel('P(touching databse element) = nprobe / sqrt(db) * c')

        # plot 3: accuracy by c * nprobe
        sub_df.plot.scatter(x='c', y='nprobe', c='f1', colormap='viridis')

        sub_df.groupby(['c','nprobe'])['f1'].mean().reset_index().plot.scatter(x='c', y='nprobe',c='f1', colormap='viridis')

        sub_df.groupby(['c','nprobe'])['time'].mean().reset_index().plot.scatter(x='c', y='nprobe',c='time', colormap='viridis')


        sub_df.plot.scatter(x='time', y='f1', c='c', s=1, colormap='viridis')

        ivf_experiments_df.groupby('db_size')['time'].mean()

        sns.kdeplot(data=sub_df[(sub_df['c'] == 15) | (sub_df['c'] == 1)], x='time', y='f1', hue='c', fill=True, common_norm=False, alpha=0.5)

        sub_df[(sub_df['c'] == 15) | (sub_df['c'] == 1)].plot.scatter(x='time', y='f1', c='c', s=1, colormap='viridis')

        
        ivf_experiments_df['nlist'] = ivf_experiments_df['c'] * np.sqrt(ivf_experiments_df['db_size'])
        ivf_experiments_df['bucketsize'] = ivf_experiments_df['db_size'] / ivf_experiments_df['nlist']
        ivf_experiments_df['bucketxnprobe'] = ivf_experiments_df['bucketsize'] * ivf_experiments_df['nprobe']
        ivf_experiments_df['dbxnprobe'] = ivf_experiments_df['db_size'] * ivf_experiments_df['nprobe']
        ivf_experiments_df['p_probe'] = ivf_experiments_df['nprobe'] / ivf_experiments_df['nlist']

        sns.regplot(data=ivf_experiments_df, x='bucketxnprobe', y='time', lowess=True, scatter_kws={"alpha":0.1, 's':5, 'color':'grey'})
        
        sns.regplot(data=ivf_experiments_df, x='db_size', y='time', lowess=True, scatter_kws={"alpha":0.1, 's':5, 'color':'grey'})


        sns.scatterplot(data=ivf_experiments_df, x='p_probe', y='f1', hue='k')

        sns.scatterplot(data=ivf_experiments_df, x='k', y='mrr', hue='distribution')

        sns.scatterplot(data=ivf_experiments_df, x='c', y='nprobe', hue='f1')

        sns.scatterplot(data=ivf_experiments_df, x='c', y='nprobe', hue='time')
        ivf_experiments_df.groupby(['c', 'nprobe'])['time'].mean()


    def ivfmpc():
        ivf_MPC_experiments_df = pd.read_csv('results/IVF_MPC_experiments_df.csv')
        sub_MPC_df = ivf_MPC_experiments_df.copy()
        sub_MPC_df = sub_MPC_df[(sub_MPC_df['c']==3) & (sub_MPC_df['nprobe']==10)]

        sub_MPC_df.plot.scatter(x='db_size', y='time')

        ivf_MPC_experiments_df.groupby('distribution')['time'].mean()
        ivf_MPC_experiments_df.groupby('distribution')['default_time'].mean()
        ivf_MPC_experiments_df.groupby('distribution')['f1'].mean()
        ivf_MPC_experiments_df.groupby('distance_func')['time'].mean()
        ivf_MPC_experiments_df.groupby('distance_func')['f1'].mean()

        ivf_MPC_experiments_df['sqrtdb'] = np.sqrt(ivf_MPC_experiments_df['db_size'])
        ivf_MPC_experiments_df['sqrtdbxprobe'] = ivf_MPC_experiments_df['sqrtdb'] *ivf_MPC_experiments_df['nprobe']

        ivf_MPC_experiments_df['nlist'] = ivf_MPC_experiments_df['c'] * np.sqrt(ivf_MPC_experiments_df['db_size'])
        ivf_MPC_experiments_df['bucketsize'] = ivf_MPC_experiments_df['db_size'] / ivf_MPC_experiments_df['nlist']
        ivf_MPC_experiments_df['bucketxnprobe'] = ivf_MPC_experiments_df['c']* ivf_MPC_experiments_df['bucketsize'] * ivf_MPC_experiments_df['nprobe']
        ivf_MPC_experiments_df['dbxnprobe'] = ivf_MPC_experiments_df['db_size'] * ivf_MPC_experiments_df['nprobe']
        ivf_MPC_experiments_df['p_probe'] = ivf_MPC_experiments_df['nprobe'] / ivf_MPC_experiments_df['nlist']

        sns.scatterplot(data=ivf_MPC_experiments_df, x='sqrtdbxprobe', y='time', hue='db_size')

        sns.scatterplot(data=ivf_MPC_experiments_df, x='db_size', y='time', hue='nprobe')

        sns.scatterplot(data=ivf_MPC_experiments_df, x='nprobe', y='f1', hue='nlist')


        #  c * db_size / c sqrt(dbsize)  * nprobe


