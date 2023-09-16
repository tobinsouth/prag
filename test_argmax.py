import crypten
import torch, numpy as np
import crypten.mpc as mpc
import time, pickle
from crypten.config import cfg
from tqdm import tqdm
import pandas as pd
import seaborn as sns, matplotlib.pyplot as plt

@mpc.run_multiprocess(world_size=2)
def test_argmax(v: torch.Tensor):
    v_enc = crypten.cryptensor(v)
    v_argmax = v_enc.argmax()
    return pickle.dumps(v_argmax.get_plain_text())

size = 1000


timesandargmax = []
for size in tqdm(torch.logspace(2,9,steps=100)):
    size = int(size.item())
    v = torch.rand(size)
    for max_method in ["log_reduction", "pairwise"]:
        cfg.functions.max_method = max_method
        start = time.time()
        v_argmax = test_argmax(v)
        end = time.time()
        timetaken = end - start
        timesandargmax.append((size, timetaken, max_method))

timesandargmax_df = pd.DataFrame(timesandargmax, columns=["size", "time", "max_method"])

timesandargmax_df['size'] = timesandargmax_df['size'].apply(lambda x: int(x.item()))

sns.lineplot(data=timesandargmax_df, x="size", y="time", hue="max_method")

