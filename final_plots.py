import pandas as pd, matplotlib.pyplot as plt, seaborn as sns, numpy as np
import tueplots

from tueplots import bundles
plt.rcParams.update(bundles.icml2022())


e2e_result_df = pd.read_csv('results/e2e_result_df.csv')


correct_e2e = e2e_result_df[(e2e_result_df['distribution'] != 'binary') & (e2e_result_df['function'] == 'cos_sim') & (e2e_result_df['embedding_dim'] == 256)]


fig, ax = plt.subplots(figsize=(3.25, 2.25))
sns.scatterplot(data=correct_e2e, x='db_size', y='time', hue='k', s=20, ax=ax)
ax.set_xlabel("Database size (N)")
ax.set_ylabel("Query time (s), MPC exact")
plt.show()


dataset_name = "fiqa_full_mdb3"
recall_by_nprobe_df = pd.read_csv(f"results/ivf_f1_recall_by_nprobe_{dataset_name}.csv")

recall_by_nprobe_df['axis_prop'] = recall_by_nprobe_df['nprobe']/max(recall_by_nprobe_df['nprobe'])

fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(3.25, 3.25), sharex=True)
recall_by_nprobe_df.plot(x ='axis_prop', y='recall', kind = 'scatter', ax=ax1, c='red', s=1, alpha=0.3)
# recall_by_nprobe_df.groupby('axis_prop')['recall'].mean().plot(ax=ax1, c='red')
ax1.set_ylabel("Recall")

recall_by_nprobe_df.plot(x ='axis_prop', y='mpc_time', kind = 'scatter', ax=ax2,  alpha=0.3, c='purple', s=1)
# recall_by_nprobe_df.groupby('axis_prop')['mpc_time'].mean().plot(ax=ax2, c='purple')
ax2.set_ylabel("Query time (s)")
ax2.set_xlabel("Fraction of IVF index searched")
# fig.title("MPC IVF Query on fiqa dataset with mdb3")
plt.show()