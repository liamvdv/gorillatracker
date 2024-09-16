# nohup python3 cluster_sweep.py > output.log 2> error.log &
from gorillatracker.classification.clustering import sweep_clustering_algorithms, EXT_MERGED_DF, configs

print("Number of configurations:", len(configs))
print("Number of algorithm configurations:", sum(len(params) for _, _, _, params in configs))
results_df = sweep_clustering_algorithms(EXT_MERGED_DF, configs, cache_dir="./cache_cluster_sweep")
results_df.to_pickle("results4.pkl")
