# nohup python3 cluster_sweep.py > output.log 2> error.log &
from gorillatracker.classification.clustering import sweep_clustering_algorithms, EXT_MERGED_DF, configs

print("Number of configurations:", len(configs))
print("Number of algorithm configurations:", sum(len(params) for _, _, _, params in configs))
results_df = sweep_clustering_algorithms(EXT_MERGED_DF, configs, cache_dir="./cache_cluster_sweep_sep26")
results_df.to_pickle("sep29_clustering_results_euclidean.pkl")



from gorillatracker.classification.clustering_cosine import sweep_clustering_algorithms_cosine

print("Number of configurations:", len(configs))
print("Number of algorithm configurations:", sum(len(params) for _, _, _, params in configs))
results_df = sweep_clustering_algorithms_cosine(EXT_MERGED_DF, configs, cache_dir="./cosine_cache_cluster_sweep_sep29")
results_df.to_pickle("sep29_clustering_results_cosine.pkl")