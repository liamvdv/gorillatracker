# nohup python3 cluster_sweep.py > output.log 2> error.log &
from gorillatracker.classification.reid import sweep_configs, EXT_MERGED_DF, configs
import pickle

print("Number of configurations:", len(configs))
results = sweep_configs(EXT_MERGED_DF, configs, resolution=100, metric="cosine", cache_dir="sep29_reid_cache_cosine")

with open("sep29_reid_results_cosine.pkl", "wb") as f:
    pickle.dump(results, f)

# results = sweep_configs(
#     EXT_MERGED_DF, configs, resolution=100, metric="euclidean", cache_dir="sep29_reid_cache_euclidean"
# )

# with open("sep29_reid_results_euclidean.pkl", "wb") as f:
#     pickle.dump(results, f)
