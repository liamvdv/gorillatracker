# nohup python3 cluster_sweep.py > output.log 2> error.log &
from gorillatracker.classification.reid import sweep_configs, EXT_MERGED_DF, configs
import pickle

print("Number of configurations:", len(configs))
results = sweep_configs(EXT_MERGED_DF, configs, resolution=100, cache_dir="reid_cache_sep26")

with open("sep26_reid_results.pkl", "wb") as f:
    pickle.dump(results, f)
