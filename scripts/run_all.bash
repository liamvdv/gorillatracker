#!/bin/bash
./scripts/kisz-run-in-docker.sh -g 0 python scripts/multi_config_train.py --config_path scripts/multi_configs/baseline_sweep_1.json
./scripts/kisz-run-in-docker.sh -g 1 python scripts/multi_config_train.py --config_path scripts/multi_configs/baseline_sweep_2.json
./scripts/kisz-run-in-docker.sh -g 2 python scripts/multi_config_train.py --config_path scripts/multi_configs/baseline_sweep_qat_and_kd_1.json
./scripts/kisz-run-in-docker.sh -g 3 python scripts/multi_config_train.py --config_path scripts/multi_configs/baseline_sweep_qat_and_kd_2.json
./scripts/kisz-run-in-docker.sh -g 4 python scripts/multi_config_train.py --config_path scripts/multi_configs/baseline_sweep_qat_and_kd_3.json
./scripts/kisz-run-in-docker.sh -g 5 python scripts/multi_config_train.py --config_path scripts/multi_configs/baseline_sweep_quantization_aware_train_1.json
./scripts/kisz-run-in-docker.sh -g 6 python scripts/multi_config_train.py --config_path scripts/multi_configs/baseline_sweep_quantization_aware_train_2.json
./scripts/kisz-run-in-docker.sh -g 7 python scripts/multi_config_train.py --config_path scripts/multi_configs/baseline_sweep_quantization_aware_train_3.json