#!/bin/bash

# Configuration
config_path="/workspaces/gorillatracker/cfgs/visiontransformer_simclr_face.yml"
project_name="Embedding-ViTLarge-SSL-Confidence"
entity="gorillas"
run_name="sweep-simclr-face-confidence"

# Array of feature types
feature_types=("face_45" "face_90")

confidence=(0.3 0.5 0.7 0.9)

for feature_type in "${feature_types[@]}"; do
for conf in "${confidence[@]}"; do
    echo "Running training with feature_type=${feature_type} and confidence=${conf}"
    python train.py \
      --config_path "${config_path}" \
      --min_confidence ${conf} \
      --project_name "${project_name}" \
      --run_name "${run_name}-${feature_type}-${conf}" \
      --feature_type "${feature_type}"
  done
done
