#!/bin/bash

# Configuration
config_path="/workspaces/gorillatracker/cfgs/visiontransformer_simclr_face.yml"
project_name="Embedding-ViTLarge-SSL-Height"
entity="gorillas"
run_name="sweep-simclr-face-height"

# Array of feature types
feature_types=("face_45" "face_90")

# Array of height ranges
height_ranges=(
  "0 50"
  "50 100"
  # "100 150"
  # "150 200"
  # "200 250"
  # "250 300"
  # "300 350"
  # "350 400"
  # "400 450"
  # "450 5000"
)

# Loop through each feature type and height range and run the training command
for feature_type in "${feature_types[@]}"; do
  for height_range in "${height_ranges[@]}"; do
    echo "Running training with feature_type=${feature_type} and height_range=${height_range}"
    python train.py \
      --config_path "${config_path}" \
      --height_range ${height_range} \
      --project_name "${project_name}" \
      --run_name "${run_name}-${feature_type}-${height_range// /_}" \
      --feature_type "${feature_type}"
  done
done
