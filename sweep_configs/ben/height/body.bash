#!/bin/bash

# Configuration
config_path="/workspaces/gorillatracker/cfgs/visiontransformer_simclr_body.yml"
project_name="Embedding-ViTLarge-SSL-Height"
entity="gorillas"
run_name="sweep-simclr-body-height"

# Array of feature types
feature_types=("body" "body_with_face")

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
  # "450 500"
  # "500 550"
  # "550 600"
  # "600 650"
  # "650 700"
  # "700 750"
  # "750 5000"
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
