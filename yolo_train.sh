#!/bin/bash

# Define paths and settings
MODEL_PATH="/workspaces/gorillatracker/yolov8n.pt"
RUN_NAME="#198-tune-45-degree-yolov8n"
DATA_PATH="/workspaces/gorillatracker/cfgs/yolo_detection_face_45.yaml"
BATCH_SIZES=(4 8 16 32 64 -1) # Array of batch sizes
PYTHON_SCRIPT="yolo_train.py" # Name of your Typer script

# Loop through each batch size for training
for BATCH_SIZE in "${BATCH_SIZES[@]}"
do
    echo "Training with batch size: $BATCH_SIZE"
    # Adjust this line to call your Python Typer script with the `train` command
    python $PYTHON_SCRIPT train "$MODEL_PATH" "${RUN_NAME}-bs-${BATCH_SIZE}" "$DATA_PATH" --epochs 200 --batch-size $BATCH_SIZE --patience 20
done

# Tune the model after training
echo "Tuning model..."
# Adjust this line to call your Python Typer script with the `tune` command
python $PYTHON_SCRIPT tune "$MODEL_PATH" "$RUN_NAME" "$DATA_PATH" --iterations 5

echo "Done."
