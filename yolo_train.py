from ultralytics import YOLO

WANDB_PROJECT = "Detection-YoloV8-CXL_Body"
WANDB_ENTITY = "gorillas"

model_paths = {
        "yolov8n": "./models/yolov8n.pt",
        "yolov8m": "./models/yolov8m.pt",
        "yolov8x": "./models/yolov8x.pt",
    }

def train(model_name, training_name, data="./gorilla.yaml",  epochs=100, batch_size = -1, patience=40):
    model = YOLO(model_paths[model_name])
    result = model.train(project=WANDB_PROJECT, name=training_name, data=data, epochs=epochs, batch=batch_size, patience=patience)

if __name__ == "__main__":
    train("yolov8n", "#36-test")
    