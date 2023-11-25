import os
import shutil
from ultralytics import YOLO

# Receive arguments from command line

model_paths = {
    "yolov8n": "/workspaces/gorillatracker/yolov8n.pt",
    "yolov8m": "/workspaces/gorillatracker/yolov8m.pt",
    "yolov8x": "/workspaces/gorillatracker/yolov8x.pt",
}


def train_yolo(model_name, epochs, batch_size):
    model = YOLO(model_paths[model_name])
    training_name = f"{model_name}-e{epochs}-b{batch_size}"

    print(f"Training model {model_name} with {epochs} epochs and batch size of {batch_size}")

    result = model.train(name=training_name, data="/workspaces/gorillatracker/src/gorillatracker/scripts/gorilla.yaml", 
                        epochs=epochs, batch=batch_size, patience=10)

    print(f"Training finished for {training_name}")
    return model,result


def build_yolo_dataset(base_dir, annotation_dir, dest_dir, file_extension=".jpg", undo=False):
    def load_annotation(annotation_dir, destination_dir):
        for image_file in os.listdir(destination_dir):
            annotation_filename = image_file.replace(file_extension, ".txt")
            
            with open(os.path.join(annotation_dir, annotation_filename), 'r') as annotation_file:
                lines = annotation_file.readlines()
                new_lines = []
                for line in lines:
                    line = line.strip()
                    if line:
                        line = line.split(" ")
                        line[0] = "0"
                        line = " ".join(line)
                        new_lines.append(line)
                new_annotation = os.path.join(destination_dir, annotation_filename)
                with open(new_annotation, 'w') as new_annotation_file:
                    new_annotation_file.write("\n".join(new_lines))
    
    def undo_annotation(annotation_dir, destination_dir):
        for image_file in os.listdir(destination_dir):
            annotation_filename = image_file.replace(file_extension, ".txt")
            
            shutil.copyfile(os.path.join(annotation_dir, annotation_filename), os.path.join(destination_dir, annotation_filename))
                  
    train_dir = os.path.join(dest_dir, "train")
    val_dir = os.path.join(dest_dir, "val")
    test_dir = os.path.join(dest_dir, "test")
    
    if undo:
        undo_annotation(annotation_dir, train_dir)
        undo_annotation(annotation_dir, val_dir)
        undo_annotation(annotation_dir, test_dir)
    else:
        load_annotation(annotation_dir, train_dir)
        load_annotation(annotation_dir, val_dir)
        load_annotation(annotation_dir, test_dir)


def detect_gorillafaces_cxl(model, dir):
    image_files = os.listdir(dir)
    image_files = filter(lambda x: x.endswith(".png"), image_files)
    
    for image_file in image_files:
        image_path = os.path.join(dir, image_file)
        result = model(image_path)
        # delete the old .txt file
        if os.path.exists(image_path.replace(".png", ".txt")):
            os.remove(image_path.replace(".png", ".txt"))
        result[0].save_txt(image_path.replace(".png", ".txt"), save_conf=True) # NOTE: simply appends to the .txt file
        # print(f"Saved result for {image_path}")


if __name__ == "__main__":
    model_name = "yolov8x"
    epochs = 30
    batch_size = 16
    
    annotation_folder = "/workspaces/gorillatracker/data/ground_truth/bristol/full_images_face_bbox"
    image_folder = "/workspaces/gorillatracker/data/splits/ground_truth-bristol-full_images-openset-reid-val-10-test-10-mintraincount-3-seed-43-train-70-val-15-test-15"
    
    # build_yolo_dataset(image_folder, annotation_folder)
    model, result = train_yolo(model_name, epochs, batch_size)
    # save model
    model.export() # should be saved somehow (somehow because ultralytics docs are trash)
    
    # predict on /workspaces/gorillatracker/data/splits/ground_truth-cxl-full_images-openset-reid-val-10-test-10-mintraincount-3-seed-43-train-70-val-15-test-15
    base_dir = "/workspaces/gorillatracker/data/splits/ground_truth-cxl-full_images-openset-reid-val-10-test-10-mintraincount-3-seed-43-train-70-val-15-test-15"
    detect_gorillafaces_cxl(model, base_dir + "/test")
    detect_gorillafaces_cxl(model, base_dir + "/val")
    detect_gorillafaces_cxl(model, base_dir + "/train")
    
    