import os
import shutil
from datetime import datetime
from ultralytics import YOLO

# Receive arguments from command line

model_paths = {
    "yolov8n": "/workspaces/gorillatracker/yolov8n.pt",
    "yolov8m": "/workspaces/gorillatracker/yolov8m.pt",
    "yolov8x": "/workspaces/gorillatracker/yolov8x.pt",
}


def train_yolo(model_name, epochs, batch_size, dataset_yml, wandb_project, wandb_run_name):
    """Train a YOLO model with the given parameters."""
    
    model = YOLO(model_paths[model_name])
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    training_name = f"{model_name}-e{epochs}-b{batch_size}-{timestamp}"

    print(f"Training model {model_name} with {epochs} epochs and batch size of {batch_size}")

    result = model.train(name=training_name, data=dataset_yml, epochs=epochs, batch=batch_size, patience=10, project=wandb_project)

    print(f"Training finished for {training_name}")
    return model,result


def set_annotation_class_0(annotation_dir, dest_dir):
    """Set the class of all annotations to 0 (gorilla face) and save them in the destination directory.
    
    Args:
        annotation_dir (str): Directory containing the annotation files.
        dest_dir (str): Directory to save the new annotation files to.
        file_extension (str, optional): File extension of the images. Defaults to ".jpg".
    """
    for annotation_filename in os.listdir(annotation_dir):
        if not annotation_filename.endswith(".txt"):
            continue
        
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
            new_annotation = os.path.join(dest_dir, annotation_filename)
            with open(new_annotation, "+w") as new_annotation_file:
                new_annotation_file.write("\n".join(new_lines))


def join_annotations_and_imgs(image_dir, annotation_dir, output_dir, file_extension=".jpg"):
    """Build a dataset for yolo using the given image and annotation directories.
    
    Args:
        image_dir (str): Directory containing the images.
        annotation_dir (str): Directory containing the annotation files.
        output_dir (str): Directory to merge the images and annotations into.
        file_extension (str, optional): File extension of the images. Defaults to ".png".
    """
    image_files = os.listdir(image_dir)
    image_files = filter(lambda x: x.endswith(file_extension), image_files)
    
    for image_file in image_files:
        annotation_file = image_file.replace(file_extension, ".txt")
        annotation_path = os.path.join(annotation_dir, annotation_file)
        if not os.path.exists(annotation_path):
            raise Exception(f"Annotation file {annotation_path} does not exist")
        
        shutil.copyfile(annotation_path, os.path.join(output_dir, annotation_file))
        if not os.path.exists(os.path.join(output_dir, image_file)):
            shutil.copyfile(os.path.join(image_dir, image_file), os.path.join(output_dir, image_file))


def remove_annotations_from_dir(annotation_dir, file_extension=".txt"):
    for annotation_file in os.listdir(annotation_dir):
        if annotation_file.endswith(file_extension):
            os.remove(os.path.join(annotation_dir, annotation_file))


def detect_gorillafaces_cxl(model, image_dir, output_dir, file_extension=".png"):
    """Detect gorilla faces in the given directory and save the results in the output directory using the given yolo model."""
    image_files = os.listdir(image_dir)
    image_files = filter(lambda x: x.endswith(file_extension), image_files)
    
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        result = model(image_path)
        
        annotation_path = os.path.join(output_dir, image_file.replace(file_extension, ".txt"))
        if os.path.exists(annotation_path):
            os.remove(annotation_path)
        result[0].save_txt(annotation_path, save_conf=True) # NOTE: simply appends to the .txt file
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
    
    