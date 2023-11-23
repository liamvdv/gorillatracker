import os
import shutil
import sys
import ultralytics
from gorillatracker.scripts.dataset_splitter import generate_split
from gorillatracker.scripts.ensure_integrity_openset import ensure_integrity
from gorillatracker.scripts.train_yolo import build_yolo_dataset, train_yolo, detect_gorillafaces_cxl
from gorillatracker.scripts.crop_dataset import crop_images

if __name__ == "__main__":
    # build an openset split
    
    # 1. split the dataset into train, val and test 
    # bristol_dir = generate_split(dataset="ground_truth/bristol/full_images", mode="openset", seed=43, reid_factor_test=10, reid_factor_val=10)
    # print("Created bristol split")
    # cxl_dir = generate_split(dataset="ground_truth/cxl/full_images", mode="openset", seed=43, reid_factor_test=10, reid_factor_val=10)
    # print("Created cxl split")
    
    # 2. ensure integrity of the split
    bristol_dir = "/workspaces/gorillatracker/data/splits/ground_truth-bristol-full_images-openset-reid-val-10-test-10-mintraincount-3-seed-43-train-70-val-15-test-15"
    cxl_dir = "/workspaces/gorillatracker/data/splits/ground_truth-cxl-full_images-openset-reid-val-10-test-10-mintraincount-3-seed-43-train-70-val-15-test-15"
    train_dir = os.path.join(bristol_dir, "train")
    val_dir = os.path.join(bristol_dir, "val")
    test_dir = os.path.join(bristol_dir, "test")
    bbox_dir = "/workspaces/gorillatracker/data/ground_truth/bristol/full_images_face_bbox"
    
    ensure_integrity(train_dir, val_dir, test_dir, bbox_dir)
    
    # # 3. train yolo model on the bristol dataset split (openset)
    # model_name = "yolov8x"
    # epochs = 30
    # batch_size = 32
    
    # # build yolo dataset 
    build_yolo_dataset(bristol_dir, bbox_dir)
    
    # train_yolo(model_name, epochs, batch_size)
    
    # # 4. predict on the cxl dataset split (openset)
    model_path = "/workspaces/gorillatracker/runs/detect/yolov8x-e30-b16/weights/best.pt"
    model = ultralytics.YOLO(model_path)
    detect_gorillafaces_cxl(model, os.path.join(cxl_dir, "test"))
    detect_gorillafaces_cxl(model, os.path.join(cxl_dir, "val"))
    detect_gorillafaces_cxl(model, os.path.join(cxl_dir, "train"))
    
    # 5. join into one dataset (openset) and  6. crop the images (openset)
    joined_base_dir = "/workspaces/gorillatracker/data/joined_splits"
    # TODO(rob2u): find nice path for this
    # joined_dir = os.path.join(joined_base_dir, cxl_dir.split("/")[-1] + "_combined_" + bristol_dir.split("/")[-1]) 
    joined_dir = os.path.join(joined_base_dir, "combined")
    
    
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(joined_dir, split), exist_ok=True)

        # copy the files
        for file in os.listdir(os.path.join(bristol_dir, split)):
            shutil.copyfile(os.path.join(bristol_dir, split, file), os.path.join(joined_dir, split, file))
        print(f"Copied bristol files to {split}")
        
        for file in os.listdir(os.path.join(cxl_dir, split)):
            shutil.copyfile(os.path.join(cxl_dir, split, file), os.path.join(joined_dir, split, file)) 
        print(f"Copied cxl files to {split}")  
        
        os.makedirs(os.path.join(joined_dir + "_cropped", split), exist_ok=True) 
        crop_images(os.path.join(joined_dir, split),  os.path.join(joined_dir, split), os.path.join(joined_dir + "_cropped", split), file_extension='.jpg')
        crop_images(os.path.join(joined_dir, split), os.path.join(joined_dir, split), os.path.join(joined_dir + "_cropped", split), file_extension='.png', is_bristol=False)
             
    
    # 7. enjoy
    # ground_truth-cxl-full_images-openset-reid-val-10-test-10-mintraincount-3-seed-43-train-70-val-15-test-15_combined_ground_truth-bristol-full_images-openset-reid-val-10-test-10-mintraincount-3-seed-43-train-70-val-15-test-15
    # ground_truth-cxl-full_images-openset-reid-val-10-test-10-mintraincount-3-seed-43-train-70-val-15-test-15_combined_ground_truth-bristol-full_images-openset-reid-val-10-test-10-mintraincount-3-seed-43-train-70-val-15-test-15_cropped