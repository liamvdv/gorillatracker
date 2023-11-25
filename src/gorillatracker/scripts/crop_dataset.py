from PIL import Image
import os


index_to_name = {
    0: "afia",
    1: "ayana",
    2: "jock",
    3: "kala",
    4: "kera",
    5: "kukuena",
    6: "touni"
}

def crop_and_save_image(image_path, x, y, w, h, output_path):
    # Open the image
    img = Image.open(image_path)

    # Calculate pixel coordinates from relative coordinates
    img_width, img_height = img.size
    left = int((x - w/2) * img_width)
    right = int((x + w/2) * img_width)
    top = int((y - h/2) * img_height)
    bottom = int((y + h/2) * img_height)

    # Crop the image
    cropped_img = img.crop((left, top, right, bottom))

    # Save the cropped image to the output folder
    cropped_img.save(output_path)
   
   
def read_bbox_data(bbox_path):
    # check if the file exists
    if not os.path.exists(bbox_path):
        print("warning: no bounding box found for image with path " + bbox_path)
        return []
    
    bbox_data_lines = []
    with open(bbox_path, 'r') as bbox_file:
        bbox_data_lines = bbox_file.read().strip().split("\n")
    
    bbox_data_lines = [list(map(float, bbox_data_line.strip().split(" "))) for bbox_data_line in bbox_data_lines]
    
    return bbox_data_lines
    
    
def crop_bristol(image_path, bbox_path, output_dir): 
    bbox_data_lines = read_bbox_data(bbox_path)
    
    for index, x, y, w, h in bbox_data_lines:
        name = index_to_name[index]
        file_name = name + "_" + os.path.basename(image_path)
        output_path = os.path.join(output_dir, file_name)
        crop_and_save_image(image_path, x, y, w, h, output_path)


def crop_cxl(image_path, bbox_path, output_dir):    
    bbox_data_lines = read_bbox_data(bbox_path)
    bbo_max_confidence_idx, bbox_max_confidence = max(enumerate(bbox_data_lines), key=lambda x: x[1][-1], default=(-1, -1)) # get the shortest line in the file
                    
    if bbo_max_confidence_idx != -1: # when there is a bounding box
        index , x, y, w, h, _ = bbox_max_confidence
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        crop_and_save_image(image_path, x, y, w, h, output_path)
        

# TODO: only keep the bounding box with the highest confidence score (if there are multiple)
def crop_images(image_dir, bbox_dir, output_dir, file_extension='.jpg', is_bristol = True):
    # Ensure output folder exists
    os.makedirs(output_dir, exist_ok=True)

    # Get list of image files
    image_files = [f for f in os.listdir(image_dir) if f.endswith(file_extension)]

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        bbox_path = os.path.join(bbox_dir, image_file.replace(file_extension, '.txt'))

        if not os.path.exists(bbox_path):
            print("warning: no bounding box found for image " + image_file)
            continue
        
        if is_bristol:
            crop_bristol(image_path, bbox_path, output_dir)
        else:
            crop_cxl(image_path, bbox_path, output_dir)
        


     
if __name__ == "__main__":
    
    full_images_folder = "/workspaces/gorillatracker/data/ground_truth/bristol/full_images"
    bbox_folder = "/workspaces/gorillatracker/data/ground_truth/bristol/full_images_face_bbox"
    output_folder = "/workspaces/gorillatracker/data/ground_truth/bristol/cropped_images_face"

    # crop_images(full_images_folder, bbox_folder, output_folder)
    test_bbox = "/workspaces/gorillatracker/data/joined_splits/combined/train/afia-1-img-0.txt"
    data = read_bbox_data(test_bbox)
    print(data)