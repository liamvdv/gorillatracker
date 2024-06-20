import os
import shutil

def read_label_in_dirname(datadir_path: str) -> dict[str, str]:
    """read labels for images from directory name"""
    path_to_label_dict = {}
    for label in os.listdir(datadir_path):
        for file in os.listdir(os.path.join(datadir_path, label)):
            path_to_label_dict[os.path.join(datadir_path, label, file)] = label 
    return path_to_label_dict

def write_dir(output_path: str, path_to_label_dict: dict[str, str]):
    """write images to new directory with label in filename"""
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for filepath, id in path_to_label_dict.items():
        new_file_name = f"{id}_{os.path.basename(filepath)}"
        new_file_name = new_file_name.lower()
        new_file_name = new_file_name.replace(" ", "_")
        new_filepath = os.path.join(output_path, new_file_name)
        shutil.copy2(filepath, new_filepath)

if __name__ == "__main__":
    datadir_path = "path/to/data/dir"
    output_path = "path/to/output/dir"
    path_to_label_dict = read_label_in_dirname(datadir_path)
    write_dir(output_path, path_to_label_dict)