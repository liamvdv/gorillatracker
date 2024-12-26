# gorillatracker
This repository is the byproduct of the Gorillatracker Bachelors Project 2023/2024 at HPI. It can be used for easily training computer vision models supervised aswell as self supervised.

# Setup
## Clone Repository
```git clone https://github.com/joschaSchroff/gorillatracker.git```

## Build Docker Image
```docker build -t gorillatracker:latest .```

## Devcontainer
Install [VS-Code devcontainer extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

### Adjust `devcontainer.json`

Ensure the following directories and files exist, and verify that the correct paths are set in your `devcontainer.json` file:

#### Required Directories
- `.ssh`
- `.cache`

#### Required Files
- `.netrc`
- `.gitconfig`

### Open in Devcontainer

1. Open the project in Visual Studio Code.
2. Use the **Command Palette** (`Ctrl+Shift+P` or `Cmd+Shift+P` on macOS) and select:
   - `Dev Containers: Reopen in Container`.
3. Wait for the container to build and start. Once completed, your environment will be ready for development.


### Update `.gitconfig`
You can prevent readding your git name and email after devcontainer rebuild by 
placing them in a `.gitconfig`. It will not be commited to remote.

```
[user]
    name = Your Name
    email = some.body@student.hpi.de
``` 

# Architecture
## Adding a Dataset

1. Create a Dataset that supports __getitem__(self, idx: int) (read: single element access) in `gorillatracker.datasets.<name>.py`.  
If you need to do custom transformations (except resizing), you can also declare a classmethod `get_transforms(cls)`.

2. Select the dataset from your cfgs/<yourconfigname>.yml `dataset_class`.

You can now use the dataset for online and offline triplet loss. All the sampling 
for triplet generation is ready build. 

## Where to transforms go? `dataset_class.get_transforms()` vs  `model_class.get_tensor_transforms()`
The model class should many apply a Resize to it's expected size and if needed enforce number of channels needed.
The dataset class should specify all other transforms and MUST at least transform `torchvision.transforms.ToTensor`.

# Training
## train a model
### inside of devcontainer
- make sure you have mounted the right gpu in devcontainer.json
- run: ```python train.py --config_path cfgs/<yourconfigname>.yml```
### without devcontainer
```bash run-in-docker.sh -g [GPUs] python train.py --config_path cfgs/<yourconfigname>.yml```

## sweep for best hyperparameters
### setup sweep config
set up a json file(see examples in ```/sweep_configs```) with the parameters you want to sweep over.
### start sweep
```bash scripts/run-in-docker.sh -g [GPUs] python init_sweep.py --sweep_config_file sweep_configs/<yoursweepconfigname>.json```









