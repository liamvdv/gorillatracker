# gorillatracker
The GorillaTracker BP


# Data
Our data setup `/scratch2/gorillatracker` as base.
`data/` contains our data.
    `ground_truth/<dataset>/` only contains , e. g. `bristol`, `cxl` 
    `derived_dataset/<dataset>`
    `splits/<split-id>`
`models/` contains our trained model weights (push them to W&B too though!)

# Architecture
### Adding a Dataset
1. create a Dataset that supports __getitem__(self, idx: int) (read: single element access) in `gorillatracker.dataset_<name>.py`. 
2. create a subclass from `gorillatracker.data_modules.TripletDataModule` and `gorillatracker.data_modules.QuadletDataModule` in `gorillatracker.data_module_<name>.py`, implement `get_dataset_class` and `get_transforms`.

You can now use the data module exposed in `gorillatracker.data_module_<name>.py` for online and offline triplet loss. All the sampling for triplet generation is ready build. 

# Statistcs

```bash
ground_truth/ 
  cxl/ contains 952 images of 120 individuals  
```


#### Development Setup
You can prevent readding your git name and email after devcontainer rebuild by 
placing them in a `.gitconfig`. It will not be commited to remote.

```
[user]
    name = Your Name
    email = some.body@student.hpi.de
``` 