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
```bash
/scratch1/
    wildlife_conservation_data/spac_gorillas_converted/
        12425 videos as .mp4
        total lenght: 8249.79 min.
        average video length: 39.83 sec
        total number of frames: 29710260
        average fps: 60.0
        resolution: 1920/1080
        used cameras: 40
        filmed on 349 days between 27.05.2021 and 09.03.2023

    wildlife_conservation/data/all_gorillas/bristol/full_images
        images: 5409
        individuals: 7
        images/individual
            min: 488
            max: 957
            avg: 772.71

    wildlife_conservation/data/all_gorillas/cxl/full_images
        images: 944
        individuals: 120
        images/individual
            min: 1
            max: 55
            avg: 7.87
        camera/individual: 1.158
        days/individual: 1.625
	    >=3 images
	        individuals: 106
	        avg image/individual: 8.72

    wildlife_conservation/data/all_gorillas/cxl/face_images
        images: 854
        individuals: 115
        images/individual
            min: 1
            max: 50
            avg: 7.43
        camera/individual: 1.148
        days/individual: 1.617
	    >=3 images
	        individuals: 96
	        avg image/individual: 8.57

    wildlife_conservation/data/all_gorillas/cxl/body_images
        images: 846
        individuals: 115
        images/individual
            min: 1
            max: 50
            avg: 7.36
        camera/individual: 1.148
        days/individual: 1.617
	    >=3 images
	        individuals: 96
	        avg image/individual: 8.50
```


#### Development Setup
You can prevent readding your git name and email after devcontainer rebuild by 
placing them in a `.gitconfig`. It will not be commited to remote.

```
[user]
    name = Your Name
    email = some.body@student.hpi.de
``` 
