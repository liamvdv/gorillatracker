# gorillatracker
The GorillaTracker BP


# Data 
Our data setup `/scratch2/gorillatracker` as base.\
`data/` contains our data.\
&nbsp;&nbsp;&nbsp;&nbsp;`ground_truth/<dataset>/` only contains , e. g. `bristol`, `cxl` \
&nbsp;&nbsp;&nbsp;&nbsp;`derived_dataset/<dataset>`\
&nbsp;&nbsp;&nbsp;&nbsp;`splits/<split-id>`\
&nbsp;&nbsp;&nbsp;&nbsp;`derived_data/`\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`spac_gorillas_labels/` contains a v.json for every v.mp4 where the bounding box of frame i can be accessed through v\["labels"\]\[i\]\
`models/` contains our trained model weights (push them to W&B too though!)
<!-- sorry for the ugly formatting, I couldn't get it to show up otherwise -->


# Architecture
### Adding a Dataset

1. Create a Dataset that supports __getitem__(self, idx: int) (read: single element access) in `gorillatracker.datasets.<name>.py`.  
If you need to do custom transformations (except resizing), you can also declare a classmethod `get_transforms(cls)`.

2. Select the dataset from your cfgs/<yourconfigname>.yml `dataset_class`.

You can now use the dataset for online and offline triplet loss. All the sampling 
for triplet generation is ready build. 

### Where to transforms go? `dataset_class.get_transforms()` vs  `model_class.get_tensor_transforms()`
The model class should many apply a Resize to it's expected size and if needed enforce number of channels needed.
The dataset class should specify all other transforms and MUST at least transform `torchvision.transforms.ToTensor`.


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


#### How to access the Grafana Dashboard?
1. Forward the remote port 3000 to your machine on e. g. port 8000.
To forward the remote Grafana Dashboard, which runs on port 3000 to your laptop port 8000, run
```
ssh -L 8000:localhost:3000 gpuserver2
```

Note that `gpuserver2` is my ssh host alias for the remote host, change it to 
whatever you have named it.

2. Login 
student
AIisthebest





