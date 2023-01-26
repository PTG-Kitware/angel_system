## Installation

### Requirements
- Linux or macOS with Python ≥ 3.6
- PyTorch ≥ 1.8 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this
- OpenCV is optional but needed by demo and visualization


### Build detectron2 
Step 1: Install Pytroch: following the instruction in https://pytorch.org/ to install the latest version of pytorch.

Step 2: Following the corresponding structure, clone the code, and run:
```
cd ./angel_system/berkeley

python -m pip install.
```

### Download the pre-trained model

Download our [model](https://drive.google.com/file/d/1CfOVLWW7HPLQmndgJ15C70QgjV8UTYJu/view?usp=sharing) and save it to ```./weights``` folder

## Test the model

```
python test_integration.py
```

Please ignore the reading images algorithm in ```test_integration.py```, and directly use the ```predict``` function in it.


## The results you will get by running the model

You will get a dict including the frame level preditions, with the structure of


```
├── frame_id
│   ├── object1
│   │   ├── class
│   │   └── confidence score
|   |   └── bbox
│   ├── object2
│   │   ├── ......
│   │   └── ......
│   ├── object3
│   │   ├── ......
│   │   ├── ......
│   ├   └── ......
│── frame_id

```

You will also get an array with a dimension of ```N * H * W * 3``` of the visualized frames with predicted bounding box in it.

Notice that the keys of the dict should be equal or less than the input number of frames, including no empty predictions, but the visualized images should be the same number of the input images.

