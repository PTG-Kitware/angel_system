# Step-by-step how to run the Angel System pipeline

## Table of Contents
- [Local installation](#localinstallation)
- [Docker installation](#dockerinstallation)
- [Data and pretrained models](#data)
- [Training](#training)
- [Docker local testing with pre-recorded data](#local)
- [Real-time](#realtime)

## Local Installation <a name = "localinstallation"></a>

Follow the following steps (the optional steps are for active development purposes):

##### Get required repositories:
```
(optional) git clone git@github.com:PTG-Kitware/TCN_HPL.git
(optional) git clone git@github.com:PTG-Kitware/yolov7.git
git clone git@github.com:PTG-Kitware/angel_system.git
cd angel_system
git submodule update --init --recursive
```

##### Create the environment
```
conda create --name angel_systen python=3.8.10
conda activate angel_test_env
poetry lock --no-update
poetry install
```

##### 

## Docker Installation <a name = "dockerinstallation"></a>

Follow the following steps (the optional steps are for active development purposes):
##### Get required repositories:
```
(optional) git clone git@github.com:PTG-Kitware/TCN_HPL.git
(optional) git clone git@github.com:PTG-Kitware/yolov7.git
git clone git@github.com:PTG-Kitware/angel_system.git
cd angel_system
git submodule update --init --recursive
```
##### Create the environment
```
./angel-docker-build.sh -f
./angel-workspace-shell.sh
```

##### Inside the docker container:
```
./workspace_build.sh; source install/setup.sh
```

## Data <a name = "data"></a>
- On gyges, raw data is located at `/data/PTG/medical/bbn_data/Release_v0.5/v0.56/<task>`
- pre-trained models are available on `https://data.kitware.com/#collection/62cc5eb8bddec9d0c4fa9ee1/folder/6605bc558b763ca20ae99f55`
- In this pipeline we are only provided with object detection ground truth training data, which is located `/data/PTG/medical/object_anns/<task>`
- For real-time execution, we store our models in /angel_system/model_files

##### Examples of files and what they are used for:

- `/angel_system/tmux/demos/medical/M3-Kitware.yml`: defines complete ROS configuration for all nodes
- `/angel_system/model_files/coco/r18_test_activity_preds.mscoco.json`: required to determine the step activation threshold for the GSP
- `/angel_system/model_files/data_mapping/r18_mapping.txt`: required to define model activity step classes
- `/angel_system/model_files/models/r18_tcn.ckpt`: TCN model checkpoint
- `/angel_system/model_files/models/hands_model.pt`: hand detection trained model
- `/angel_system/model_files/models/r18_det.pt`: object detection trained model


## Training Procedure <a name = "training"></a>

We take the following steps:
1. train object detection model
2. predict objects in the scene
3. predict poses and patient bounding boxes in the scene
4. generate interaction feature vectors for the TCN
5. train the TCN

##### Example with R18
```
cd yolo7
python yolov7/train.py --workers 8 --device 0 --batch-size 4 --data configs/data/PTG/medical/r18_task_objects.yaml --img 768 768 --cfg configs/model/training/PTG/medical/yolov7_r18.yaml --weights weights/yolov7.pt --project /data/PTG/medical/training/yolo_object_detector/train/ --name r18_all_v1_example
python yolov7/detect_ptg.py --tasks r18 --weights /data/PTG/medical/training/yolo_object_detector/train/r18_all_v1_example/weights/best.pt --project /data/PTG/medical/training/yolo_object_detector/detect/ --name r18_all_example --device 0 --img-size 768 --conf-thres 0.25
cd TCN_HPL/tcn_hpl/data/utils/pose_generation/configs
```

with the above scripts, we should get a kwcoco file at:
```
/data/PTG/medical/training/yolo_object_detector/detect/r18_all_example/
```

Edit `TCN_HPL/tcn_hpl/data/utils/pose_generation/configs/main.yaml` with the task in hand (here, we use r18), the path to the output detection kwcoco, and where to output kwcoco files from our pose generation step.
```
cd ..
python generate_pose_data.py
cd TCN_HPL/tcn_hpl/data/utils
```
At this stage, there should be a new kwcoco file generated in the field defined at `main.yaml`:
```
data:
    save_root: <path-to-kwcoco-file-with-pose-and-detections>
```

Next, edit the `/TCN_HPL/configs/experiment/r18/feat_v6.yaml` file with the correct experiment name and kwcoco file in the following fields:

```
exp_name: <experiment-name>
path:
    dataset_kwcoco: <path-to-kwcoco-with-poses-and-dets>
```

Then run the following commands to generate features:
```
python  ptg_datagenerator –task r18 --data_type <bbn or gyges> --config-root <root-to-TCN-HPL-configs> --ptg-root <path-to-local-angel-systen-repo>
cd TCN_HPL/tcn_hpl
python train.py experiment=r18/feat_v6
```

==At this point, we have our trained model at the path specified in our config file. For real-time execurtion, we would need to copy it over to angel_system/model_files==


## Docker local testing <a name = "local"></a>

***to start the service run:***
```
./angel-workspace-shell.sh
tmuxinator start demos/medical/Kitware-R18
```

This will execute the yaml file defined at `/angel_system/ros/demos/medical/Kitware-R18.yaml`. We can use the pre-recorded data by ensuring the `sensor_input` line points to a pre-recorded data as such:

`- sensor_input: ros2 bag play ${ANGEL_WORKSPACE_DIR}/ros_bags/rosbag2_2024_04_02-19_57_59-R18_naked_mannequin_no_blue_tarp-timestamp_unconfident/rosbag2_2024_04_02-19_57_59_0.db3`

***to end the service run:***
```
tmuxinator stop demos/medical/Kitware-R18
```

***Any modifications to the code would require to stop the service and re-build the workspace***
```
tmuxinator stop demos/medical/Kitware-R18
./workspace_build.sh; source install/setup.sh
```


## Docker real-time <a name = "realtime"></a>

This step requires a user on the BBN systems to login to the Kitware machine. After it is set up:

```
ssh <username>@kitware-ptg-magic.eln.bbn.com
cd angel/angel_system
git pull
git submodule update --init --recursive
./angel-workspace-shell.sh
./workspace_build.sh; source install/setup.sh
tmuxinator start demos/medical/BBN-integrate-Kitware-R18
```

