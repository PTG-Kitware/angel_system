# Step-by-step how to run the Angel System pipeline

## Table of Contents
- [Local installation](#local-installation)
- [Docker installation](#docker-installation)
- [Data and pretrained models](#data)
- [Training](#training-procedure)
- [Training on lab data](#example-with-r18)
- [Docker local testing with pre-recorded data](#docker-local-testing)
- [Real-time](#docker-real-time)

## Local Installation

Follow the following steps (the optional steps are for active development purposes):

##### Get required repositories:
```
git clone git@github.com:PTG-Kitware/angel_system.git
cd angel_system
git submodule update --init --recursive
```

##### Create the environment
```
# IF YOU DON'T ALREADY HAVE PYTHON 3.8.10 AVAILABLE
conda create --name angel_systen python=3.8.10
conda activate angel_test_env
poetry install

# OR JUST
poetry install
```

## Docker Installation

Follow the following steps (the optional steps are for active development purposes):
### Get required repositories:
```
git clone git@github.com:PTG-Kitware/angel_system.git
cd angel_system
git submodule update --init --recursive
```
### Create the environment
```
./angel-docker-build.sh -f
./angel-workspace-shell.sh
```

### Inside the docker container:
```
./workspace_build.sh; source install/setup.sh
```

## Data
See our data "Where things are" document on Google Drive
[here](https://docs.google.com/document/d/13etNTetmAEuUxbZdrxmHvQ83g3fyORYCHIYASLhS344/edit?).

### Object Detection Source Data
TODO

### Pose Detection Source Data
TODO

### TCN Source Data

#### BBN Medical Datasets
Source data from BBN can be acquired from https://bbn.com/private/ptg-magic/.
Consult a team member for login information.

Each task ("skill") has their own sub-page ("In-Lab Data" link) from which sets
of data are described and referred to.
Data is stored on their SFTP server, to which the "Click to Download" links
refer to.

Storage of downloaded ZIP archives, and their subsequent extractions, should
follow the pattern below.
```
bbn_data/
├── README.md  # Indicate where we have acquired this BBN data.
└── lab_data-golden/
    ├── m2_tourniquet/
    │   ├── positive/
    │   │   ├── Fri-Apr-21/
    │   │   │   ├── 20230420_122603_HoloLens.mp4
    │   │   │   ├── 20230420_122603_HoloLens.skill_labels_by_frame.txt
    │   │   │   ├── 20230420_123212_HoloLens.mp4
    │   │   │   ├── 20230420_123212_HoloLens.skill_labels_by_frame.txt
    │   │   │   ├── 20230420_124541_HoloLens.mp4
    │   │   │   ├── 20230420_124541_HoloLens.skill_labels_by_frame.txt
    │   │   │   ├── 20230420_125033_HoloLens.mp4
    │   │   │   ├── 20230420_125033_HoloLens.skill_labels_by_frame.txt
    │   │   │   ├── 20230420_125517_HoloLens.mp4
    │   │   │   └── 20230420_125517_HoloLens.skill_labels_by_frame.txt
    │   │   ├── Fri-Apr-21.zip
    │   │   ├── Mon-Apr-17/
    │   │   │   ...
    │   │   └── Mon-Apr-17.zip
    │   │       ...
    │   └── negative/
    │       ├── Fri-Aug-25/
    │       │   ...
    │       └── Fri-Aug-25.zip
    ├── m3_pressure_dressing/
    │   ...
    └── r18_chest_seal/
        ...
```
A script is provided at `scripts/extract_bbn_video_archives.bash` to automate
the recursive extraction of such ZIP files.
To operate this script, change directories to be in a parent directory
under which the ZIP files to be extracted are located, and then execute the
script:
```bash
cd ${PATH_TO_DATA}/
bash ${PATH_TO_ANGEL_SYSTEM}/scripts/extract_bbn_video_archives.bash
```

Golden data should be marked as read-only after downloading and extracting to
prevent accidental modification of the files:
```
chmod a-w -R bbn_data/lab_data-golden/
```

##### Extracting Truth COCO and frame image files
BBN archives provide MP4 videos, however we will need individual image frames
for the following steps.
The script to convert BBN Truth data into a COCO format will also, by necessity
for down-stream processes, extract frames and dump them into a symmetric layout
in another writable location:
```bash
python-tpl/TCN_HPL/tcn_hpl/data/utils/bbn.py ...
# OR use the console-script entrypoint installed with the package
bbn_create_truth_coco \
  ./bbn_data/lab_data-golden \
  ./bbn_data/lab_data-working
  ../../config/activity_labels/medical/m2.yaml \
  activity-truth-COCO.json
```


### Storage on Gyges
- On gyges, raw data is located at
  - `/data/PTG/medical/bbn_data/Release_v0.5/v0.56/<task>`
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


## Training Procedure

We take the following steps:

1. train object detection model
2. Generate activity classification truth COCO file.
3. predict objects in the scene
4. predict poses and patient bounding boxes in the scene
5. generate interaction feature vectors for the TCN
6. train the TCN

### Example with M2
Contents:
- [Train Object Detection Model](#train-object-detection-model)
- [Generate activity classification truth COCO file](#generate-activity-classification-truth-coco-file)
- [Generate Object Predictions in the Scene](#generate-object-predictions-in-the-scene)
- [Generate Pose Predictions](#generate-pose-predictions)
- [Configure TCN Training Experiment](#configure-tcn-training-experiment)
- [Run TCN Training](#run-tcn-training)

#### Train Object Detection Model
First we train the detection model on annotated data.
This would be the same data source for both the lab and professional data.
```
python3 python-tpl/yolov7/yolov7/train.py \
  --workers 8 --device 0 --batch-size 4 \
  --data configs/data/PTG/medical/m2_task_objects.yaml \
  --img 768 768 \
  --cfg configs/model/training/PTG/medical/yolov7_m2.yaml \
  --weights weights/yolov7.pt \
  --project /data/PTG/medical/training/yolo_object_detector/train/ \
  --name m2_all_v1_example
```

#### Generate activity classification truth COCO file
Generate the truth MS-COCO file for per-frame activity truth annotations.
This example presumes we are using BBN Medical data as our source (as of
2024/10/15).
```
python-tpl/TCN_HPL/tcn_hpl/data/utils/bbn.py \
  ~/data/darpa-ptg/bbn_data/lab_data-golden/m2_tourniquet \
  ~/data/darpa-ptg/bbn_data/lab_data-working/m2_tourniquet \
  ~/dev/darpa-ptg/angel_system/config/activity_labels/medical/m2.yaml \
  ~/data/darpa-ptg/bbn_data/lab_data-working/m2_tourniquet-activity_truth.coco.json
```

Train, validation, and testing splits can be split from COCO files.
The `kwcoco split` tool may be utilized to create splits at the video level,
otherwise splits may be created manually.

For example:
```bash
kwcoco split \
  --src ~/data/darpa-ptg/bbn_data/lab_data-working/m2_tourniquet-activity_truth.coco.json \
  --dst1 TRAIN-activity_truth.coco.json \
  --dst2 REMAINDER-activity_truth.coco.json \
  --splitter video \
  --rng 12345 \
  --factor 2
kwcoco split \
  --src REMAINDER-activity_truth.coco.json \
  --dst1 VALIDATION-activity_truth.coco.json \
  --dst2 TEST-activity_truth.coco.json \
  --splitter video \
  --rng 12345 \
  --factor 2
# Protect your files!
chmod a-w \
  TRAIN-activity_truth.coco.json \
  REMAINDER-activity_truth.coco.json \
  VALIDATION-activity_truth.coco.json \
  TEST-activity_truth.coco.json
```

#### Generate Object Predictions in the Scene
Note that the input COCO file is that which was generated in the previous step.
This is to ensure that all represented videos and image frames are predicted on
and present in both COCO files.
```
python-tpl/yolov7/yolov7/detect_ptg.py \
  -i TRAIN-activity_truth.coco.json \
  -o test_det_output.coco.json
  --model-hands ./model_files/object_detector/hands_model.pt \
  --model-objects ./model_files/object_detector/m2_det.pt \
  --model-device 0 \
  --img-size 768 \
# Repeat for other relevant activity truth inputs
```
Additional debug outputs may optionally be generated.
See the `-h`/`--help` options for more details.

#### Generate Pose Predictions
Note that the input COCO file is that which was generated in the
`Generate activity classification truth COCO file` section.
```
python-tpl/TCN_HPL/tcn_hpl/data/utils/pose_generation/generate_pose_data.py \\
  -i ~/data/darpa-ptg/bbn_data/lab_data-working/m2_tourniquet/activity_truth.coco.json \\
  -o ./test_pose_output.coco.json \\
  --det-config ./python-tpl/TCN_HPL/tcn_hpl/data/utils/pose_generation/configs/medic_pose.yaml \\
  --det-weights ./model_files/pose_estimation/pose_det_model.pth \\
  --pose-config ./python-tpl/TCN_HPL/tcn_hpl/data/utils/pose_generation/configs/ViTPose_base_medic_casualty_256x192.py \\
  --pose-weights ./model_files/pose_estimation/pose_model.pth
# Repeat for other relevant activity truth inputs
```

#### Configure TCN Training Experiment
Create a new version of, or modify an existing (preferring the former) and
modify attributes appropriately for your experiment.

* `task_name` -- This may be updated with a unique name that identifies this
  training experiment. This is mostly encouraged for when the `paths:root_dir`
  is shared between many experiments (see below).
* `paths:root_dir` -- Update with the path to where the training "logs/"
  directory should go. This may be shared between training experiments, in
  which case customizing your "task_name" is important for separation, or set
  to be a unique directory per experiment.
* `data:coco_*` -- Update with the appropriate paths to the COCO input files to
  be trained over.
* `data:target_framerate` -- Update with the target framerate for input data to
  ensure consistent temporal spacing in the dataset content loading.
* `data:epoch_length` -- Update to a length appropriate for the quantity of
  windows in the dataset. This may also be increased yet more if vector
  augmentation is being utilized to increase the variety of window variations
  seen during a single epoch.
* `data:train_dataset:window_size` -- Update with the desired window size for
  this experiment.
* `data:train_dataset:vectorizer` -- Update with the type and hyperparameters
  for the specific vectorizer to utilize for this experiment.
* `data:train_dataset:transform:transforms` -- Update to include any vector
  generalized transformations/augmentations that should be utilized during
  dataset iteration.
  * The transforms utilized for train, validation and testing may be customized
    independently. By default, the test set will share the validation set's
    transforms, however the hyperparameters for the test dataset can of course
    be manually specified if something different is desired.
  * Currently, the hyperparameters for the test dataset is what will be 
    utilized by the ROS2 node integration.
* `model:num_classes` -- Update with the total number of activity
  classification classes.
* `model:net:dim` -- Update with the dimensionality of the feature vector the
  configured vectorizer class will produce.

If a "master" activity truth COCO was specifically split for a training setup,
then video/frame related object detections and pose estimations may be subset
from equivalently "master" prediction COCO files via a utility provided in 
TCN-HPL:
```bash
# Detections
kwcoco_guided_subset \
  object_detections-yolov7_baseline_20241030.coco.json \
  TRAIN-activity_truth.coco.json \
  TRAIN-object_detections.coco.json
kwcoco_guided_subset \
  object_detections-yolov7_baseline_20241030.coco.json \
  VALIDATION-activity_truth.coco.json \
  VALIDATION-object_detections.coco.json
kwcoco_guided_subset \
  object_detections-yolov7_baseline_20241030.coco.json \
  TEST-activity_truth.coco.json \
  TEST-object_detections.coco.json
# Poses
kwcoco_guided_subset \
  pose_estimations-mmpose_baseline_20241030.coco.json \
  TRAIN-activity_truth.coco.json \
  TRAIN-pose_estimations.coco.json
kwcoco_guided_subset \
  pose_estimations-mmpose_baseline_20241030.coco.json \
  VALIDATION-activity_truth.coco.json \
  VALIDATION-pose_estimations.coco.json
kwcoco_guided_subset \
  pose_estimations-mmpose_baseline_20241030.coco.json \
  TEST-activity_truth.coco.json \
  TEST-pose_estimations.coco.json
```

#### Run TCN Training
TODO

## Example with R18

First we train the detection model on annotated data. This would be the same
data source for both the lab and professional data
```
cd yolo7
python yolov7/train.py \
  --workers 8 \
  --device 0 \
  --batch-size 4 \
  --data configs/data/PTG/medical/r18_task_objects.yaml \
  --img 768 768 \
  --cfg configs/model/training/PTG/medical/yolov7_r18.yaml \
  --weights weights/yolov7.pt \
  --project /data/PTG/medical/training/yolo_object_detector/train/ \
  --name r18_all_v1_example
```

###### Note on training on lab data <a name = "lab_data"></a>:
since we do not have detection GT for lab data, this is our start point for training the TCN on the lab data

Next, we generate detection predictions in kwcoco file using the following script. Note that this
```
python yolov7/detect_ptg.py \
  --tasks r18 \
  --weights /data/PTG/medical/training/yolo_object_detector/train/r18_all_v1_example/weights/best.pt \
  --project /data/PTG/medical/training/yolo_object_detector/detect/ \
  --name r18_all_example \
  --device 0 \
  --img-size 768 \
  --conf-thres 0.25
cd TCN_HPL/tcn_hpl/data/utils/pose_generation/configs
```

with the above scripts, we should get a kwcoco file at:
```
/data/PTG/medical/training/yolo_object_detector/detect/r18_all_example/
```

Edit `TCN_HPL/tcn_hpl/data/utils/pose_generation/configs/main.yaml` with the
task in hand (here, we use r18), the path to the output detection kwcoco, and
where to output kwcoco files from our pose generation step.
```
cd ..
python generate_pose_data.py
cd TCN_HPL/tcn_hpl/data/utils
```
At this stage, there should be a new kwcoco file generated in the field defined
at `main.yaml`:
```
data:
    save_root: <path-to-kwcoco-file-with-pose-and-detections>
```

Next, edit the `/TCN_HPL/configs/experiment/r18/feat_v6.yaml` file with the
correct experiment name and kwcoco file in the following fields:
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
The TCN training script produced a `text_activity_preds.mscoco.json` which is used by the Global Step Predictor. That file should be copied to `/angel_system/model_files/coco/`.


## Docker local testing

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


## Docker real-time

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
