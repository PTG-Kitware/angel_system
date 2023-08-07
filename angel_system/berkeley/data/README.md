Task Recognition Training Roadmap
=================================

# Stage 1


# Stage 2
## Generate "ground truth" data
- make sure that `demo` points to the stage1 model and `conf_thr` is 0.4
- make sure `stage` is set to "stage2"

python data/create_contact_metadata.py

## Create config file that points to the files egenrated above

## Train the model
python train_net.py --config-file ../configs/MC50-InstanceSegmentation/cooking/coffee/stage2/mask_rcnn_R_50_FPN_1x_fine-tuning.yaml --num-gpus 1

# Train the Activity Classifier
## Generate training data for the activity classifier
- make sure that `demo` points to the stage2 model and `conf_thr` is 0.01
- make sure `stage` is set to "results"

python data/create_contact_metadata.py

