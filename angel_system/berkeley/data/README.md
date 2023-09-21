Task Recognition Training Roadmap
=================================

# Stage 1
## Train the model
```bash
python train_net.py --config-file ../configs/MC50-InstanceSegmentation/cooking/coffee/stage1/mask_rcnn_R_50_FPN_1x_fine-tuning.yaml --num-gpus 1
```

# Stage 2
## Generate "ground truth" data
- make sure that `demo` points to the stage1 model and `conf_thr` is 0.4
- make sure `stage` is set to "stage2"

```bash
python data/create_contact_metadata.py
```

## Create config file that points to the files egenrated above

## Train the model
```bash
python train_net.py --config-file ../configs/MC50-InstanceSegmentation/cooking/coffee/stage2/mask_rcnn_R_50_FPN_1x_fine-tuning.yaml --num-gpus 1
```

# Train the Activity Classifier
## Generate training data for the activity classifier
- make sure that `demo` points to the stage2 model and `conf_thr` is 0.01
- make sure `stage` is set to "results"
```bash
python data/create_contact_metadata.py
```

# Train the activity classifier
```bash
python angel_system/activity_hmm/scripts/activity_detection_exploration/train_activity_classifier.py --pred-fnames coffee_no_background_results_test.mscoco.json coffee_no_background_results_train_activity.mscoco.json coffee_no_background_results_val.mscoco.json --act-label-yaml ~/projects/PTG/angel_system/config/activity_labels/recipe_coffee.yaml --output-dir activity_classifier_training/
```

