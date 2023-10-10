---------------
OBJECT DETECTOR
---------------

COPY DIVE FILES
===============
scp <fn> gyges:/data/PTG/cooking/object_anns/coffee/berkeley/dive

DIVE to KWCOCO
==============
$ cd angel_system/angel_system
$ conda activate ptg

(for berkeley coffee data remove the video_id stuff)

$ python data/common/cli/dive_to_kwcoco.py --recipe coffee

# saves to {obj_dets_dir}/berkeley/coffee_obj_annotations_v2.3.mscoco.json

(repeat for tea)

Add background images 
=====================
$ kwcoco union --src /data/PTG/cooking/images/coffee/berkeley/background_images/bkgd.mscoco.json /data/PTG/cooking//object_anns/coffee/berkeley//coffee_obj_annotations_v2.3.mscoco.json --dst /data/PTG/cooking//object_anns/coffee/berkeley//coffee_obj_annotations_v2.3_plus_bkgd.mscoco.json --remember_parent True

(repeat for tea)

Combine files
=============
$ kwcoco union --src /data/PTG/cooking/object_anns/coffee/berkeley/coffee_obj_annotations_v2.3_plus_bkgd.mscoco.json /data/PTG/cooking/object_anns/tea/berkeley/tea_obj_annotations_v2.2_plus_bkgd.mscoco.json --dst /data/PTG/cooking/object_anns/coffee+tea/berkeley/coffee_v2.3_and_tea_v2.2_obj_annotations_plus_bkgd.mscoco.json --remember_parent True


Visualize Training Data
=======================
$ python data/common/cli/visualize_kwcoco.py  --dset /data/PTG/cooking/object_anns/coffee+tea/berkeley/coffee_v2.3_and_tea_v2.2_obj_annotations_plus_bkgd.mscoco.json --save_dir /data/PTG/cooking/object_anns/coffee+tea/berkeley/visualization/


(double check that the objects look okay here)

Detectron2 Setup
================
$ vim angel_system/angel_system/berkeley/configs/MC50-InstanceSegmentation/cooking/coffee+tea/stage1/mask_rcnn_R_50_FPN_1x_fine-tuning.yaml

(edit NUM_CLASSES, OUTPUT_DIR, TOTAL_IMAGE_NUMBER)

$ vim angel_system/angel_system/berkeley/detectron2/data/datasets/kitware_cooking.py

(edit _PREDEFINED_SPLITS_KITWARE_COOKING["KITWARE_COOKING_COFFEE_TEA"], ALL_COOKING_CATEGORIES["KITWARE_COOKING_COFFEE_TEA"])

Train
=====
$ cd angel_system/angel_system/berkeley/tools
$ python train_net.py --config-file ../configs/MC50-InstanceSegmentation/cooking/coffee+tea/stage1/mask_rcnn_R_50_FPN_1x_fine-tuning.yaml --num-gpus 4



-------------------
ACTIVITY CLASSIFIER
-------------------

Get Object Detection Results
============================
$ cp angel_system/angel_system/berkeley/configs/MC50-InstanceSegmentation/cooking/coffee+tea/stage1/mask_rcnn_R_50_FPN_1x_fine-tuning.yaml
angel_system/angel_system/berkeley/configs/MC50-InstanceSegmentation/cooking/coffee+tea/stage1/mask_rcnn_R_50_FPN_1x_demo.yaml
$ vim angel_system/angel_system/berkeley/configs/MC50-InstanceSegmentation/cooking/coffee+tea/stage1/mask_rcnn_R_50_FPN_1x_demo.yaml

(edit MODEL.WEIGHTS to OUTPUT_DIR/model_final.pth)

$ vim angel_system/angel_system/berkeley/data/run_object_detector.py

(edit main() to call coffee_main(),
edit coffee_main to call the config file above)

$ mkdir coffee && cd coffee
$ python run_object_detector.py

$ rm -r coffee/temp
$ mv coffee /data/PTG/cooking/annotations/coffee+tea/results/coffee_and_tea

(repeat for tea)


Add HoloLens Hands
==================


Combine Results
===============
$ kwcoco union --src /data/PTG/cooking/annotations/coffee+tea/results/coffee_and_tea/coffee/coffee_and_tea_results_{split}.mscoco.json
/data/PTG/cooking/annotations/coffee+tea/results/coffee_and_tea/tea/coffee_and_tea_results_{split}.mscoco.json
--dst /data/PTG/cooking/annotations/coffee+tea/results/coffee_and_tea/coffee_and_tea_results_{split}.mscoco.json
--remember_parent True

(for each split)

TCN Training Data
=================
$ cd tcn_hpl
$ vim tcn_hpl/data/utils/ptg_datagenerator.py

(edit ....)

$ python tcn_hpl/data/utils/ptg_datagenerator.py

Train
=====
$ vim tcn_hpl/configs/experiment/coffee+tea/feat_v5.yaml

(edit exp_name, paths.data_dir, all_transforms)

$ python src/train.py experiment=coffee+tea/feat_v5

