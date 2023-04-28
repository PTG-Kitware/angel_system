
## Get Step-level Predition

We provide a command line tool to run a simple demo of step-level prediciton:

```
python ./demo/demo_step_pred.py --input your_images_path --output your_save_path
```
for eample:
```
python ./demo/demo_step_pred.py --input /shared/niudt/DATASET/PTG_Kitware/all_activities_6/images/*.png --output /shared/niudt/detectron2/DEMO_Results/test_integration
```

This will save the predictions in the ```your_save_path```.

Note that you may need to modify the ```Re_order``` function in line 138, this fuction is to re-order the images in your ```input_images_path```, the goal is to order the images in a sequential order, the current version is customized to images based on the name of the frames with the format of  ```frame_XXXXX_XXXXXXXXXXX_XXXXXXXXXXX.png```.
