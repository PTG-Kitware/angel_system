============================================================
Iterations of Activity Detection and Object Detection Models
============================================================

Activity Detection Models
=========================

UHO Models 
----------

Baseline Faster-RCNN detector
-----------------------------
This UHO model was trained and validated on our own cooking data. Hand features were generated 
from each hand pose using the Unified FCN model (uho/src/models/components/fcn.py). Object features were generated
by passing each of our frames through a Faster-RCNN model pretrained on the visual genome dataset, and collecting
the features from the bboxes that were detected. 

Fine tuned Faster-RCNN on Berkley data
--------------------------------------
This UHO model was trained and validated on our own cooking data. Hand features were generated 
from each hand pose using the Unified FCN model (uho/src/models/components/fcn.py). Object features were generated
by passing each of our frames through a Faster-RCNN model pretrained on coco and fine-tuned on the object detection
labels given to us from Berkley and collecting the features from the bboxes that were detected. 

Object Detection Models
=======================

Faster-RCNN Models
------------------

Baseline detector
-----------------
Out of the box detector pretrained on the visual genome dataset

Fine tuned on Berkley Data
--------------------------
Baseline faster-rcnn detector trained on coco, then fine tuned on the object labels provided by Berkley
