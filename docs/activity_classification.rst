=======================
Activity Classification
=======================

Angel-System Package Components
###############################

Early Plugin-based effort
-------------------------
Interface defined internally within the `angel_system` package in the file
:file:`angel_system/interfaces/detect_activities.py`.

PyTorch Video Slow-Fast r50
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Implemented in :file:`angel_system/impls/detect_activities/pytorchvideo_slow_fast_r50.py`.

This was the initial activity classification model used in the system that
utilized a pretrained model.

SwinB Transformer
^^^^^^^^^^^^^^^^^
Implemented in :file:`angel_system/impls/detect_activities/two_stage/two_stage_detect_activities.py`.

First model we attempted training specifically on domain-specific data from our
ROS2 system.

Two-stage transformer model
---------------------------
This "plugin" deviated from the initial interface referenced above.
The use of an interface here was ultimately premature and currently sees no use
other than in this "implementation".

Interface is defined in :file:`angel_system/interfaces/mm_detect_activities.py`.
Implementation is in :file:`angel_system/impls/detect_activities/two_stage/two_stage_detect_activities.py`.
This implementation contains only prediction capabilities.
Training functionality lives in a different repository `here
<https://github.com/PTG-Kitware/ptg-activity-recognition>`_.

This method introduced a "multi-modal" approach and takes, in addition to a
window of image frames, the left and right 3D hand poses of the user.

Hand poses are communicated at a different rate than the images so individual
poses are associated with an image frame by nearest temporal distance.

UHO Model
---------
Evolution of the "Two-stage" model above that incorporated additional
modalities.

Currently incorporates the additional modalities of:

    * left/right 3D hand pose positions
    * object detection top-K bounding boxes
    * object detection top-K descriptors

The model is implemented under the ``angel_system.uho`` python module.

Training
^^^^^^^^
TODO: Introduce how to train this model.

Prediction
^^^^^^^^^^
Prediction functions underneath the module ``angel_system/uho/prediction.py``.

.. autofunction:: angel_system.uho.prediction.get_uho_classifier

.. autofunction:: angel_system.uho.prediction.get_uho_classifier_labels

.. autofunction:: angel_system.uho.prediction.predict


ROS2 Nodes
##########
Reminder: ROS2 python system nodes are located here:
:file:`ros/angel_system_nodes/angel_system_nodes`.

``activity_detector.py``
------------------------
Initial plugin-based activity detector node.
This node is utilizes the ``DetectActivities`` interface whose prediction is
purely based on image data (no multi-modal auxiliary data).

``uho_activity_detector.py``
----------------------------
This node makes specific use of multi-model input and currently just the UHO
model for prediction (:ref:`see above reference <Prediction>`).


As such this node takes in data from:

    * RGB image frames
    * left/right hand poses
    * 2D object detection predictions/descriptors (same message)

This node is set up to perform asynchronous data input and prediction, which is
centered on the :ref:`InputBuffer` structure.
This structure's job is to accumulate sensor data into buffers and manage the
ability to extract "windows" of data from the buffers such that auxiliary data
is sparsely associated with an image frame.

Ground-Truth Mode
^^^^^^^^^^^^^^^^^
There is a secondary mode this node may take on when it is passed the ``gt``
option, whose value should be the filesystem path to an activity ground truth
file. When his is provided, the activity classifier is not actually invoked,
but the node will "simulate" activity classification prediction via known
activity time regions that the ground truth file specified.
When paired with the appropriate input dataset that the ground truth describes,
the activity classifier node will now output "perfect" classification
predictions as the ground truth describes.
