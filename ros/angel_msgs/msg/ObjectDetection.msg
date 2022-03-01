#
# Message for transporting a single object detection and classification
# inference prediction.
#
# This is modeled after the `smqtk_detection.DetectImageObjects` return
# structure.
#

# std_msgs/Header header

# Image-space axis-aligned bounding box. Origin is upper-left corner.
float32 left
float32 right
float32 top
float32 bottom

# Prediction classification labels.
string[] label_vec

# Prediction label confidences. Should be same length as `label_vec`
float64[] label_confidence_vec
