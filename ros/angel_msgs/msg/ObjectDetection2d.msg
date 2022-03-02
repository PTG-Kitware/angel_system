#
# A single object detection on a 2D image plane.
# This includes a predicted set of classifications.
#
# This is modeled after the `smqtk_detection.DetectImageObjects` return
# structure.
#
# TODO: maybe drop labels and provide a service for consumers to query for
#       label list just once.
#

# Image-space axis-aligned bounding box. Origin is upper-left corner.
float32 left
float32 right
float32 top
float32 bottom

# Prediction label confidences. Should be same length as `label_vec`
float64[] label_confidence_vec

# Prediction classification labels.
string[] label_vec
