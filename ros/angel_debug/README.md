# Lauch-file examples

## Off-line object detection example
This connects a simple image generation node that will try to pull from your
video device ID0 (usually your builtin webcam, no we can't change the device 
ID used) to the detector node and detection debugger node, opening a
rqt_image_view window to view the compressed debug image output (needs manual
navigation to the topic apparently).

The intent is to show basic detection capabilities.

To run this:
```bash
./angel-workspace-shell.sh -r
ros2 launch angel_debug debug_offline_object_detection.yaml
```
