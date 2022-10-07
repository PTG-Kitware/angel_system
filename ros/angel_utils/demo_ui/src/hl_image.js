// Get data from angel_system
workspace = '/debug';
var ros = new ROSLIB.Ros({
  url : 'ws://localhost:9090'
});

// Create a listener for /angel/ActivityDetections
var image_listener = new ROSLIB.Topic({
    ros : ros,
    name : workspace + '/PVFramesRGB/compressed',
    messageType : 'sensor_msgs/msg/CompressedImage' // find: $ ros2 topic type <>
});

image_listener.subscribe(function(m) {
    // update image
    var image = document.getElementById('hl-image');
    image.src = "data:image/jpeg;base64," + m.data;
});
