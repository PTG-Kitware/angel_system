ANGEL Sensor Streaming
======================
The `HL2SS <https://github.com/jdibenes/hl2ss>`_ package is used to stream data off of the HoloLens 2 (HL2).
Running within the Unity app as a plugin, HL2SS creates a server for
each of its sensor streams on startup. Clients are then free to connect
to any of the desired streams.

The HL2SS ROS bridge node is the ROS node that creates the HL2SS clients
for the desired sensors, converts them to ROS message formats used
throughout the ANGEL system, and then publishes the ROS messages.

For a full list of supported HL2SS streams, see `supported streams <https://github.com/jdibenes/hl2ss#hololens-2-sensor-streaming>`_.


HL2 Streams published
---------------------
The following is a list of sensor streams currently published by the HL2SS
ROS bridge node.


Personal Video (PV) camera
^^^^^^^^^^^^^^^^^^^^^^^^^^
The main forward camera on the HL2. Default profile is 1280x720, 30Hz. Frame
resolution and frame rate can be modified by passing the appropriate ROS args
to the HL2SS ROS bridge node.

Network data rate measured by HL2 device portal: ~5Mbps

ROS message type: `sensor_msgs.msg.Image`


Spatial Input Data
^^^^^^^^^^^^^^^^^^
Provides head tracking, eye tracking, and hand tracking data in a single data packet.
Currently, the HL2SS ROS bridge node only publishes the hand tracking data at a rate
of ~60Hz per hand. Hand tracking data is only published when the hands are within the
FOV of the HL2.

Network data rate measured by HL2 device portal: ~1Mbps

ROS message type: `angel_msgs.msg.HandJointPosesUpdate`


Audio
^^^^^
Audio streaming from the HL2 microphone. The HL2SS ROS bridge node uses the compressed
audio profile, AAC 24000, provided by HL2SS. The HL2SS audio client decodes the audio
data before providing it to the HL2SS ROS bridge node.

The audio data is 2 channel, 48000Hz sample rate.

Network data rate measured by HL2 device portal: ~300-400Kbps

ROS message type: `angel_msgs.msg.HeadsetAudioData`
