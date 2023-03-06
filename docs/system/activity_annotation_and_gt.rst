====================================
Activity Annotation and Ground Truth
====================================

Annotation Process
##################
TODO: Use more words

1) Run through task with the annotation node running and foot-pedal at the
   ready, pressing the pedal appropriately when true activities/errors occur.
2) Load into DIVE to refine annotations.
3) Export refined annotations from DIVE into a CSV file format where 2-state
   tracks are activity start and end times.

Ground Truth Data
#################
Ground truth annotations are acquired in a preliminary way via the ROS system,
and then refined using the DIVE annotation system (https://viame.kitware.com/
or via a local installation).
Thus, the original ground truth is in the DIVE export format, which is a CSV
where activities take the form of two-state "tracks".
Each event has a unique ID within a video clip, and only two states should be
present for a single ID: the activity starting time and ending time.

The DIVE format only records frame numbers, so the image file names are our
connection back to a time stamp as the images exported from the bag extraction
are named such that their seconds + nanoseconds are encoded in the file name
in a deterministic manner::

    frame_00109_01664199528_381554365.png --> 1664199528.381554365 seconds
