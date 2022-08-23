#!/usr/bin/env python3
import csv
import json
import os
import pandas as pd
import argparse

def main(args):
    """
    This script is for parsing annotation_event_data.txt files.The data from the files can be formatted in either
    the PTG format or the DIVE annotation JSON format for further refinement in the DIVE annotation tool.

    :param args.image_path: Absolute path to the 'images' folder as is output by the exploded ros bag script
    :param args.ann_path: Absolute path to the annotation_event_data.txt file as is output by the exploded ros bag
                          script
    :param args.rosbag_name: A string containing the name of the rosbag that is being processed. This is used to 
                             create a relative filepath to the exploded rosbag image folder for evaluation.
    :param args.activity: Only used when parsing directly to the PTG labels, this is for when a recording only contains
                     a single activity being performed repeatedly. The parameter is a string of the name of the activity
    :param args.ptg_out_file: The absolute path of the feather file we will output that will contain the PTG data format.
    :param args.dive_out_file: The absolute path of the DIVE annotation JSON file that will be output.
    :param args.ptg_labels: A boolean that is true when a .feather file in the PTG format should be output and is false
                            when a DIVE annotation JSON file should be output. 
    """

    # Create the events dictionary and initialize the number of error and annotations seen
    events = {}
    num_anns = 0
    num_errors = 0

    imgs = sorted(os.listdir(args.image_path))
    event_data = []
    with open(args.ann_path, "r") as inFile:
        event_data = json.loads(inFile.read())

    # Extract the combined timestamp from the annotation_event_data.txt file and count number of annoataions and errors
    # Within the event_data file, each event is generally classified as an annotation or error depending on the button
    # that was pressed to trigger the event. Additionally each event marks either the begin or end of the annotation/error.
    # It is guaranteed to have a stop event for each start event. 
    for ann in event_data:
        if ann['description'] == 'Start annotation':
            events['event_{}'.format(num_anns)] = {'start_time': int(ann['time_sec']) + (int(ann['time_nanosec'])*1e-9)}
        elif ann['description'] == 'Start error':
            events['error_{}'.format(num_errors)] = {'start_time': int(ann['time_sec']) + (int(ann['time_nanosec'])*1e-9)}
        elif ann['description'] == 'Stop annotation':
            events['event_{}'.format(num_anns)]['end_time'] = int(ann['time_sec']) + (int(ann['time_nanosec'])*1e-9)
            num_anns += 1
        elif ann['description'] == 'Stop error':
            events['error_{}'.format(num_errors)]['end_time'] = int(ann['time_sec']) + (int(ann['time_nanosec'])*1e-9)
            num_errors += 1

    # Find the first frame after the start time and the last frame before the end time of each event
    for ann_key in events.keys():
        ann = events[ann_key]
        for i in range(len(imgs)):
            fnum, ts, tns = map(int, imgs[i].strip().split('.')[0].split('_')[1:])
            tsec = ts+(tns*1e-9)
            if tsec >= ann['start_time'] and 'start_frame' not in ann.keys():
                ann['start_frame'] = imgs[i]
            if tsec > ann['end_time'] and 'end_frame' not in ann.keys():
                ann['end_frame'] = imgs[i-1]
                break

    if args.ptg_labels:
        # Format the start and end frames into the PTG format
        classes = []
        start_frames = []
        end_frames = []
        exploded_ros_bag_paths = []
        for key in sorted(events.keys()):
            if "event" in key:
                classes.append(args.activity)
                start_frames.append(events[key]['start_frame'])
                end_frames.append(events[key]['end_frame'])
                exploded_ros_bag_paths.append(args.rosbag_name + "/_extracted/images/")
        ptg_anns = {'class': classes, 'start_frame': start_frames, 'end_frame': end_frames, 'exploded_ros_bag_path': exploded_ros_bag_paths}
        eval_df = pd.DataFrame(ptg_anns)
        eval_df.to_feather(args.ptg_out_file)

                
    else:
        # Format the start and end frames into tracks into tracks for DIVE to process
        track_json = {'version': 2, 'tracks': {}, 'groups': {}}
        for key in sorted(events.keys()):
            if "event" in key:
                track_dict = {}

                detection = key.split("_")[-1]
                track_dict['id'] = detection
                track_dict['meta'] = {}
                track_dict['attributes'] = {}
                
                track_dict['confidencePairs'] = [[detection, 1]]

                start_frame_id = events[key]['start_frame'].split("_")[1].lstrip('0')
                start_detection = [detection, events[key]['start_frame'], start_frame_id,0,0,1,1,1,-1,key,1]

                track_dict['features'] = [{'frame': int(start_frame_id), 'bounds': [0,0,1,1]}] 

                end_frame_id = events[key]['end_frame'].split("_")[1].lstrip('0')
                end_detection = [detection, events[key]['end_frame'], end_frame_id,0,0,1,1,1,-1,key,1]

                track_dict['features'].append({'frame': int(end_frame_id), 'bounds': [0,0,1,1]})
                track_dict['begin'] = int(start_frame_id)
                track_dict['end'] = int(end_frame_id)

                track_json['tracks'][detection] = track_dict
        
        with open(args.dive_out_file, "w") as outfile:
            json.dump(track_json, outfile)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--image_path', default='')
    ap.add_argument('--ann_path', default='')
    ap.add_argument('--rosbag_name', default='')
    ap.add_argument('--activity', default='')
    ap.add_argument('--ptg_out_file', default='')
    ap.add_argument('--dive_out_file', default='')
    ap.add_argument('--ptg_labels', default=False, type=bool)

    args = ap.parse_args()
    main(args)
