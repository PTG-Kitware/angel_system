#!/usr/bin/env python3
import argparse
import cv2
import os
import pandas as pd
import shutil
from PIL import Image


def main(args):
    """
    This is a script to transform data into the format that the LEARN platform
    (https://gitlab.kitware.com/darpa_learn/learn) expects for video classification.
    The input can currently take two different forms, an mp4 video with a label file or
    folder containing one or more subfolders of frames with a folder containing one or more 
    label files of the coco format as exported by the DIVE annotation tool. 

    This script will output a series of folders each containing the frames of a single
    instance of an activity. It will also create a set of .feather annotation files 
    which are expected by the LEARN platform.

    :param args.label_file: Only used when processing an .mp4 video. The absolute path to the 
                            label file. label file where each line is a single activity
                            with the structure `start_frame end_frame class_label`

    :param args.label_dir: Only used when processing frames. The absolute path to the
                              folder containing one or more .csv annotation files in the 
                              coco format as exported by the DIVE annoation tool. The 
                              annotation files must have the same name as the corresponding
                              folder containing the frames within image_dir.

    :param args.video_file: Only used when processing an .mp4 video. The absolute path to the
                            video file
    
    :param args.image_dir: Only used when processing frames. The absolute path to the folder
                           containing one or more subfolders holding frames to be processed.
                           The subfolders must have the same name as the corresponding annotation
                           file within label_dir.

    :param args.data_out_path: The aboslute path to the desired directory to contain the processed
                               image folders. A subfolder will be made in this directory for each 
                               individual activity instance in the input data
    
    :param args.label_out_path: The absolute path to the desired directory to contain the created
                                annotation files. For each input video it will create two files of
                                the form labels_{phase}_x.feather and meta_{phase}_x.feather where
                                phase is train or test and x if an integer. It will then combine 
                                these files into labels_{phase}.feather and meta_{phase}.feather files.

    :param args.phase: A string with the value `train` or `test` which will indicate what phase of
                       the training the dataset is used for.
    
    :param args.from_video: a boolean indicating whether the data will be processed from an mp4 file
                            or a set of frames
    """
    label_dict = dict()
    label_counter = 0
    for i in range(len(os.listdir(args.label_dir))):
        label_name = os.listdir(args.label_dir)[i]
        labels = ""
        if not args.from_video:
            df = pd.read_csv(args.label_dir + os.listdir(args.label_dir)[i])
            for label in range(0, (len(pd.unique(df["# 1: Detection or Track-id"]))-1)):
                temp_df = df[df["# 1: Detection or Track-id"] == str(label)]
                if temp_df.iloc[0]["10-11+: Repeated Species"] not in label_dict.keys():
                    label_dict[temp_df.iloc[0]["10-11+: Repeated Species"]] = label_counter
                    label_counter += 1
                min_frame = pd.to_numeric(temp_df["3: Unique Frame Identifier"]).min()
                max_frame = pd.to_numeric(temp_df["3: Unique Frame Identifier"]).max()
                labels += "{} {} {}\n".format(min_frame, max_frame, label_dict[temp_df.iloc[0]["10-11+: Repeated Species"]])
            lines = [line for line in labels.split("\n") if line != ""]
            print(lines)
        if args.from_video:
            lines = open(args.label_file, "r").readlines()
            cap = cv2.VideoCapture(args.video_file)
            frame_count = 0
        label_anns = {"id": [], "video_id":[], "class": [], "start_frame": [], "end_frame": []}
        meta_anns = {"id": [], "video_id": [], "start_frame": [], "end_frame": []}
        for line in lines:
            data = line.split(" ")
            frame_start = int(data[0])
            frame_end = int(data[1])
            label = data[2]
            
            num_labels = len([file for file in os.listdir("{}/{}/".format(args.data_out_path, args.phase)) if int(label) == int(file.split("_")[0])])
            label_folder = '{}_{}'.format(label, num_labels)
            os.makedirs("{}/{}/{}".format(args.data_out_path, args.phase, label_folder))
            label_anns['id'].append(label_folder)
            label_anns['video_id'].append(label_folder)
            label_anns['class'].append(label)
            label_anns['start_frame'].append(frame_start)
            label_anns['end_frame'].append(frame_end)

            meta_anns['id'].append(label_folder)
            meta_anns['video_id'].append(label_folder)
            meta_anns['start_frame'].append(frame_start)
            meta_anns['end_frame'].append(frame_end)

            if args.from_video:
                while frame_start > frame_count:
                    retval, frame = cap.read()
                    frame_count += 1
                while frame_end > frame_count:
                    retval, frame = cap.read()
                    if retval:
                        pass
                        cv2.imwrite('{}/{}/{}/{}.jpg'.format(args.data_out_path, args.phase, label_folder, frame_count), frame)
                    frame_count += 1
            
            else:
                image_folder = "{}/{}/".format(args.image_dir, label_name.split(".")[0])
                for frame in sorted(os.listdir(image_folder)):
                    if int(frame[6:11]) >= frame_start and int(frame[6:11]) <= frame_end:
                        im = Image.open(image_folder + frame)
                        im.save('{}/{}/{}/{}.jpg'.format(args.data_out_path, args.phase, label_folder, frame[6:11].lstrip('0')))

        if args.phase == "train":
            train_df = pd.DataFrame(label_anns)
            train_meta_df = pd.DataFrame(meta_anns)
            train_df.to_feather(args.label_out_path + "labels_train_{}.feather".format(i))
            train_meta_df.to_feather(args.label_out_path + "meta_train_{}.feather".format(i))
        else:
            test_df = pd.DataFrame(label_anns)
            test_meta_df = pd.DataFrame(meta_anns)

            test_df.to_feather(args.label_out_path + "labels_test_{}.feather".format(i)) 
            test_meta_df.to_feather(args.label_out_path + "meta_test_{}.feather".format(i)) 
                
    files = ["labels_test", "meta_test", "labels_train", "meta_train"]
    for file in files:
        final_df = pd.DataFrame()
        for i in range(len(os.listdir(args.label_dir))):
            temp_df = pd.read_feather(args.label_out_path + "{}_{}.feather".format(file, i))
            final_df = pd.concat([final_df, temp_df]).reset_index(drop=True, inplace=False)
        print(final_df)
        final_df.to_feather(args.label_out_path + "{}.feather".format(file))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--label_file', default='')
    ap.add_argument('--label_folder', default='')
    ap.add_argument('--video_file', default=None)
    ap.add_argument('--image_dir', default='')
    ap.add_argument('--data_out_path', default="")
    ap.add_argument('--label_out_path', default="")
    ap.add_argument('--phase', default='train')
    ap.add_argument('--from_video', default=False, type=bool)
    args = ap.parse_args()
    main(args)

