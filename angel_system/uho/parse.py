import os
from os import path as osp
import shutil
import tqdm
import json
import numpy as np
import csv
import pandas
import pdb


def parse_video(root_dir, data_dir, label_file, annotations, overlap, sta_row, cnt):
    # Load data
    img_dir = osp.join(data_dir, "images")
    hand_file = osp.join(data_dir, "hand_pose_data.txt")
    if not os.path.exists(hand_file):
        hand_file = osp.join(data_dir, "hand_pose_data.json")
    hand = json.load(open(hand_file))
    imgs = os.listdir(img_dir)
    print(data_dir)
    print(len(hand), len(imgs))

    # Sync the multi-streamed data
    imgs = sorted(imgs)
    hand_ts = [x["time_sec"] + (x["time_nanosec"] * 1e-9) for x in hand]

    print("Is hand pose data sorted? ", hand_ts == sorted(hand_ts))
    slop = 1e-1  # In seconds
    mm_dict = {fn: [] for fn in imgs}
    i, j = 0, 0

    # Each frame has multiple hand samples. The unassigned hand
    # samples are assigned to the most recent frame (if it is
    # within slop interval) We assign the most recent detected hand
    # to the frame within slop parameter.
    while i < len(imgs) and j < len(hand_ts):
        fnum, ts, tns = map(int, imgs[i].strip().split(".")[0].split("_")[1:])
        tsec = ts + (tns * 1e-9)
        if (tsec - hand_ts[j]) < slop:
            if tsec >= hand_ts[j]:
                mm_dict[imgs[i]].append(hand[j])
                j += 1
            else:
                i += 1
        else:
            j += 1

    # Save the data to disk
    new_imgs = write_synced_data(mm_dict, data_dir)

    # Create activity class annotations
    labels = {annotations[i].lower(): str(i) for i in range(len(annotations))}
    h2o_data = create_activity_annotations(
        data_dir, label_file, labels, imgs, overlap, sta_row, cnt
    )

    return h2o_data, new_imgs


# Helpers to save synced data to disk
def get_hand_pose_from_msg(msg):
    hand_joints = [
        {"joint": m["joint"], "position": m["position"]} for m in msg["joint_poses"]
    ]

    # Rejecting joints not in OpenPose hand skeleton format
    reject_joint_list = [
        "ThumbMetacarpalJoint",
        "IndexMetacarpal",
        "MiddleMetacarpal",
        "RingMetacarpal",
        "PinkyMetacarpal",
    ]
    joint_pos = []
    for j in hand_joints:
        if j["joint"] not in reject_joint_list:
            joint_pos.append(j["position"])
    joint_pos = np.array(joint_pos).flatten()

    # Appending 1 as per H2O requirement
    if msg["hand"] == "Right":
        rhand = np.concatenate([[1], joint_pos])
        lhand = np.zeros_like(rhand)
    elif msg["hand"] == "Left":
        lhand = np.concatenate([[1], joint_pos])
        rhand = np.zeros_like(lhand)
    else:
        lhand = np.zeros_like(len(joint_pos) + 1)
        rhand = np.zeros_like(len(joint_pos) + 1)

    return lhand, rhand


def write_synced_data(mmdata, fpath):
    if not osp.isdir(osp.join(fpath, "rgb")):
        os.mkdir(osp.join(fpath, "rgb"))

    if not osp.isdir(osp.join(fpath, "hand_pose")):
        os.mkdir(osp.join(fpath, "hand_pose"))

    new_imgs = []
    idx = 0
    format_str = "{:06d}"
    for fn in tqdm.tqdm(mmdata):
        dst_img_name = osp.join(fpath, "rgb", format_str.format(idx) + ".png")
        if not osp.exists(dst_img_name):
            src_img_name = osp.join(fpath, "images", fn)
            shutil.copy(src_img_name, dst_img_name)

        tmp = dst_img_name.split("/")
        img_name = "/".join(tmp[6:])
        new_imgs.append(img_name)

        hpose_name = osp.join(fpath, "hand_pose", format_str.format(idx) + ".txt")
        if not osp.exists(hpose_name):
            if len(mmdata[fn]) > 0:
                # Get the most recent hand pose info for each frame
                lhand, rhand = get_hand_pose_from_msg(mmdata[fn][-1])
            else:
                lhand = np.zeros(64)
                rhand = np.zeros(64)
            hpose = np.concatenate([lhand, rhand])
            np.savetxt(hpose_name, hpose, newline=" ")

        idx += 1

    return new_imgs


def create_activity_annotations(
    data_dir, csv_name, annotations, imgs, overlap, sta_row, cnt
):
    csv_reader = csv.reader(open(csv_name))
    rows = [x for x in csv_reader]

    sta_fr_set = [
        int(rows[i][2]) for i in range(sta_row, len(rows) - 1, 2) if len(rows[i][2]) > 0
    ]
    ind = sta_row + np.argsort(sta_fr_set) * 2  # index of sorted clips
    h2o_data = []
    max_fr = len(imgs)
    num_fr = 32

    tmp = 0
    last_end = 0
    action_step = num_fr // 4 if overlap else num_fr
    for i in ind:
        if len(rows[i][0]) == 0:
            break
        idx = str(cnt + tmp)
        action = rows[i][-2]
        if action[:1] == " ":
            action = action[1:]
        if action[-3:] == ".  ":
            action = action[:-3]
        if action[-2:] == ". ":
            action = action[:-2]
        if action[-1] == " " or action[-1] == ".":
            action = action[:-1]
        try:
            action_label = annotations[action.lower()]
        except:
            pdb.set_trace()
        start_act = rows[i][2]
        end_act = rows[i + 1][2]
        start_frame = "0"
        end_frame = str(max_fr)

        # gap between two actions
        cnt1 = 0
        if int(start_act) - last_end > num_fr // 2:
            for k in range(last_end, int(start_act), num_fr):
                sta_idx = str(k)
                end_idx = str(k + num_fr - 1)
                if int(end_idx) > int(start_act) + num_fr // 3:
                    continue
                act_idx = "0"
                h2o_data.append(
                    [
                        str(int(idx) + cnt1),
                        data_dir,
                        act_idx,
                        sta_idx,
                        end_idx,
                        start_frame,
                        end_frame,
                    ]
                )
                cnt1 += 1
                tmp += 1
            # print(csv_name, act_idx, int(start_act), last_end)
        # within the action
        if int(end_act) - int(start_act) > num_fr // 2:
            for k in range(int(start_act), int(end_act), action_step):
                sta_idx = str(k)
                end_idx = str(k + num_fr - 1)
                if int(end_idx) > int(end_act) + num_fr // 3 or int(end_idx) > max_fr:
                    continue
                act_idx = action_label
                h2o_data.append(
                    [
                        str(int(idx) + cnt1),
                        data_dir,
                        act_idx,
                        sta_idx,
                        end_idx,
                        start_frame,
                        end_frame,
                    ]
                )
                cnt1 += 1
                tmp += 1

        last_end = int(end_act)
        # print(csv_name, act_idx, int(start_act), last_end)

    # after last action
    if max_fr - last_end > num_fr:
        for k in range(last_end, max_fr, num_fr):
            sta_idx = str(k)
            end_idx = str(k + num_fr - 1)
            if int(end_idx) > max_fr:
                continue
            act_idx = "0"
            h2o_data.append(
                [
                    str(int(idx) + cnt1),
                    data_dir,
                    act_idx,
                    sta_idx,
                    end_idx,
                    start_frame,
                    end_frame,
                ]
            )
            cnt1 += 1
            tmp += 1

    h2o_data = [" ".join(x) for x in h2o_data]
    # pdb.set_trace()
    return h2o_data


if __name__ == "__main__":
    annotations = [
        "Background",
        "Measure 12 ounces of water in the liquid measuring cup",
        "Pour the water from the liquid measuring cup into the electric kettle",
        "Turn on the Kettle",
        "Place the dripper on top of the mug",
        "Take the coffee filter and fold it in half to create a semi-circle",
        "Fold the filter in half again to create a quarter-circle",
        "Place the folded filter into the dripper such that the the point of the quarter-circle rests in the center of the dripper",
        "Spread the filter open to create a cone inside the dripper",
        "Turn on the kitchen scale",
        "Place a bowl on the scale",
        "Zero the scale",
        "Add coffee beans to the bowl until the scale reads 25 grams",
        "Pour the measured coffee beans into the coffee grinder",
        "Set timer for 20 seconds",
        "Turn on the timer",
        "Grind the coffee beans by pressing and holding down on the black part of the lid",
        "Pour the grounded coffee beans into the filter cone prepared in step 2",
        "Turn on the thermometer",
        "Place the end of the thermometer into the water",
        "Set timer to 30 seconds",
        "Pour a small amount of water over the grounds in order to wet the grounds",
        "Slowly pour the water over the grounds in a circular motion. Do not overfill beyond the top of the paper filter",
        "Allow the rest of the water in the dripper to drain",
        "Remove the dripper from the cup",
        "Remove the coffee grounds and paper filter from the dripper",
        "Discard the coffee grounds and paper filter",
    ]

    subset = "all_activities_"  # "all_activities_" "brian_coffee_"
    root_dir = "/data/dawei.du/datasets/ROS"
    train_data = ["id path action_label start_act end_act start_frame end_frame"]
    val_data = ["id path action_label start_act end_act start_frame end_frame"]
    test_data = ["id path action_label start_act end_act start_frame end_frame"]
    if subset == "brian_coffee_":
        train_id = []
        val_id = []
        test_id = [1, 2, 3, 4]
        sta_row = 2
    else:
        # fmt: off
        #train_id = [1,2,3,4,5,6,7,8,9,10,11,12,41,42,43,44,45] #[32,33,34,35,36,46,47,48,49,50]###[21,22,23,24,41,42,43,44,45] #[1,2,3,4,5,6,7,8,9,10,11,12] #[1,2,3,4,5,6,7,8,9,10,11,12,37,38,39,40,41,42,43,44,45,46,47,48,49,50]
        train_id = []#[1,2,3,4,5,6,7,8,9,10,11,12,37,38,39,40,46,47,48,49,50]#[25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]
        val_id = []#[13,14,15,16,17,18,19,20]#[25,26,27,28,29,30,31,32,33,34,35,36]
        test_id = []#[23,24]#[21,22,23,24,46,47,48,49,50]#[13,14,15,16,17,18,19,20,21,22,23,24]
        train_id = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,46,47,48,49,50]
        val_id = [23,24,41,42,43,44,45]
        # fmt: on

        sta_row = 1
    all_id = train_id + val_id + test_id
    train_frs, val_frs, test_frs = [], [], []
    train_cnt, val_cnt, test_cnt = 0, 0, 0

    for k in all_id:
        data_dir = osp.join(root_dir, "Data", subset + str(k))
        label_file = osp.join(root_dir, "Labels", subset + str(k) + ".csv")
        if k in train_id:
            h2o_data, h2o_frs = parse_video(
                root_dir, data_dir, label_file, annotations, True, sta_row, train_cnt
            )
            train_data = train_data + h2o_data
            train_frs = train_frs + h2o_frs
            train_cnt += len(h2o_data)
        elif k in val_id:
            h2o_data, h2o_frs = parse_video(
                root_dir, data_dir, label_file, annotations, False, sta_row, val_cnt
            )
            val_data = val_data + h2o_data
            val_frs = val_frs + h2o_frs
            val_cnt += len(h2o_data)
        elif k in test_id:
            h2o_data, h2o_frs = parse_video(
                root_dir, data_dir, label_file, annotations, False, sta_row, test_cnt
            )
            test_data = test_data + h2o_data
            test_frs = test_frs + h2o_frs
            test_cnt += len(h2o_data)

    np.savetxt(
        osp.join(root_dir, "Data", "label_split", subset + "action_train4.txt"),
        np.array(train_data, dtype=object),
        delimiter=" ",
        fmt="%s",
    )
    # np.savetxt(osp.join(root_dir, "Data", "label_split", subset+'pose_train1.txt'), np.array(train_frs, dtype=object), delimiter=" ", fmt="%s")
    np.savetxt(
        osp.join(root_dir, "Data", "label_split", subset + "action_val4.txt"),
        np.array(val_data, dtype=object),
        delimiter=" ",
        fmt="%s",
    )
    # np.savetxt(osp.join(root_dir, "Data", "label_split", subset+'pose_val1.txt'), np.array(val_frs, dtype=object), delimiter=" ", fmt="%s")
    # np.savetxt(osp.join(root_dir, "Data", "label_split", 'hannah_'+'action_test.txt'), np.array(test_data, dtype=object), delimiter=" ", fmt="%s")
    # np.savetxt(osp.join(root_dir, "Data", "label_split", subset+'pose_test1.txt'), np.array(test_frs, dtype=object), delimiter=" ", fmt="%s")
