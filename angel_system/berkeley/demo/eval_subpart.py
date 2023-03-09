import os
import pandas as pd
import numpy as np


# pred file

def calculate_iou(gt, pred):
    gt = gt[1:]
    pred = pred[1:]
    p_end = int(pred[1])
    y_end = int(gt[1])
    p_start = int(pred[0])
    y_start = int(pred[0])
    intersection = min(p_end, y_end) - max(p_start, y_start)
    union = max(p_end, y_end) - min(p_start, y_start)
    iou = intersection/union
    return iou


def align_name(data):
    """

    :param data:
    :return: hot index vector
    """

    _data = data[0]
    if _data in [' Pour 12 ounces of water into liquid measuring cup', 'Measure 12 ounces of cold water', 'Measure 12 ounces of water in the liquid measuring cup', ' Pour 12 ounces of water into liquid measuring cup', ' Measure 12 ounces of water in the liquid measuring cup']:
        index = 1
    elif _data in ['Transfer the water to a kettle', 'Pour the water from the liquid measuring cup into the electric kettle']:
        index = 2
    elif _data in ['Turn on the kettle', 'Turn on the Kettle']:
        index = 3
    elif _data in ['Place the Dripper on top of the mug', 'Place the dripper on top of the mug']:
        index = 4
    elif _data in ['Fold the coffee filter again to create a quarter circle.', 'Fold the filter in half again to create a quarter-circle']:
        index = 6
    elif _data in ['Place the folder paper into the dripper.', 'Place the folded filter into the dripper such that the the point of the quarter-circle rests in the center of the dripper']:
        index = 7
    elif _data in ['Spread the filter open to create a cone inside the dripper', 'Spread the filter open to create a cone inside the dripper.']:
        index = 8
    elif _data in ['Turn on the kitchen scale.', 'Turn on the kitchen scale']:
        index = 9
    elif _data in ['Place a bowl on the scale.', 'Place a bowl on the scale']:
        index = 10
    elif _data in ['Zero out the kitchen scale.', 'Zero the scale']:
        index = 11
    elif _data in ['Add coffee beans into the bowl until read 25 grams.', 'Add coffee beans to the bowl until the scale reads 25 grams']:
        index = 12
    elif _data in ['Poured the measured beans into the coffee grinder.', 'Pour the measured coffee beans into the coffee grinder']:
        index = 13
    elif _data in ['Use timer', 'Set timer for 20 seconds', 'Turn on the timer', 'Set timer to 30 seconds']:
        index = 14
    elif _data in ['Grind the coffee beans by pressing and holding down the back part', 'Grind the coffee beans by pressing and holding down on the black part of the lid']:
        index = 15
    elif _data in ['Pour the grinded coffee beans into the filter cone.', 'Pour the grounded coffee beans into the filter cone prepared in step 2', 'Pour the grounded coffee beans into the filter cone prepared in step 2 ']:
        index = 16
    elif _data in ['Turn on the thermometer.', 'Turn on the thermometer']:
        index = 17
    elif _data in ['Place the end of the thermometer into the water.', 'Place the end of the thermometer into the water. ']:
        index = 18
    elif _data in ['Pour small amount of water onto the grounds.', 'Pour a small amount of water over the grounds in order to wet the grounds. ', 'Pour a small amount of water over the grounds in order to wet the grounds.']:
        index = 19
    elif _data in ['Slowly pour water into the grounds circular motion.', 'Slowly pour the water over the grounds in a circular motion. Do not overfill beyond the top of the paper filter']:
        index = 20
    elif _data in ['Remove dripper from cup.', 'Remove the dripper from the cup']:
        index = 21
    elif _data in ['Remove the coffee grounds and paper filter from the dripper.', 'Remove the coffee grounds and paper filter from the dripper. ']:
        index = 22
    elif _data in ['Discard the coffee grounds and paper filter.', 'Discard the coffee grounds and paper filter. ', 'Discard the coffee grounds and paper filter.  ']:
        index = 23
    elif _data in ['Take the coffee filter and fold it in half to create a semi-circle', 'Take the coffee filter and fold it in half into a semicircle.']:
        index = 24
    elif _data in ['Allow the rest of the water in the dripper to drain']:
        index = 25
    else:
        index = -1
    return index
    # elif _data in ['Place the end of the thermometer into the water.']:
    #     index = 18
    # elif _data in ['Place the end of the thermometer into the water.']:
    #     index = 18
    # elif _data in ['Place the end of the thermometer into the water.']:
    #     index = 18
    # elif _data in ['Place the end of the thermometer into the water.']:
    #     index = 18


root = '/shared/niudt/detectron2/DEMO_Results/2022-11-07/eval'

_list = os.listdir(root)

iou_thresh = 0
results = []
for _pred in _list:
    sub_result = []
    pred_dir = os.path.join(root, _pred)
    #read
    pred_list = pd.read_csv(pred_dir).values[:, 1:].tolist()


    # read gt
    index = pred_dir.split('_')[-1][:-4]
    if int(index) in [1,3,7,8,11,12,13,14,25,31]:
        continue
    print('*'*100)
    print('Processing sequence %s'%index)
    print('*' * 100)
    sub_result.append(index)
    gt_dir = '/shared/niudt/DATASET/PTG_Kitware/labels/' + 'all_activities_' + index + '/' + 'all_activities_' + index + '.csv'
    if not os.path.exists(gt_dir):
        print('gt %s not find'%gt_dir)

    df_gt = pd.read_csv(gt_dir, header=None).values

    s = 1

    #transfer the format of gt
    c = 0
    for x in df_gt:
        if x[1].split('_')[0] == 'frame':
            break
        c = c + 1
    gt_list = []
    for i in range(c, df_gt.shape[0] - 1, 2):
        j = i + 1

        # print(ann[i, 9])
        # print(ann[j, 9])
        if str(df_gt[i, 9]) == 'nan' or str(df_gt[j, 9]) == 'nan':
            continue
        if df_gt[i, 9] != df_gt[j, 9]:
            print('error' * 50)
            continue
        start_frame = int(df_gt[i, 2])
        end_frame = int(df_gt[j, 2])
        gt_list.append([df_gt[j, 9], start_frame, end_frame])

    # align gt
    for _gt_ in gt_list:
        if align_name(_gt_) == -1:
            print(_gt_[0])
        else:
            _gt_[0] = align_name(_gt_)

    # align pred
    for _pred_ in pred_list:
        if align_name(_pred_) == -1:
            print(_pred_[0])
        else:
            _pred_[0] = align_name(_pred_)
    s = 1


    # calculate the recall
    total_gt_num = len(gt_list)
    tp = 0
    fn = 0
    for gt_ in gt_list:
        flag_true = 0
        sub_step_gt = gt_[0]
        start_frame_gt = gt_[1]
        end_frame_gt = gt_[2]


        # find it in the pred
        for pred_ in pred_list:
            sub_step_pred = pred_[0]
            start_frame_pred = pred_[1]
            end_frame_pred = pred_[2]
            s = 1

            if sub_step_gt == sub_step_pred:
                iou = calculate_iou(gt_, pred_)
                if iou > iou_thresh:
                    flag_true = 1
                    break
        if flag_true == 1:
            tp = tp + 1
        else:
            fn = fn + 1

    if tp + fn != total_gt_num:
        print('error' * 100)
    recall = tp / (tp + fn)
    sub_result.append(recall)

    # calculate the percision
    total_pred_num = len(pred_list)
    tp = 0
    fp = 0
    for pred_ in pred_list:
        flag_true = 0
        sub_step_pred = pred_[0]
        start_frame_pred = pred_[1]
        end_frame_pred = pred_[2]


        # find it in the gt
        for gt_ in gt_list:
            sub_step_gt = gt_[0]
            start_frame_gt = gt_[1]
            end_frame_gt = gt_[2]
            s = 1

            if sub_step_gt == sub_step_pred:
                iou = calculate_iou(gt_, pred_)
                if iou > iou_thresh:
                    flag_true = 1
                    break
        if flag_true == 1:
            tp = tp + 1
        else:
            fp = fp + 1

    if tp + fp != total_pred_num:
        print('error' * 100)
    percison = tp / (tp + fp)
    sub_result.append(percison)
    results.append(sub_result)

print(results)

df = pd.DataFrame(results, index=None)
df.to_csv('/shared/niudt/detectron2/DEMO_Results/2022-11-06/eval_iou0.csv', header=['sequence', 'recall', 'precision'])











