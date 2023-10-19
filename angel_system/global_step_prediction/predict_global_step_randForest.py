import yaml
import os
import seaborn as sn
import numpy as np
import kwcoco
import matplotlib.pyplot as plt
import sklearn
import sklearn.ensemble
from sklearn.metrics import confusion_matrix
import scipy.ndimage as ndi
import torch

def sanitize_str(str_: str):
    """
    Convert string to lowercase and emove trailing whitespace and period.

    :param str_: Input text

    :return: ``str_`` converted to lowercase and stripped of trailing whitespace and period.
    :rtype: str
    """
    return str_.lower().strip(" .")

def bilateralFtr1D(y, sSpatial = 5, sIntensity = 1):
    '''
    The equation of the bilateral filter is
    
            (       dx ^ 2       )       (         dI ^2        )
    F = exp (- ----------------- ) * exp (- ------------------- )
            (  sigma_spatial ^ 2 )       (  sigma_Intensity ^ 2 )
        ~~~~~~~~~~~~~~~~~~~~~~~~~~
        This is a guassian filter!
        dx - The 'geometric' distance between the 'center pixel' and the pixel
         to sample
    dI - The difference between the intensity of the 'center pixel' and
         the pixel to sample
    sigma_spatial and sigma_Intesity are constants. Higher values mean
    that we 'tolerate more' higher value of the distances dx and dI.
    
    Dependencies: numpy, scipy.ndimage.gaussian_filter1d
    
    calc gaussian kernel size as: filterSize = (2 * radius) + 1; radius = floor (2 * sigma_spatial)
    y - input data
    '''

    # gaussian filter and parameters
    radius = np.floor (2 * sSpatial)
    filterSize = ((2 * radius) + 1)
    ftrArray = np.zeros(int(filterSize))
    ftrArray[int(radius)] = 1
    
    # Compute the Gaussian filter part of the Bilateral filter
    gauss = ndi.gaussian_filter1d(ftrArray, sSpatial)

    # 1d data dimensions
    width = y.size

    # 1d resulting data
    ret = np.zeros (width)

    for i in range(width):

        ## To prevent accessing values outside of the array
        # The left part of the lookup area, clamped to the boundary
        xmin = max(i - radius, 1);
        # How many columns were outside the image, on the left?
        dxmin = xmin - (i - radius);

        # The right part of the lookup area, clamped to the boundary
        xmax = min(i + radius, width);
        # How many columns were outside the image, on the right?
        dxmax = (i + radius) - xmax;

        # The actual range of the array we will look at
        area = y [int(xmin):int(xmax)]

        # The center position
        center = y[i]

        # The left expression in the bilateral filter equation
        # We take only the relevant parts of the matrix of the
        # Gaussian weights - we use dxmin, dxmax, dymin, dymax to
        # ignore the parts that are outside the image
        expS = gauss[int((1+dxmin)):int((filterSize-dxmax))]

        # The right expression in the bilateral filter equation
        dy = y [int(xmin):int(xmax)] - y[i]
        dIsquare = (dy * dy)
        expI = np.exp (- dIsquare / (sIntensity * sIntensity))

        # The bilater filter (weights matrix)
        F = expI * expS

        # Normalized bilateral filter
        Fnormalized = F / sum(F)

        # Multiply the area by the filter
        tempY = y [int(xmin):int(xmax)] * Fnormalized

        # The resulting pixel is the sum of all the pixels in
        # the area, according to the weights of the filter
        # ret(i,j,R) = sum (tempR(:))
        ret[i] = sum (tempY)
    
    return ret


def get_average_TP_activations(coco, clf):
    # For each activity, given the Ground Truth-specified
    # frame subset where that activity is happening, get the
    # average activation of that class.

    all_activity_ids = np.unique(np.asarray(coco.images().lookup('activity_gt')))
    all_vid_ids = np.unique(np.asarray(coco.images().lookup('video_id')))

    activity_confs = torch.asarray(coco.images().lookup("activity_conf"))
    new_probs = clf.predict_proba(activity_confs)
    new_probs_all_classes = np.zeros((new_probs.shape[0], new_probs.shape[1]+1))
    new_probs_all_classes[:,0:17] = new_probs[:,0:17]
    new_probs_all_classes[:,18:] = new_probs[:,17:]

    avg_probs = np.zeros(max(all_activity_ids) + 1)

    for activity_id in all_activity_ids:
        image_ids = [img['id'] for img in coco.videos(video_ids=all_vid_ids).images[0].objs if img['activity_gt'] == activity_id]
        probs_for_true_inds = np.asarray(new_probs_all_classes)[image_ids][:,activity_id]
        avg_prob = np.mean(probs_for_true_inds)
        avg_probs[activity_id] = avg_prob

    # import ipdb; ipdb.set_trace()

    return avg_probs

def train_random_forest(coco):
    activity_confs = torch.asarray(coco.images().lookup("activity_conf"))
    activity_preds = torch.asarray(coco.images().lookup("activity_pred"))
    activity_gt = torch.asarray(coco.images().lookup("activity_gt"))
    n_classes = len(activity_confs[0])
    clf = sklearn.ensemble.RandomForestClassifier(n_estimators = 100, max_depth=2, random_state=0) #, class_weight="balanced")
    # training
    clf.fit(activity_confs,activity_gt)

    # Sanity check: print out training dataset performance
    y_hat= clf.predict(activity_confs)

    TP = np.sum(activity_gt.numpy()==y_hat)
    n = y_hat.shape[0]
    print(f'{TP}/{n} Train RF Accuracy {100*TP/n:0.2f}%')

    TP = np.sum(activity_gt.numpy()==activity_preds.numpy())
    n = y_hat.shape[0]
    print(f'{TP}/{n} TCN Accuracy {100*TP/n:0.2f}%')

    return clf


config_fn = "config/tasks/task_steps_cofig-recipe-coffee-shortstrings.yaml"
with open(config_fn, "r") as stream:
    config = yaml.safe_load(stream)
labels = [sanitize_str(l["description"]) for l in config["steps"]]
steps = config['steps']
if steps[0]['id'] == 1:
    config['steps'].insert(0, {'id':0,
        'activity_id':0,
        'description':'background',
        'median_duration_seconds':0.5,
        'mean_conf':0.5,
        'std_conf':0.2,
        })

coco_val = kwcoco.CocoDataset("model_files/val_activity_preds_epoch40.mscoco.json")
coco_test = kwcoco.CocoDataset("model_files/test_activity_preds.mscoco.json")

image_ids = coco_test.index.vidid_to_gids[3]
video_dset = coco_test.subset(gids=image_ids, copy=True)

# "Training": for each activity class, see what the average "true positive"
# activation was.
clf = train_random_forest(coco_test)
avg_probs = get_average_TP_activations(coco_test, clf)
print(f"average_probs = {avg_probs}")

all_vid_ids = np.unique(np.asarray(coco_val.images().lookup('video_id')))

for vid_id in all_vid_ids:
    print(f"vid_id {vid_id}")

    image_ids = coco_test.index.vidid_to_gids[vid_id]
    video_dset = coco_test.subset(gids=image_ids, copy=True)

    # All N activity confs x each video frame
    activity_confs = video_dset.images().lookup("activity_conf")
    new_probs = clf.predict_proba(activity_confs)
    new_probs_all_classes = np.zeros((new_probs.shape[0], new_probs.shape[1]+1))
    new_probs_all_classes[:,0:17] = new_probs[:,0:17]
    new_probs_all_classes[:,18:] = new_probs[:,17:]


    next_step = 1
    step_predictions = []
    num_frames_activated = 0

    # Predicted step: confidence has been above threshold for 5 frames.
    for activity_conf in new_probs_all_classes:
        # Next step
        next_activity_id = steps[next_step]['activity_id']

        next_activity_conf = activity_conf[next_activity_id]

        avg_prob_next_activity = avg_probs[next_activity_id]

        if next_activity_conf > 0.8 * avg_prob_next_activity:
            num_frames_activated += 1
        else:
            num_frames_activated = 0

        if num_frames_activated >= 8:
            if next_step < 23:
                next_step += 1
            num_frames_activated = 0

        step_predictions.append(next_step-1)

    # Ground truth step:
    activity_gts = video_dset.images().lookup("activity_gt")
    step_gts = []
    step_gts_no_background = []
    current_step = 0
    for activity_gt in activity_gts:
        # convert activity id to step id
        step_id = next(int(item['id']) for item in steps if item['activity_id'] == activity_gt)
        step_gts.append(step_id)

        # A version of GT that never jumps back to 0
        if step_id > 0:
            current_step = step_id
        step_gts_no_background.append(current_step)


    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(100, 100))
    cm = confusion_matrix(step_gts_no_background, step_predictions)
    sn.heatmap(cm, annot=True, fmt="g", ax=ax)
    sn.set(font_scale=4)
    ax.set(
            title="Confusion Matrix",
            xlabel="Predicted Label",
            ylabel="True Label",)
    fig.savefig(f"./outputs/plot_confusion_mat_vid{vid_id}.png")

    # Plot gt vs predicted class across all vid frames
    fig = plt.figure()
    sn.set(font_scale=1)
    step_gts = [float(i) for i in step_gts]
    plt.plot(step_gts, label = 'gt')
    plt.plot(step_predictions, label = 'estimated')
    #plt.plot(inliers-0.5, label = 'inliers')
    plt.plot(10*np.asarray(activity_confs)[:,17]-5, label = 'act_preds[17]')
    plt.plot(10*np.asarray(activity_confs)[:,18]-5, label = 'act_preds[18]')
    plt.plot(10*np.asarray(activity_confs)[:,19]-5, label = 'act_preds[19]')

    plt.plot(bilateralFtr1D(10*np.asarray(activity_confs)[:,17])-10, label = 'act_preds_bilateral[17]')
    plt.plot(bilateralFtr1D(10*np.asarray(activity_confs)[:,18])-10, label = 'act_pred_bilateral[18]')
    plt.plot(bilateralFtr1D(10*np.asarray(activity_confs)[:,19])-10, label = 'act_preds_bilateral[19]')
    #plt.plot(10*X_conf_incremental, label = 'confidence')
    #plt.plot(10*vid_acts[:,10], label = act_labels[10])
    #plt.plot(10*vid_acts[:,11], label = act_labels[11])
    #plt.plot(10*vid_acts[:,12], label = act_labels[12])
    plt.legend()
    fig.savefig(f"./outputs/plot_pred_vs_gt_vid{vid_id}.png")


