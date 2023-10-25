import yaml
import os
import seaborn as sn
import numpy as np
import kwcoco
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import scipy.ndimage as ndi

def sanitize_str(str_: str):
    """
    Convert string to lowercase and emove trailing whitespace and period.

    :param str_: Input text

    :return: ``str_`` converted to lowercase and stripped of trailing whitespace and period.
    :rtype: str
    """
    return str_.lower().strip(" .")

def plot_steps_confusion_matrix(step_gts_no_background, step_predictions):
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(100, 100))
    cm = confusion_matrix(step_gts_no_background, step_predictions, normalize="true")
    sn.heatmap(cm, annot=True, fmt="0.0%", ax=ax, linewidth=.5)
    sn.set(font_scale=4)
    ax.set(
            title="Confusion Matrix",
            xlabel="Predicted Label",
            ylabel="True Label",)
    fig.savefig(f"./outputs/plot_confusion_mat_vid{vid_id}.png")

def plot_positive_GT_conf_distributions(activity_confs, activity_gt):
    """
    plot_TP_conf_distributions:
    For each activity, plot the distribution of confidences when ground
    truth indicates that activity is happening.

    i.e.: for activity x, for frames in which ground truth = x, plot 
    the distribution of confidences.

    Inputs:
    activity_confs: frames x class-wise-confidences. Given a kwcoco
        dataset called "coco":
        ```
        activity_confs = torch.asarray(coco.images().lookup("activity_conf"))
        ```
        (49K x 25 for coffee val set.)
    activity_gt: frames x ground truth activity_id.
        Given a kwcoco dataset called "coco":
        ```
        activity_gt = torch.asarray(coco.images().lookup("activity_gt"))
        ```
    """

    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    # Get data together
    true_confs = [float(activity_confs[i,truth_ind]) for i, truth_ind in enumerate(activity_gt)]
    data = {"true_conf":true_confs, "gt":activity_gt}
    df = pd.DataFrame(data)

    false_confs = np.array([[a for i, a in enumerate(act_conf) if i != gt] for act_conf, gt in zip(activity_confs, activity_gt)]).flatten()
    false_gt = np.array([[gt for i, a in enumerate(act_conf) if i != gt] for act_conf, gt in zip(activity_confs, activity_gt)]).flatten()
    data_opposite = {"true_conf":false_confs, "gt":false_gt}
    df_opposite = pd.DataFrame(data_opposite)

    def plot(df):
        # Initialize the FacetGrid object
        pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
        g = sns.FacetGrid(df, row="gt", hue="gt", aspect=15, height=.5, palette=pal)

        # Draw the densities in a few steps
        g.map(sns.kdeplot, "true_conf",
              bw_adjust=.5, clip_on=False,
              fill=True, alpha=1, linewidth=1.5)
        g.map(sns.kdeplot, "true_conf", clip_on=False, color="w", lw=2, bw_adjust=.5)

        # passing color=None to refline() uses the hue mapping
        g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)
        
        # Define and use a simple function to label the plot in axes coordinates
        def label(x, color, label):
            ax = plt.gca()
            ax.text(0, .2, label, fontweight="bold", color=color,
                    ha="left", va="center", transform=ax.transAxes)
        g.map(label, "true_conf")

        # Set the subplots to overlap
        g.figure.subplots_adjust(hspace=-.25)

        # Remove axes details that don't play well with overlap
        g.set_titles("")
        g.set(yticks=[], ylabel="")
        g.despine(bottom=True, left=True)

        # save
        plt.savefig("./outputs/plot_positive_GT_conf_distributions.png")

def plot_gt_vs_predicted_one_recipe(step_gts, step_predictions, activity_confs, avg_probs):
    # Plot gt vs predicted class across all vid frames
    fig = plt.figure()
    sn.set(font_scale=1)
    step_gts = [float(i) for i in step_gts]
    plt.plot(step_gts, label = 'gt')
    plt.plot(step_predictions, label = 'estimated')

    starting_zero_value = 0
    for i in range(len(avg_probs)):
        starting_zero_value -= 2
        plot_line = np.asarray(activity_confs)[:,i]
        plt.plot(2*plot_line-starting_zero_value, label = f'act_preds[{i}]')
        plt.plot([2*int(j < avg_probs[i])+starting_zero_value for j in plot_line], label = f'act_preds_threshold[{i}]')
    plt.legend()
    fig.savefig(f"./outputs/plot_pred_vs_gt_vid{vid_id}.png")
    plt.show()

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


def get_average_TP_activations(coco):
    # For each activity, given the Ground Truth-specified
    # frame subset where that activity is happening, get the
    # average activation of that class.

    all_activity_ids = np.unique(np.asarray(coco.images().lookup('activity_gt')))
    all_vid_ids = np.unique(np.asarray(coco.images().lookup('video_id')))

    avg_probs = np.zeros(max(all_activity_ids) + 1)

    for activity_id in all_activity_ids:
        #image_ids = coco.index.vidid_to_gids[vid_id]
        image_ids = [img['id'] for img in coco.videos(video_ids=all_vid_ids).images[0].objs if img['activity_gt'] == activity_id]
        sub_dset = coco.subset(gids=image_ids, copy=True)
        probs_for_true_inds = np.asarray(
                sub_dset.images().lookup("activity_conf"))[:,activity_id]
        avg_prob = np.mean(probs_for_true_inds)
        avg_probs[activity_id] = avg_prob

    return avg_probs

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
avg_probs = get_average_TP_activations(coco_test)
print(f"average_probs = {avg_probs}")

all_vid_ids = np.unique(np.asarray(coco_val.images().lookup('video_id')))

for vid_id in all_vid_ids:
    print(f"vid_id {vid_id}")

    image_ids = coco_test.index.vidid_to_gids[vid_id]
    video_dset = coco_test.subset(gids=image_ids, copy=True)

    # All N activity confs x each video frame
    activity_confs = video_dset.images().lookup("activity_conf")

    next_step = 1
    step_predictions = []
    num_frames_activated = 0

    # Predicted step: confidence has been above threshold for 5 frames.
    threshold_frame_count = 8
    for i, activity_conf in enumerate(activity_confs):

        # Check if we're done: if so, append last step & continue
        if next_step == len(steps):
            step_predictions.append(next_step-1)
            continue
        # Next step
        next_activity_id = steps[next_step]['activity_id']
        next_next_activity_id = steps[min(len(steps)-1,next_step + 1)][
                'activity_id']

        next_activity_conf = activity_conf[next_activity_id]
        next_next_activity_conf = activity_conf[next_next_activity_id]

        avg_prob_next_activity = avg_probs[next_activity_id]
        avg_prob_next_next_activity = avg_probs[next_next_activity_id]
        if i > 15:
            threshold_frame_count = 16

        # Check next activity. If conf > threshold, designate as 
        # "activated frame". Else, reset the counter.
        if next_activity_conf > 0.8 * avg_prob_next_activity:
            num_frames_activated += 1
        else:
            num_frames_activated = 0
        # Check 2 steps ahead too.
        if next_next_activity_conf > 0.8 * avg_prob_next_next_activity:
            num_skip2_frames_activated += 1
        else:
            num_skip2_frames_activated = 0

        # If the next step (or the one after) is activated long enough,
        # jump ahead.
        if num_frames_activated >= threshold_frame_count:
            next_step += 1
            num_frames_activated = 0
            num_skip2_frames_activated = 0
        elif num_skip2_frames_activated >= threshold_frame_count:
            next_step = min(next_step + 2, len(steps))
            num_frames_activated = 0
            num_skip2_frames_activated = 0
            print(f"hit a skip-step!! Jumped to step {next_step - 1}.")

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

    if True:
        plot_steps_confusion_matrix(step_gts_no_background, step_predictions)
        plot_gt_vs_predicted_one_recipe(step_gts, step_predictions, activity_confs, avg_probs)

    if False:
        plot_positive_GT_conf_distributions(activity_confs, activity_gt)



