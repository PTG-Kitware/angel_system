import yaml
import os
import seaborn as sn
import numpy as np
import kwcoco
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import scipy.ndimage as ndi

class GlobalStepPredictor(
        ):
    def __init__(self,
        max_step_jump=2,
        threshold_multiplier=0.8,
        threshold_multiplier_weak=0.0,
        threshold_frame_count=8,
        threshold_frame_count_weak=0.0,
        deactivate_thresh_mult = 0.3,
        deactivate_thresh_frame_count = 20,
        recipe_configs=["coffee"],
        ):

        '''
        GlobalStepPredctor: based on a TCN activity classifier's activity classification
        outputs, 
        '''

        # maximum number of steps that can be "jumped" to.
        # i.e. if max_step_jump is 1, from step 2, you can only jump to 3.
        self.max_step_jump = max_step_jump
        # Earlier, we've caluclated the average activation for each activity,
        # when GT says that activity was happening.
        # Suppose that avg activation was 0.9. And threshold_multiplier = 0.8.
        # Then, the threshold to "activate" this step is 0.72.
        # activation must be > 0.72 for (int(threshold_frame_count)) frames.
        self.threshold_multiplier = threshold_multiplier
        # If max_step jump is 2, you could jump 2 frames, e.g. from 2 to 4.
        # The "weak threshold multiplier" is the minimum activation of the 
        # "next step" needed to jump "two steps.".
        # Example: the tracker is on step 2. If step 3 "weakly activates", the 
        # tracker still reports that it's on step 2, but it's now able to jump
        # to step 4 if step 4 activates.
        self.threshold_multiplier_weak = threshold_multiplier_weak
        self.threshold_frame_count = threshold_frame_count
        self.threshold_frame_count_weak = threshold_frame_count_weak

        # Threshold & nuber of consecutive below-threshold frames required
        # to deactivate an activity instance.
        self.deactivate_thresh_mult = deactivate_thresh_mult
        self.deactivate_thresh_frame_count = deactivate_thresh_frame_count


        # Glabally track current frame. When a new tracker is added,
        # The tracker dict "starting_frame_index" will indicate
        # how many frames were loaded before that. The first trackers
        # all start at frame 30, since the TCN takes in 30 frames.
        self.current_frame = 30

        self.activity_conf_history = np.empty((0,25))


        # Array of tracker dicts
        self.trackers = [] 
        for recipe in recipe_configs:
            self.initialize_new_recipe_tracker(recipe)

        # activated_activites: shape = (number of activity indexes) x 2
        # - [ [ activity_index_0, num. activating/deactivating frames ],
        #       ...
        #   ]
        # When the left column is 0, the right column indicates the number of
        # previous frames in a row that have been "activated".
        # When the left column is 1, the right column indicates the number of
        # previous frames in a row that have been "DEactivated".
        self.activated_activities = np.zeros((np.max(np.array([
            tracker["step_to_activity_id"] for tracker in self.trackers])) + 1, 2))

    def compute_average_TP_activations(self, coco):
        # For each activity, given the Ground Truth-specified
        # frame subset where that activity is happening, get the
        # average activation of that class.

        all_activity_ids = np.unique(np.asarray(coco.images().lookup('activity_gt')))
        all_vid_ids = np.unique(np.asarray(coco.images().lookup('video_id')))

        avg_probs = np.zeros(max(all_activity_ids) + 1)

        for activity_id in all_activity_ids:
            #image_ids = coco.index.vidid_to_gids[vid_id]
            image_ids = [img['id'] for img in coco.videos(video_ids=all_vid_ids).images[
                0].objs if img['activity_gt'] == activity_id]
            sub_dset = coco.subset(gids=image_ids, copy=True)
            probs_for_true_inds = np.asarray(
                    sub_dset.images().lookup("activity_conf"))[:,activity_id]
            avg_prob = np.mean(probs_for_true_inds)
            avg_probs[activity_id] = avg_prob

        self.avg_probs = avg_probs
        return self.avg_probs

    def get_average_TP_activations_from_file(self, fpath):
        self.avg_probs = np.load(fpath)

    def get_average_TP_activations_from_array(self, avg_probs):
        self.avg_probs = avg_probs

    def initialize_new_recipe_tracker(self, recipe):
        '''
        tracker dict fields:
            {
            "recipe":"coffee"
            "current_step": 5
                - int type.
                - 0 = background.
            "total_num_steps": 23
                - int type.
            "can_skip": False
                - boolean type.
                - Switched to "True" when the "weak threshold" has been met.
                  i.e. the "weak threshold" has been achieved for
                  "threshold_frame_count_weak" number of frames.
                  When this happens, the tracker can "skip" a step.
            "step_to_activity_id": [0,1,2,3,24,4,...]
                - array type.
                - steps_to_activities[current_step] is the current activity index
                  the tracker is on. (step index != activity index).
            "step_to_activity_desc": ["background", ...]
            "starting_frame_index": 30
                - index videos frame at which point this tracker instance was initialized.
            "prediction_history": [0,0,0,...,1,1,1,...23,23]
            "active": True
                - When true, keep checking to update this recipe.
                - When false, the recipe is complete... stop adding to its history
                  or checking for a next step.

        '''
        tracker_dict = {}
        if recipe == "coffee":
            config_fn = "config/tasks/task_steps_cofig-recipe-coffee-shortstrings.yaml"
        else:
            raise ValueError(f"Invalid recipe type. Valid types: [coffee].")
        with open(config_fn, "r") as stream:
            config = yaml.safe_load(stream)
        labels = [self.sanitize_str(l["description"]) for l in config["steps"]]
        steps = config['steps']
        if steps[0]['id'] == 1:
            config['steps'].insert(0, {'id':0,
                'activity_id':0,
                'description':'background',
                'median_duration_seconds':0.5,
                'mean_conf':0.5,
                'std_conf':0.2,
                })
        tracker_dict["recipe"] = recipe
        tracker_dict["current_step"] = 0
        tracker_dict["total_num_steps"] = len(steps)
        tracker_dict["can_skip"] = False
        tracker_dict["step_to_activity_id"] = [step['activity_id'] for step in steps]
        tracker_dict["step_to_activity_desc"] = [step['description'] for step in steps]
        tracker_dict["prediction_history"] = np.array([])
        tracker_dict["active"] = True
        tracker_dict["steps"] = steps

        self.trackers.append(tracker_dict)

    def process_new_confidences(self, activity_confs):

        assert np.array(activity_confs).shape[1] <= len(self.activated_activities)

        activated_indexes = np.where(self.activated_activities[:,0] == 1)[0]
        deactivated_indexes = np.where(self.activated_activities[:,0] == 0)[0]
        assert len(activated_indexes) + len(deactivated_indexes) == len(self.activated_activities)

        # activity conf = vector of all activities for one frame.
        for i, activity_conf in enumerate(activity_confs):
            # activated_confidences = classes predicted to be "happening" now.
            activated_indexes = np.where(self.activated_activities[:,0] == 1)[0]
            # deactivated_confidences = classes predicted NOT to be happening yet.
            deactivated_indexes = np.where(self.activated_activities[:,0] == 0)[0]

            # Check for activations > 80% * (avg act threshold)
            above_pos_threshold_indexes = np.where(activity_conf > 
                    self.threshold_multiplier * self.avg_probs)[0]
            # ...and < 30% * (avg act threshold)
            below_pos_threshold_indexes = np.where(activity_conf < 
                    self.threshold_multiplier * self.avg_probs)[0]
            # Check for activations > 80% * (avg act threshold)
            above_neg_threshold_indexes = np.where(activity_conf > 
                    self.deactivate_thresh_mult * self.avg_probs)[0]
            # ...and < 30% * (avg act threshold)
            below_neg_threshold_indexes = np.where(activity_conf < 
                    self.deactivate_thresh_mult * self.avg_probs)[0]

            indexes_to_add_pos_threshold_increment = np.intersect1d(
                    np.array(deactivated_indexes), 
                    np.array(above_pos_threshold_indexes))
            self.activated_activities[indexes_to_add_pos_threshold_increment, 1] += 1
            indexes_to_add_neg_threshold_increment = np.intersect1d(
                    np.array(activated_indexes),
                    np.array(below_neg_threshold_indexes))
            self.activated_activities[indexes_to_add_neg_threshold_increment, 1] += 1
            #if len(indexes_to_add_pos_threshold_increment) > 0:
            #    import ipdb; ipdb.set_trace()
            #print(np.max(self.activated_activities))

            # Not activated AND not above activation threshold? 
            # Then set the activation clock back to zero.
            indexes_to_reset_deactive = np.intersect1d(np.array(deactivated_indexes), 
                    np.array(below_pos_threshold_indexes))
            self.activated_activities[indexes_to_reset_deactive, 1] = 0
            indexes_to_reset_active = np.intersect1d(np.array(activated_indexes), 
                    np.array(above_neg_threshold_indexes))
            self.activated_activities[indexes_to_reset_active, 1] = 0

            # Activate the inactive classes that have surpassed the 
            # activation threshold for enough consecutive frames
            flipping_on_indexes = np.intersect1d(deactivated_indexes,
                    np.where(self.activated_activities[:,1] >= self.
                    threshold_frame_count)[0])
            #flipping_on_indexes = np.where((self.activated_activities[:,0] == 0,)
            #    and (self.activated_activities[:,1] >= self.
            #        threshold_frame_count,))[1]
            if 1 in flipping_on_indexes:
                print(f"woo i = {i}")
            #if 1 in above_threshold_indexes:
                #print(f"activity 1 engaged. i = {i}")
                #print(f"self.activated_activities[1,1] = {self.activated_activities[1,1]}")
            self.activated_activities[flipping_on_indexes, 0] = 1
            self.activated_activities[flipping_on_indexes, 1] = 0

            # Deactivate the active classes that have surpassed the 
            # deactivation threshold for enough consecutive frames
            flipping_off_indexes = np.intersect1d(activated_indexes,
                    np.where(self.activated_activities[:,1] >= self.
                    deactivate_thresh_frame_count)[0])
            if 1 in flipping_off_indexes:
                print(f"boo i = {i}")
            #flipping_off_indexes = np.where((self.activated_activities[:,0] == 1,)
            #    and (self.activated_activities[:,1] <= self.
            #        deactivate_thresh_frame_count,))[1]
            self.activated_activities[flipping_off_indexes, 0] = 0
            self.activated_activities[flipping_off_indexes, 1] = 0

            # Now, go through ACTIVE trackers and see which corresponds to "flipping_on_indexes."
            for tracker_ind, tracker in enumerate(self.trackers):
                current_step = tracker["current_step"]

                # TODO: For now the tracker can jump 1 or 2 steps, if base 
                # jump criteria is met. Add "weak" threshold too.
                if current_step == tracker["total_num_steps"] - 1:
                    continue

                next_step = current_step + 1
                next_next_step = min(next_step + 1, tracker["total_num_steps"]-1)
                current_activity = tracker["step_to_activity_id"][current_step]
                next_activity = tracker["step_to_activity_id"][next_step]
                try:
                    next_next_activity = tracker["step_to_activity_id"][next_next_step]
                except:
                    import ipdb; ipdb.set_trace()

                # TODO: prioritize a 1-step jump over a 2-step jump. Create a 
                # second loop just for the 2-step jumps, after this loop has completed
                # searching for one-step jumps.
                if next_activity in flipping_on_indexes:
                    self.trackers[tracker_ind]["current_step"] = next_step
                    # Each activity activation can only be used once.
                    next_act_ind = np.argwhere(flipping_on_indexes == next_activity)
                    flipping_on_indexes = np.delete(flipping_on_indexes, next_act_ind)
                elif next_next_activity in flipping_on_indexes:
                    self.trackers[tracker_ind]["current_step"] = next_next_step
                    # Each activity activation can only be used once.
                    next_next_act_ind = np.argwhere(flipping_on_indexes == next_next_activity)
                    flipping_on_indexes = np.delete(flipping_on_indexes, next_next_act_ind)

                #TODO: Try requiring that previous step is de-activated

                #Add current preds to this tracker's prediction history
                self.trackers[tracker_ind]["prediction_history"] = \
                        np.append(self.trackers[
                            tracker_ind]["prediction_history"], current_step)
        # Update the current_frame
        self.current_frame += len(activity_confs)

        self.activity_conf_history = \
                np.append(self.activity_conf_history, activity_confs, axis=0)

    def plot_gt_vs_predicted_one_recipe(self, step_gts, fname_suffix=None):
        # Plot gt vs predicted class across all vid frames
        fig = plt.figure()
        sn.set(font_scale=1)
        step_gts = [float(i) for i in step_gts]
        plt.plot(step_gts, label = 'gt')
        for i, tracker in enumerate(self.trackers):
            step_predictions = tracker["prediction_history"]
            plt.plot(step_predictions, label = f'estimated_{i}')
        
        starting_zero_value = 0
        for i in range(len(self.avg_probs)):
            starting_zero_value -= 2
            plot_line = np.asarray(self.activity_conf_history)[:,i]
            plt.plot(2*plot_line+starting_zero_value, label = f'act_preds[{i}]')
            plt.plot([2*int(j < self.avg_probs[i])+starting_zero_value for 
                j in plot_line], label = f'act_preds_threshold[{i}]')
        plt.legend()
        if not fname_suffix:
            fname_suffix = f"vid{vid_id}"
        fig.savefig(f"./outputs/plot_pred_vs_gt_{fname_suffix}.png")
        #plt.show()

    def sanitize_str(self, str_: str):
        """
        Convert string to lowercase and emove trailing whitespace and period.

        :param str_: Input text

        :return: ``str_`` converted to lowercase and stripped of trailing whitespace and period.
        :rtype: str
        """
        return str_.lower().strip(" .")

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

def get_gt_steps_from_gt_activities(video_dset):
    activity_gts = video_dset.images().lookup("activity_gt")
    step_gts = []
    step_gts_no_background = []
    current_step = 0

    # TODO: rm these lines
    def sanitize_str(str_: str):
        return str_.lower().strip(" .")

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
    # TODO: ^^ rm these lines

    for activity_gt in activity_gts:
        # convert activity id to step id
        step_id = next(int(item['id']) for item in steps if item['activity_id'] == activity_gt)
        step_gts.append(step_id)

        # A version of GT that never jumps back to 0
        if step_id > 0:
            current_step = step_id
        step_gts_no_background.append(current_step)
    return step_gts, step_gts_no_background

