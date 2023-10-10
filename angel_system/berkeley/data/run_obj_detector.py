import os
import pickle
import glob
import ubelt as ub

from detectron2.data.detection_utils import read_image
from demo import predictor, model

from angel_system.data.common.kwcoco_utils import preds_to_kwcoco
from angel_system.data.common.load_data import Re_order, time_from_name
from angel_system.data.data_paths import grab_data


os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"


def load_model(config, conf_thr=0.1):
    # Load model
    parser = model.get_parser()
    args = parser.parse_args(
        f"--config-file {config} --confidence-threshold {conf_thr}".split()
    )
    print("Arguments: " + str(args))

    cfg = model.setup_cfg(args)
    print(f"Model: {cfg.MODEL.WEIGHTS}")

    demo = predictor.VisualizationDemo_add_smoothing(
        cfg, last_time=2, draw_output=False, tracking=False
    )

    print(f"Loaded {config}")
    return demo

def run_obj_detector(
    demo, stage, split, no_contact=False, add_hl_hands=True
):
    """
    Run object detector trained without contact information
    on all the videos associated with the task and clear any
    contact predictions
    """
    preds = {}

    for (
        video_folder
    ) in split:
        idx = 0
        video_name = video_folder.split("/")[-1]

        preds[video_name] = {}

        video_images = glob.glob(f"{video_folder}_extracted/images/*.png")
        input_list = Re_order(video_images, len(video_images))
        for image_fn in ub.ProgIter(input_list, desc=f"images in {video_name}"):
            frame, time_stamp = time_from_name(image_fn)

            image = read_image(image_fn, format="RGB")
            # Only for medical
            #if stage == "results":
            #    image = Image.fromarray(image)
            #    image = image.resize(size=(760, 428), resample=Image.BILINEAR)
            #    image = np.array(image)

            h, w, c = image.shape

            predictions, step_infos, visualized_output = demo.run_on_image_smoothing_v2(
                image, current_idx=idx
            )
            decoded_preds = model.decode_prediction(predictions)
            using_contact = True if predictions[2] is not None else False

            if decoded_preds is not None:
                preds[video_name][frame] = decoded_preds

                if no_contact and using_contact:
                    print("Clearing contact states from detections")
                    for class_, dets in preds[video_name][frame].items():
                        for i in range(len(dets)):
                            # Clear contact states
                            preds[video_name][frame][class_][i][
                                "obj_obj_contact_state"
                            ] = False
                            preds[video_name][frame][class_][i][
                                "obj_hand_contact_state"
                            ] = False

                # Image metadata needed later
                preds[video_name][frame]["meta"] = {
                    "file_name": image_fn,
                    "im_size": {"height": h, "width": w},
                    "frame_idx": frame,
                    "time_stamp": time_stamp
                }

            idx += 1

    return preds, using_contact

def main():
    experiment_name = "coffee_resnet50_plus_bkgd_original_labels"
    stage = "results"
    
    print("Experiment: ", experiment_name)
    print("Stage: ", stage)

    ###############
    # Data
    ###############
    recipe = "coffee"
    ( ptg_root,
      data_dir,
      activity_config_fn,
      activity_gt_dir, 
      ros_bags_dir,
      training_split,
      obj_dets_dir,
      obj_config ) = grab_data(recipe, "gyges")

    print("\nTraining split:")
    for split_name, videos in training_split.items():
        print(f"{split_name}: {len(videos)} videos")
        print([v.split("/")[-1] for v in videos])
    print("\n")

    splits = ["val", "train_activity", "test"]

    ###############
    # Model
    ###############
    berkeley_configs_dir = f"{ptg_root}/angel_system/berkeley/configs"
    model_dir = f"{berkeley_configs_dir}/MC50-InstanceSegmentation/cooking/coffee"

    demo = load_model(
        config=f"{model_dir}/stage1/mask_rcnn_R_50_FPN_1x_demo.yaml",
        conf_thr=0.2,
    )

    # Step map
    metadata = demo.metadata.as_dict()

    if not os.path.exists("temp"):
        os.mkdir("temp")

    ###############
    # Detect
    ###############
    for split in splits:
        print(f"{split}: {len(training_split[split])} videos")

        # Raw detector output
        preds, using_contact = run_obj_detector(
            demo,
            stage,
            training_split[split],
            add_hl_hands=False,
        )
        print(f"Using contact: {using_contact}")

        fn = f"temp/{experiment_name}_{split}_preds.pickle"
        print("temp file: ", fn)
        with open(fn, "wb") as fh:
            #preds = pickle.load(fh)
            pickle.dump(preds, fh)

        dset = preds_to_kwcoco(
            metadata,
            preds,
            "",
            save_fn=f"{experiment_name}_{stage}_{split}.mscoco.json",
        )

if __name__ == "__main__":
    main()
