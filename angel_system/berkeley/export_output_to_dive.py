from typing import Sequence
from typing import Tuple
from rich.progress import track

import numpy as np
import numpy.typing as npt

####new packages that need to import
from demo.model import *
from detectron2.data.detection_utils import read_image



def predict(
    frames: Sequence[npt.NDArray[np.uint8]],
) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
    """
    Predict activity classification for the sequence of temporally successive
    image frames.

    One activity classification is expected to cover the whole input window.
    The window frame may be allowed to be as small as one frame, or as large as
    the model and system resources can handle.

    It is currently expected that successive calls to this function will be
    called will be with a consistently sized `frames` instance.
    E.g. `frames` would be consistently the same length from call to call
    during the same runtime.

    Output confidences are expected to fall within the [0, 1] inclusive range.

    Use case example (with arbitrary input and return values):
        >>> frame_window_32: npt.NDArray = np.random.randint(0, 256, (32, 720, 1280, 3),
        ...                                                  dtype=np.uint8)
        >>> conf, labels = predict(frame_window_32)
        >>> print(conf)
        [ 0.1  0.1  0.2  0.6 ]
        >>> print(labels)
        [ "background", "measure-water", "measure-beans", "pour-water" ]

    NOTES FOR BERKELEY:
    * Additional input parameters and output values up for discussion.
    * We have a historical expectation that the label vector output is the same
      length and order from call to call. This is, yes, redundant information
      that could be extracted to a separate function that could be called just
      once.

    :param frames: Some sequence of RGB image frames in [H x W x C] shape for
        which and activity classification is desired.

    :return: Two vectors consisting of the predicted activity class labels and
        class confidences for the input window of image frames.
    """
    preds, visualized_outputs = inference(frames)
    return preds, visualized_outputs


gt_to_dive = False
if gt_to_dive:
    # Load coco file
    import kwcoco 
    coco_file = '/home/chris/Documents/video_grouped/2022-11-1/MC_13/ann_contact.json'
    output_file = 'output.csv'
    coco = kwcoco.CocoDataset(coco_file)

    # Fix the filenames
    keys = coco.imgs.keys()
    for k in keys:
        im = coco.imgs[k]
        fn_temp = im['file_name'].split('/')[-1].split('_')
        fn_temp[0] = fn_temp[0][-5:]
        im['file_name'] = '_'.join(fn_temp)

    # text = coco.dumps()
    # with open(output_file, 'w') as file:
    #     file.write(text)
    # for test:

    # - 1: Detection or Track Unique ID
    # - 2: Video or Image String Identifier
    # - 3: Unique Frame Integer Identifier
    # - 4: TL-x (top left of the image is the origin: 0,0)
    # - 5: TL-y
    # - 6: BR-x
    # - 7: BR-y
    # - 8: Auxiliary Confidence (how likely is this actually an object)
    # - 9: Target Length
    text = ''
    id = 1
    keys = coco.anns.keys()
    for k in keys:
        ann = coco.anns[k]
        text += str(id) + ','
        text += coco.imgs[ann['image_id']]['file_name'] + ','
        text += '0,'
        dets = np.array(ann['bbox'],dtype='float64')
        dets[2:] = dets[:2] -  dets[2:]
        diff = dets[:2] - dets[2:]
        dets[:2] += diff
        dets[2:] += diff
        text +=  str(dets[2]) + ','
        text +=  str(dets[3]) + ','
        text +=  str(dets[0]) + ','
        text +=  str(dets[1]) + ','
        text += '0.5,'
        text += '0,'
        text += coco.cats[ann['category_id']]['name']  + ','
        text += '0.5,'
        text += '\n'
        id += 1

        if ann['obj-obj_contact_state'] == 1:
            text += str(id) + ','
            text += coco.imgs[ann['image_id']]['file_name'] + ','
            text += '0,'
            text +=  str(dets[2]) + ','
            text +=  str(dets[3]) + ','
            text +=  str(dets[0]) + ','
            text +=  str(dets[1]) + ','
            text += '0.75,'
            text += '0,'
            text += 'obj-obj contact'  + ','
            text += '0.75,'
            text += '\n'
            id += 1



        if ann['obj-hand_contact_state'] == 1:
            text += str(id) + ','
            text += coco.imgs[ann['image_id']]['file_name'] + ','
            text += '0,'
            text +=  str(dets[2]) + ','
            text +=  str(dets[3]) + ','
            text +=  str(dets[0]) + ','
            text +=  str(dets[1]) + ','
            text += '1.0,'
            text += '0,'
            text += 'obj-hand contact'  + ','
            text += '1.0,'
            text += '\n'
            id += 1

    with open(output_file, 'w') as file:
        file.write(text)





def Re_order(image_list, image_number):
    img_id_list = []
    for img in image_list:
        img_id = int(img.split('/')[-1].split('_')[1])
        img_id_list.append(img_id)
    img_id_arr = np.array(img_id_list)
    s = np.argsort(img_id_arr)
    new_list = []
    for i in range(image_number):
        idx = s[i]
        new_list.append(image_list[idx])
    return new_list

#  Pick which video to run
data_root = '/run/user/692589645/gvfs/smb-share:server=ptg-nas-01,share=data_working/Cooking/Coffee/coffee_recordings'
video_name = 'all_activities_24'
path_root = f'{data_root}/{video_name}/{video_name}/_extracted/images'
image_output_dir = f'{data_root}/{video_name}/{video_name}/preds/'
batch_size = 500
# Glob up images
img_list = os.listdir(path_root)
image_list = []
for img in img_list:
    if 'png' in img:
        image_list.append(os.path.join(path_root, img))
# re-oder the input
input_list = Re_order(image_list, len(image_list))


import multiprocessing as mp

mp.set_start_method("spawn", force=True)
args = get_parser().parse_args()
setup_logger(name="fvcore")
logger = setup_logger()
logger.info("Arguments: " + str(args))

cfg = setup_cfg(args)


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

demo = VisualizationDemo_add_smoothing(cfg, number_frames=len(input_list), last_time=2, fps = 30)
idx = 0
preds = {}
visualized_outputs = []

for img_path in track(input_list, total=len(input_list), show_speed=True):
    idx = idx + 1
    img = read_image(img_path, format="RGB")

    frame = img[...,[2, 1, 0]]

    predictions, visualized_output = demo.run_on_image_smoothing_v2(frame, current_idx=idx)

    out_fn = image_output_dir + img_path.split('/')[-1]
    visualized_output.save(out_fn)
    # visualized_outputs.append(visualized_output)

    if decode_prediction(predictions) != None:
        preds[idx] = decode_prediction(predictions)

