#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
import argparse
import sys
from PIL import Image
from collections import OrderedDict

import detectron2.utils.comm as comm
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetMapper, build_detection_train_loader, build_detection_test_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)

from detectron2.modeling import GeneralizedRCNNWithTTA
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 2, 3, 4, 6, 7, 8, 9'
os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2, 3'
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        return CityscapesSemSegEvaluator(dataset_name)
    elif evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    elif evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, output_dir=output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        logger.setLevel(logging.DEBUG)

        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

class TrainerAug(Trainer):
    @classmethod
    def build_train_loader(cls, cfg):
        augs = [
            # TODO: Adjust zed camera images

            # Orientation
            #T.RandomRotation(angle=[-30, 30], sample_style="range", expand=True) # neck/head angle
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False), # body position relative to dummy
            T.RandomCrop(crop_type="relative_range", crop_size=(0.85, 0.85)), # camera distance to dummy 
            
            # Environment
            T.RandomBrightness(intensity_min=0.5, intensity_max=1.5),
            T.RandomContrast(intensity_min=0.75, intensity_max=1.25),
            T.RandomSaturation(intensity_min=0.5, intensity_max=1.25),
            #T.ColorTransform(),
        
            #T.Resize(shape=(428, 760), interp=Image.BILINEAR) # bring everything back to the right size
            T.Resize(shape=(720, 1280), interp=Image.BILINEAR) # bring everything back to the right size
        ]
        using_contact = True if cfg.MODEL.ROI_HEADS.NAME == "StandardROIHeads_PLUS_CONTACT" else False
        print(f"Using contact: {using_contact}")
        
        mapper = DatasetMapper(cfg, is_train=True, augmentations=augs, using_contact=using_contact)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        augs = [
            T.Resize(shape=(720, 1280), interp=Image.BILINEAR) # bring everything back to the right size
            #T.Resize(shape=(428, 760), interp=Image.BILINEAR) # bring everything back to the right size
        ]
        using_contact = True if cfg.MODEL.ROI_HEADS.NAME == "StandardROIHeads_PLUS_CONTACT" else False
        print(f"Using contact: {using_contact}")
        
        mapper = DatasetMapper(cfg, is_train=False, augmentations=augs, using_contact=using_contact)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # modify the iteration time by epoch
    cfg.defrost()
    cfg.SOLVER.MAX_ITER = int(cfg.SOLVER.EPOCH * cfg.SOLVER.TOTAL_IMAGE_NUMBER / cfg.SOLVER.IMS_PER_BATCH)
    start_decay_step = int(3 / 4 * cfg.SOLVER.MAX_ITER)
    stop_decay_step = int(8 / 9 * cfg.SOLVER.MAX_ITER)
    cfg.SOLVER.STEPS = (start_decay_step, stop_decay_step)

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)



    if args.eval_only:
        model = TrainerAug.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = TrainerAug.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(TrainerAug.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = TrainerAug(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()

def user_parser(epilog=None):
    """
    Create a parser with some common arguments used by detectron2 users.

    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="../configs/MC50-InstanceSegmentation/mask_rcnn_R_101_FPN_1x_fine-tuning.yaml", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory. "
        "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
        default=True
    )
    parser.add_argument("--eval-only", default=False, action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2**15 + 2**14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2**14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    args = user_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
