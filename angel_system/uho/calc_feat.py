import tqdm
import sys
import os
from torch import nn
import torch
import pdb

from angel_system.uho.src.datamodules.ros_datamodule import ROSFrameDataset
from angel_system.uho.src.models.components.fcn import UnifiedFCNModule

from torchvision.models import resnext50_32x4d
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms


if __name__ == "__main__":
    subset = "all_activities_"  # "brian_coffee_" # "all_activities_"
    fcn = UnifiedFCNModule("resnext").to("cuda")
    fcn.eval()

    trans = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    root_path = "/data/dawei.du/datasets/ROS/Data"
    # for key in ['all_activities_pose_train','all_activities_pose_val','brian_coffee_pose_test']:
    # for key in ['all_activities_pose_test1','all_activities_pose_train1']:
    for key in ["pose_train"]:
        frame_path = os.path.join(root_path, "label_split", key + ".txt")
        dataset = ROSFrameDataset(root_path, frame_path, trans)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            shuffle=False,
        )
        cnt = 0
        for data in tqdm.tqdm(dataloader):
            cnt += 1
            fsplit = data["fname"][0].strip().split("/")
            if not os.path.isdir("/".join(fsplit[:-2] + ["feat"])):
                os.mkdir("/".join(fsplit[:-2] + ["feat"]))
            fpath = "/".join(fsplit[:-2] + ["feat", f'{fsplit[-1].split(".")[0]}.pk'])
            # if not os.path.exists(fpath):
            data["frm"] = data["frm"].to("cuda")
            feats = fcn(data)
            labels = {k: data[k] for k in data if k != "frm"}
            sample_info = dict(feats=feats, labels=labels)
            torch.save(sample_info, fpath)

    print("All image features computed and saved.")
