from typing import Optional

import numpy as np
from ultralytics import YOLO as YOLOv8


def predict_hands(
    hand_model: YOLOv8,
    img0: np.array,
    device: str,
    imgsz: Optional[int] = None
) -> tuple:
    """Predict hands using a YOLOv8 hand model and update the labels to be
    hand(left) and hand(right)

    Boxes returned are in xyxy format (upper-left, lower-right).
    """
    width, height = img0.shape[:2]
    pred_kwargs = {}
    if imgsz:
        pred_kwargs["imgsz"] = imgsz
    hands_preds = hand_model.predict(
        source=img0, conf=0.1, device=device, verbose=False, **pred_kwargs
    )[
        0
    ]  # list of length=num images

    hand_centers = [center.xywh.tolist()[0][0] for center in hands_preds.boxes][:2]
    hands_label = []

    # Update the hand label to left and right specific labels
    if len(hand_centers) == 2:
        if hand_centers[0] > hand_centers[1]:
            hands_label.append("hand (right)")
            hands_label.append("hand (left)")
        elif hand_centers[0] <= hand_centers[1]:
            hands_label.append("hand (left)")
            hands_label.append("hand (right)")
    elif len(hand_centers) == 1:
        if hand_centers[0] > width // 2:
            hands_label.append("hand (right)")
        elif hand_centers[0] <= width // 2:
            hands_label.append("hand (left)")

    boxes, labels, confs = [], [], []

    for bbox, hand_cid in zip(hands_preds.boxes, hands_label):
        xyxy_hand = bbox.xyxy.tolist()[0]

        conf = bbox.conf.item()

        boxes.append(xyxy_hand)
        labels.append(hand_cid)
        confs.append(conf)

    return boxes, labels, confs
