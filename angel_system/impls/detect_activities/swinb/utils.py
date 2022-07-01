"""
NOTE: THIS FILE WAS COPIED FROM THE LEARN repository
      video_classification_2021 branch
"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Learning rate policy."""

import math
from . import transform as transform
import torch
import random


def get_lr_at_epoch(cfg, cur_epoch):
    """
    Retrieve the learning rate of the current epoch with the option to perform
    warm up in the beginning of the training stage.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        cur_epoch (float): the number of epoch of the current training stage.
    """
    lr = get_lr_func(cfg['lr_policy'])(cfg, cur_epoch)
    # Perform warm up.
    if cur_epoch < cfg['warmup_epoch']:
        lr_start = cfg['warmup_start_lr']
        lr_end = get_lr_func(cfg['lr_policy'])(cfg, cfg['warmup_epoch'])
        alpha = (lr_end - lr_start) / cfg['warmup_epoch']
        lr = cur_epoch * alpha + lr_start
    return lr


def lr_func_cosine(cfg, cur_epoch):
    """
    Retrieve the learning rate to specified values at specified epoch with the
    cosine learning rate schedule. Details can be found in:
    Ilya Loshchilov, and  Frank Hutter
    SGDR: Stochastic Gradient Descent With Warm Restarts.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        cur_epoch (float): the number of epoch of the current training stage.
    """
    assert cfg['cosine_end_lr'] < cfg['lr']
    return (
        cfg['cosine_end_lr']
        + (cfg['lr'] - cfg['cosine_end_lr'])
        * (math.cos(math.pi * cur_epoch / cfg['max_epoch']) + 1.0)
        * 0.5
    )


def lr_func_steps_with_relative_lrs(cfg, cur_epoch):
    """
    Retrieve the learning rate to specified values at specified epoch with the
    steps with relative learning rate schedule.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        cur_epoch (float): the number of epoch of the current training stage.
    """
    ind = get_step_index(cfg, cur_epoch)
    return cfg['lrs'][ind] * cfg['lr']


def get_step_index(cfg, cur_epoch):
    """
    Retrieves the lr step index for the given epoch.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        cur_epoch (float): the number of epoch of the current training stage.
    """
    steps = cfg['steps'] + [cfg['max_epoch']]
    for ind, step in enumerate(steps):  # NoQA
        if cur_epoch < step:
            break
    return ind - 1


def get_lr_func(lr_policy):
    """
    Given the configs, retrieve the specified lr policy function.
    Args:
        lr_policy (string): the learning rate policy to use for the job.
    """
    policy = "lr_func_" + lr_policy
    if policy not in globals():
        raise NotImplementedError("Unknown LR policy: {}".format(lr_policy))
    else:
        return globals()[policy]


def topks_correct(preds, labels, ks):
    """
    Given the predictions, labels, and a list of top-k values, compute the
    number of correct predictions for each top-k value.

    Args:
        preds (array): array of predictions. Dimension is batchsize
            N x ClassNum.
        labels (array): array of labels. Dimension is batchsize N.
        ks (list): list of top-k values. For example, ks = [1, 5] correspods
            to top-1 and top-5.

    Returns:
        topks_correct (list): list of numbers, where the `i`-th entry
            corresponds to the number of top-`ks[i]` correct predictions.
    """
    assert preds.size(0) == labels.size(
        0
    ), "Batch dim of predictions and labels must match"
    # Find the top max_k predictions for each sample
    _top_max_k_vals, top_max_k_inds = torch.topk(
        preds, max(ks), dim=1, largest=True, sorted=True
    )
    # (batch_size, max_k) -> (max_k, batch_size).
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size).
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct.
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # Compute the number of topk correct predictions for each k.
    topks_correct = [top_max_k_correct[:k, :].float().sum() for k in ks]
    return topks_correct


def spatial_sampling(
    frames,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    random_horizontal_flip=True,
    inverse_uniform_sampling=False,
    ssl_aug=False,
):
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        frames, _ = transform.random_short_side_scale_jitter(
            images=frames,
            min_size=min_scale,
            max_size=max_scale,
            inverse_uniform_sampling=inverse_uniform_sampling,
        )
        frames, _ = transform.random_crop(frames, crop_size)
        if random_horizontal_flip:
            frames, _ = transform.horizontal_flip(0.5, frames)
        if ssl_aug:
            frames = transform.color_jitter(frames, 0.8, 0.8, 0.8)
            frames = transform.random_gray_scale(frames, p=0.2)
            frames = transform.random_gaussian_blur(frames, p=0.5)
    else:
        # The testing is deterministic and no jitter should be performed.
        # min_scale, max_scale, and crop_size are expect to be the same.
        # assert len({min_scale, max_scale, crop_size}) == 1
        frames, _ = transform.random_short_side_scale_jitter(
            frames, min_scale, max_scale
        )
        frames, _ = transform.uniform_crop(frames, crop_size, spatial_idx)
    return frames


def temporal_sampling(frames, start_idx, end_idx, num_samples):
    index = torch.linspace(start_idx, end_idx, num_samples)
    index = torch.clamp(index, 0, len(frames) - 1).long()
    if isinstance(frames, list):
        frames = [frames[i] for i in index]
        frames = torch.stack(frames)
    else:
        frames = torch.index_select(frames, 0, index)
    return frames


def get_start_end_idx(video_size, clip_size, clip_idx, num_clips):
    delta = max(video_size - clip_size, 0)
    if clip_idx == -1:
        # Random temporal sampling.
        start_idx = random.uniform(0, delta)
    else:
        # Uniformly sample the clip with the given index.
        start_idx = delta * clip_idx / num_clips
    end_idx = start_idx + clip_size - 1
    return start_idx, end_idx


class CollateFn(object):
    def __init__(
        self, sampling_rate=32, clip_idx=-1, num_clips=1, num_frames=8, spatial_idx=-1
    ) -> None:
        self.sampling_rate = sampling_rate
        self.clip_idx = clip_idx
        self.num_clips = num_clips
        self.num_frames = num_frames
        self.spatial_idx = spatial_idx

    def __call__(self, batch):
        frames, labels, vid_ids = list(zip(*batch))
        # frames = [torch.stack(x) for x in frames]
        fps, target_fps = 30, 30
        clip_size = self.sampling_rate * self.num_frames / target_fps * fps
        start_end_idx = [
            get_start_end_idx(len(x), clip_size, self.clip_idx, self.num_clips)
            for x in frames
        ]
        # assert False, start_end_idx
        frames = [
            temporal_sampling(x, s, e, self.num_frames)
            for x, (s, e) in zip(frames, start_end_idx)
        ]
        frames = [x.permute(1, 0, 2, 3) for x in frames]
        frames = [spatial_sampling(x, spatial_idx=self.spatial_idx) for x in frames]
        frames = torch.stack(frames)

        labels = torch.tensor(labels).long()
        vid_ids = torch.tensor(vid_ids).long()
        return frames, labels, vid_ids
