"""
NOTE: THIS FILE WAS COPIED FROM THE LEARN repository
      video_classification_2021 branch
"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Learning rate policy."""

from . import transform as transform
import torch
import random


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
