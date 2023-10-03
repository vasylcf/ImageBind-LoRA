import torch
import pandas as pd
import numpy as np

from torchvision import transforms
from torchvision.transforms._transforms_video import NormalizeVideo, CenterCropVideo

from pytorchvideo import transforms as pv_transforms
from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler
from pytorchvideo.data.encoded_video import EncodedVideo


import os
from typing import Optional, Callable

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from models.imagebind_model import ModalityType
import data


def get_clip_timepoints(clip_sampler, duration):
    # Read out all clips in this video
    all_clips_timepoints = []
    is_last_clip = False
    end = 0.0
    while not is_last_clip:
        start, end, _, _, is_last_clip = clip_sampler(end, duration, annotation=None)
        all_clips_timepoints.append((start, end))
    return all_clips_timepoints


def load_and_transform_video_data(
    video_paths,
    device,
    clip_duration=0.5,
    clips_per_video=7,
    frames_per_clip=1,
    crop_size=224,
    center_crop=True,
    spatial_crop=False
):

    if video_paths is None:
        return None

    video_outputs = []
    if center_crop:
        video_transform = transforms.Compose(
                [
                    pv_transforms.ShortSideScale(crop_size),
                    NormalizeVideo(
                        mean=(0.48145466, 0.4578275, 0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711),
                    ),
                    CenterCropVideo(crop_size)
                ]
            )
    else:
        video_transform = transforms.Compose(
                [
                    pv_transforms.ShortSideScale(crop_size),
                    NormalizeVideo(
                        mean=(0.48145466, 0.4578275, 0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711),
                    ),
                ]
            )

    clip_sampler = ConstantClipsPerVideoSampler(
        clip_duration=clip_duration, clips_per_video=clips_per_video
    )
    frame_sampler = pv_transforms.UniformTemporalSubsample(num_samples=frames_per_clip)

    for video_path in video_paths:
        video = EncodedVideo.from_path(
            video_path,
            decode_audio=False,
        )

        all_clips_timepoints = get_clip_timepoints(clip_sampler, video.duration)

        all_video = []
        for clip_timepoints in all_clips_timepoints:
            # Read the clip, get frames
            clip = video.get_clip(clip_timepoints[0], clip_timepoints[1])
            if clip is None:
                raise ValueError("No clip found")
            video_clip = frame_sampler(clip["video"])
            video_clip = video_clip / 255.0  # since this is float, need 0-1

            all_video.append(video_clip)

        all_video = [video_transform(clip) for clip in all_video]

        if spatial_crop:
            all_video = data.SpatialCrop(crop_size, num_crops=3)(all_video)

        all_video = torch.stack(all_video, dim=0)
        video_outputs.append(all_video)
    video_tensor = torch.vstack(video_outputs).to(device)
    return video_tensor


def load_imu(imu_path, device):
    imu_tensor = torch.load(imu_path)
    if isinstance(imu_tensor, np.ndarray):
        imu_tensor = torch.from_numpy(imu_tensor)
    return imu_tensor.to(device)


class AvatarDataset(Dataset):
    def __init__(self, root_dir: str, csv_path: str = None,
                 transform: Optional[Callable] = None,
                 split: str = 'train', train_size: float = 0.95, random_seed: int = 42, device: str = 'cpu'):
        self.root_dir = root_dir
        # self.ds = pd.read_csv(csv_path)
        # print(f'avatar csv shape: {self.ds.shape}')
        self.transform = transform
        self.device = device

        # self.classes = self.ds.sentence.unique()
        self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        print(f'classes: {len(self.classes)}')
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.paths = []
        # for cls, cls_df in self.ds.groupby('sentence'):
        #     for _i, r in cls_df.iterrows():
        #         self.paths.append((os.path.join(root_dir, r['clip']),
        #                            os.path.join(root_dir, r['telemetry'])))

        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls, 'clips')
            for filename in os.listdir(cls_dir):
                if filename.endswith('.mp4'):
                    self.paths.append((os.path.join(cls_dir, filename),
                                       os.path.join(cls_dir.replace('clips', 'telemetry'), filename.replace('mp4', 'pt'))))

        # Split dataset
        train_paths, test_paths = train_test_split(self.paths, train_size=train_size, random_state=random_seed)

        if split == 'train':
            self.paths = train_paths
        elif split == 'test':
            self.paths = test_paths
        else:
            print(f'Without train/test splitting')

        print(f'{split}: Clips in dataset {len(self.paths)}')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        video_path, imu_path = self.paths[index]
        video = load_and_transform_video_data([video_path], self.device)
        imu = load_imu(imu_path, self.device)

        return video, ModalityType.VISION, imu, ModalityType.IMU
