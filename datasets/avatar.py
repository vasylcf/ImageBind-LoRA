import torch
import numpy as np
import pickle
from datetime import timedelta
import pandas as pd

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


def normalize_kpts(points, shelter_ids=[5, 6]):
    origins = points[:, shelter_ids, :].mean(axis=1, keepdims=True)
    norm_m = points - origins
    d = np.linalg.norm(norm_m[:, 5, :], axis=1, keepdims=True)
    norm_m = norm_m / np.expand_dims(d, axis=-1)
    norm_m = np.dot(norm_m, np.array([[1, 0], [0, -1]])) 
    return norm_m


def interpolate_kpts(m):
    new_shape  = (m.shape[0]*2, m.shape[1], m.shape[2])
    df = pd.DataFrame(m.reshape(m.shape[0], -1))
    df['dt'] = df.index
    df['sec'] = pd.to_datetime(df['dt'], unit='s')

    inerp_df = df.groupby(pd.Grouper(key='sec', freq='500ms')).max()
    empty_row = {k: np.nan for k in df.columns}
    _ind = df.iloc[-1,-1] + timedelta(milliseconds=500)
    inerp_df.loc[_ind] = empty_row
    inerp_df = inerp_df.reset_index(drop=True)
    inerp_df = inerp_df.drop(columns='dt')
    inerp_df = inerp_df.interpolate()

    return inerp_df.values.reshape(new_shape)


def load_imu(imu_path, device):
    kpt_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    brows_ids = [40, 42, 44, 45, 47, 49]
    chin_ids = [23, 25, 27, 29, 31, 33, 35, 37, 39]
    mouth_ids = [71, 74, 77, 80]
    vis_points = kpt_ids + brows_ids + chin_ids + mouth_ids

    with open(imu_path, 'rb') as f:
        kpts_data = pickle.load(f)

    chunk_points = [i['kpts'] for i in kpts_data]
    m = np.array([i[vis_points]for i in chunk_points])
    m = interpolate_kpts(m)
    norm_m = normalize_kpts(m, shelter_ids=[5, 6])
    imu_tensor = torch.from_numpy(norm_m)
    imu_tensor = imu_tensor.permute(2, 1, 0)
    imu_tensor = imu_tensor.reshape(6, 2000)
    imu_tensor = imu_tensor.to(dtype=torch.float32)
    return imu_tensor.to(device)


def load_kpts(imu_path, interpolate=True):
    kpt_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    brows_ids = [40, 42, 44, 45, 47, 49]
    chin_ids = [23, 25, 27, 29, 31, 33, 35, 37, 39]
    mouth_ids = [71, 74, 77, 80]
    vis_points = kpt_ids + brows_ids + chin_ids + mouth_ids

    with open(imu_path, 'rb') as f:
        kpts_data = pickle.load(f)

    chunk_points = [i['kpts'] for i in kpts_data]
    m = np.array([i[vis_points]for i in chunk_points])
    if interpolate:
        m = interpolate_kpts(m)
    norm_m = normalize_kpts(m, shelter_ids=[5, 6])
    return norm_m


class AvatarDataset(Dataset):
    def __init__(self, root_dir: str,
                 transform: Optional[Callable] = None,
                 split: str = 'train', train_size: float = 0.9, random_seed: int = 42, device: str = 'cpu'):
        self.root_dir = root_dir
        self.transform = transform
        self.device = device

        self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        print(f'classes: {len(self.classes)}')
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.paths = []

        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls, 'video')
            for filename in os.listdir(cls_dir):
                if filename.endswith('.mp4'):
                    self.paths.append((os.path.join(cls_dir, filename),
                                       os.path.join(cls_dir.replace('video', 'kpts'), filename.replace('mp4', 'pickle'))))

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
