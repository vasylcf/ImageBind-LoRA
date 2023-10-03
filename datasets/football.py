import torch
import pandas as pd

from torchvision import transforms
from torchvision.transforms._transforms_video import NormalizeVideo

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
    clip_duration=2,
    clips_per_video=4, # 5
    sample_rate=16000,
):

    if video_paths is None:
        return None

    video_outputs = []
    video_transform = transforms.Compose(
        [
            pv_transforms.ShortSideScale(224),
            NormalizeVideo(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    clip_sampler = ConstantClipsPerVideoSampler(
        clip_duration=clip_duration, clips_per_video=clips_per_video
    )
    frame_sampler = pv_transforms.UniformTemporalSubsample(num_samples=clip_duration)

    for video_path in video_paths:
        video = EncodedVideo.from_path(
            video_path,
            # decoder="decord",
            decode_audio=False,
            # **{"sample_rate": sample_rate},
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
        all_video = data.SpatialCrop(224, num_crops=1)(all_video) #num_crops=3

        all_video = torch.stack(all_video, dim=0)
        video_outputs.append(all_video)
    video_tensor = torch.vstack(video_outputs).to(device)
    return video_tensor


def load_imu(imu_path, device):
    event = pd.read_csv(imu_path)
    imu_seq = event[['vx','vy','v','ax','ay','a']].values
    imu_seq = imu_seq[:2000].T.astype('float32')
    imu_tensor = torch.from_numpy(imu_seq)
    return imu_tensor.to(device)


class FootballDataset(Dataset):
    def __init__(self, root_dir: str, transform: Optional[Callable] = None,
                 split: str = 'train', train_size: float = 0.8, random_seed: int = 42, device: str = 'cpu'):
        self.root_dir = root_dir
        self.transform = transform
        self.device = device

        self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        print(f'classes: {len(self.classes)}')
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.paths = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls, 'clips')
            for filename in os.listdir(cls_dir):
                if filename.endswith('.mp4'):
                    self.paths.append((os.path.join(cls_dir, filename), 
                                       os.path.join(cls_dir.replace('clips', 'telemetry'), filename.replace('mp4', 'csv'))))
        print(f'Clips in dataset: {len(self.paths)}')
        # Split dataset
        train_paths, test_paths = train_test_split(self.paths, train_size=train_size, random_state=random_seed)

        if split == 'train':
            self.paths = train_paths
        elif split == 'test':
            self.paths = test_paths
        else:
            raise ValueError(f"Invalid split argument. Expected 'train' or 'test', got {split}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        video_path, imu_path = self.paths[index]
        video = load_and_transform_video_data([video_path], self.device)

        # if self.transform is not None:
        #     image = images[0]
        #     images = self.transform(image)

        imu = load_imu(imu_path, self.device)

        return video, ModalityType.VISION, imu, ModalityType.IMU

