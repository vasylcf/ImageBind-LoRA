from datasets.avatar import AvatarDataset
from datasets.football import FootballDataset
import os
from torch.utils.data import DataLoader


datasets_dir = '/home/vash/apps/avatar/ImageBindLora_tuning/ImageBind-LoRA/.datasets'
train_dataset = AvatarDataset(
    root_dir=os.path.join(datasets_dir, "avatar"), split="train", transform=None)

# datasets_dir = '/home/vash/apps/telemetry_similarity/ds'
# train_dataset = FootballDataset(
#     root_dir=os.path.join(datasets_dir, "RBLeipzig-WerderBremen_14052023_1T"), split="train", transform=None)

train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=False,
        drop_last=True,
        pin_memory=False,
        num_workers=1,
    )

for i in train_loader:
    print(i)
    break