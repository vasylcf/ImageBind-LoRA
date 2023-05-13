# Based on PyTorch Lightning Tutorial 13 -
# SSL : https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/13-contrastive-learning.html
# Modified by Fares Abawi (@fabawi).
import logging
import os

from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
import lightning as L
from lightning.pytorch import Trainer, seed_everything
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

from models import imagebind_model
from models import lora
from models.imagebind_model import ModalityType
import data


class ContrastiveTransformations:
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for _ in range(self.n_views)]


class DreamBoothDataset(Dataset):
    def __init__(self, root_dir, transform=None, split='train', train_size=0.8, random_seed=42):
        self.root_dir = root_dir
        self.transform = transform

        self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]  # list of classes        # list of classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}  # map class name to index

        self.paths = []  # list of (path, class) tuples
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for filename in os.listdir(cls_dir):
                if filename.endswith('.jpg'):
                    self.paths.append((os.path.join(cls_dir, filename), cls))

        # split dataset
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
        img_path, class_text = self.paths[index]
        images = data.load_and_transform_vision_data([img_path], device, to_tensor=False)

        if self.transform is not None:
            image = images[0]
            images = self.transform(image)

        texts = data.load_and_transform_text([class_text], device)

        return images, ModalityType.VISION, texts, ModalityType.TEXT


# TODO (fabawi): WIP
class LoRATrain(L.LightningModule):
    def __init__(self, lr=5e-4, weight_decay=1e-4, max_epochs=500,  # TODO (fabawi): dummy values. Change soon!
                 temperature=0.07, lora_rank=4, lora_checkpoint_dir="./.checkpoints/lora",
                 lora_layer_idxs=None, lora_modality_names=None
                 ):
        super().__init__()
        self.save_hyperparameters()

        # image bind model (load pretrained model)
        self.model = imagebind_model.imagebind_huge(pretrained=True)
        self.model.modality_trunks = lora.apply_lora_modality_trunks(self.model.modality_trunks, rank=lora_rank,
                                                                     layer_idxs=self.hparams.lora_layer_idxs,
                                                                     modality_names=self.hparams.lora_modality_names)
        try:
            # now load LoRA params if found
            lora.load_lora_modality_trunks(self.model.modality_trunks, checkpoint_dir=lora_checkpoint_dir)
        except FileNotFoundError:
            logging.warning("No LoRA checkpoint found. Training LoRA layers from scratch!")

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50
        )
        return [optimizer], [lr_scheduler]

    def info_nce_loss(self, batch, mode="train"):
        data_a, class_a, data_b, class_b = batch

        # class_a is always 'vision' according to ImageBind
        feats_a = [self.model({class_a[0]: data_a_i}) for data_a_i in data_a]
        feats_a_tensor = torch.cat([list(dict_.values())[0] for dict_ in feats_a], dim=0)
        # class_b could be any modality
        feats_b = [self.model({class_b[idx]: data_b_i}) for idx, data_b_i in enumerate(data_b)]
        feats_b_tensor = torch.cat([list(dict_.values())[0] for dict_ in feats_b], dim=0)

        feats_a_b_tensor = torch.cat([feats_a_tensor.chunk(2)[0], feats_b_tensor], dim=0)
        # Accumulate self-contrastive loss for image and its augmentation, and modailty with image
        dual_nll = False
        for feats_tensor in [feats_a_tensor, feats_a_b_tensor]:
            # Calculate cosine similarity
            cos_sim = F.cosine_similarity(feats_tensor[:, None, :], feats_tensor[None, :, :], dim=-1)
            # Mask out cosine similarity to itself
            self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
            cos_sim.masked_fill_(self_mask, -9e15)
            # Find positive example -> batch_size//2 away from the original example
            pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
            # InfoNCE loss
            cos_sim = cos_sim / self.hparams.temperature
            nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
            nll = nll.mean()
            if not dual_nll:
                dual_nll = nll
            else:
                dual_nll += nll
                dual_nll /= 2
            # Logging loss
            self.log(mode + "_loss", nll)
            # Get ranking position of positive example
            comb_sim = torch.cat(
                [cos_sim[pos_mask][:, None], cos_sim.masked_fill(pos_mask, -9e15)],  # First position positive example
                dim=-1,
            )
            sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
            # Logging ranking metrics
            self.log(mode + "_acc_top1", (sim_argsort == 0).float().mean())
            self.log(mode + "_acc_top5", (sim_argsort < 5).float().mean())
            self.log(mode + "_acc_mean_pos", 1 + sim_argsort.float().mean())

        return nll

    def training_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self.info_nce_loss(batch, mode="val")


if __name__ == "__main__":
    # constants
    batch_size = 12
    seed_everything(43, workers=True)
    device_name = "cuda:0" if torch.cuda.is_available() else "cpu"

    torch.backends.cudnn.determinstic = True
    device = torch.device(device_name)

    contrast_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size=224),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)],
                                   p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=9),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    train_dataset = DreamBoothDataset(root_dir='./.datasets/dreambooth/dataset', split='train', transform=ContrastiveTransformations(contrast_transforms, n_views=2))
    test_dataset = DreamBoothDataset(root_dir='./.datasets/dreambooth/dataset', split='test', transform=ContrastiveTransformations(contrast_transforms, n_views=2))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
        num_workers=0,
    )
    val_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=False,
        num_workers=0,
    )

    # DEBUG
    # Visualize some examples
    NUM_IMAGES = batch_size
    imgs = [torch.stack(train_dataset[idx][0], dim=0) for idx in range(NUM_IMAGES)]
    imgs = torch.stack(imgs, dim=0)
    img_grid = torchvision.utils.make_grid(imgs.reshape(-1, *imgs.shape[2:]), nrow=6, normalize=True, pad_value=0.9)
    img_grid = img_grid.permute(1, 2, 0)

    plt.figure(figsize=(10, 5))
    plt.title("Augmented image examples of the Dreambooth dataset")
    plt.imshow(img_grid.cpu())
    plt.axis("off")
    plt.show()
    plt.close()
    ## END DEBUG

    # train dataset
    model = LoRATrain(lora_layer_idxs={ModalityType.VISION: [0, 1, 2, 3, 4, 5, 6, 7, 8],
                                  ModalityType.TEXT: [0, 1, 2, 3, 4, 5, 6, 7, 8]},
                      lora_modality_names=[ModalityType.VISION, ModalityType.TEXT])
    trainer = Trainer(accelerator=device_name, devices=1, max_epochs=500)
    trainer.fit(model, train_loader, val_loader)

