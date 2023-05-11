# Based on PyTorch Lightning Tutorial 13 -
# SSL : https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/13-contrastive-learning.html
# Modified by Fares Abawi (@fabawi).
import logging

import lightning as L
from lightning.pytorch import Trainer, seed_everything
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models import imagebind_model
from models import lora

# make training deterministic for reproducibility. REMOVE when testing on different runs
seed_everything(42, workers=True)


# TODO (fabawi): WIP
class LoRATrain(L.LightningModule):
    def __init__(self, lora_rank=4, lora_checkpoint_dir="./.checkpoints/lora"):
        super().__init__()
        self.save_hyperparameters()

        # image bind model (load pretrained model)
        self.model = imagebind_model.imagebind_huge(pretrained=True)
        self.model.modality_trunks = lora.apply_lora_modality_trunks(self.model.modality_trunks, rank=lora_rank)
        try:
            # now load LoRA params if found
            lora.load_lora_modality_trunks(self.model.modality_trunks)
        except FileNotFoundError:
            logging.warning("No LoRA checkpoint found. Training LoRA layers from scratch!")

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50
        )
        return [optimizer], [lr_scheduler]

    def info_nce_loss(self, batch, mode="train"):
        imgs, _ = batch
        imgs = torch.cat(imgs, dim=0)

        # TODO (fabawi): change the feats format, and make it generalizable to all supported i/o formats
        feats = self.model(imgs)
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.hparams.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

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
    model = LoRATrain()
    trainer = Trainer(accelerator="cpu")
    trainer.fit(model)

