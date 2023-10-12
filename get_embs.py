import torch
from datasets.avatar import AvatarDataset
from lightning.pytorch import Trainer, seed_everything

from torch.utils.data import DataLoader

from models import imagebind_model
from models.imagebind_model import ModalityType, load_module
from models import lora as LoRA

import os
import numpy 
import pickle

from tqdm import tqdm


if __name__ == '__main__':

    seed = 43
    seed_everything(seed, workers=True)
    torch.backends.cudnn.determinstic = True
    
    batch_size = 1
    device_name =  "cuda:0"
    device = torch.device(device_name)

    datasets_dir = '.datasets'

    train_dataset = AvatarDataset(
            root_dir=os.path.join(datasets_dir, "avatar"), 
            split=None, # a whole ds withpout train/test split
            transform=None)
    
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=False,
        num_workers=12,
    )

    # Instantiate model
    model = imagebind_model.imagebind_huge(pretrained=True)

    model.modality_trunks.update(
        LoRA.apply_lora_modality_trunks(model.modality_trunks, rank=4,
                                        modality_names=[ModalityType.VISION, ModalityType.IMU]))

    # Load LoRA params if found
    LoRA.load_lora_modality_trunks(model.modality_trunks,
                                checkpoint_dir=".checkpoints/lora/10_ted", postfix="_last")

    # Load postprocessors & heads
    load_module(model.modality_postprocessors, module_name="postprocessors",
                checkpoint_dir=".checkpoints/lora/10_ted", postfix="_last")
    load_module(model.modality_heads, module_name="heads",
                checkpoint_dir=".checkpoints/lora/10_ted", postfix="_last")

    model.eval()
    model.to(device)

    dls = {'train': train_loader}
    dss = {'train': train_dataset}

    for phase in dls:
        video_embs, imu_embs = [], []
        for batch in tqdm(dls[phase]):
            data_a, class_a, data_b, class_b = batch
            
            inputs = {
                ModalityType.VISION: data_a.to(device),
                ModalityType.IMU: data_b.to(device),
            }
            
            with torch.no_grad():
                embeddings = model(inputs)

            video_embs.append(embeddings[ModalityType.VISION].cpu().numpy())
            imu_embs.append(embeddings[ModalityType.IMU].cpu().numpy())
        
        with open(f'ted_{phase}_embs.pickle', 'wb') as f:
            pickle.dump({'video_embs': video_embs, 'imu_embs': imu_embs, 'files': dss[phase].paths}, f)
        
        





