import logging
import torch
import data

from models import imagebind_model
from models.imagebind_model import ModalityType
from models import lora as LoRA

logging.basicConfig(level=logging.INFO)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
lora = True

text_list=["Dog",
           "Car",
           "Bird",
           "monster_toy",
           "bear_plushie",
           "poop_emoji"]
image_paths=[".assets/dog_image.jpg",
             ".assets/car_image.jpg",
             ".assets/bird_image.jpg",
             ".assets/monster_toy.jpg",
             ".assets/bear_plushie.jpg",
             ".assets/poop_emoji.jpg"]
audio_paths=[".assets/dog_audio.wav",
             ".assets/car_audio.wav",
             ".assets/bird_audio.wav"]

# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)
if lora:
    model.modality_trunks.update(
        LoRA.apply_lora_modality_trunks(model.modality_trunks, rank=4,
                                        layer_idxs={ModalityType.TEXT: [0, 1, 2, 3, 4, 5, 6, 7, 8],
                                                    ModalityType.VISION: [0, 1, 2, 3, 4, 5, 6, 7, 8]},
                                        modality_names=[ModalityType.TEXT, ModalityType.VISION]))

    # Load LoRA params if found
    LoRA.load_lora_modality_trunks(model.modality_trunks, checkpoint_dir="./.checkpoints/lora", postfix="_last")


model.eval()
model.to(device)

# Load data
inputs = {
    ModalityType.TEXT: data.load_and_transform_text(text_list, device),
    ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device, to_tensor=True),
    ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, device),
}

with torch.no_grad():
    embeddings = model(inputs)

print(
    "Vision x Text: ",
    torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T, dim=-1),
)
print(
    "Audio x Text: ",
    torch.softmax(embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.TEXT].T, dim=-1),
)
print(
    "Vision x Audio: ",
    torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.AUDIO].T, dim=-1),
)

