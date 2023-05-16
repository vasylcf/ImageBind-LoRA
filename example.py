import logging
import torch
import data

from models import imagebind_model
from models.imagebind_model import ModalityType
from models import lora as LoRA

logging.basicConfig(level=logging.INFO)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
# TODO (fabawi): This should be adjusted for during training or on saving the weights.
#  For now, we just set it to the max batch size we used during training / temperature.
lora_factor = 12 / 0.07

lora = True

text_list=["bird",
           "car",
           "dog3",
           "dog5",
           "dog8",
           "grey_sloth_plushie"]
image_paths=[".assets/bird_image.jpg",
             ".assets/car_image.jpg",
             ".assets/dog3.jpg",
             ".assets/dog5.jpg",
             ".assets/dog8.jpg",
             ".assets/grey_sloth_plushie.jpg",]
audio_paths=[".assets/bird_audio.wav",
             ".assets/car_audio.wav",
             ".assets/dog_audio.wav"]

# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)
if lora:
    model.modality_trunks.update(
        LoRA.apply_lora_modality_trunks(model.modality_trunks, rank=4,
                                        layer_idxs={ModalityType.TEXT: [0, 1, 2, 3, 4, 5, 6, 7, 8],
                                                    ModalityType.VISION: [0, 1, 2, 3, 4, 5, 6, 7, 8]},
                                        modality_names=[ModalityType.TEXT, ModalityType.VISION]))

    # Load LoRA params if found
    LoRA.load_lora_modality_trunks(model.modality_trunks, checkpoint_dir="./.checkpoints/lora", postfix="-dreambooth_last")


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
    torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T * (lora_factor if lora else 1), dim=-1),
)
print(
    "Audio x Text: ",
    torch.softmax(embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.TEXT].T * (lora_factor if lora else 1), dim=-1),
)
print(
    "Vision x Audio: ",
    torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.AUDIO].T, dim=-1),
)

