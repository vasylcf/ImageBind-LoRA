# Sheng Wang at Feb 22 2023
# Original version of LoRA-ViT: https://github.com/JamesQFreeman/LoRA-ViT/blob/main/lora.py
# Modified by Fares Abawi (@fabawi).


import math
from typing import Optional, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor
from torch.nn.parameter import Parameter

from transformer import SimpleTransformer


def apply_lora_modality_trunks(modality_trunks: Dict[SimpleTransformer], rank: int,
                               lora_layer_idxs: Optional[List[int]] = None, modality_names: List[str] = None):
    if modality_names is None:
        modality_names = list(modality_trunks.keys())
    return {modality_name: LoRA_SimpleTransformer(modality_trunk, rank, lora_layer_idxs) for
            modality_name, modality_trunk in modality_trunks.items() if modality_name in modality_names}


class _LoRALayer(nn.Module):
    def __init__(self, w: nn.Module, w_a: nn.Module, w_b: nn.Module):
        super().__init__()
        self.w = w
        self.w_a = w_a
        self.w_b = w_b

    def forward(self, x):
        x = self.w(x) + self.w_b(self.w_a(x))
        return x


class LoRA_SimpleTransformer(nn.Module):
    """Applies low-rank adaptation to simple transformer with pytorch multihead attention.

    Args:
        transformer_model: a vision transformer model, see base_vit.py
        rank: rank of LoRA
        lora_layer_idxs: which layer we apply LoRA.

    Examples::
        >>> model = SimpleTransformer()
        >>> lora_model = LoRA_SimpleTransformer(model, rank=4)
        >>> preds = lora_model(img)
        >>> print(preds.shape)
        torch.Size([1, 1000])
    """

    def __init__(self, transformer_model: SimpleTransformer, rank: int, lora_layer_idxs: Optional[List[int]] = None):
        super(LoRA_SimpleTransformer, self).__init__()

        assert rank > 0
        base_dim = transformer_model.transformer.blocks[0].attn.in_proj_bias.size()[0]
        dim = base_dim
        if lora_layer_idxs is not None:
            self.lora_layer_idxs = lora_layer_idxs
        else:
            self.lora_layer_idxs = list(range(len(transformer_model.blocks)))
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []

        # lets freeze first
        for param in transformer_model.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        for t_layer_idx, blk in enumerate(transformer_model.transformer.blocks):
            # If we only want few lora layer instead of all
            if t_layer_idx not in self.lora_layer_idxs:
                continue
            w_q_linear = blk.attn.q_proj_weight
            w_v_linear = blk.attn.v_proj_weight
            w_a_linear_q = nn.Linear(dim, rank, bias=False)
            w_b_linear_q = nn.Linear(rank, dim, bias=False)
            w_a_linear_v = nn.Linear(dim, rank, bias=False)
            w_b_linear_v = nn.Linear(rank, dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.proj_q = _LoRALayer(w_q_linear, w_a_linear_q, w_b_linear_q)
            blk.attn.proj_v = _LoRALayer(w_v_linear, w_a_linear_v, w_b_linear_v)
        if self.training:
            self.reset_parameters()
        self.lora_model = transformer_model

    def save_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensors if you do not have one installed yet.
        """

        assert filename.endswith(".safetensors")

        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}

        merged_dict = {**a_tensors, **b_tensors}
        save_file(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensors if you do not have one installed yet.
        """

        assert filename.endswith(".safetensors")

        with safe_open(filename, framework="pt") as f:
            for i, w_A_linear in enumerate(self.w_As):
                saved_key = f"w_a_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_A_linear.weight = Parameter(saved_tensor)

            for i, w_B_linear in enumerate(self.w_Bs):
                saved_key = f"w_b_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_B_linear.weight = Parameter(saved_tensor)

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.lora_model(x)

