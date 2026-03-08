"""
Multi-layer token feature extractors for CLIP, DINO and MAE.
"""
from typing import Dict, Iterable, List, Optional

import torch

from .clip_model import CLIPModel
from .dino_model import DINOModel
from .mae_model import MAEModel


def _normalize_layer_indices(layer_indices: Optional[Iterable[int]], num_layers: int) -> List[int]:
    if layer_indices is None:
        return list(range(num_layers))
    normalized: List[int] = []
    for idx in layer_indices:
        i = int(idx)
        if i < 0:
            i = num_layers + i
        if i < 0 or i >= num_layers:
            raise ValueError(f"Layer index {idx} out of range [0, {num_layers - 1}]")
        normalized.append(i)
    return sorted(set(normalized))


class CLIPMultiLayerModel(CLIPModel):
    """Extract CLIP token features from arbitrary transformer layers."""

    def get_num_layers(self) -> int:
        return len(self.model.visual.transformer.resblocks)

    def extract_batch_multilayer_features(
        self,
        images: torch.Tensor,
        layer_indices: Optional[Iterable[int]] = None,
    ) -> Dict[int, torch.Tensor]:
        if images.dim() == 3:
            images = images.unsqueeze(0)
        images = images.to(self.device)

        layer_ids = _normalize_layer_indices(layer_indices, self.get_num_layers())
        layer_set = set(layer_ids)
        outputs: Dict[int, torch.Tensor] = {}

        with torch.no_grad():
            visual = self.model.visual
            images = images.to(visual.conv1.weight.dtype)
            x = visual.conv1(images)  # [B, C, H, W]
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # [B, N, C]

            cls = visual.class_embedding.to(x.dtype)
            cls = cls + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
            x = torch.cat([cls, x], dim=1)
            x = x + visual.positional_embedding.to(x.dtype)
            x = visual.ln_pre(x)

            x = x.permute(1, 0, 2)  # [T, B, C]
            for idx, block in enumerate(visual.transformer.resblocks):
                x = block(x)
                if idx in layer_set:
                    outputs[idx] = x.permute(1, 0, 2).contiguous()  # [B, T, C]

        return outputs


class DINOMultiLayerModel(DINOModel):
    """Extract DINO token features from arbitrary transformer layers."""

    def get_num_layers(self) -> int:
        return len(self.model.blocks)

    def extract_batch_multilayer_features(
        self,
        images: torch.Tensor,
        layer_indices: Optional[Iterable[int]] = None,
    ) -> Dict[int, torch.Tensor]:
        if images.dim() == 3:
            images = images.unsqueeze(0)
        images = images.to(self.device)

        layer_ids = _normalize_layer_indices(layer_indices, self.get_num_layers())
        layer_set = set(layer_ids)
        outputs: Dict[int, torch.Tensor] = {}

        with torch.no_grad():
            x = self.model.prepare_tokens(images)  # [B, T, C]
            for idx, block in enumerate(self.model.blocks):
                x = block(x)
                if idx in layer_set:
                    outputs[idx] = self.model.norm(x).contiguous()

        return outputs


class MAEMultiLayerModel(MAEModel):
    """Extract MAE token features from arbitrary transformer layers."""

    def get_num_layers(self) -> int:
        return self.model.config.num_hidden_layers

    def extract_batch_multilayer_features(
        self,
        images: torch.Tensor,
        layer_indices: Optional[Iterable[int]] = None,
    ) -> Dict[int, torch.Tensor]:
        if images.dim() == 3:
            images = images.unsqueeze(0)
        images = images.to(self.device)

        layer_ids = _normalize_layer_indices(layer_indices, self.get_num_layers())
        outputs: Dict[int, torch.Tensor] = {}

        with torch.no_grad():
            model_out = self.model(
                pixel_values=images,
                output_hidden_states=True,
                return_dict=True,
            )
            # hidden_states[0] is patch embedding output; transformer layers start from index 1
            hidden_states = model_out.hidden_states[1:]
            for idx in layer_ids:
                outputs[idx] = hidden_states[idx].contiguous()

        return outputs
