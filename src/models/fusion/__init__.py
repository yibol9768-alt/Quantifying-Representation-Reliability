"""Fusion module - all multi-model fusion methods."""

from typing import Dict, Optional, Sequence

from .simple import (
    MultiModelConcatExtractor,
    MultiModelProjectedConcatExtractor,
    MultiModelWeightedSumExtractor,
    MultiModelGatedFusionExtractor,
)
from .interaction import (
    MultiModelDifferenceAwareExtractor,
    MultiModelHadamardExtractor,
    MultiModelBilinearExtractor,
)
from .attention import (
    MultiModelFiLMExtractor,
    MultiModelContextGatingExtractor,
    MultiModelLMFExtractor,
    MultiModelSEFusionExtractor,
    MultiModelLateFusionExtractor,
)
from .token import COMMStrictFusionExtractor, MMViTStrictFusionExtractor
from .routing import (
    MultiModelTopKRouterExtractor,
    MultiModelMoERouterExtractor,
    MultiModelAttentionRouterExtractor,
)
from ..extractor import FeatureExtractor


def get_extractor(
    model_type: str,
    fusion_method: str = "concat",
    fusion_models: Optional[Sequence[str]] = None,
    fusion_kwargs: Optional[Dict] = None,
    model_dir: str = "./models",
):
    """Factory function to create extractors."""
    if model_type != "fusion":
        return FeatureExtractor(model_type, normalize_input=False, model_dir=model_dir)

    model_types = list(fusion_models) if fusion_models is not None else ["mae", "clip", "dino"]
    fusion_method = fusion_method.lower()
    fusion_kwargs = {} if fusion_kwargs is None else dict(fusion_kwargs)

    _REGISTRY = {
        "concat": lambda: MultiModelConcatExtractor(
            model_types, output_dim=fusion_kwargs.get("fusion_output_dim"), model_dir=model_dir,
        ),
        "proj_concat": lambda: MultiModelProjectedConcatExtractor(
            model_types, proj_dim=fusion_kwargs.get("proj_dim", 256), model_dir=model_dir,
        ),
        "weighted_sum": lambda: MultiModelWeightedSumExtractor(
            model_types, proj_dim=fusion_kwargs.get("proj_dim", 512), model_dir=model_dir,
        ),
        "gated": lambda: MultiModelGatedFusionExtractor(
            model_types, proj_dim=fusion_kwargs.get("proj_dim", 512),
            hidden_dim=fusion_kwargs.get("hidden_dim", 128), model_dir=model_dir,
        ),
        "difference_concat": lambda: MultiModelDifferenceAwareExtractor(
            model_types, proj_dim=fusion_kwargs.get("proj_dim", 256), model_dir=model_dir,
        ),
        "hadamard_concat": lambda: MultiModelHadamardExtractor(
            model_types, proj_dim=fusion_kwargs.get("proj_dim", 256), model_dir=model_dir,
        ),
        "bilinear_concat": lambda: MultiModelBilinearExtractor(
            model_types, proj_dim=fusion_kwargs.get("proj_dim", 64), model_dir=model_dir,
        ),
        "film": lambda: MultiModelFiLMExtractor(
            model_types, proj_dim=fusion_kwargs.get("proj_dim", 512), model_dir=model_dir,
        ),
        "context_gating": lambda: MultiModelContextGatingExtractor(
            model_types, proj_dim=fusion_kwargs.get("proj_dim", 256), model_dir=model_dir,
        ),
        "lmf": lambda: MultiModelLMFExtractor(
            model_types, proj_dim=fusion_kwargs.get("proj_dim", 512),
            rank=fusion_kwargs.get("lmf_rank", 16),
            output_dim=fusion_kwargs.get("lmf_output_dim", 512), model_dir=model_dir,
        ),
        "se_fusion": lambda: MultiModelSEFusionExtractor(
            model_types, proj_dim=fusion_kwargs.get("proj_dim", 512),
            reduction=fusion_kwargs.get("se_reduction", 4), model_dir=model_dir,
        ),
        "late_fusion": lambda: MultiModelLateFusionExtractor(
            model_types, num_classes=fusion_kwargs.get("num_classes", 100), model_dir=model_dir,
        ),
        "comm": lambda: COMMStrictFusionExtractor(
            model_types=model_types,
            dino_mlp_blocks=fusion_kwargs.get("comm_dino_mlp_blocks", 2),
            dino_mlp_ratio=fusion_kwargs.get("comm_dino_mlp_ratio", 8.0),
            output_dim=fusion_kwargs.get("fusion_output_dim"), model_dir=model_dir,
        ),
        "mmvit": lambda: MMViTStrictFusionExtractor(
            model_types=model_types,
            base_dim=fusion_kwargs.get("mmvit_base_dim", 96),
            mlp_ratio=fusion_kwargs.get("mmvit_mlp_ratio", 4.0),
            num_heads=fusion_kwargs.get("mmvit_num_heads", 8),
            max_position_tokens=fusion_kwargs.get("mmvit_max_position_tokens", 256),
            output_dim=fusion_kwargs.get("fusion_output_dim"), model_dir=model_dir,
        ),
        "topk_router": lambda: MultiModelTopKRouterExtractor(
            model_types, proj_dim=fusion_kwargs.get("proj_dim", 512),
            hidden_dim=fusion_kwargs.get("hidden_dim", 128),
            router_k=fusion_kwargs.get("router_k", 2), model_dir=model_dir,
        ),
        "moe_router": lambda: MultiModelMoERouterExtractor(
            model_types, proj_dim=fusion_kwargs.get("proj_dim", 512),
            hidden_dim=fusion_kwargs.get("hidden_dim", 128), model_dir=model_dir,
        ),
        "attention_router": lambda: MultiModelAttentionRouterExtractor(
            model_types, proj_dim=fusion_kwargs.get("proj_dim", 512),
            num_heads=fusion_kwargs.get("attention_router_heads", 4), model_dir=model_dir,
        ),
    }

    if fusion_method not in _REGISTRY:
        raise ValueError(
            f"Unsupported fusion method: {fusion_method}. "
            f"Choose from {sorted(_REGISTRY.keys())}."
        )

    return _REGISTRY[fusion_method]()
