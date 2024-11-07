import torch
import torch.nn as nn
import models_vit

from timm.models.layers import trunc_normal_
from util.pos_embed import interpolate_pos_embed
from pathlib import Path
from typing import Literal


class ViTClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
        hidden_features: int = 512,
        dropout: float = 0.2,
        activation: Literal[nn.GELU, nn.ReLU] = nn.GELU,
        vit_arch: str = "vit_large_patch16",
        vit_kwargs: dict = {
            "img_size": 224,
            "num_classes": 1000,
            "drop_path_rate": 0.1,
            "global_pool": True
        },
        vit_weights: Path | str | None = '/autofs/space/crater_001/datasets/private/mee_parkinsons/models/RETFound_cfp_weights.pth'
    ):
        super().__init__()

        self.model = models_vit.__dict__[vit_arch](**vit_kwargs)

        if vit_weights:
            checkpoint = torch.load(vit_weights, map_location="cpu")

            print(f"Loading ViT weights from {vit_weights}")
            checkpoint_model = checkpoint["model"]
            state_dict = self.model.state_dict()
            for k in ["head.weight", "head.bias"]:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

            # interpolate position embedding
            interpolate_pos_embed(self.model, checkpoint_model)

            # load pre-trained model
            msg = self.model.load_state_dict(checkpoint_model, strict=False)
            # print(msg)

            if vit_kwargs.get("global_pool", False):
                assert set(msg.missing_keys) == {"head.weight", "head.bias", "fc_norm.weight", "fc_norm.bias"}
            else:
                assert set(msg.missing_keys) == {"head.weight", "head.bias"}

            # manually initialize fc layer
            trunc_normal_(self.model.head.weight, std=2e-5)

        classifier_head = models_vit.Mlp(
            in_features=self.model.head.in_features,
            hidden_features=hidden_features,
            out_features=num_classes,
            act_layer=activation,
            drop=dropout
        )

        self.model.head = classifier_head



    def forward(self, x: torch.Tensor):
        return self.model(x)
