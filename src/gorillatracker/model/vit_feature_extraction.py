import copy
from typing import Any, Callable

import numpy as np
import timm
import torch
import torch.nn as nn
from torchvision.transforms import v2 as transforms_v2
from torchvision.transforms import transforms

from gorillatracker.transform_utils import PlanckianJitter
from gorillatracker.model.base_module import BaseModule

# TODO(rob2u): test
# TODO(rob2u): improve embedding_layer
# TODO(rob2u): improve f2a and aam
# TODO(rob2u): try using positional embeddings
# TODO(rob2u): try combining different layers of the model
class ViT_FeatureExtraction(BaseModule):
    def __init__(  # type: ignore
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        
        self.model = timm.create_model("vit_large_patch14_dinov2.lvd142m", pretrained=not self.from_scratch, img_size=224)    
        self.feature_dim = 1024
        
        self.f2a = FeatureToAttributeProjector(self.feature_dim, self.feature_dim, self.feature_dim) # NOTE(rob2u): Project CLS token to synthetic image attribute
        self.aam = AttributeAttentionModule(self.feature_dim, self.feature_dim) # NOTE(rob2u): extract features from patches using synthetic image attribute
        self.embedding_layer = torch.nn.Linear(self.feature_dim * 2, self.embedding_size) # NOTE(rob2u): reduce concatenated feature dimension to embedding size
        
        model_cpy = copy.deepcopy(self.model)
        model_cpy.head = torch.nn.Identity()
        self.set_losses(model=model_cpy, **kwargs)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        vit_features = self.model.forward_features(x)
        cls_feature = vit_features[:, 0]
        patch_features = vit_features[:, 1:]
        
        synthetic_image_attribute = self.f2a(cls_feature)
        aam_feature, patch_attention_weight = self.aam(patch_features, synthetic_image_attribute)
        
        features = torch.cat([cls_feature, aam_feature], dim=1)
        output = self.embedding_layer(features)
        return output

    def get_grad_cam_layer(self) -> torch.nn.Module:
        # see https://github.com/jacobgil/pytorch-grad-cam/blob/master/tutorials/vision_transformers.md#how-does-it-work-with-vision-transformers
        return self.model.blocks[-1].norm1

    def get_grad_cam_reshape_transform(self) -> Any:
        # see https://github.com/jacobgil/pytorch-grad-cam/blob/master/tutorials/vision_transformers.md#how-does-it-work-with-vision-transformers
        def reshape_transform(tensor: torch.Tensor, height: int = 14, width: int = 14) -> torch.Tensor:
            result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

            result = result.transpose(2, 3).transpose(1, 2)
            return result

        return reshape_transform

    @classmethod
    def get_training_transforms(cls) -> Callable[[torch.Tensor], torch.Tensor]:
        return transforms.Compose(
            [
                transforms_v2.ToPILImage(),
                PlanckianJitter(),
                transforms_v2.ToTensor(),
                # transforms_v2.RandomPhotometricDistort(p=0.5),
                transforms_v2.RandomHorizontalFlip(p=0.5),
                transforms_v2.RandomErasing(p=0.5, value=0, scale=(0.02, 0.13)),
                transforms_v2.RandomRotation(60, fill=0),
                transforms_v2.RandomResizedCrop(224, scale=(0.75, 1.0)),
                
            ]
        )

    
class FeatureToAttributeProjector(nn.Module): 
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.leaky_relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, cls_feature):
        x = self.fc1(cls_feature)
        x = self.leaky_relu(x)
        x = self.fc2(x)
        return x


class AttributeAttentionModule(nn.Module):
    def __init__(self, input_dim, proj_dim):
        super().__init__()
        self.query_proj = nn.Linear(input_dim, proj_dim)
        self.key_proj = nn.Linear(input_dim, proj_dim)
        self.value_proj = nn.Linear(input_dim, proj_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, patch_features, synthetic_image_attribute):
        Q = self.query_proj(synthetic_image_attribute)  # Query
        K = self.key_proj(patch_features)               # Key
        V = self.value_proj(patch_features)             # Value

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (K.shape[-1] ** 0.5)
        attention_weights = self.softmax(attention_scores)
        weighted_sum = torch.matmul(attention_weights, V)

        return weighted_sum, attention_weights
    

# TODO(rob2u): test
class LearnedPositionalEmbeddings(nn.Module):
    def __init__(self, num_patches, dim):
        super().__init__()
        self.positional_embeddings = nn.Parameter(get_sinusoid_encoding(num_patches, dim))

    def forward(self, x):
        return x + self.positional_embeddings
    

# FROM: https://towardsdatascience.com/position-embeddings-for-vision-transformers-explained-a6f9add341d5
def get_sinusoid_encoding(num_tokens, token_len):
    """ Make Sinusoid Encoding Table

        Args:
            num_tokens (int): number of tokens
            token_len (int): length of a token
            
        Returns:
            (torch.FloatTensor) sinusoidal position encoding table
    """

    def get_position_angle_vec(i):
        return [i / np.power(10000, 2 * (j // 2) / token_len) for j in range(token_len)]

    sinusoid_table = np.array([get_position_angle_vec(i) for i in range(num_tokens)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) 

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

# TODO(rob2u): test
class SinusoidalPositionalEmbeddings(nn.Module):
    def __init__(self, num_patches, dim):
        super().__init__()
        self.positional_embeddings = torch.Tensor(get_sinusoid_encoding(num_patches, dim))
        

    def forward(self, x):
        return x + self.positional_embeddings
