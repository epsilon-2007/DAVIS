import json
import torch
import requests
from torch import nn
from PIL import Image
from torchvision import transforms

from .route import RouteDICE


class PatchEmbedding(nn.Module):
    """
    Splits an image into patches and embeds them.

    Args:
        img_size (int): Size of the input image (e.g., 224 for ImageNet).
        patch_size (int): Size of each patch (e.g., 16).
        in_channels (int): Number of input channels (e.g., 3 for RGB).
        embed_dim (int): The embedding dimension for each patch.
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # A convolution layer to convert the image into patches and embed them
        self.projection = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # Input x: (batch_size, in_channels, height, width)
        x = self.projection(x)  # (batch_size, embed_dim, num_patches_h, num_patches_w)
        x = x.flatten(2)       # (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (batch_size, num_patches, embed_dim)
        return x

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism.

    Args:
        embed_dim (int): The embedding dimension.
        num_heads (int): The number of attention heads.
        dropout (float): Dropout probability.
    """
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        # Input x: (batch_size, num_patches, embed_dim)
        B, N, C = x.shape
        
        # (B, N, C) -> (B, N, 3 * C) -> (B, N, 3, num_heads, head_dim) -> (3, B, num_heads, N, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled Dot-Product Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale # (B, num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # (B, num_heads, N, N) @ (B, num_heads, N, head_dim) -> (B, num_heads, N, head_dim)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    """
    Multi-Layer Perceptron block.
    
    Args:
        in_features (int): Number of input features.
        hidden_features (int): Number of hidden features.
        out_features (int): Number of output features.
        dropout (float): Dropout probability.
    """
    def __init__(self, in_features, hidden_features, out_features, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the Transformer Encoder.

    Args:
        embed_dim (int): The embedding dimension.
        num_heads (int): The number of attention heads.
        mlp_ratio (float): Determines the hidden dimension of the MLP.
        dropout (float): Dropout probability.
    """
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(
            in_features=embed_dim, 
            hidden_features=mlp_hidden_dim, 
            out_features=embed_dim,
            dropout=dropout
        )

    def forward(self, x):
        # Attention block with residual connection
        x = x + self.attn(self.norm1(x))
        # MLP block with residual connection
        x = x + self.mlp(self.norm2(x))
        return x

# ==============================================================================
# 2. The Main Vision Transformer (ViT) Model with Global Average Pooling
# ==============================================================================

class VisionTransformer(nn.Module):
    """
    Vision Transformer with Global Average Pooling.

    This implementation omits the [CLS] token and uses global average
    pooling over the patch embeddings for classification, as requested.
    """
    def __init__(
        self, 
        img_size=224, 
        patch_size=16, 
        in_channels=3, 
        num_classes=1000, 
        embed_dim=768, 
        depth=12, 
        num_heads=12, 
        mlp_ratio=4.0, 
        dropout=0.0
    ):
        super().__init__()
        self.dim_in = embed_dim
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # 1. Patch Embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # 2. Positional Embedding (learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # 3. Transformer Encoder
        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # 4a. Final normalization
        self.norm = nn.LayerNorm(embed_dim)

        # # Initialize weights
        # self._init_weights()

    def _init_weights(self):
        # Initialize positional embedding with a truncated normal distribution
        torch.nn.init.trunc_normal_(self.pos_embed, std=.02)
        # Initialize all linear layers
        self.apply(self._init_linear_weights)

    def _init_linear_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def my_encoder(self, x):
        x = self.patch_embed(x) # [bt, num_patches, embed_dim)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        # reshaping as [bt, channels, h, w]
        x = x.transpose(1, 2)
        _, _, N = x.shape
        H = W = int(N ** 0.5) # assuming a square number of patches (e.g., 196 -> 14x14)
        assert H*W == N
        x = x.reshape(-1, self.embed_dim, H, W)
        return x # returns [bt, channels, h, w]


class ViT(VisionTransformer):
    def __init__(self, args, **vit_base_config):
        super(ViT, self).__init__(**vit_base_config)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 4b. Classification Head
        if args.p is None:
            self.head = nn.Linear(self.dim_in, args.num_classes)
        else:
            self.head = RouteDICE(self.dim_in, args.num_classes, device=args.device, p=args.p, info=args.info)

        # initialize weights
        self._init_weights()

    def my_features(self, x):
        x = self.my_encoder(x)
        # x = self.norm(x)
        x = self.avgpool(x)
        x = x.view(-1, self.dim_in)
        return x

    def forward(self, x):
        x = self.my_features(x)
        x = self.head(x)
        return x


    
def load_pretrained_weights(model, model_name="vit_base_patch16_224"):
    """
    Loads pre-trained weights from a timm model into our custom ViT.
    This function carefully handles the absence of a [CLS] token in our model.
    """
    print(f"Loading weights for {model_name}...")
    # Load a pre-trained model from torch.hub (timm library)
    timm_model = torch.hub.load('facebookresearch/deit:main', model_name, pretrained=True)
    timm_state_dict = timm_model.state_dict()
    
    # Create a new state dict for our custom model
    custom_state_dict = model.state_dict()

    for name, param in timm_state_dict.items():
        # Map timm's weight names to our model's names
        # **FIXED LOGIC**: Handle specific cases for 'pos_embed' and 'patch_embed' first.
        if 'pos_embed' in name:
            # IMPORTANT: The timm model has a positional embedding for the [CLS] token
            # at index 0. Our model does not. We must skip it.
            # timm pos_embed shape: (1, 1 + num_patches, embed_dim)
            # our pos_embed shape:  (1, num_patches, embed_dim)
            custom_state_dict['pos_embed'] = param[:, 1:, :]
            print(f"  - Adjusted 'pos_embed' shape from {param.shape} to {custom_state_dict['pos_embed'].shape}")
        elif 'patch_embed.proj' in name:
            # Map patch embedding weights
            new_name = name.replace('patch_embed.proj', 'patch_embed.projection')
            if new_name in custom_state_dict:
                custom_state_dict[new_name] = param
        elif name in custom_state_dict:
            # Direct mapping if names match
            custom_state_dict[name] = param

    # Load the mapped state dictionary
    model.load_state_dict(custom_state_dict, strict=True)
    print("Pre-trained weights loaded successfully.")
    return model


def get_vit(args, pretrained=False):
    vit_base_config = {
    "img_size": 224,
    "patch_size": 16,
    "embed_dim": 768,
    "depth": 12,
    "num_heads": 12,
    "mlp_ratio": 4.0,
    "num_classes": args.num_classes
    }
    model = ViT(args, **vit_base_config)
    if pretrained == True:
        model = load_pretrained_weights(model)
    return model