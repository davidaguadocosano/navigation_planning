import torch


class VIT(torch.nn.Module):
    
    def __init__(
        self,
        image_size=64,
        patch_size=8,
        hidden_dim=128,
        num_heads=8,
        num_blocks=2,
        num_channels=3,
        hidden_dim_ff=512,
        dropout=0.1,
        *args, **kwargs):
        super(VIT, self).__init__()
        
        # Dimensions
        self.patch_size = patch_size
        self.patch_dim = num_channels * patch_size ** 2
        self.num_patches = (image_size // patch_size) * (image_size // patch_size)

        # Patch Embedding
        self.patch_embedding = torch.nn.Linear(self.patch_dim, hidden_dim)

        # Positional Embedding
        self.positional_embedding = torch.nn.Parameter(torch.randn(1, self.num_patches, hidden_dim))

        # Transformer Encoder
        encoder_layer = torch.nn.TransformerEncoderLayer(
            hidden_dim, num_heads, dim_feedforward=hidden_dim_ff, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)

    def forward(self, data):
        image = data['binary_map']
        
        # Input shape: (batch_size, num_channels, image_size, image_size) to (batch_size, image_size, image_size, num_channels)
        image = image.permute(0, 2, 3, 1)
        batch_size, _, _, channels = image.shape

        # Divide image into patches: (batch_size, num_patches, patch_size, patch_size, num_channels)
        patches = image.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(batch_size, -1, self.patch_size, self.patch_size, channels)

        # Flatten patches: (batch_size, num_patches, patch_dim), where patch_dim = num_channels * patch_size^2
        patches = patches.view(batch_size, -1, self.patch_dim)

        # Patch embedding projection: (batch_size, num_patches, hidden_dim)
        patches = self.patch_embedding(patches)

        # Add positional embeddings: (batch_size, num_patches, hidden_dim)
        positional_embeddings = self.positional_embedding.repeat(batch_size, 1, 1)
        patches += positional_embeddings

        # Transformer Encoder: (batch_size, num_patches, hidden_dim)
        image_embedding = self.transformer_encoder(patches)
        return image_embedding
