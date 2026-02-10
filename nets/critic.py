import copy
import torch


class Critic(torch.nn.Module):
    
    def __init__(self, model, hidden_dim: int = 128, *args, **kwargs) -> None:
        super().__init__()
        self.encoder = copy.deepcopy(model.encoder)
        for parameter in self.encoder.parameters():
            parameter.requires_grad = True
        self.value = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, data):
        
        # Get embeddings
        embedding = self.encoder(data).mean(dim=1)
        
        # Predict critic value
        return self.value(embedding).squeeze(dim=1)
