import torch
    

class GTN(torch.nn.Module):
    
    def __init__(
        self,
        node_dim1=3,
        node_dim2=None,
        num_heads=8,
        num_blocks=2,
        hidden_dim=128,
        hidden_dim_ff=512,
        dropout=0.1,
        *args, **kwargs):
        super(GTN, self).__init__()
        
        # To map input to embedding space
        self.init_embed1 = torch.nn.Linear(node_dim1, hidden_dim)
        self.init_embed2 = torch.nn.Linear(node_dim2, hidden_dim) if (node_dim2 is not None) else None
        self.init_embed = torch.nn.Linear(hidden_dim, hidden_dim)

        # Transformer Encoder
        encoder_layer = torch.nn.TransformerEncoderLayer(
            hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim_ff, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)

    def forward(self, data):
        nodes, obstacles = data['nodes'], data.get('obstacles', None)
        
        # Initial embedding for nodes: (batch_size, num_nodes, node_dim1) to (batch_size, num_nodes, hidden_dim)
        node_embedding = self.init_embed1(nodes)
        
        # Initial embedding for obstacles: (batch_size, num_nodes, node_dim2) to (batch_size, num_obstacles, hidden_dim)
        obstacle_embedding = None if obstacles is None else self.init_embed2(obstacles)
        
        # Create graph from nodes and obstacles (if any): (batch_size, num_nodes + num_obstacles, hidden_dim)
        graph = node_embedding if obstacles is None else torch.cat((node_embedding, obstacle_embedding), dim=1)
        
        # Initial graph embedding: batch_size, num_nodes + num_obstacles, hidden_dim
        initial_embedding = self.init_embed(graph)

        # Transformer Encoder: (batch_size, num_nodes + num_obstacles, hidden_dim)
        graph_embedding = self.transformer_encoder(initial_embedding)
        return graph_embedding
