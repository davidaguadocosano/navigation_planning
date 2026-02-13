import os
import torch

from utils import get_module, load_checkpoint, set_dataparallel, set_rng_state, set_train
from envs import special_args, get_state
from .encoders import *
from .decoders import *


GRAPH_ENCODERS = encoders()['graph_encoders']
IMAGE_ENCODERS = encoders()['image_encoders']
DECODERS = decoders()


def models():
    return {
        'encoders': encoders(),
        'decoders': decoders(),
    }
    

def load_model(opts, train=False, dataparallel=True):
    
    # Load checkpoint
    checkpoint, args, first_epoch = load_checkpoint(load_path=opts.load_path, opts=opts)
    
    # Create model
    model = get_model(
        # model=opts.model,
        # env=opts.env,
        # scenario=opts.scenario,
        # device=opts.device,
        checkpoint=checkpoint,
        **args
    )
    
    # Set rng state
    if train:
        set_rng_state(
            checkpoint=checkpoint,
            use_cuda=opts.use_cuda
        )
        
    # Train / eval mode
    model = set_train(
        model=model,
        train=train,
        return_actions=not train
    )
    
    # Set DataParallel
    if dataparallel:
        model = set_dataparallel(
            model=model,
            device=opts.device,
            use_cuda=opts.use_cuda,
            use_distributed=opts.use_distributed
        )
    
    # Return model, checkpoint data, and first epoch (to resume training)
    return model, checkpoint, first_epoch


def get_model(model, env, scenario, device, freeze_encoder=False, checkpoint=None, *args, **kwargs):
    use_contrastive = scenario == 'contrastive'
    
    # Get state
    state = get_state(env=env)
    
    # Get special args
    node_dim1, node_dim2, num_channels = special_args(env)
    
    # Select modules
    graph_encoder, image_encoder, decoder = select_modules(model, scenario)
    del kwargs['graph_encoder'], kwargs['image_encoder'], kwargs['decoder']
    
    # Load encoder
    encoder = Encoder(
        graph_encoder=graph_encoder,
        image_encoder=image_encoder,
        checkpoint=checkpoint,
        node_dim1=node_dim1,
        node_dim2=node_dim2,
        num_channels=num_channels,
        *args, **kwargs
    )
    if freeze_encoder and not use_contrastive:
        for parameter in encoder.parameters():
            parameter.requires_grad = False
    
    # Load decoder (if not contrastive learning)
    decoder = None if use_contrastive else Decoder(
        decoder=decoder,
        checkpoint=checkpoint,
        *args, **kwargs
    )
        
    # Create model (encoder + decoder)
    return Model(encoder=encoder, decoder=decoder, state=state).to(device)

#dac
def select_modules(model, scenario):
    module1, module2 = model.split('_')
    if scenario == 'contrastive':
        graph_encoder = module1
        image_encoder = None
        decoder = None
    if scenario == 'graph':
        graph_encoder = module1
        image_encoder = None
        decoder = module2
    elif scenario == 'image':
        graph_encoder = None
        image_encoder = module1
        decoder = module2
    else:
        graph_encoder = module1
        image_encoder = module2
        decoder = None
    return graph_encoder, image_encoder, decoder


class Model(torch.nn.Module):
    
    def __init__(self, encoder, decoder=None, state=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.state = state
    
    def set_train(self, train=False, return_actions=False, *args, **kwargs):
        if self.decoder is not None:
            self.decoder.set_train(train=train, return_actions=return_actions)
   
   #dac
    def forward(self, data):
        
        # Define state (if not contrastive learning)
        state = None if (self.state is None) else self.state.initialize(data)
        
        # Calculate embeddings from encoder
        embeddings = self.encoder(data)

        # Si el encoder devolvió una tupla (original y rotado) pero tenemos un decoder,
        # solo pasamos el embedding original al decoder.
        if self.decoder is not None and isinstance(embeddings, tuple):
            embeddings = embeddings[0]
        
        # Make predictions from encoder (if not contrastive learning)
        predictions = None if (self.decoder is None) else self.decoder(embeddings, state)
        
        # Return either predictions or embeddings
        return embeddings if (predictions is None) else predictions


class Encoder(torch.nn.Module):
    
    def __init__(self, graph_encoder: str = '', image_encoder: str = '', *args, **kwargs) -> None:
        super().__init__()
        
        # Create graph encoder
        if graph_encoder in GRAPH_ENCODERS:
            
            # Create graph encoder module
            self.graph_encoder = get_module(
                module_name=graph_encoder,
                module_dict=GRAPH_ENCODERS,
                module_type='graph_encoder',
                module_path=graph_encoder if os.path.exists(graph_encoder) else kwargs['load_path'],
                *args, **kwargs
            )
        
        # Do not use graph encoder
        else:
            self.graph_encoder = None
        
        # Create image encoder
        if image_encoder in IMAGE_ENCODERS:
            self.image_encoder = get_module(
                module_name=image_encoder,
                module_dict=IMAGE_ENCODERS,
                module_type='image_encoder',
                module_path=image_encoder if os.path.exists(image_encoder) else kwargs['load_path'],
                *args, **kwargs
            )
        
        else:
            self.image_encoder = None

    #dac modificar encoder para que procese 2 grafos    
    def forward(self, data):

        if 'nodes_rotated' in data:
            # Pasamos el original y el rotado por el MISMO encoder de grafos (comparten pesos)
            emb1 = self.graph_encoder({'nodes': data['nodes']})
            emb2 = self.graph_encoder({'nodes': data['nodes_rotated']})
            return emb1, emb2
    
        # Mantener lógica original para otros casos
        graph_embedding = self.graph_encoder(data) if (self.graph_encoder is not None) else None
        return graph_embedding
    """    
        # Graph embeddings
        graph_embedding = self.graph_encoder(data) if (self.graph_encoder is not None) else None
        
        # Image embeddings
        image_embedding = self.image_encoder(data) if (self.image_encoder is not None) else None
        
        # Return calculated embeddings
        if graph_embedding is None:
            return image_embedding
        if image_embedding is None:
            return graph_embedding
        return graph_embedding, image_embedding
    """


class Decoder(torch.nn.Module):
    
    def __init__(self, decoder: str = '', *args, **kwargs) -> None:
        super().__init__()
        
        # Create decoder module
        self.decoder = get_module(
            module_name=decoder,
            module_dict=DECODERS,
            module_type='decoder',
            module_path=decoder if os.path.exists(decoder) else kwargs['load_path'],
            *args, **kwargs
        )
    
    def set_train(self, train=False, return_actions=True, *args, **kwargs):
        self.decoder.set_train(train=train, return_actions=return_actions)
        
    def forward(self, embeddings, state):
        
        # Make predictions
        predictions = self.decoder(embeddings, state)
        return predictions
