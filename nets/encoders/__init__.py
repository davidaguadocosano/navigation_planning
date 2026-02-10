from .graph_encoders.gtn import GTN
from .image_encoders.vit import VIT


def encoders():
    return {
        'graph_encoders': {
            'gtn': GTN,
        },
        'image_encoders': {
            'vit': VIT,
        },
    }
