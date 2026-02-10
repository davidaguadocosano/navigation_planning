from .tsp.ar_decoder import TSPAutoRegressive


def decoders():
    return {
        'tsp-ar': TSPAutoRegressive
    }
