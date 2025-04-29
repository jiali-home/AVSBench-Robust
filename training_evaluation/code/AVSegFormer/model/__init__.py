from .AVSegFormer import AVSegFormer
from .AVSegFormer_robust import AVSegFormer_robust


def build_model(type, **kwargs):
    if type == 'AVSegFormer':
        return AVSegFormer(**kwargs)
    if type == 'AVSegFormer_robust':
        return AVSegFormer_robust(**kwargs)

    else:
        raise ValueError
