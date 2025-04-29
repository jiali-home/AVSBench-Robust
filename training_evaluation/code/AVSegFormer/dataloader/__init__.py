from .v2_dataset import V2Dataset, get_v2_pallete
from .s4_dataset import S4Dataset, S4Dataset_slience, S4Dataset_ood_silence
from .ms3_dataset import MS3Dataset, MS3Dataset_slience, MS3Dataset_ood_silence
# from mmcv import Config
from mmengine.config import Config


def build_dataset(type, split, **kwargs):

    if type == 'S4Dataset':
        return S4Dataset(split=split, cfg=Config(kwargs))
    elif type == 'S4Dataset_slience':
        return S4Dataset_slience(split=split, cfg=Config(kwargs))   
    elif type == 'S4Dataset_ood_silence':
        return S4Dataset_ood_silence(split=split, cfg=Config(kwargs))   
        
    elif type == 'MS3Dataset':
        return MS3Dataset(split=split, cfg=Config(kwargs))
    elif type == 'MS3Dataset_slience':
        return MS3Dataset_slience(split=split, cfg=Config(kwargs))
    elif type == 'MS3Dataset_ood_silence':
        return MS3Dataset_ood_silence(split=split, cfg=Config(kwargs))    
    
    else:
        raise ValueError


__all__ = ['build_dataset', 'get_v2_pallete']
