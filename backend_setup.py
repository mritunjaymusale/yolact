from data import set_cfg
import torch

def initial_setup():
    torch.backends.cudnn.fastest = True
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    set_cfg('yolact_plus_resnet50_config')