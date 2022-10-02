import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Model):
    def __init__(self):
        super().__init__()
        
