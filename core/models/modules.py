import torch.nn as nn
from mamba_ssm import Mamba
from core.models.hyena import HyenaEncoderLayer
from core.models.bi_mamba import BiMambaWrapper
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MODULE_REGISTRY = {}

def register_module(name):
    def decorator(cls):
        MODULE_REGISTRY[name.lower()] = cls
        return cls
    return decorator

def build_module(name, input_dim, output_dim):
    name = name.lower()
    if name not in MODULE_REGISTRY:
        raise ValueError(f"Unknown module type: {name}")
    return MODULE_REGISTRY[name](input_dim, output_dim)


@register_module("cnn")
class CNNModule(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        kernel_size = 9
        padding = (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.BatchNorm1d(output_dim)
        )

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.net(x)
        x = x.permute(0,2,1)
        return x

@register_module("hyena")
class HyenaModule(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.project_in = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.hyena = HyenaEncoderLayer(d_model=output_dim, l_max=1024)
        self.project_out = nn.Identity()

    def forward(self, x):
        x = self.project_in(x)
        x = self.hyena(x)
        return self.project_out(x)


@register_module("mamba")
class MambaModule(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.project_in = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.norm = nn.LayerNorm(output_dim)
        self.mamba = Mamba(d_model=output_dim)

    def forward(self, x):
        x = self.project_in(x)
        x = self.norm(x)  
        x = self.mamba(x)
        return x

@register_module("onlymamba")
class OnlyMambaModule(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.project_in = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.norm = nn.LayerNorm(output_dim)
        self.mamba = Mamba(d_model=output_dim)

    def forward(self, x):
        x = self.project_in(x)
        identity = x
        x_norm = self.norm(x)  
        x = self.mamba(x_norm)
        return identity + x

@register_module("bimamba")
class BiMambaModule(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.project_in = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.mamba = BiMambaWrapper(d_model=output_dim, bidirectional=True)

    def forward(self, x):
        x = self.project_in(x)
        x = self.mamba(x)
        return x
        
@register_module("lstm")
class LSTMModule(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        #output_dim = int(output_dim/2)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=output_dim, dropout=0.4, batch_first=True)
        for name, param in self.lstm.named_parameters():
            if 'weight_hh' in name:
                nn.init.orthogonal_(param) # 对循环权重应用正交初始化
            elif 'weight_ih' in name:
                nn.init.xavier_uniform_(param) # 对输入权重应用Xavier初始化
            elif 'bias' in name:
                nn.init.zeros_(param)
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1.0) # 将遗忘门偏置设为1
    def forward(self, x):
        x, _ = self.lstm(x)
        return x

'''
@register_module("lstm")
class LSTMModule(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        output_dim = int(output_dim/2)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=output_dim, dropout=0.4, batch_first=True, bidirectional=True)
        for name, param in self.lstm.named_parameters():
            if 'weight_hh' in name:
                nn.init.orthogonal_(param) # 对循环权重应用正交初始化
            elif 'weight_ih' in name:
                nn.init.xavier_uniform_(param) # 对输入权重应用Xavier初始化
            elif 'bias' in name:
                nn.init.zeros_(param)
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1.0) # 将遗忘门偏置设为1
    def forward(self, x):
        x, _ = self.lstm(x)
        return x
'''

@register_module("gru")
class GRUModule(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        output_dim = int(output_dim/2)
        self.gru = nn.GRU(input_size=input_dim, hidden_size=output_dim, batch_first=True, bidirectional=True)

    def forward(self, x):
        x, _ = self.gru(x)
        return x


@register_module("rnn")
class RNNModule(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        output_dim = int(output_dim/2)
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=output_dim, batch_first=True, bidirectional=True)

    def forward(self, x):
        x, _ = self.rnn(x)
        return x


@register_module("transformer")
class TransformerModule(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        nhead = int(output_dim / 64)
        self.project_in = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.encoder = nn.TransformerEncoderLayer(d_model=output_dim, nhead=nhead, batch_first=True)

    def forward(self, x):
        x = self.project_in(x)
        x = self.encoder(x)
        return x
