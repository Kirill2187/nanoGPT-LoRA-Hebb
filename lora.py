import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from typing import Literal

# Code is based on https://github.com/danielgrittner/nanoGPT-LoRA

class LoRALinear(nn.Linear):

    def __init__(self,
                 # nn.Linear parameters
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 device=None,
                 dtype=None,
                 # LoRA parameters
                 lora_rank: int = 0,
                 lora_alpha: float = 0.0,
                 lora_dropout: float = 0.0,
                 a_train: Literal['backprop', 'hebb', 'frozen'] = 'bp',
                 b_train: Literal['backprop', 'hebb', 'frozen'] = 'bp',
                 a_init: Literal['gaussian', 'he'] = 'he',
                 b_init: Literal['gaussian', 'zero'] = 'zero',
                 # Hebb parameters
                 hebb_lr: float = 0.001,
                 hebb_temp: float = 0.2
                ) -> None:
        nn.Linear.__init__(
            self,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype
        )

        # LoRA stuff
        self.has_weights_merged = False
        if lora_rank > 0:
            self.lora_dropout = nn.Dropout(lora_dropout)

            self.lora_scaling = lora_alpha / lora_rank
            
            if a_train == 'hebb':
                self.lora_A = SoftHebbLinear(in_features, lora_rank, dtype=dtype, learning_rate=hebb_lr, temperature=hebb_temp)
            else:
                self.lora_A = nn.Linear(in_features, lora_rank, bias=False, device=device, dtype=dtype)
                
            if b_train == 'hebb':
                self.lora_B = SoftHebbLinear(lora_rank, out_features, dtype=dtype, learning_rate=hebb_lr, temperature=hebb_temp)
            else:
                self.lora_B = nn.Linear(lora_rank, out_features, bias=False, device=device, dtype=dtype)

            self.lora_A.weight.requires_grad = False
            self.lora_B.weight.requires_grad = False
            
            self.a_init = a_init
            self.b_init = b_init
            
            self.lora_A.weight.training_method = a_train
            self.lora_B.weight.training_method = b_train

            self.reset_parameters()

    def is_lora(self) -> bool:
        return hasattr(self, 'lora_A')

    def reset_parameters(self) -> None:
        nn.Linear.reset_parameters(self)
        if self.is_lora():
            if self.a_init == 'gaussian':
                torch.nn.init.normal_(self.lora_A.weight, mean=0.0, std=0.01)
            elif self.a_init == 'he':
                # Same as nn.Linear
                torch.nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))

            if self.b_init == 'gaussian':
                torch.nn.init.normal_(self.lora_B.weight, mean=0.0, std=0.01)
            elif self.b_init == 'zero':
                torch.nn.init.zeros_(self.lora_B.weight)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = nn.Linear.forward(self, input)
        if not self.has_weights_merged and self.is_lora():
            # h = Wx + BAx * scaling
            x += self.lora_scaling * self.lora_B(self.lora_A(self.lora_dropout(input)))
        return x

    def extra_repr(self) -> str:
        out = nn.Linear.extra_repr(self)
        if self.is_lora():
            out += f', lora_rank={self.lora_A.weight.shape[0]}, lora_scaling={self.lora_scaling}, lora_dropout={self.lora_dropout.p}'
            out += f', lora_A_train={self.lora_A.weight.training_method}, lora_B_train={self.lora_B.weight.training_method}'
        return out

    def train(self, mode: bool = True) -> "LoRALinear":
        nn.Linear.train(self, mode)
        if self.has_weights_merged and self.is_lora():
            self.lora_A.train(mode)
            self.lora_B.train(mode)
            # de-merge weights, i.e., remove BA from W = W + BA
            self.weight.data -= self.lora_scaling * self.lora_B.weight @ self.lora_A.weight
            self.has_weights_merged = False
        return self

    def eval(self) -> "LoRALinear":
        nn.Linear.eval(self)
        if not self.has_weights_merged and self.is_lora():
            self.lora_A.eval()
            self.lora_B.eval()
            # merge weights, i.e., add BA to W
            self.weight.data += self.lora_scaling * self.lora_B.weight @ self.lora_A.weight
            self.has_weights_merged = True
        return self
    

class SoftHebbLinear(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            learning_rate: float = 0.001,
            temperature: float = 0.2,
            dtype=None,
    ) -> None:    
        super(SoftHebbLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.learning_rate = learning_rate
        self.temperature = temperature
        
        self.weight = nn.Parameter(torch.empty((out_features, in_features), dtype=dtype), requires_grad=False)
            
    def forward(self, x):
        preactivations = F.linear(x, self.weight)
        activations = F.softmax(preactivations / self.temperature, dim=-1)

        if self.training:
            self._hebbian_update(x, activations, preactivations)

        return activations
    
    @torch.no_grad()
    def _hebbian_update(self, x, activations, preactivations):        
        x = x.view(-1, self.in_features)
        activations = activations.view(-1, self.out_features)
        preactivations = preactivations.view(-1, self.out_features)
        
        yx = torch.matmul(activations.t(), x)
        yu = torch.multiply(activations, preactivations)
        yu = torch.sum(yu.t(), dim=1).unsqueeze(1)
        dw = yx - yu.view(-1, 1) * self.weight
        
        self.weight += self.learning_rate * dw


def get_lora_model(model: nn.Module) -> nn.Module:
    trainable_params = 0
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = (param.training_method == 'backprop')
        else:
            param.requires_grad = False
        trainable_params += param.numel() if param.requires_grad else 0
    print(f"Backprop trainable parameters: {trainable_params}")
    return model


if __name__ == "__main__":    
    layer = SoftHebbLinear(2, 3, learning_rate=0.01)
    torch.nn.init.normal_(layer.weight, mean=0.0, std=1)
    
    import numpy as np
    
    n = 1000
    data = []
    centers = np.array([[-2, 0], [1, 0], [0, 2], [0, -1]])
    for center in centers:
        data.append(center + np.random.randn(n, 2) * 2)
    
    data = np.concatenate(data)
    data -= data.mean(axis=0)
    data /= data.std(axis=0)
    
    data = torch.tensor(data, dtype=torch.float32)
    
    batch_size = 8
    norms = []
    for _ in range(1000):
        layer(data[torch.randint(0, len(data), (batch_size,))])
        norms.append(np.linalg.norm(layer.weight.detach().numpy(), axis=1))
        
    layer.eval()
    for center in centers:
        print(layer(torch.tensor(center, dtype=torch.float32).unsqueeze(0)))
        
    print(layer.weight)

    from matplotlib import pyplot as plt
    plt.plot(np.array(norms))
    plt.show()    
    
