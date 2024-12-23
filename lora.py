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
            self.lora_A = nn.Parameter(torch.empty((lora_rank, self.in_features), device=device, dtype=dtype))
            self.lora_B = nn.Parameter(torch.empty((self.out_features, lora_rank), device=device, dtype=dtype))

            self.lora_A.requires_grad = False
            self.lora_B.requires_grad = False
            
            self.a_init = a_init
            self.b_init = b_init
            
            self.lora_A.training_method = a_train
            self.lora_B.training_method = b_train

            self.reset_parameters()

    def is_lora(self) -> bool:
        return hasattr(self, 'lora_A')

    def reset_parameters(self) -> None:
        nn.Linear.reset_parameters(self)
        if self.is_lora():
            if self.a_init == 'gaussian':
                torch.nn.init.normal_(self.lora_A, mean=0.0, std=0.01)
            elif self.a_init == 'he':
                # Same as nn.Linear
                torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
                
            if self.b_init == 'gaussian':
                torch.nn.init.normal_(self.lora_B, mean=0.0, std=0.01)
            elif self.b_init == 'zero':
                torch.nn.init.zeros_(self.lora_B)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = nn.Linear.forward(self, input)
        if not self.has_weights_merged and self.is_lora():
            # h = Wx + BAx * scaling
            x += self.lora_scaling * F.linear(
                F.linear(
                    self.lora_dropout(input),
                    self.lora_A
                ),
                self.lora_B
            )
        return x

    def extra_repr(self) -> str:
        out = nn.Linear.extra_repr(self)
        if self.is_lora():
            out += f', lora_rank={self.lora_A.shape[0]}, lora_scaling={self.lora_scaling}, lora_dropout={self.lora_dropout.p}'
            out += f', lora_A_train={self.lora_A.training_method}, lora_B_train={self.lora_B.training_method}'
        return out

    def train(self, mode: bool = True) -> "LoRALinear":
        nn.Linear.train(self, mode)
        if self.has_weights_merged and self.is_lora():
            # de-merge weights, i.e., remove BA from W = W + BA
            self.weight.data -= self.lora_scaling * self.lora_B @ self.lora_A
            self.has_weights_merged = False
        return self

    def eval(self) -> "LoRALinear":
        nn.Linear.eval(self)
        if not self.has_weights_merged and self.is_lora():
            # merge weights, i.e., add BA to W
            self.weight.data += self.lora_scaling * self.lora_B @ self.lora_A
            self.has_weights_merged = True
        return self

def get_lora_model(model: nn.Module) -> nn.Module:
    trainable_params = 0
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = (param.training_method == 'backprop')
        else:
            param.requires_grad = False
        trainable_params += param.numel() if param.requires_grad else 0
    print(f"Trainable parameters: {trainable_params}")
    return model
