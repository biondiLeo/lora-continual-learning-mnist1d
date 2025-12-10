# src/models_cl.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    MLP semplice per MNIST-1D.
    Input: vettore [B, L] (già flatten).
    """
    def __init__(self, input_dim: int, h1: int = 256, h2: int = 256,
                 num_classes: int = 10, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward_features(self, x):
        """
        Restituisce le feature del penultimo layer (embedding).
        Dimensione = h2.
        """
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return x

    def forward(self, x):
        feat = self.forward_features(x)
        logits = self.fc3(feat)
        return logits

# src/models_cl.py (aggiungi sotto la classe MLP)

class LoRALinear(nn.Module):
    """
    Layer lineare con adattamento LoRA:
    output = W x + (alpha / r) * B(Ax)
    dove W è il peso "base" (pre-addestrato, spesso congelato),
    A e B sono i pesi low-rank allenabili.
    """
    def __init__(self, in_features: int, out_features: int,
                 rank: int = 8, alpha: float = 1.0):
        super().__init__()
        self.base = nn.Linear(in_features, out_features)
        self.rank = rank
        self.alpha = alpha

        if rank > 0:
            self.lora_A = nn.Linear(in_features, rank, bias=False)
            self.lora_B = nn.Linear(rank, out_features, bias=False)
            # init: A "normale", B a zero → partiamo vicini al backbone
            nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
            nn.init.zeros_(self.lora_B.weight)
        else:
            self.lora_A = None
            self.lora_B = None

    def forward(self, x):
        out = self.base(x)
        if self.rank > 0 and self.lora_A is not None:
            delta = self.lora_B(self.lora_A(x)) * (self.alpha / self.rank)
            out = out + delta
        return out


class MLPWithLoRA(nn.Module):
    """
    MLP con LoRA sui primi due layer (fc1, fc2).
    I pesi base vengono di solito inizializzati da un MLP pre-addestrato
    e congelati; si allenano solo i pesi LoRA.
    """
    def __init__(self, input_dim: int, h1: int = 256, h2: int = 256,
                 num_classes: int = 10, dropout: float = 0.1,
                 rank: int = 8, alpha: float = 1.0):
        super().__init__()
        self.fc1 = LoRALinear(input_dim, h1, rank=rank, alpha=alpha)
        self.fc2 = LoRALinear(h1, h2, rank=rank, alpha=alpha)
        self.fc3 = nn.Linear(h2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward_features(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return x

    def forward(self, x):
        feat = self.forward_features(x)
        logits = self.fc3(feat)
        return logits


def init_lora_from_mlp(mlp: MLP, lora_mlp: MLPWithLoRA,
                       freeze_base: bool = True,
                       freeze_head: bool = True):
    """
    Copia i pesi da un MLP backbone dentro un MLPWithLoRA:
    - fc1.base, fc2.base, fc3 <- pesi backbone
    - opzionalmente congela i pesi base e la testa.

    Con LoRA e continual learning ho cambiato "init_lora_from_mlp(mlp: MLP, lora_mlp: MLPWithLoRA, freeze_base: bool = True, freeze_head: bool = True)"
    in init_lora_from_mlp(backbone, lora_model, freeze_base=True, freeze_head=False) --> non congelo la head

    i parametri con requires_grad=False non ricevono gradiente → non vengono aggiornati.
    """
    with torch.no_grad():
        # fc1
        lora_mlp.fc1.base.weight.copy_(mlp.fc1.weight)
        lora_mlp.fc1.base.bias.copy_(mlp.fc1.bias)
        # fc2
        lora_mlp.fc2.base.weight.copy_(mlp.fc2.weight)
        lora_mlp.fc2.base.bias.copy_(mlp.fc2.bias)
        # fc3 (head)
        lora_mlp.fc3.weight.copy_(mlp.fc3.weight)
        lora_mlp.fc3.bias.copy_(mlp.fc3.bias)

    if freeze_base:
        for p in lora_mlp.fc1.base.parameters():
            p.requires_grad_(False)
        for p in lora_mlp.fc2.base.parameters():
            p.requires_grad_(False)

    if freeze_head:
        for p in lora_mlp.fc3.parameters():
            p.requires_grad_(False)
