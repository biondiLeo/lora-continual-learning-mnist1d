# src/training_cl.py

from dataclasses import dataclass
from typing import Tuple, Dict, Any
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch.nn.functional as F


@dataclass
class TrainConfig:
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    print_every: int = 1


def train_one_epoch(model, loader, optim, device, desc="train"):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(
        loader,
        desc=desc,
        leave=True,           # mantiene la barra alla fine dell’epoca
        dynamic_ncols=True
    )

    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100.0 * correct / total:.2f}%"
        })

    return total_loss / total, 100.0 * correct / total



def evaluate(model, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            total_loss += loss.item() * x.size(0)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / total
    avg_acc = 100.0 * correct / total
    return avg_loss, avg_acc


def train_model(
    model,
    train_loader: DataLoader,
    test_loader: DataLoader,
    cfg: TrainConfig,
) -> Dict[str, Any]:
    device = torch.device(cfg.device)
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    # meglio inizializzare best_acc < 0, così la prima epoca entra sempre
    best_acc = -1.0
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optim, device, desc=f"train | epoch {epoch}/{cfg.epochs}")
        te_loss, te_acc = evaluate(model, test_loader, device)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)

        if epoch % cfg.print_every == 0 or epoch == cfg.epochs:
            print(
                f"Epoch {epoch:02d}/{cfg.epochs} | "
                f"train_loss={tr_loss:.4f} acc={tr_acc:.2f}% | "
                f"test_loss={te_loss:.4f} acc={te_acc:.2f}%"
            )

        if te_acc > best_acc:
            best_acc = te_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # (opzionale ma safe) se per qualche motivo best_state fosse None:
    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return {
        "history": history,
        "best_acc": best_acc,
        "best_state": best_state,
    }

def extract_features(
    model,
    loader: DataLoader,
    device: torch.device,
):
    """
    Estrae le feature interne (penultimo layer) e le etichette
    per tutti i campioni nel loader.

    Ritorna:
      feats: np.ndarray [N, D]
      labels: np.ndarray [N]
    """
    model.eval()
    feats = []
    labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            # usiamo forward_features definita in MLP e MLPWithLoRA
            if hasattr(model, "forward_features"):
                z = model.forward_features(x)
            else:
                # fallback, se mai servisse
                z = model(x)

            feats.append(z.cpu())
            labels.append(y.cpu())

    feats = torch.cat(feats, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    return feats, labels