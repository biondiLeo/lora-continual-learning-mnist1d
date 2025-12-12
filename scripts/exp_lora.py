# scripts/exp_lora.py

"""
Esperimento 2: LoRA in Continual Learning su MNIST-1D.

- Task A: cifre 0-4 (train backbone MLP)
- Task B: cifre 5-7 (congelo backbone, alleno SOLO LoRA)

Pipeline:
1) Allena MLP su Task A -> backbone
2) Valuta su Task A -> Acc_A_before
3) Crea MLPWithLoRA inizializzato dal backbone, congela base + head
4) Allena su Task B aggiornando SOLO LoRA
5) Valuta su Task A e Task B -> Acc_A_after, Acc_B_after
6) Calcola forgetting = Acc_A_after - Acc_A_before
"""

from pathlib import Path
from src.data_cl import PROJECT_ROOT
import torch

from src.data_cl import get_task_loaders
from src.models_cl import MLP, MLPWithLoRA, init_lora_from_mlp
from src.training_cl import TrainConfig, train_model, evaluate, extract_features
import numpy as np


def run_lora_experiment(
    h1: int = 256,
    h2: int = 256,
    dropout: float = 0.1,
    rank: int = 8,
    alpha: float = 1.0,
    epochs_A: int = 30,
    epochs_B: int = 30,
    lr_backbone: float = 1e-3,
    lr_lora: float = 1e-3,
    weight_decay: float = 0.0,
    batch_size: int = 128,
    num_workers: int = 0,
    freeze_base: bool = True,
    freeze_head: bool = False,
    device: str | None = None,
    out_dir: str | Path = PROJECT_ROOT / "outputs_lora_cl",
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev_t = torch.device(device)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1) TASK A: training backbone MLP su cifre 0-4
    # ---------------------------------------------------------
    print("=== TASK A (classi 0-4) — training backbone ===")
    trainA, testA, input_dim, num_classes = get_task_loaders(
        "A",
        batch_size=batch_size,
        num_workers=num_workers,
    )
    print(f"Input dim = {input_dim}, num_classes = {num_classes}")

    backbone = MLP(input_dim, h1=h1, h2=h2, num_classes=num_classes, dropout=dropout)

    cfgA = TrainConfig(
        epochs=epochs_A,
        lr=lr_backbone,
        weight_decay=weight_decay,
        device=device,
    )
    print(f"--- Inizio training backbone Task A per {epochs_A} epoche ---")
    resA = train_model(backbone, trainA, testA, cfgA)

    backbone.load_state_dict(resA["best_state"])
    _, accA_before = evaluate(backbone, testA, dev_t)
    print(f"\nAccuracy Task A (prima di Task B, backbone): {accA_before:.2f}%")

    # --- feature backbone dopo Task A (prima di LoRA su B) ---
    featA_backbone_afterA, yA_backbone_afterA = extract_features(backbone, testA, dev_t)
    np.savez(
        out_path / f"features_lora_backbone_taskA_afterA_r{rank}.npz",
        features=featA_backbone_afterA,
        labels=yA_backbone_afterA,
    )
    print(f"Salvate feature backbone Task A (dopo A) in: "
          f"{out_path / f'features_lora_backbone_taskA_afterA_r{rank}.npz'}")

    torch.save(backbone.state_dict(), out_path / "backbone_taskA.pt")
    print(f"Salvato backbone in: {out_path / 'backbone_taskA.pt'}")

    # ---------------------------------------------------------
    # 2) Inizializza modello LoRA dal backbone
    # ---------------------------------------------------------
    print("\n=== Inizializzazione MLPWithLoRA dal backbone ===")
    lora_model = MLPWithLoRA(
        input_dim,
        h1=h1,
        h2=h2,
        num_classes=num_classes,
        dropout=dropout,
        rank=rank,
        alpha=alpha,
    )
    init_lora_from_mlp(backbone, lora_model, freeze_base=freeze_base, freeze_head=freeze_head)

    # controlliamo quanti parametri sono trainabili
    total_params = sum(p.numel() for p in lora_model.parameters())
    trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    print(f"Parametri totali LoRA: {total_params:,}")
    print(f"Parametri trainabili : {trainable_params:,} "
          f"({100.0 * trainable_params / total_params:.2f}% dei parametri)")

    # ---------------------------------------------------------
    # 3) TASK B: training SOLO LoRA su cifre 5-7
    # ---------------------------------------------------------
    print("\n=== TASK B (classi 5-7) — training SOLO LoRA ===")
    trainB, testB, _, _ = get_task_loaders(
        "B",
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Usa train_model normalmente: aggiornerà solo i parametri con requires_grad=True
    cfgB = TrainConfig(
        epochs=epochs_B,
        lr=lr_lora,
        weight_decay=weight_decay,
        device=device,
    )
    print(f"--- Inizio training LoRA Task B per {epochs_B} epoche ---")
    resB = train_model(lora_model, trainB, testB, cfgB)

    lora_model.load_state_dict(resB["best_state"])

    # ---------------------------------------------------------
    # 4) Valutazione finale
    # ---------------------------------------------------------
    _, accA_after = evaluate(lora_model, testA, dev_t)
    _, accB_after = evaluate(lora_model, testB, dev_t)

    # --- feature LoRA dopo Task B ---
    featA_lora_afterB, yA_lora_afterB = extract_features(lora_model, testA, dev_t)
    featB_lora_afterB, yB_lora_afterB = extract_features(lora_model, testB, dev_t)

    np.savez(
        out_path / f"features_lora_taskA_afterB_r{rank}.npz",
        features=featA_lora_afterB,
        labels=yA_lora_afterB,
    )
    np.savez(
        out_path / f"features_lora_taskB_afterB_r{rank}.npz",
        features=featB_lora_afterB,
        labels=yB_lora_afterB,
    )

    print(f"Salvate feature LoRA Task A (dopo B) in: "
          f"{out_path / f'features_lora_taskA_afterB_r{rank}.npz'}")
    print(f"Salvate feature LoRA Task B (dopo B) in: "
          f"{out_path / f'features_lora_taskB_afterB_r{rank}.npz'}")

    forgetting = accA_after - accA_before

    print("\n=== RISULTATI LoRA CL ===")
    print(f"Acc Task A prima di B : {accA_before:.2f}%")
    print(f"Acc Task A dopo B     : {accA_after:.2f}%")
    print(f"Acc Task B dopo B     : {accB_after:.2f}%")
    print(f"Forgetting (A)        : {forgetting:.2f} punti percentuali")

    torch.save(lora_model.state_dict(), out_path / f"lora_taskAB_r{rank}.pt")
    print(f"Salvato modello LoRA finale in: {out_path / f'lora_taskAB_r{rank}.pt'}")

    return {
        "accA_before": accA_before,
        "accA_after": accA_after,
        "accB_after": accB_after,
        "forgetting": forgetting,
        "best_acc_A": resA["best_acc"],
        "best_acc_B": resB["best_acc"],
        "total_params": total_params,
        "trainable_params": trainable_params,
    }


if __name__ == "__main__":
    # run di default con rank=8
    run_lora_experiment(rank=16)
