# scripts/exp_full_ft.py

"""
Esperimento 1: Full fine-tuning (baseline) in Continual Learning su MNIST-1D.

- Task A: cifre 0-4
- Task B: cifre 5-7

Pipeline:
1) Allena MLP su Task A (classi 0-4)
2) Valuta su Task A -> Acc_A_before
3) Continua training sul Task B aggiornando TUTTI i pesi
4) Valuta su Task A e Task B -> Acc_A_after, Acc_B_after
5) Calcola forgetting = Acc_A_after - Acc_A_before

È scritto come funzione riusabile (run_full_finetuning_experiment) e
come script eseguibile da terminale.
"""

from pathlib import Path
from src.data_cl import PROJECT_ROOT
import torch

from src.data_cl import get_task_loaders
from src.models_cl import MLP
from src.training_cl import TrainConfig, train_model, evaluate, extract_features
import numpy as np

def run_full_finetuning_experiment(
    h1: int = 256,
    h2: int = 256,
    dropout: float = 0.1,
    epochs_A: int = 15,
    epochs_B: int = 30,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    batch_size: int = 128,
    num_workers: int = 0,
    device: str | None = None,
    out_dir: str | Path = PROJECT_ROOT / "outputs_full_ft",
):
    """
    Esegue l'esperimento di full fine-tuning A -> B e restituisce un dict
    con le metriche principali.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device_t = torch.device(device)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1) TASK A: training MLP su cifre 0-4
    # ---------------------------------------------------------
    print("=== TASK A (classi 0-4) — training completo ===")
    trainA, testA, input_dim, num_classes = get_task_loaders(
        "A",
        batch_size=batch_size,
        num_workers=num_workers,
    )
    print(f"Input dim = {input_dim}, num_classes = {num_classes}")

    model = MLP(input_dim, h1=h1, h2=h2, num_classes=num_classes, dropout=dropout)

    cfgA = TrainConfig(
        epochs=epochs_A,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
    )
    print(f"--- Inizio training Task A per {epochs_A} epoche ---")
    resA = train_model(model, trainA, testA, cfgA)

    # carica i pesi migliori su Task A
    model.load_state_dict(resA["best_state"])
    _, accA_before = evaluate(model, testA, device_t)
    print(f"\nAccuracy Task A (prima di Task B): {accA_before:.2f}%")

    # --- salvataggio feature dopo Task A (baseline full FT) ---
    featA_afterA, yA_afterA = extract_features(model, testA, device_t)
    np.savez(
        out_path / "features_fullft_taskA_afterA.npz",
        features=featA_afterA,
        labels=yA_afterA,
    )
    print(f"Salvate feature Task A (dopo Task A) in: {out_path / 'features_fullft_taskA_afterA.npz'}")

    torch.save(model.state_dict(), out_path / "mlp_taskA_full_ft.pt")
    print(f"Salvato modello dopo Task A in: {out_path / 'mlp_taskA_full_ft.pt'}")

    # ---------------------------------------------------------
    # 2) TASK B: continua il training su cifre 5-7 (full FT)
    # ---------------------------------------------------------
    print("\n=== TASK B (classi 5-7) — full fine-tuning ===")
    trainB, testB, _, _ = get_task_loaders(
        "B",
        batch_size=batch_size,
        num_workers=num_workers,
    )

    cfgB = TrainConfig(
        epochs=epochs_B,
        lr=lr,                 # puoi usare lr diverso per B se vuoi
        weight_decay=weight_decay,
        device=device,
    )
    print(f"--- Inizio training Task B per {epochs_B} epoche ---")
    resB = train_model(model, trainB, testB, cfgB)

    # modello alla fine di Task B (miglior checkpoint su B)
    model.load_state_dict(resB["best_state"])

    # ---------------------------------------------------------
    # 3) Valutazione finale
    # ---------------------------------------------------------
    _, accA_after = evaluate(model, testA, device_t)
    _, accB_after = evaluate(model, testB, device_t)

    # --- salvataggio feature dopo Task B (baseline full FT) ---
    featA_afterB, yA_afterB = extract_features(model, testA, device_t)
    featB_afterB, yB_afterB = extract_features(model, testB, device_t)

    np.savez(
        out_path / "features_fullft_taskA_afterB.npz",
        features=featA_afterB,
        labels=yA_afterB,
    )
    np.savez(
        out_path / "features_fullft_taskB_afterB.npz",
        features=featB_afterB,
        labels=yB_afterB,
    )

    print(f"Salvate feature Task A (dopo B) in: {out_path / 'features_fullft_taskA_afterB.npz'}")
    print(f"Salvate feature Task B (dopo B) in: {out_path / 'features_fullft_taskB_afterB.npz'}")

    forgetting = accA_after - accA_before

    print("\n=== RISULTATI FULL FINE-TUNING ===")
    print(f"Acc Task A prima di B : {accA_before:.2f}%")
    print(f"Acc Task A dopo B     : {accA_after:.2f}%")
    print(f"Acc Task B dopo B     : {accB_after:.2f}%")
    print(f"Forgetting (A)        : {forgetting:.2f} punti percentuali")

    torch.save(model.state_dict(), out_path / "mlp_taskAB_full_ft.pt")
    print(f"Salvato modello finale (dopo A+B) in: {out_path / 'mlp_taskAB_full_ft.pt'}")

    return {
        "accA_before": accA_before,
        "accA_after": accA_after,
        "accB_after": accB_after,
        "forgetting": forgetting,
        "best_acc_A": resA["best_acc"],
        "best_acc_B": resB["best_acc"],
    }


if __name__ == "__main__":
    # Default run da terminale:
    run_full_finetuning_experiment()
