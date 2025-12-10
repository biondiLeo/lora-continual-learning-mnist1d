# scripts/plot_features.py

from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#per distinguere il terminale da eventuali notebook colab/jupyter
try:
    from IPython.display import display
    IN_NOTEBOOK = True
except ImportError:
    IN_NOTEBOOK = False

# -----------------------------
# Path progetto
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FULL_FT_DIR = PROJECT_ROOT / "outputs_full_ft"
LORA_DIR = PROJECT_ROOT / "outputs_lora_cl"


# -----------------------------
# Utility
# -----------------------------
def load_features(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")
    data = np.load(path)
    return data["features"], data["labels"]


def plot_pca(
    features,
    labels,
    title,
    save_path,
    pca=None,
    show: bool = True,
):
    if pca is None:
        pca = PCA(n_components=2)
        Z = pca.fit_transform(features)
    else:
        Z = pca.transform(features)

    plt.figure(figsize=(6, 5))
    sc = plt.scatter(Z[:, 0], Z[:, 1], c=labels, s=6, alpha=0.7)
    plt.colorbar(sc, label="classe")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.tight_layout()

    # salva sempre
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)

    # mostra solo se notebook
    if show and IN_NOTEBOOK:
        display(plt.gcf())

    plt.close()
    print(f"Salvato: {save_path}")

    return pca


# -----------------------------
# Main
# -----------------------------
def main(rank: int):
    print(f"Plot feature PCA per LoRA rank = {rank}")

    # ---------- BASELINE FULL FT ----------
    fA_afterA = FULL_FT_DIR / "features_fullft_taskA_afterA.npz"
    fA_afterB = FULL_FT_DIR / "features_fullft_taskA_afterB.npz"

    XA_A, yA_A = load_features(fA_afterA)
    XA_B, yA_B = load_features(fA_afterB)

    # PCA condivisa
    pca_full = PCA(n_components=2)
    pca_full.fit(np.vstack([XA_A, XA_B]))

    plot_pca(
        XA_A,
        yA_A,
        "Baseline Full FT – Task A dopo Task A",
        FULL_FT_DIR / "pca_fullft_taskA_afterA.png",
        pca=pca_full,
    )

    plot_pca(
        XA_B,
        yA_B,
        "Baseline Full FT – Task A dopo Task B",
        FULL_FT_DIR / "pca_fullft_taskA_afterB.png",
        pca=pca_full,
    )

    # ---------- LoRA ----------
    f_backbone_A = LORA_DIR / f"features_lora_backbone_taskA_afterA_r{rank}.npz"
    f_lora_A_B   = LORA_DIR / f"features_lora_taskA_afterB_r{rank}.npz"

    X_backA, y_backA = load_features(f_backbone_A)
    X_loraA, y_loraA = load_features(f_lora_A_B)

    # PCA condivisa LoRA
    pca_lora = PCA(n_components=2)
    pca_lora.fit(np.vstack([X_backA, X_loraA]))

    plot_pca(
        X_backA,
        y_backA,
        f"LoRA r={rank} – Task A dopo Task A (backbone)",
        LORA_DIR / f"pca_lora_r{rank}_taskA_afterA.png",
        pca=pca_lora,
    )

    plot_pca(
        X_loraA,
        y_loraA,
        f"LoRA r={rank} – Task A dopo Task B",
        LORA_DIR / f"pca_lora_r{rank}_taskA_afterB.png",
        pca=pca_lora,
    )


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot PCA delle feature MNIST-1D")
    parser.add_argument(
        "--rank",
        type=int,
        required=True,
        help="Rank LoRA usato per generare le feature (es. 8, 16, ...)",
    )
    args = parser.parse_args()

    main(rank=args.rank)
