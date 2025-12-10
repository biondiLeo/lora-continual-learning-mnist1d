# scripts/plot_features.py

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FULL_FT_DIR = PROJECT_ROOT / "outputs_full_ft"
LORA_DIR = PROJECT_ROOT / "outputs_lora_cl"


def load_features(path: Path):
    data = np.load(path)
    return data["features"], data["labels"]


def plot_pca_2d(features, labels, title, save_path=None, show=False):
    pca = PCA(n_components=2)
    Z = pca.fit_transform(features)

    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(Z[:, 0], Z[:, 1], c=labels, s=5, alpha=0.7)
    plt.colorbar(scatter, label="classe")
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Salvata figura in: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_all_features(rank: int = 16, show: bool = False):
    # ---------------------
    # 1) Baseline full FT
    # ---------------------
    fA_A = FULL_FT_DIR / "features_fullft_taskA_afterA.npz"
    fA_B = FULL_FT_DIR / "features_fullft_taskA_afterB.npz"

    X_AA, y_AA = load_features(fA_A)
    X_AB, y_AB = load_features(fA_B)

    # stesso PCA per confrontare geometria: fit su union
    X_all = np.vstack([X_AA, X_AB])
    pca = PCA(n_components=2).fit(X_all)
    Z_AA = pca.transform(X_AA)
    Z_AB = pca.transform(X_AB)

    # plot prima di B
    plt.figure(figsize=(6, 5))
    sc1 = plt.scatter(Z_AA[:, 0], Z_AA[:, 1], c=y_AA, s=5, alpha=0.7)
    plt.colorbar(sc1, label="classe")
    plt.title("Baseline full FT - Task A (dopo A)")
    plt.tight_layout()
    save1 = FULL_FT_DIR / "pca_fullft_taskA_afterA.png"
    save1.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save1, dpi=300)
    print(f"Salvata figura in: {save1}")
    if show:
        plt.show()
    else:
        plt.close()

    # plot dopo B
    plt.figure(figsize=(6, 5))
    sc2 = plt.scatter(Z_AB[:, 0], Z_AB[:, 1], c=y_AB, s=5, alpha=0.7)
    plt.colorbar(sc2, label="classe")
    plt.title("Baseline full FT - Task A (dopo B)")
    plt.tight_layout()
    save2 = FULL_FT_DIR / "pca_fullft_taskA_afterB.png"
    plt.savefig(save2, dpi=300)
    print(f"Salvata figura in: {save2}")
    if show:
        plt.show()
    else:
        plt.close()

    # ---------------------
    # 2) LoRA (rank variabile)
    # ---------------------
    f_backbone_A = LORA_DIR / f"features_lora_backbone_taskA_afterA_r{rank}.npz"
    f_lora_A_B = LORA_DIR / f"features_lora_taskA_afterB_r{rank}.npz"

    X_backA, y_backA = load_features(f_backbone_A)
    X_loraA_B, y_loraA_B = load_features(f_lora_A_B)

    X_all_lora = np.vstack([X_backA, X_loraA_B])
    pca_lora = PCA(n_components=2).fit(X_all_lora)
    Z_backA = pca_lora.transform(X_backA)
    Z_loraA_B = pca_lora.transform(X_loraA_B)

    # LoRA backbone dopo A
    plt.figure(figsize=(6, 5))
    sc3 = plt.scatter(Z_backA[:, 0], Z_backA[:, 1], c=y_backA, s=5, alpha=0.7)
    plt.colorbar(sc3, label="classe")
    plt.title(f"LoRA r={rank} - Task A (backbone dopo A)")
    plt.tight_layout()
    save3 = LORA_DIR / f"pca_lora_backbone_taskA_afterA_r{rank}.png"
    plt.savefig(save3, dpi=300)
    print(f"Salvata figura in: {save3}")
    if show:
        plt.show()
    else:
        plt.close()

    # LoRA dopo B
    plt.figure(figsize=(6, 5))
    sc4 = plt.scatter(Z_loraA_B[:, 0], Z_loraA_B[:, 1], c=y_loraA_B, s=5, alpha=0.7)
    plt.colorbar(sc4, label="classe")
    plt.title(f"LoRA r={rank} - Task A (dopo B)")
    plt.tight_layout()
    save4 = LORA_DIR / f"pca_lora_taskA_afterB_r{rank}.png"
    plt.savefig(save4, dpi=300)
    print(f"Salvata figura in: {save4}")
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    plot_all_features(rank=16, show=False)
