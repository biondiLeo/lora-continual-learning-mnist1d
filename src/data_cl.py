# src/data_cl.py

import os
import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

# URL ufficiale del file dati MNIST-1D su GitHub
DATA_URL = "https://github.com/greydanus/mnist1d/raw/master/mnist1d_data.pkl"

# === NUOVA PARTE: definiamo la root del progetto e il path fisso del dataset ===
# this file = .../mnist1d-lora-cl/src/data_cl.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]   # sali da src/ alla root del progetto
DATA_DIR = PROJECT_ROOT / "data"
DATA_FILE = DATA_DIR / "mnist1d_data.pkl"


def download_mnist1d(data_path: str | Path = DATA_FILE):
    """
    Scarica il file mnist1d_data.pkl da GitHub se non esiste già in locale.
    Il path è fissato in PROJECT_ROOT/data/mnist1d_data.pkl
    """
    data_path = Path(data_path)
    data_path.parent.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        import urllib.request
        print(f"Scarico MNIST-1D da {DATA_URL} ...")
        urllib.request.urlretrieve(DATA_URL.as_posix() if isinstance(DATA_URL, Path) else DATA_URL,
                                   data_path)
        print(f"Salvato in {data_path}")
    else:
        print(f"Trovato MNIST-1D locale in {data_path}")
    return data_path


def load_mnist1d(data_path: str | Path = DATA_FILE):
    """
    Carica MNIST-1D da file pickle.
    """
    data_path = download_mnist1d(data_path)
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    X = data["x"].astype("float32")  # shape [N, L]
    y = data["y"].astype("int64")    # shape [N]
    return X, y


def split_train_test(X, y, test_ratio=0.2, seed=42):
    """
    Split semplice in train/test con shuffle.
    """
    rng = np.random.RandomState(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    X = X[idx]
    y = y[idx]
    n_test = int(len(X) * test_ratio)
    X_test, y_test = X[:n_test], y[:n_test]
    X_train, y_train = X[n_test:], y[n_test:]
    return X_train, y_train, X_test, y_test


def get_task_mask(y, task: str):
    """
    Task A: cifre 0-4
    Task B: cifre 5-7
    """
    task = task.upper()
    if task == "A":
        classes = [0, 1, 2, 3, 4]
    elif task == "B":
        classes = [5, 6, 7]
    else:
        raise ValueError(f"Task sconosciuto: {task} (usa 'A' o 'B')")
    mask = np.isin(y, classes)
    return mask


def get_task_loaders(
    task: str,
    batch_size: int = 128,
    num_workers: int = 0,
    test_ratio: float = 0.2,
    seed: int = 42,
):
    """
    Restituisce (train_loader, test_loader, input_dim, num_classes)
    per il Task A o B di MNIST-1D.
    """
    X, y = load_mnist1d()  # adesso usa il pickle scaricato da GitHub
    mask = get_task_mask(y, task)
    X_task = X[mask]
    y_task = y[mask]

    X_tr, y_tr, X_te, y_te = split_train_test(X_task, y_task, test_ratio=test_ratio, seed=seed)

    X_tr = torch.from_numpy(X_tr)
    y_tr = torch.from_numpy(y_tr)
    X_te = torch.from_numpy(X_te)
    y_te = torch.from_numpy(y_te)

    train_ds = TensorDataset(X_tr, y_tr)
    test_ds = TensorDataset(X_te, y_te)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    input_dim = X.shape[1]
    num_classes = 10  # manteniamo sempre 10 classi nel classifier

    return train_loader, test_loader, input_dim, num_classes
