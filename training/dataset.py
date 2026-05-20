import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class LandmarkDataset(Dataset):
    """
    Loads pre-collected landmark sequences saved as .npz files.
    Each .npz contains:
      sequences : (N, window_size, 63) float32
      labels    : (N,) int64
    """

    LABEL_MAP = {
        "idle": 0, "reach": 1, "pick": 2,
        "inspect": 3, "place": 4, "verify": 5
    }

    def __init__(self, data_dir: str, window_size: int = 20, augment: bool = False):
        self.window_size = window_size
        self.augment = augment
        self.sequences: list[np.ndarray] = []
        self.labels: list[int] = []

        data_path = Path(data_dir)
        for npz_file in sorted(data_path.glob("*.npz")):
            data = np.load(npz_file)
            self.sequences.extend(data["sequences"])
            self.labels.extend(data["labels"].tolist())

        if not self.sequences:
            raise FileNotFoundError(
                f"No .npz files found in {data_dir}. "
                "Run scripts/collect_data.py first to record training samples."
            )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx].astype(np.float32)

        if self.augment:
            # Gaussian noise augmentation
            seq += np.random.normal(0, 0.01, seq.shape).astype(np.float32)
            # Random time warp: drop/duplicate a frame
            if np.random.random() < 0.3:
                drop = np.random.randint(0, len(seq))
                seq = np.delete(seq, drop, axis=0)
                seq = np.vstack([seq, seq[-1:]])  # pad back to length

        return (
            torch.tensor(seq, dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )
