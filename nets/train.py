import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os

# Import your provided classes
from model import HEEGNet
from trainer import Trainer


# 1. Custom Dataset for Preprocessed DEAP Data
class DEAPDataset(Dataset):
    def __init__(self, data_path, target='arousal', threshold=5):
        """
        data_path: Path to .npz file for a specific subject
        target: 'arousal' or 'valence'
        threshold: Threshold for binary classification (e.g., 5)
        """
        data = np.load(data_path)
        # Assuming shape (2400, 32, 128) from your preprocessing script
        self.inputs = torch.from_numpy(data['X_RAW']).float()

        # Binary labels: 1 if target > threshold, else 0
        raw_labels = data[f'y_{target}']
        self.labels = torch.from_numpy((raw_labels > threshold).astype(int)).long()

        # Domain ID for DSMDBN (Subject ID)
        subject_id = int(os.path.basename(data_path).split('_')[1].split('.')[0])
        self.domain_ids = torch.full((len(self.labels),), subject_id).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # HEEGNet expects a dictionary for the forward call (**features)
        return {
            'inputs': self.inputs[idx],
            'domains': self.domain_ids[idx]
        }, self.labels[idx]


# 2. Main Training Function
def train_subject_independent():
    # --- CONFIGURATION ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DTYPE = torch.float32  # Use float32 for typical EEG DL tasks
    PROCESSED_DATA_DIR = "data/processed/"  # Update to your path
    BATCH_SIZE = 64
    EPOCHS = 100
    LR = 0.001

    print(f"Starting HEEGNet training on {DEVICE}...")

    # Load all subject files
    subject_files = [os.path.join(PROCESSED_DATA_DIR, f) for f in os.listdir(PROCESSED_DATA_DIR) if f.endswith('.npz')]

    # Simple Split: Subject-Independent (Subjects 1-25 for train, 26-32 for val)
    train_files = subject_files[:25]
    val_files = subject_files[25:]

    train_ds = torch.utils.data.ConcatDataset([DEAPDataset(f) for f in train_files])
    val_ds = torch.utils.data.ConcatDataset([DEAPDataset(f) for f in val_files])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # --- INITIALIZE MODEL ---
    # F1=16, F2=32 for increased capacity as discussed
    model = HEEGNet(
        chunk_size=124,
        num_electrodes=32,
        F1=16,
        F2=32,
        num_classes=2,
        device=DEVICE,
        dtype=DTYPE,
        domain_adaptation=True,
        domains=list(range(1, 33))  # List of subject IDs 1-32
    )

    # --- INITIALIZE TRAINER ---
    # swd_weight is the HHSW loss weight (0.5 recommended in paper for non-emotion VEP/iEEG) [cite: 607]
    trainer = Trainer(
        max_epochs=EPOCHS,
        callbacks=[],  # Add your custom callbacks if needed
        loss=nn.CrossEntropyLoss(),
        device=DEVICE,
        dtype=DTYPE,
        lr=LR,
        swd_weight=0.5
    )

    # --- START FIT ---
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    train_subject_independent()