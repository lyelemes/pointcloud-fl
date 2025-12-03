import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import glob
import os

class PointCloudDataset(Dataset):
    def __init__(self, data_folder):
        self.file_paths = glob.glob(os.path.join(data_folder, "*.npy"))
        print(f"Found {len(self.file_paths)} point clouds")
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # Load point cloud: shape (600, 11)
        point_cloud = np.load(self.file_paths[idx])
        return torch.FloatTensor(point_cloud)

def get_train_dataloader(data_folder, batch_size=32, num_workers=4, worker_init_fn=None, seed=42):
    # Create dataset
    train_dataset = PointCloudDataset(data_folder)

    g = torch.Generator()
    g.manual_seed(seed)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            num_workers=num_workers, 
                            drop_last=True,
                            worker_init_fn=worker_init_fn, 
                            generator=g)
    
    print(f"Train: {len(train_dataset)} samples")
    return train_loader

def load_pointcloud_data(data_folder, batch_size=32, num_workers=4, deterministic=False, worker_init_fn=None, seed=42):
    """
    Infinite generator that yields batches of point clouds and empty dicts
    (like the ImageDataset version, for TrainLoop compatibility).
    """
    dataset = PointCloudDataset(data_folder)

    g = torch.Generator()
    g.manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not deterministic,
        num_workers=num_workers,
        drop_last=True,
        worker_init_fn=worker_init_fn, 
        generator=g
    )

    while True:
        yield from loader