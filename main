import torch
import torch.nn as nn
import numpy as np
from torchdiffeq import odeint
from model import unet
from model import autoencoder 
from loader import get_train_dataloader
import time
import random
import os

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def print_parameter_count(model: nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")

def sample_random_times(n: int) -> torch.Tensor:
    t_i = torch.rand(n)
    return t_i

def euclidean_conditional_vf(xt: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return (x-xt)/(1-t)

def euclidean_vf_loss(out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.mse_loss(out, target)

def train(
        epochs: int,
        device: torch.device,
        model: torch.nn.Module,
        autoencoder: torch.nn.Module, 
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        ) -> tuple[torch.nn.Module, list[float]]:
    
    print("Training started...")
    print_parameter_count(model)
    record_losses = []
    loss_best = float('inf')
    for epoch in range(epochs):
        losses = []
        model.train()
        autoencoder.eval()
        for x in dataloader:  # x1 ~ p_1
            x = x.to(device)
            with torch.no_grad():
                _, x1 = autoencoder(x)  # Encode to latent space
            optimizer.zero_grad()
            x0 = torch.randn_like(x1, requires_grad=False, device=device)  # x0 ~ p_0 euclidean_prior
            t = sample_random_times(x1.size(0)).to(device)
            t_batch = t.view(-1, *[1] * (x1.ndim - 1))
            xt = (1.0 - t_batch) * x0 + t_batch * x1  # linear interpolation
            y = euclidean_conditional_vf(xt, x1, t_batch)
            y_pred = model(xt, t)
            l = euclidean_vf_loss(y_pred, y)
            l.backward()
            optimizer.step()
            losses += [l.item()]

        current_loss = np.mean(losses)
        if current_loss < loss_best:
            loss_best = current_loss
            torch.save(model.state_dict(), f"results_fl/steps/best_matching_unet_{epoch}.pth")
            print(f"New best model saved with loss: {loss_best:.4f}")
        # Report loss
        record_losses += [np.mean(losses)]
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs}, Loss: {record_losses[-1]:.4f}, LR: {current_lr:.2e}")
        if scheduler is not None:
            scheduler.step(record_losses[-1])
    
    print("Training completed.")
    return model

if __name__ == "__main__":
    # Hyperparameters
    seed = 42
    set_seed(seed)

    epochs = 1000
    batch_size = 128
    learning_rate = 1e-4
    min_lr = 1e-6
    patience = 5
    factor = 0.99

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_samples = 1536
    # DataLoader
    data_dir = "../1.data/1542_original_grains"
    checkpoint_path = "../2.autoencoder/results/best_autoencoder.pth"
    # Model parameters
    in_channels = 1
    base_channel = 32
    num_res_blocks = 1  
    scheme = [1, 2, 4, 8]  # [C, 2C, 4C, 8C]
    resolutions = [32, 16, 8]  # Attention resolutions

    

    # Model, Autoencoder, Optimizer, DataLoader should be defined here
    print("Initializing model...")
    model = unet.UNet(in_channels, 
                           base_channel,
                           num_res_blocks,
                           scheme,
                           resolutions)  # Replace with your model
    model.to(device)
    print("Model initialized.")

    print("Loading autoencoder...")
    autoencoder = autoencoder.PointCloudAutoencoder()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    print("Loading autoencoder from:", checkpoint_path)
    autoencoder.load_state_dict(checkpoint['model_state_dict'])
    autoencoder.to(device)
    print("Autoencoder loaded.")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                          mode='min', 
                                                          patience=patience, 
                                                          factor=factor,  
                                                          min_lr=min_lr)

    data_loader = get_train_dataloader(data_dir, batch_size=batch_size, worker_init_fn=seed_worker, seed=seed)

    # Training
    model = train(epochs, device, model, autoencoder, data_loader, optimizer, scheduler=scheduler)
    torch.save(model.state_dict(), "results_fl/steps/matching_unet_checkpoint.pth")
