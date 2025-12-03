import os
import torch
import torch.nn as nn
import numpy as np
from torchdiffeq import odeint
from model import unet
from model import autoencoder 
import time

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
    
def sample(
    device: torch.device,
    model: torch.nn.Module,
    autoencoder: torch.nn.Module,
    num_samples: int = 50,
    steps: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:

    print("Sampling started...")
    start_time = time.time()
    
    model.eval()
    autoencoder.eval()

    with torch.no_grad():
        # Initial latent noise
        x0 = torch.randn(num_samples, 1, 32, 32, device=device)  # match latent shape

        # Function for ODE
        def func(t, x):
            t_batch = torch.full((x.size(0),), t.item(), device=x.device)
            return model(x, t_batch)

        # ODE integration
        out = odeint(
            func,
            x0,  # initial condition
            torch.tensor([0.0, 1.0], dtype=torch.float32),
            method="dopri5",
            atol=1e-5,
            rtol=1e-5,
            )[-1]
    

        latent_emb = out.squeeze(1)  # [len(times), B, 32, 32]

        flat_latent = latent_emb.view(-1, 32*32)  # [len(times)*B, 1024]
        decoded = autoencoder.decoder(flat_latent)
        reconstructed = decoded.view(num_samples, 600, 3)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Sampling completed. Time taken: {:.2f} seconds".format(elapsed_time))
    return latent_emb.cpu().numpy(), reconstructed.cpu().numpy()

@torch.inference_mode()  # turns off gradient computations for faster inference
def euclidean_euler_sampling(
    model: nn.Module,
    autoencoder: nn.Module,
    device: torch.device,
    num_samples: int = 1000,
    num_steps: int = 100,
) -> list[torch.Tensor]:
    
    model.eval()
    autoencoder.eval()

    with torch.no_grad():
        # Initial latent noise
        x = torch.randn(num_samples, 1, 32, 32, device=device) 
        emb_traj = [x]
        
        latent_emb = x.squeeze()  
        flat_latent = latent_emb.view(-1, 32*32)  
        decoded = autoencoder.decoder(flat_latent)
        reconstructed = decoded.view(num_samples, 600, 3)
        rec_traj = [reconstructed]

        t = torch.linspace(0, 1, num_steps+1)
        print("Sampling started...")
        start_time = time.time()
        for t0, t1 in zip(t[:-1], t[1:]):
            t0_batch = torch.full((x.size(0),1, 1, 1), t0.item(), device=x.device)
            t1_batch = torch.full((x.size(0),1, 1, 1), t1.item(), device=x.device)
            t0_input = torch.full((x.size(0),), t0.item(), device=x.device)
            x = x + (t1_batch-t0_batch) * model(x, t0_input)

            latent_emb = x.squeeze(1)  # [B, 32, 32]
            flat_latent = latent_emb.view(-1, 32*32)  # [B, 1024]
            decoded = autoencoder.decoder(flat_latent)
            reconstructed = decoded.view(num_samples, 600, 3)
            
            emb_traj.append(latent_emb.cpu().numpy())
            rec_traj.append(reconstructed.cpu().numpy())
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Sampling completed. Time taken: {:.2f} seconds".format(elapsed_time))
    return emb_traj, rec_traj

if __name__ == "__main__":
    # Hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    print("Model initialized.")

    print("Loading autoencoder...")
    autoencoder = autoencoder.PointCloudAutoencoder()
    print("Autoencoder loaded.")
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    print("Loading autoencoder from:", checkpoint_path)
    autoencoder.load_state_dict(checkpoint['model_state_dict'])
    autoencoder.to(device)
    
    #Sampling
    # emd_sampled, rec_sampled = euclidean_euler_sampling(
    #     model,
    #     autoencoder,
    #     device,
    #     num_samples=num_samples,
    #     num_steps=250,
    # )
    checkpoints = [
        # "results_ddpm/ddpm_best_step_547.pth",
        #"results_ddpm/ddpm_best_step_1982.pth",
        "results_fl/steps/best_matching_unet_3.pth",
        "results_fl/steps/best_matching_unet_6.pth",
        "results_fl/steps/best_matching_unet_10.pth"
        "results_fl/steps/best_matching_unet_16.pth",
        "results_fl/steps/best_matching_unet_28.pth",
        "results_fl/steps/best_matching_unet_34.pth",
        "results_fl/steps/best_matching_unet_41.pth"

        # "results_fl/steps/best_matching_unet_190.pth",       # Early
        # "results_fl/steps/best_matching_unet_378.pth",      # Mid
        # "results_fl/steps/best_matching_unet_824.pth"      # Late
    ]

    num_samples = 50  # number of samples per checkpoint

    save_root = "results_fl/sampled_checkpoints"

    os.makedirs(save_root, exist_ok=True)
    
    for ckpt_path in checkpoints:
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint {ckpt_path} not found, skipping.")
            continue

        print(f"Loading checkpoint: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path))
        model.to(device)
        emd_sampled, rec_sampled = sample(
            device,
            model,
            autoencoder,
            num_samples=50,
            steps=250,
        )
        ckpt_name = os.path.basename(ckpt_path).replace(".pth", "")
        save_dir = os.path.join(save_root, ckpt_name)
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, f"{ckpt_name}_embeddings.npy"), emd_sampled)
        np.save(os.path.join(save_dir, f"{ckpt_name}_reconstructions.npy"), rec_sampled)