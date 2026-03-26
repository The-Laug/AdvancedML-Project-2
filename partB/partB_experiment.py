import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from ensambleVAE import EnsembleGaussianDecoder, GaussianEncoder, VAE, GaussianPrior, decoder_nets, encoder_net, latent_dim, data_loader, num_decoders, train
from partB import  optimize_geodesic_ensemble
    
# --- 1. Get your FIXED Image Pairs (Run this ONCE outside the loops) ---
def get_fixed_image_pairs(data_loader, num_pairs=10):
    """Grabs random pairs of raw images to use across all models."""
    images = []
    for x, _ in data_loader:
        images.append(x)
        if sum(len(b) for b in images) >= num_pairs * 2:
            break
            
    images = torch.cat(images, dim=0)
    
    # Randomly select indices
    indices = torch.randperm(images.shape[0])[:num_pairs * 2]
    selected_images = images[indices]
    
    y_starts = selected_images[:num_pairs]
    y_ends = selected_images[num_pairs:2*num_pairs]
    return y_starts, y_ends

# --- 2. Calculate Distance Functions ---
def calc_euclidean_dist(z_start, z_end):
    """Standard L2 distance in the 2D latent space."""
    return torch.norm(z_start - z_end, p=2).item()

def calc_geodesic_dist(optimized_curve, ensemble_decoder, num_decoders, mc_samples=10):
    """
    Distance is the length of the curve. Under the pull-back metric, 
    this is the sum of the distances between consecutive decoded points.
    """
    N = optimized_curve.shape[0]
    all_decoded = torch.stack([ensemble_decoder(optimized_curve, idx=m).mean for m in range(num_decoders)])
    # all_decoded = ensemble_decoder(optimized_curve)  # Shape: (num_decoders, N, 1, 28, 28)
    
    total_mc_length = 0.0
    for _ in range(mc_samples):
        L = torch.randint(0, num_decoders, (N - 1,))
        K = torch.randint(0, num_decoders, (N - 1,))
        
        segment_starts = all_decoded[L, torch.arange(N - 1)] 
        segment_ends = all_decoded[K, torch.arange(1, N)]    
        
        # Distance (length), NOT energy (squared). 
        # So we use sqrt of the sum of squared differences for each segment.
        segment_lengths = torch.sqrt(torch.sum((segment_starts - segment_ends) ** 2, dim=(1,2,3)))
        total_mc_length += torch.sum(segment_lengths).item()
        
    return total_mc_length / mc_samples

def decoder_net():
    # A simple MLP decoder architecture. 
    return torch.nn.Sequential(
        nn.Linear(latent_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 784),
        nn.Sigmoid(), 
        nn.Unflatten(-1, (1, 28, 28)) 
    )

def build_ensemble_vae(num_decoders):
    # Build a BRAND NEW VAE with `num_decoders` decoders
    # For simplicity, we can just reuse the same decoder architecture but create new instances.
    new_decoder_nets = [decoder_net() for _ in range(num_decoders)]
    ensemble_decoder = EnsembleGaussianDecoder(new_decoder_nets)
    encoder = GaussianEncoder(encoder_net)
    model = VAE(GaussianPrior(latent_dim), ensemble_decoder, encoder)
    return model

# --- 3. The Grand Outer Loop (Optimized) ---
def run_cov_experiments(data_loader, device):
    y_starts, y_ends = get_fixed_image_pairs(data_loader, num_pairs=10)
    y_starts = y_starts.to(device)
    y_ends = y_ends.to(device)
    
    decoder_counts = [1, 2, 3] # As required by the project
    M_retrainings = 10         # Number of VAE retrainings
    
    # Use dictionaries to store the matrices for 1, 2, and 3 decoders
    dist_euc = {1: np.zeros((10, M_retrainings)), 2: np.zeros((10, M_retrainings)), 3: np.zeros((10, M_retrainings))}
    dist_geo = {1: np.zeros((10, M_retrainings)), 2: np.zeros((10, M_retrainings)), 3: np.zeros((10, M_retrainings))}
    
    print(f"\n=== Starting Optimized Experiments ({M_retrainings} Total Trainings) ===")
    
    for run in range(M_retrainings):
        print(f"\n  Training VAE {run + 1}/{M_retrainings} (with 3 Decoders)...")
        
        # 1. Initialize and train a VAE with ALL 3 decoders
        model = build_ensemble_vae(num_decoders=3).to(device) 
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train(model, optimizer, data_loader, epochs=50, device=device) 
        
        model.eval()
        
        # 2. Find the latent coordinates for our fixed images
        with torch.no_grad():
            x_starts = model.encoder(y_starts).mean 
            x_ends = model.encoder(y_ends).mean 
            
        # 3. Calculate distances for all 10 pairs on THIS model
        for pair_idx in tqdm(range(10), desc=f"Optimizing pairs for Run {run + 1}"):
            z_start = x_starts[pair_idx]
            z_end = x_ends[pair_idx]
            
            # The Euclidean distance only relies on the encoder, so it is 
            # identical for 1, 2, and 3 decoders on this specific model run.
            euc_distance = calc_euclidean_dist(z_start, z_end)
            
            for num_decoders in decoder_counts:
                dist_euc[num_decoders][pair_idx, run] = euc_distance
                
                # By passing `num_decoders` here, your calc_energy_ensemble and calc_geodesic_dist 
                # functions will automatically only loop up to that index (e.g., only using idx=0 for num_decoders=1),
                # perfectly simulating a smaller ensemble!
                optimized_curve = optimize_geodesic_ensemble(
                    z_start, z_end, model, num_decoders=num_decoders, num_nodes=50, num_steps=500, lr=0.01
                )
                dist_geo[num_decoders][pair_idx, run] = calc_geodesic_dist(optimized_curve, model.decoder, num_decoders)
                
    # --- Calculate Final CoV ---
    avg_cov_euclidean = []
    avg_cov_geodesic = []
    
    print("\n=== Final Results ===")
    for num_decoders in decoder_counts:
        # Equation 2: CoV = std / mean
        cov_euc_per_pair = np.std(dist_euc[num_decoders], axis=1) / np.mean(dist_euc[num_decoders], axis=1)
        cov_geo_per_pair = np.std(dist_geo[num_decoders], axis=1) / np.mean(dist_geo[num_decoders], axis=1)
        
        avg_cov_euclidean.append(np.mean(cov_euc_per_pair))
        avg_cov_geodesic.append(np.mean(cov_geo_per_pair))
        
        print(f"  {num_decoders} Decoder(s):")
        print(f"  -> Avg Euclidean CoV: {avg_cov_euclidean[-1]:.4f}")
        print(f"  -> Avg Geodesic CoV:  {avg_cov_geodesic[-1]:.4f}")
        
        with open("cov_results.txt", "a") as f:
            f.write(f"{num_decoders} Decoders: Avg Euclidean CoV = {avg_cov_euclidean[-1]:.4f}, Avg Geodesic CoV = {avg_cov_geodesic[-1]:.4f}\n")

    return decoder_counts, avg_cov_euclidean, avg_cov_geodesic


def plot_cov_results(decoder_counts, avg_cov_euclidean, avg_cov_geodesic):
    plt.figure(figsize=(8, 6))
    plt.plot(decoder_counts, avg_cov_euclidean, marker='o', label='Euclidean Distance', linestyle='--')
    plt.plot(decoder_counts, avg_cov_geodesic, marker='s', label='Geodesic Distance', linewidth=2)
    
    plt.title("Reliability of Distances (Coefficient of Variation)")
    plt.xlabel("Number of Ensemble Decoders")
    plt.ylabel("Average CoV (Lower is more reliable)")
    plt.xticks(decoder_counts)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save it for your report!
    plt.savefig("cov_results.pdf", bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    decoder_counts, avg_cov_euclidean, avg_cov_geodesic = run_cov_experiments(data_loader, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    plot_cov_results(decoder_counts, avg_cov_euclidean, avg_cov_geodesic)
