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


# --- 3. The Grand Outer Loop ---
def run_cov_experiments(data_loader, device):
    y_starts, y_ends = get_fixed_image_pairs(data_loader, num_pairs=10)
    y_starts = y_starts.to(device)
    y_ends = y_ends.to(device)
    
    num_max_decoders = 3  # Train with all 3
    decoder_counts = [1, 2, 3]  # Test with subsets
    M_retrainings = 10
    
    avg_cov_euclidean = []
    avg_cov_geodesic = []
    
    for run in range(M_retrainings):
        print(f"\nTraining Model {run + 1}/{M_retrainings} with {num_max_decoders} decoders...")
        
        # 1. Train ONCE with all decoders
        model = build_ensemble_vae(num_max_decoders).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train(model, optimizer, data_loader, epochs=50, device=device)
        model.eval()
        
        # 2. Get latent codes (same for all subset tests)
        with torch.no_grad():
            x_starts = model.encoder(y_starts).mean
            x_ends = model.encoder(y_ends).mean
        
        # 3. Test each decoder count by holding out decoders
        for num_decoders in decoder_counts:
            print(f"  Testing with {num_decoders} decoder(s)...")
            
            dist_euc = np.zeros((10, 1))
            dist_geo = np.zeros((10, 1))
            
            for pair_idx in range(10):
                z_start = x_starts[pair_idx]
                z_end = x_ends[pair_idx]
                
                # Euclidean
                dist_euc[pair_idx, 0] = calc_euclidean_dist(z_start, z_end)
                
                # Geodesic (only use first num_decoders)
                optimized_curve = optimize_geodesic_ensemble(
                    z_start, z_end, model, num_decoders, num_nodes=50, num_steps=500, lr=0.01
                )
                dist_geo[pair_idx, 0] = calc_geodesic_dist(optimized_curve, model.decoder, num_decoders)
            
            # Store results
            if run == 0:
                avg_cov_euclidean.append(np.zeros(M_retrainings))
                avg_cov_geodesic.append(np.zeros(M_retrainings))
            
            idx = decoder_counts.index(num_decoders)
            avg_cov_euclidean[idx][run] = np.mean(np.std(dist_euc, axis=1) / np.mean(dist_euc, axis=1))
            avg_cov_geodesic[idx][run] = np.mean(np.std(dist_geo, axis=1) / np.mean(dist_geo, axis=1))
    
    # Average CoV across the M_retrainings
    final_cov_euc = [np.mean(cov) for cov in avg_cov_euclidean]
    final_cov_geo = [np.mean(cov) for cov in avg_cov_geodesic]
    
    for i, num_dec in enumerate(decoder_counts):
        print(f"{num_dec} Decoders: Euclidean CoV = {final_cov_euc[i]:.4f}, Geodesic CoV = {final_cov_geo[i]:.4f}")
        with open("cov_results.txt", "a") as f:
            f.write(f"{num_dec} Decoders: Euclidean CoV = {final_cov_euc[i]:.4f}, Geodesic CoV = {final_cov_geo[i]:.4f}\n")
    
    return decoder_counts, final_cov_euc, final_cov_geo


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
