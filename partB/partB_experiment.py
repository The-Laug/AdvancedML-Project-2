import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from ensambleVAE import EnsembleGaussianDecoder, GaussianEncoder, VAE, GaussianPrior, decoder_nets, encoder_net, latent_dim, data_loader, num_decoders, train
from partB import  optimize_geodesic_ensemble
    
# --- 1. Get your FIXED Image Pairs (Run this ONCE outside the loops) ---
def get_fixed_image_pairs(data_loader, num_pairs=10):
    """Grabs 10 fixed pairs of raw images to use across all models."""
    images = []
    for x, _ in data_loader:
        images.append(x)
        if sum(len(b) for b in images) >= num_pairs * 2:
            break
            
    images = torch.cat(images, dim=0)
    
    # Select our fixed y_i and y_j images
    y_starts = images[:num_pairs]
    y_ends = images[num_pairs:2*num_pairs]
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
    # A simple MLP decoder architecture. You can customize this as needed.
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
    
    decoder_counts = [1, 2, 3] # As required by the project
    M_retrainings = 10         # Number of VAE retrainings per count
    
    # Dictionaries to store the average CoV across the 10 pairs for each decoder count
    avg_cov_euclidean = []
    avg_cov_geodesic = []
    
    for num_decoders in decoder_counts:
        print(f"\n=== Starting experiments for {num_decoders} Decoder(s) ===")
        
        # Arrays to hold distances for all 10 pairs across all 10 runs
        # Shape: (num_pairs, num_runs) -> (10, 10)
        dist_euc = np.zeros((10, M_retrainings))
        dist_geo = np.zeros((10, M_retrainings))
        
        for run in range(M_retrainings):
            print(f"  Training Model {run + 1}/{M_retrainings}...")
            
            # 1. Initialize a BRAND NEW VAE with `num_decoders` decoders
            model = build_ensemble_vae(num_decoders).to(device) 
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            # 2. Train the model (e.g., 50-100 epochs)
            # Make sure you are using your ensemble ELBO function here!
            train(model, optimizer, data_loader, epochs=100, device=device) 
            
            model.eval()
            
            # 3. Find the current latent coordinates for our fixed images
            with torch.no_grad():
                x_starts = model.encoder(y_starts).mean 
                x_ends = model.encoder(y_ends).mean 
                
            # 4. Calculate distances for all 10 pairs on THIS model
            for pair_idx in range(10):
                z_start = x_starts[pair_idx]
                z_end = x_ends[pair_idx]
                
                # Euclidean
                dist_euc[pair_idx, run] = calc_euclidean_dist(z_start, z_end)
                
                # Geodesic
                # Re-use the optimization function we built earlier!
                optimized_curve = optimize_geodesic_ensemble(
                    z_start, z_end, model, num_decoders, num_nodes=50, num_steps=500, lr=0.01
                )
                dist_geo[pair_idx, run] = calc_geodesic_dist(optimized_curve, model.decoder, num_decoders)
                
        # --- Calculate CoV for this number of decoders ---
        # Equation: CoV = std / mean
        # We calculate CoV for each point pair across the 10 models, then average them.
        cov_euc_per_pair = np.std(dist_euc, axis=1) / np.mean(dist_euc, axis=1)
        cov_geo_per_pair = np.std(dist_geo, axis=1) / np.mean(dist_geo, axis=1)
        
        avg_cov_euclidean.append(np.mean(cov_euc_per_pair))
        avg_cov_geodesic.append(np.mean(cov_geo_per_pair))
        print(f"  -> Avg Euclidean CoV: {avg_cov_euclidean[-1]:.4f}")
        print(f"  -> Avg Geodesic CoV:  {avg_cov_geodesic[-1]:.4f}")

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
