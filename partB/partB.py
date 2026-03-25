
import tqdm
from ensambleVAE import EnsembleGaussianDecoder, GaussianEncoder, VAE, GaussianPrior, decoder_nets, encoder_net, latent_dim, data_loader, num_decoders
import torch
import torch.nn as nn
from partA import get_random_latent_pairs, plot_latent_geodesics



def calc_energy_ensemble(curve, ensemble_decoder, num_decoders, mc_samples=10):
    """
    Calculates the model-average curve energy using Monte Carlo approximation.
    
    Args:
        curve (torch.Tensor): Tensor of shape (N, 2) representing N points in 2D space.
        ensemble_decoder (nn.Module): The EnsembleGaussianDecoder network.
        num_decoders (int): The number of decoders in the ensemble (M).
        mc_samples (int): Number of Monte Carlo draws to approximate the expectation.
        
    Returns:
        torch.Tensor: Scalar tensor representing the approximated curve energy.
    """
    N = curve.shape[0]
    
    # 1. Decode the curve using ALL decoders in the ensemble upfront to save massive compute time.
    # We use the `idx` argument we added earlier to pick specific decoders.
    # all_decoded shape: (num_decoders, N, 1, 28, 28)
    all_decoded = torch.stack([ensemble_decoder(curve, idx=m).mean for m in range(num_decoders)])
    
    total_mc_energy = 0.0
    
    # 2. Monte Carlo approximation of the expectation
    for _ in range(mc_samples):
        # Randomly draw decoder indices l and k uniformly for each segment
        # L and K will have shape (N-1,) with integer values from 0 to num_decoders - 1
        L = torch.randint(0, num_decoders, (N - 1,), device=curve.device)
        K = torch.randint(0, num_decoders, (N - 1,), device=curve.device)
        
        # Gather the decoded images for the start (i) and end (i+1) of each segment
        # segment_starts corresponds to f_l(c(t_i))
        # segment_ends corresponds to f_k(c(t_{i+1}))
        segment_starts = all_decoded[L, torch.arange(N - 1)] 
        segment_ends = all_decoded[K, torch.arange(1, N)]    
        
        # Calculate the squared Euclidean distance for these segments
        squared_diff = (segment_starts - segment_ends) ** 2
        
        # Sum the distances across all pixels and all segments for this MC sample
        mc_energy = torch.sum(squared_diff)
        total_mc_energy += mc_energy
        
    # 3. Average the accumulated energy over the number of Monte Carlo samples
    average_energy = total_mc_energy / mc_samples
    
    return average_energy

def optimize_geodesic_ensemble(start_point, end_point, model, num_decoders, num_nodes=50, num_steps=1000, lr=0.01):
    """
    Finds the geodesic by minimizing the model-average curve energy.
    """
    alphas = torch.linspace(0, 1, num_nodes).unsqueeze(1).to(start_point.device)
    initial_curve = (1 - alphas) * start_point + alphas * end_point
    
    internal_nodes = initial_curve[1:-1].clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([internal_nodes], lr=lr)
    
    model.eval() 
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        full_curve = torch.cat([
            start_point.unsqueeze(0), 
            internal_nodes, 
            end_point.unsqueeze(0)
        ], dim=0)
        
        # KEY CHANGE: Use the ensemble energy function!
        energy = calc_energy_ensemble(full_curve, model.decoder, num_decoders, mc_samples=10)
        
        energy.backward()
        optimizer.step()
        
    final_curve = torch.cat([
        start_point.unsqueeze(0), 
        internal_nodes.detach(), 
        end_point.unsqueeze(0)
    ], dim=0)
    
    return final_curve



def parameterize_curve(start, end, num_points):
    # Create a linear space of points between start and end
    t = torch.linspace(0, 1, num_points).unsqueeze(1)  # Shape: (num_points, 1)
    curve = start + t * (end - start)  # Linear interpolation (num_points, 2)

    # We only want the interior points to be trainable parameters. Create
    # an nn.Parameter for the interior points and return it alongside a
    # helper that can assemble the full curve for optimization.
    if num_points <= 2:
        # No interior points to optimize
        return curve, None

    interior_init = curve[1:-1].clone().detach()
    interior_param = nn.Parameter(interior_init)
    return curve, interior_param


if __name__ == "__main__":
    model = VAE(GaussianPrior(latent_dim), EnsembleGaussianDecoder(decoder_nets), GaussianEncoder(encoder_net))
    model.load_state_dict(torch.load("partB/ensamble_gauss_vae.pth", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    model.eval()
    
    # Example usage of calc_energy_ensemble
    start = torch.tensor([0.0, 0.0])
    end = torch.tensor([1.0, 1.0])
    num_points = 10
    # Create a column of weights from 0 to 1
    alphas = torch.linspace(0, 1, num_points).unsqueeze(1).to(start.device)

    # Linearly interpolate between the start and end coordinates
    curve = (1 - alphas) * start + alphas * end
    
    energy = calc_energy_ensemble(curve, model.decoder, num_decoders=num_decoders, mc_samples=10)
    print(f"Estimated curve energy under the ensemble pull-back metric: {energy.item()}")

    #Accessing decoder for energy calculation
    model = VAE(GaussianPrior(latent_dim), EnsembleGaussianDecoder(decoder_nets), GaussianEncoder(encoder_net))
    model.load_state_dict(torch.load("partB/ensamble_gauss_vae.pth", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    model.eval()
    decoder = model.decoder
    
    print("Generating plot...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    start_pts, end_pts = get_random_latent_pairs(model, data_loader, num_pairs=25, device=device)
    
    optimized_curves = []
    for i in tqdm.tqdm(range(25), desc="Optimizing Geodesics"):
        curve = optimize_geodesic_ensemble(start_pts[i], end_pts[i], model, num_nodes=50, num_steps=1000, lr=0.01, num_decoders=num_decoders)
        optimized_curves.append(curve)
        
    plot_latent_geodesics(model, data_loader, optimized_curves)