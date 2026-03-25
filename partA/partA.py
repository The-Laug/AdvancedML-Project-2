


import tqdm
from gaussVAE import GaussianDecoder, GaussianEncoder, VAE, GaussianPrior, decoder_net, encoder_net, latent_dim, data_loader
import torch
import torch.nn as nn





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

def calc_energy(curve, decoder):
    """
    Calculates the energy of a discrete curve under the pull-back metric.

    Args:
        curve (torch.Tensor): Tensor of shape (N, 2) representing N points in 2D space.
        decoder (torch.nn.Module): The VAE decoder network.

    Returns:
        torch.Tensor: Scalar tensor representing the total energy of the curve.
    """
    # 1. Decode all points on the curve to get the sequence of images.
    # decoder(curve) returns a Distribution. We use .mean to get the actual image pixels.
    # Shape of decoded_images: (N, 1, 28, 28)
    decoded_images = decoder(curve).mean
    
    # 2. Calculate the difference between consecutive decoded images.
    # Shape of diff: (N-1, 1, 28, 28)
    diff = decoded_images[1:] - decoded_images[:-1]
    
    # 3. Square the differences and sum them all up.
    # This perfectly calculates the energy under the pull-back metric!
    total_energy = torch.sum(diff ** 2)
    
    return total_energy



def train_curve(start, end, num_points, num_iterations, learning_rate,decoder):
    base_curve, interior = parameterize_curve(start, end, num_points)

    # If there are interior parameters, optimize them; otherwise nothing to do
    if interior is not None:
        optimizer = torch.optim.Adam([interior], lr=learning_rate)
    else:
        optimizer = None

    for iteration in range(num_iterations):
        if optimizer is not None:
            optimizer.zero_grad()

        # assemble full curve: start, interior (if any), end
        if interior is None:
            curve = base_curve
        else:
            curve = torch.vstack((start.unsqueeze(0), interior, end.unsqueeze(0)))

        energy = calc_energy(curve, decoder)
        # if there's nothing to optimize, just compute and break
        if optimizer is None:
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Energy: {energy.item()}")
            continue

        energy.backward()
        optimizer.step()

        if iteration % 10 == 0:
            print(f"Iteration {iteration}, Energy: {energy.item()}")

    # return the final assembled curve
    if interior is None:
        return base_curve
    else:
        return torch.vstack((start.unsqueeze(0), interior.detach(), end.unsqueeze(0)))
    

import matplotlib.pyplot as plt
import torch

def plot_latent_geodesics(model, data_loader, optimized_curves):
    """
    Plots the 2D latent space and the computed geodesics.
    
    Args:
        model: Your trained VAE model.
        data_loader: The DataLoader containing your MNIST subset.
        optimized_curves: A list of 25 tensors, each shape (N, 2), representing your geodesics.
    """
    model.eval()
    all_z = []
    all_labels = []

    # 1. Encode all data to create the background map
    with torch.no_grad():
        for x, y in data_loader:
            # Get the mean of the encoder's distribution for the true locations
            z_mean = model.encoder(x).mean 
            all_z.append(z_mean)
            all_labels.append(y)

    all_z = torch.cat(all_z, dim=0).cpu().numpy()
    all_labels = torch.cat(all_labels, dim=0).cpu().numpy()

    plt.figure(figsize=(10, 8))
    
    # 2. Scatter plot the latent codes, colored by digit class
    scatter = plt.scatter(all_z[:, 0], all_z[:, 1], c=all_labels, cmap='tab10', alpha=0.6, s=15)
    #label the colors
    # Extract the distinct color handles from the scatter object
    handles, _ = scatter.legend_elements()
    
    # Create a clean, categorical legend
    plt.legend(handles, ['Digit 0', 'Digit 1', 'Digit 2'], title="MNIST Classes")
    # plt.colorbar(scatter, label="MNIST Digit Class")

    # 3. Overlay the geodesics
    # Different color for each curve, no transparency by default (set alpha_value < 1.0 for more transparency)
    alpha_value = 0.8  # set to e.g. 0.4 if you want more transparent curves
    cmap = plt.get_cmap('tab20', len(optimized_curves))
    for i, curve in enumerate(optimized_curves):
        curve_np = curve.detach().cpu().numpy()
        color = cmap(i)  # distinct color per curve

        # Plot the curved path (alpha controlled by alpha_value)
        plt.plot(curve_np[:, 0], curve_np[:, 1], color=color, linewidth=1.5, alpha=alpha_value)

        # Mark the start and end points, keep them visible on top
        plt.scatter([curve_np[0, 0], curve_np[-1, 0]],
                    [curve_np[0, 1], curve_np[-1, 1]],
                    facecolors=[color, color],
                    edgecolors='black',
                    s=40,
                    zorder=5,
                    alpha=alpha_value)

    plt.title("VAE Latent Space with Pull-back Geodesics")
    plt.xlabel("Latent Dimension 1 (z1)")
    plt.ylabel("Latent Dimension 2 (z2)")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig("latent_geodesics.png", dpi=150, bbox_inches='tight')
    plt.show()


def get_random_latent_pairs(model, data_loader, num_pairs=25, device='cpu'):
    """Encodes the data and selects random pairs of latent points."""
    model.eval()
    z_points = []
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            # Use the mean of the encoder to get deterministic points
            z = model.encoder(x).mean 
            z_points.append(z)
            if sum(len(b) for b in z_points) >= num_pairs * 2:
                break
                
    z_points = torch.cat(z_points, dim=0)
    
    # Shuffle the indices to get random pairs
    indices = torch.randperm(len(z_points))
    start_points = z_points[indices[:num_pairs]]
    end_points = z_points[indices[num_pairs:2*num_pairs]]
    
    return start_points, end_points

def optimize_geodesic(start_point, end_point, model, num_nodes=50, num_steps=1000, lr=0.01):
    """
    Finds the geodesic between two points by minimizing curve energy.
    """
    # 1. Initialize the curve as a straight Euclidean line
    alphas = torch.linspace(0, 1, num_nodes).unsqueeze(1).to(start_point.device)
    initial_curve = (1 - alphas) * start_point + alphas * end_point
    
    # 2. Extract internal nodes and make them optimizable
    # We DO NOT optimize the first and last points, they must stay fixed!
    internal_nodes = initial_curve[1:-1].clone().detach().requires_grad_(True)
    
    # 3. Setup the optimizer (Adam is usually a good starting point)
    optimizer = torch.optim.Adam([internal_nodes], lr=lr)
    
    model.eval() # Ensure the VAE weights are frozen
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # Reconstruct the full curve with the fixed endpoints
        full_curve = torch.cat([
            start_point.unsqueeze(0), 
            internal_nodes, 
            end_point.unsqueeze(0)
        ], dim=0)
        
        # Calculate energy (using the pull-back function we wrote earlier!)
        energy = calc_energy(full_curve, model.decoder)
        
        # Backpropagate to adjust the internal nodes
        energy.backward()
        optimizer.step()
        
    # Return the final detached curve for plotting
    final_curve = torch.cat([
        start_point.unsqueeze(0), 
        internal_nodes.detach(), 
        end_point.unsqueeze(0)
    ], dim=0)
    
    return final_curve

if __name__ == "__main__":
    start = torch.tensor([1.0, -1.0])
    end = torch.tensor([1.0, 1.0])
    num_points = 50
    num_iterations = 1000
    learning_rate = 0.01
    
    #Accessing decoder for energy calculation
    model = VAE(GaussianPrior(latent_dim), GaussianDecoder(decoder_net), GaussianEncoder(encoder_net))
    model.load_state_dict(torch.load("partA/gauss_vae_good.pth", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    model.eval()
    decoder = model.decoder


    optimized_curve = train_curve(start, end, num_points, num_iterations, learning_rate, decoder)
    print("Optimized Curve:")
    print(optimized_curve)

    print("Generating plot...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    start_pts, end_pts = get_random_latent_pairs(model, data_loader, num_pairs=25, device=device)
    
    optimized_curves = []
    for i in tqdm.tqdm(range(25), desc="Optimizing Geodesics"):
        curve = optimize_geodesic(start_pts[i], end_pts[i], model, num_nodes=50, num_steps=1000, lr=0.01)
        optimized_curves.append(curve)
        
    plot_latent_geodesics(model, data_loader, optimized_curves)