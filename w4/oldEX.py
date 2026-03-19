
import torch


def latent_curve_length_from_points(points):
     """
     Compute the Euclidean length of a curve in latent space from sampled points.

     Parameters:
     points: [torch.Tensor]
         Tensor of shape `(N, M)` where `N` is the number of points and `M` is
         latent dimension.

     Returns:
     length: [torch.Tensor]
         Scalar tensor with the curve length.
     """
     if points.ndim != 2:
          raise ValueError("`points` must have shape (N, M)")
     if points.shape[0] < 2:
          raise ValueError("`points` must contain at least 2 points")

     deltas = points[1:] - points[:-1]
     segment_lengths = torch.linalg.norm(deltas, dim=-1)
     return segment_lengths.sum()


def latent_curve_length(curve_fn, t_start=0.0, t_end=1.0, num_points=1000, device='cpu'):
     """
     Compute the Euclidean length of a parametric latent-space curve.

     Parameters:
     curve_fn: [callable]
         Function mapping `t` with shape `(N,)` to latent points of shape `(N, M)`.
     t_start: [float]
         Start of parameter interval.
     t_end: [float]
         End of parameter interval.
     num_points: [int]
         Number of sampling points along the curve.
     device: [str or torch.device]
         Device used for the computation.

     Returns:
     length: [torch.Tensor]
         Scalar tensor with the curve length.
     """
     if num_points < 2:
          raise ValueError("`num_points` must be at least 2")

     t = torch.linspace(t_start, t_end, num_points, device=device)
     points = curve_fn(t)

     if not isinstance(points, torch.Tensor):
          points = torch.as_tensor(points, device=device)

     return latent_curve_length_from_points(points)
 
 
 
def make_latent_curve_fn(curve_type, start, end, control, latent_dim, device='cpu'):
    """
    Build a parametric curve function `curve_fn(t)` in latent space.

    The curve is defined in the first two latent dimensions and padded with
    zeros for remaining dimensions.
    """
    if latent_dim < 2:
        raise ValueError("latent_dim must be at least 2 to define a 2D curve")

    start_xy = torch.tensor(start, dtype=torch.float32, device=device)
    end_xy = torch.tensor(end, dtype=torch.float32, device=device)

    if control is None:
        midpoint = 0.5 * (start_xy + end_xy)
        direction = end_xy - start_xy
        perp = torch.tensor([-direction[1], direction[0]], device=device)
        perp_norm = torch.linalg.norm(perp)
        if perp_norm > 0:
            perp = perp / perp_norm
        control_xy = midpoint + 0.25 * torch.linalg.norm(direction) * perp
    else:
        control_xy = torch.tensor(control, dtype=torch.float32, device=device)

    def embed_xy(xy):
        n_points = xy.shape[0]
        z = torch.zeros((n_points, latent_dim), dtype=xy.dtype, device=xy.device)
        z[:, :2] = xy
        return z

    if curve_type == 'line':
        def curve_fn(t):
            t = t.reshape(-1, 1)
            xy = (1 - t) * start_xy + t * end_xy
            return embed_xy(xy)
    elif curve_type == 'quadratic':
        def curve_fn(t):
            t = t.reshape(-1, 1)
            xy = (1 - t)**2 * start_xy + 2 * (1 - t) * t * control_xy + t**2 * end_xy
            return embed_xy(xy)
    else:
        raise ValueError(f"Unsupported curve_type: {curve_type}")

    return curve_fn






def latent_curve_energy(curve_fn, t_start=0.0, t_end=1.0, num_points=1000, device='cpu'):
    """
    Compute the energy of a parametric latent-space curve.

    Energy is defined as the integral of the squared speed along the curve.
    """
    if num_points < 2:
        raise ValueError("`num_points` must be at least 2")

    t = torch.linspace(t_start, t_end, num_points, device=device)
    points = curve_fn(t)

    if not isinstance(points, torch.Tensor):
        points = torch.as_tensor(points, device=device)

    deltas = points[1:] - points[:-1]
    segment_lengths_squared = torch.sum(deltas**2, dim=-1)
    energy = segment_lengths_squared.sum() * (t_end - t_start) / (num_points - 1)
    return energy







def plot_curve(curve_fn, t_start=0.0, t_end=1.0, num_points=1000):
    import matplotlib.pyplot as plt

    t = torch.linspace(t_start, t_end, num_points)
    points = curve_fn(t).cpu().numpy()
    plt.plot(points[:, 0], points[:, 1])
    plt.scatter(points[0, 0], points[0, 1], color='green', label='Start')
    plt.scatter(points[-1, 0], points[-1, 1], color='red', label='End')
    plt.title('Latent Curve')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.legend()
    plt.axis('equal')
    plt.show()
    

if __name__ == "__main__":
    # Example usage
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    curve_fn = make_latent_curve_fn(
        curve_type='quadratic',
        start=[0.0, 0.0],
        end=[1.0, 1.0],
        control=[0.5, 1.5],
        latent_dim=10,
        device=device
    )
    length = latent_curve_length(curve_fn, num_points=10000, device=device)
    print(f"Curve length: {length.item():.4f}")
    plot_curve(curve_fn)
    