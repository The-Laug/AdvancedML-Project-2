# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.2 (2024-02-06)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torch.nn import functional as F
from tqdm import tqdm


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


def decode_mean_from_latent(model, latent_points):
     """
     Decode latent points to mean images (Bernoulli probabilities).

     Parameters:
     model: [VAE]
         Trained VAE model.
     latent_points: [torch.Tensor]
         Tensor of shape `(N, M)`.

     Returns:
     decoded: [torch.Tensor]
         Tensor of shape `(N, D)` where D is flattened data dimension.
     """
     logits = model.decoder.decoder_net(latent_points)
     probs = torch.sigmoid(logits)
     return probs.reshape(probs.shape[0], -1)


def data_curve_length_from_latent_points(model, latent_points):
     """
     Compute curve length in data space for a latent-space polyline.

     Parameters:
     model: [VAE]
         Trained VAE model.
     latent_points: [torch.Tensor]
         Tensor of shape `(N, M)`.

     Returns:
     length: [torch.Tensor]
         Scalar tensor with data-space curve length.
     """
     decoded = decode_mean_from_latent(model, latent_points)
     deltas = decoded[1:] - decoded[:-1]
     segment_lengths = torch.linalg.norm(deltas, dim=-1)
     return segment_lengths.sum()


def optimize_data_geodesic(model, z_start, z_end, num_segments=32, iters=300, lr=1e-2, latent_reg=1e-4):
     """
     Optimize an approximate geodesic in latent space under data-space length.

     The objective is:
          data_length(path) + latent_reg * latent_length(path)
     with fixed endpoints z_start and z_end.
     """
     if num_segments < 2:
          raise ValueError("`num_segments` must be at least 2")

     t = torch.linspace(0.0, 1.0, num_segments + 1, device=z_start.device).reshape(-1, 1)
     initial_path = (1 - t) * z_start.reshape(1, -1) + t * z_end.reshape(1, -1)

     interior = nn.Parameter(initial_path[1:-1].clone())
     optimizer = torch.optim.Adam([interior], lr=lr)

     best_loss = float('inf')
     best_path = initial_path.detach().clone()

     for _ in range(iters):
          optimizer.zero_grad()
          path = torch.cat([z_start.reshape(1, -1), interior, z_end.reshape(1, -1)], dim=0)

          data_length = data_curve_length_from_latent_points(model, path)
          latent_length = latent_curve_length_from_points(path)
          loss = data_length + latent_reg * latent_length

          loss.backward()
          optimizer.step()

          if loss.item() < best_loss:
                best_loss = loss.item()
                best_path = path.detach().clone()

     best_data_length = data_curve_length_from_latent_points(model, best_path).item()
     best_latent_length = latent_curve_length_from_points(best_path).item()
     return best_path, best_data_length, best_latent_length


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


class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int] 
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)


class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]             
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)


class BernoulliDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters: 
        encoder_net: [torch.nn.Module]             
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(BernoulliDecoder, self).__init__()
        self.decoder_net = decoder_net
        self.std = nn.Parameter(torch.ones(28, 28)*0.5, requires_grad=True)

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor] 
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        logits = self.decoder_net(z)
        return td.Independent(td.Bernoulli(logits=logits), 2)


class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """
    def __init__(self, prior, decoder, encoder):
        """
        Parameters:
        prior: [torch.nn.Module] 
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """
            
        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

    def elbo(self, x):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """
        q = self.encoder(x)
        z = q.rsample()
        elbo = torch.mean(self.decoder(z).log_prob(x) - td.kl_divergence(q, self.prior()), dim=0)
        return elbo

    def sample(self, n_samples=1):
        """
        Sample from the model.
        
        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()
    
    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x)


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """
    model.train()

    total_steps = len(data_loader)*epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    for epoch in range(epochs):
        data_iter = iter(data_loader)
        for x in data_iter:
            x = x[0].to(device)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()


if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image, make_grid
    import glob

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'eval','plot-latent', 'plot-latent-curve', 'plot-geodesic'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=32, metavar='N', help='dimension of latent variable (default: %(default)s)')
    parser.add_argument('--curve-type', type=str, default='quadratic', choices=['line', 'quadratic'], help='curve type for plot-latent-curve mode (default: %(default)s)')
    parser.add_argument('--curve-start', type=float, nargs=2, default=[-2.0, -2.0], metavar=('X', 'Y'), help='start point (x y) in latent plane for plot-latent-curve mode')
    parser.add_argument('--curve-end', type=float, nargs=2, default=[2.0, 2.0], metavar=('X', 'Y'), help='end point (x y) in latent plane for plot-latent-curve mode')
    parser.add_argument('--curve-control', type=float, nargs=2, default=None, metavar=('X', 'Y'), help='control point (x y) for quadratic curve; if omitted, auto-generated')
    parser.add_argument('--curve-points', type=int, default=400, metavar='N', help='number of samples along the curve for plotting/length (default: %(default)s)')
    parser.add_argument('--geo-start', type=float, nargs=2, default=[-2.0, -2.0], metavar=('X', 'Y'), help='start point (x y) for geodesic optimization in latent plane')
    parser.add_argument('--geo-end', type=float, nargs=2, default=[2.0, 2.0], metavar=('X', 'Y'), help='end point (x y) for geodesic optimization in latent plane')
    parser.add_argument('--geo-segments', type=int, default=32, metavar='N', help='number of segments for geodesic polyline (default: %(default)s)')
    parser.add_argument('--geo-iters', type=int, default=300, metavar='N', help='optimization iterations for geodesic (default: %(default)s)')
    parser.add_argument('--geo-lr', type=float, default=1e-2, help='learning rate for geodesic optimization (default: %(default)s)')
    parser.add_argument('--geo-latent-reg', type=float, default=1e-4, help='small latent-length regularizer for geodesic optimization (default: %(default)s)')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device

    # Load MNIST as binarized at 'thresshold' and create data loaders
    thresshold = 0.5
    mnist_train_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=True, download=True,
                                                                    transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                                    batch_size=args.batch_size, shuffle=True)
    mnist_test_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=False, download=True,
                                                                transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                                    batch_size=args.batch_size, shuffle=True)

    # Define prior distribution
    M = args.latent_dim
    prior = GaussianPrior(M)

    # Define encoder and decoder networks
    encoder_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, M*2),
    )

    decoder_net = nn.Sequential(
        nn.Linear(M, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 784),
        nn.Unflatten(-1, (28, 28))
    )

    # Define VAE model
    decoder = BernoulliDecoder(decoder_net)
    encoder = GaussianEncoder(encoder_net)
    model = VAE(prior, decoder, encoder).to(device)

    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train model
        train(model, optimizer, mnist_train_loader, args.epochs, args.device)

        # Save model
        torch.save(model.state_dict(), args.model)

    elif args.mode == 'sample':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        # Generate samples
        model.eval()
        with torch.no_grad():
            samples = (model.sample(64)).cpu() 
            save_image(samples.view(64, 1, 28, 28), args.samples)
    
    elif args.mode == 'eval':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        # Evaluate model
        model.eval()
        with torch.no_grad():
            elbo = 0
            for x in mnist_test_loader:
                x = x[0].to(device)
                elbo += model.elbo(x).item()
    elif args.mode == 'plot-latent':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        # Plot latent space
        model.eval()
        with torch.no_grad():
            z = []
            labels = []
            for x, y in mnist_test_loader:
                x = x.to(device)
                z.append(model.encoder(x).mean.cpu())
                labels.append(y)
            z = torch.cat(z)
            labels = torch.cat(labels)

        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 10))
        scatter = plt.scatter(z[:, 0], z[:, 1], c=labels, cmap='tab10', alpha=0.5)
        plt.colorbar(scatter, ticks=range(10))
        plt.xlabel('z[0]')
        plt.ylabel('z[1]')
        plt.title('Latent Space')
        plt.show()

    elif args.mode == 'plot-latent-curve':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        if M < 2:
            raise ValueError("plot-latent-curve requires --latent-dim >= 2")

        model.eval()
        with torch.no_grad():
            z = []
            labels = []
            for x, y in mnist_test_loader:
                x = x.to(device)
                z.append(model.encoder(x).mean.cpu())
                labels.append(y)
            z = torch.cat(z)
            labels = torch.cat(labels)

        curve_fn = make_latent_curve_fn(
            curve_type=args.curve_type,
            start=args.curve_start,
            end=args.curve_end,
            control=args.curve_control,
            latent_dim=M,
            device=device,
        )
        curve_length = latent_curve_length(curve_fn, t_start=0.0, t_end=1.0, num_points=args.curve_points, device=device).item()

        t_plot = torch.linspace(0.0, 1.0, args.curve_points, device=device)
        curve_points = curve_fn(t_plot).detach().cpu()

        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 10))
        scatter = plt.scatter(z[:, 0], z[:, 1], c=labels, cmap='tab10', alpha=0.5)
        plt.plot(curve_points[:, 0], curve_points[:, 1], color='red', linewidth=2.5, label=f'curve length = {curve_length:.4f}')
        plt.colorbar(scatter, ticks=range(10))
        plt.xlabel('z[0]')
        plt.ylabel('z[1]')
        plt.title('Latent Space with Curve')
        plt.legend(loc='upper right')
        print(f'Curve length: {curve_length:.6f}')
        plt.show()

    elif args.mode == 'plot-geodesic':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        if M < 2:
            raise ValueError("plot-geodesic requires --latent-dim >= 2")

        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)

        with torch.no_grad():
            z = []
            labels = []
            for x, y in mnist_test_loader:
                x = x.to(device)
                z.append(model.encoder(x).mean.cpu())
                labels.append(y)
            z = torch.cat(z)
            labels = torch.cat(labels)

        z_start = torch.zeros(M, dtype=torch.float32, device=device)
        z_end = torch.zeros(M, dtype=torch.float32, device=device)
        z_start[:2] = torch.tensor(args.geo_start, dtype=torch.float32, device=device)
        z_end[:2] = torch.tensor(args.geo_end, dtype=torch.float32, device=device)

        geodesic_path, geodesic_data_length, geodesic_latent_length = optimize_data_geodesic(
            model=model,
            z_start=z_start,
            z_end=z_end,
            num_segments=args.geo_segments,
            iters=args.geo_iters,
            lr=args.geo_lr,
            latent_reg=args.geo_latent_reg,
        )

        straight_t = torch.linspace(0.0, 1.0, args.geo_segments + 1, device=device).reshape(-1, 1)
        straight_path = (1 - straight_t) * z_start.reshape(1, -1) + straight_t * z_end.reshape(1, -1)
        straight_data_length = data_curve_length_from_latent_points(model, straight_path).item()

        geodesic_path_cpu = geodesic_path.detach().cpu()

        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 10))
        scatter = plt.scatter(z[:, 0], z[:, 1], c=labels, cmap='tab10', alpha=0.5)
        plt.plot(
            geodesic_path_cpu[:, 0],
            geodesic_path_cpu[:, 1],
            color='red',
            linewidth=2.5,
            label=f'geodesic data-length = {geodesic_data_length:.4f}'
        )
        plt.scatter([args.geo_start[0], args.geo_end[0]], [args.geo_start[1], args.geo_end[1]], color='black', s=80, marker='x', label='endpoints')
        plt.colorbar(scatter, ticks=range(10))
        plt.xlabel('z[0]')
        plt.ylabel('z[1]')
        plt.title('Latent Space with Data-Space Geodesic')
        plt.legend(loc='upper right')

        print(f'Geodesic distance (data space): {geodesic_data_length:.6f}')
        print(f'Straight path distance (data space): {straight_data_length:.6f}')
        print(f'Optimized path length (latent space): {geodesic_latent_length:.6f}')
        plt.show()
                