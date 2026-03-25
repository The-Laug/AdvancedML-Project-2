# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-01-27)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py
#
# Significant extension by Søren Hauberg, 2024

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
import numpy as np
import random
from tqdm import tqdm
from copy import deepcopy
import os
import math
import matplotlib.pyplot as plt

class Curve():
    def __init__(self, points):
        self.n = len(points)
        self.start_point = points[0]
        self.end_point = points[-1]
        self.internal_points = points[1:-1].clone().detach().requires_grad_(True)

    def get_length(self):
        points = torch.stack([self[i] for i in range(self.n)], dim=0)
        diffs = points[1:] - points[:-1]
        return torch.sum(torch.norm(diffs, dim=1))

    def __getitem__(self, idx):
        if idx == 0:
            return self.start_point
        if idx == self.n - 1:
            return self.end_point
        if idx > 0 and idx < self.n - 1:
            return self.internal_points[idx-1] 
        
        raise RuntimeError("Invalid curve point requested")


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


class GaussianDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters:
        encoder_net: [torch.nn.Module]
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(GaussianDecoder, self).__init__()
        self.decoder_net = decoder_net
        # self.std = nn.Parameter(torch.ones(28, 28) * 0.5, requires_grad=True) # In case you want to learn the std of the gaussian.

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor]
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        means = self.decoder_net(z)
        return td.Independent(td.Normal(loc=means, scale=1e-1), 3)

class EnsembleVAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """

    def __init__(self, prior, decoders, encoder):
        """
        Parameters:
        prior: [torch.nn.Module]
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """

        super(EnsembleVAE, self).__init__()
        self.prior = prior
        self.decoders = nn.ModuleList(decoders)
        self.encoder = encoder

    @property
    def decoder(self):
        return random.choice(self.decoders)

    def elbo(self, x, n_samples=20):
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

        elbo = torch.mean(
            self.decoder(z).log_prob(x) - q.log_prob(z) + self.prior().log_prob(z)
        )
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

    def avg_curve_energy(self, curve, n=20):
        """
        Compute the model-average curve energy for the given curve.

        Parameters:
        curve: [torch.Tensor]
           A Curve object`
        """
        device = curve.start_point.device 
        
        points = torch.stack([curve[i] for i in range(curve.n)])
        
        starts = points[:-1]
        ends = points[1:]
        
        curve_energy = torch.zeros((), device=device)
        for _ in range(n):   # recall: self.decoder returns a RANDOM decoder
            diff = self.decoder(starts).mean - self.decoder(ends).mean
            curve_energy += diff.pow(2).sum() / n

        return curve_energy

    def optimize_geodesics(self, start, end, num_nodes=50, steps=200, lr=0.005):
        t = torch.linspace(0.0, 1.0, num_nodes, device=start.device).unsqueeze(1)
        points = start + t * (end - start)  # straight line from start to end
        curve = Curve(points)

        optimizer = torch.optim.Adam([curve.internal_points], lr=lr)

        pbar = tqdm(range(steps), desc="Optimizing geodesics...")
        for _ in pbar:
            optimizer.zero_grad()
            energy = self.avg_curve_energy(curve)
            energy.backward()
            optimizer.step()

            ### Update progress bar
            point_coords = curve.internal_points[8].detach().cpu().tolist()  # follow the progress of this random point
            formatted_coords = [round(c, 4) for c in point_coords]
            pbar.set_postfix({
                "energy": f"{energy.item():.4f}",
                "pt8": formatted_coords
            })
        
        return curve

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

    num_steps = len(data_loader) * epochs
    epoch = 0

    def noise(x, std=0.05):
        eps = std * torch.randn_like(x)
        return torch.clamp(x + eps, min=0.0, max=1.0)

    with tqdm(range(num_steps)) as pbar:
        for step in pbar:
            try:
                x = next(iter(data_loader))[0]
                x = noise(x.to(device))
                model = model
                optimizer.zero_grad()
                # from IPython import embed; embed()
                loss = model(x)
                loss.backward()
                optimizer.step()

                # Report
                if step % 5 == 0:
                    loss = loss.detach().cpu()
                    pbar.set_description(
                        f"total epochs ={epoch}, step={step}, loss={loss:.1f}"
                    )

                if (step + 1) % len(data_loader) == 0:
                    epoch += 1
            except KeyboardInterrupt:
                print(
                    f"Stopping training at total epoch {epoch} and current loss: {loss:.1f}"
                )
                break


if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        type=str,
        default="train",
        choices=["train", "sample", "eval", "geodesics", "cov"],
        help="what to do when running the script (default: %(default)s)",
    )
    parser.add_argument(
        "--experiment-folder",
        type=str,
        default="experiment",
        help="folder to save and load experiment results in (default: %(default)s)",
    )
    parser.add_argument(
        "--samples",
        type=str,
        default="samples.png",
        help="file to save samples in (default: %(default)s)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="torch device (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="batch size for training (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs-per-decoder",
        type=int,
        default=50,
        metavar="N",
        help="number of training epochs per each decoder (default: %(default)s)",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=2,
        metavar="N",
        help="dimension of latent variable (default: %(default)s)",
    )
    parser.add_argument(
        "--num-decoders",
        type=int,
        default=5,
        metavar="N",
        help="number of decoders in the ensemble (default: %(default)s)",
    )
    parser.add_argument(
        "--num-reruns",
        type=int,
        default=10,
        metavar="N",
        help="number of reruns (default: %(default)s)",
    )
    parser.add_argument(
        "--num-curves",
        type=int,
        default=10,
        metavar="N",
        help="number of geodesics to plot (default: %(default)s)",
    )
    parser.add_argument(
        "--num-t",  # number of points along the curve
        type=int,
        default=20,
        metavar="N",
        help="number of points along the curve (default: %(default)s)",
    )

    args = parser.parse_args()
    print("# Options")
    for key, value in sorted(vars(args).items()):
        print(key, "=", value)
    print("")

    device = args.device

    # Load a subset of MNIST and create data loaders
    def subsample(data, targets, num_data, num_classes):
        idx = targets < num_classes
        new_data = data[idx][:num_data].unsqueeze(1).to(torch.float32) / 255
        new_targets = targets[idx][:num_data]

        return torch.utils.data.TensorDataset(new_data, new_targets)

    num_train_data = 2048
    num_classes = 3
    train_tensors = datasets.MNIST(
        "data/",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    test_tensors = datasets.MNIST(
        "data/",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    train_data = subsample(
        train_tensors.data, train_tensors.targets, num_train_data, num_classes
    )
    test_data = subsample(
        test_tensors.data, test_tensors.targets, num_train_data, num_classes
    )

    mnist_train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True
    )
    mnist_test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False
    )

    # Define prior distribution
    M = args.latent_dim

    def new_encoder():
        encoder_net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.Softmax(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.Softmax(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(512, 2 * M),
        )
        return encoder_net

    def new_decoder():
        decoder_net = nn.Sequential(
            nn.Linear(M, 512),
            nn.Unflatten(-1, (32, 4, 4)),
            nn.Softmax(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=0),
            nn.Softmax(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.Softmax(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
        )
        return decoder_net

    # Choose mode to run
    if args.mode == "train":

        experiments_folder = args.experiment_folder
        os.makedirs(f"{experiments_folder}", exist_ok=True)

        model = EnsembleVAE(
            GaussianPrior(M),
            [GaussianDecoder(new_decoder()) for _ in range(args.num_decoders)],
            GaussianEncoder(new_encoder()),
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train(
            model,
            optimizer,
            mnist_train_loader,
            args.epochs_per_decoder * args.num_decoders,
            args.device,
        )
        os.makedirs(f"{experiments_folder}", exist_ok=True)

        torch.save(
            model.state_dict(),
            f"{experiments_folder}/model.pt",
        )

    elif args.mode == "sample":
        model = EnsembleVAE(
            GaussianPrior(M),
            [GaussianDecoder(new_decoder()) for _ in range(args.num_decoders)],
            GaussianEncoder(new_encoder()),
        ).to(device)
        model.load_state_dict(torch.load(args.experiment_folder + "/model.pt"))
        model.eval()

        with torch.no_grad():
            samples = (model.sample(64)).cpu()
            save_image(samples.view(64, 1, 28, 28), args.samples)

            data = next(iter(mnist_test_loader))[0].to(device)
            recon = model.decoder(model.encoder(data).mean).mean
            save_image(
                torch.cat([data.cpu(), recon.cpu()], dim=0), "reconstruction_means.png"
            )

    elif args.mode == "eval":
        # Load trained model
        model = EnsembleVAE(
            GaussianPrior(M),
            [GaussianDecoder(new_decoder()) for _ in range(args.num_decoders)],
            GaussianEncoder(new_encoder()),
        ).to(device)
        model.load_state_dict(torch.load(args.experiment_folder + "/model.pt"))
        model.eval()

        elbos = []
        with torch.no_grad():
            for x, y in mnist_test_loader:
                x = x.to(device)
                elbo = model.elbo(x)
                elbos.append(elbo)
        mean_elbo = torch.tensor(elbos).mean()
        print("Print mean test elbo:", mean_elbo)

    elif args.mode == "geodesics":
        model = EnsembleVAE(
            GaussianPrior(M),
            [GaussianDecoder(new_decoder()) for _ in range(args.num_decoders)],
            GaussianEncoder(new_encoder()),
        ).to(device)
        model.load_state_dict(torch.load(args.experiment_folder + "/model.pt"))
        model.eval()

        ### Encode test set samples and pick two random endpoints
        num_points = 3000
        encoded_list = []
        labels_list = []
        with torch.no_grad():
            for xb, yb in tqdm(mnist_test_loader, desc="Encoding test set samples"):
                xb = xb.to(device)
                means = model.encoder(xb).mean
                encoded_list.append(means)
                labels_list.append(yb)
                if sum(m.shape[0] for m in encoded_list) >= num_points:
                    break
        encoded = torch.cat(encoded_list, dim=0)[:num_points]
        labels = torch.cat(labels_list, dim=0)[:num_points]

        i, j = random.sample(range(encoded.shape[0]), 2)
        start = encoded[i]
        end = encoded[j]

        ### Compute linear curve energy
        t = torch.linspace(0.0, 1.0, 50, device=device).unsqueeze(1)
        points = start + t * (end - start)  # straight line from start to end
        curve = Curve(points)
        preoptim_avg_energy = model.avg_curve_energy(curve)
        print(f"Model-average curve energy (linear curve): {preoptim_avg_energy}")

        ### Compute optimized curve energy
        optimized_curve = model.optimize_geodesics(start, end, num_nodes=50)
        postoptim_avg_energy = model.avg_curve_energy(optimized_curve)
        print(f"Model-average curve energy (optimized curve): {postoptim_avg_energy}")

        ### Plot latent space and curves
        straight_pts = torch.stack([curve[i] for i in range(curve.n)])
        opt_pts = torch.stack([optimized_curve[i] for i in range(optimized_curve.n)])
        straight_np = straight_pts.detach().cpu().numpy()
        opt_np = opt_pts.detach().cpu().numpy()
        test_np = encoded.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()

        plt.figure()
        classes = np.unique(labels_np)
        cmap = plt.get_cmap('tab10')
        for idx, cls in enumerate(classes):
            mask = labels_np == cls
            plt.scatter(test_np[mask, 0], test_np[mask, 1], s=8, color=cmap(idx), label=f'class {int(cls)}', alpha=0.6)

        plt.plot(straight_np[:, 0], straight_np[:, 1], marker='o', linestyle='-', color='k', label='straight')
        plt.plot(opt_np[:, 0], opt_np[:, 1], marker='o', linestyle='--', color='r', label='optimized')
        plt.scatter([straight_np[0,0], straight_np[-1,0]], [straight_np[0,1], straight_np[-1,1]], c=['g','m'], s=50)
        plt.title('Latent space geodesics')
        plt.legend()
        plt.show()

    elif args.mode == "cov":
        # Simple selection of 10 random test data pairs (each pair is two sample points)
        x_batch, _ = next(iter(mnist_test_loader))
        idx = torch.randperm(x_batch.size(0))[:20]
        test_pairs = [
            (x_batch[idx[2 * i]], x_batch[idx[2 * i + 1]])
            for i in range(10)
        ]

        euc_covs = []
        geo_covs = []
        for d in [1, 2, 3, 5]:
            euclidean_distances = []
            geodesics_distances = []
            for i in range(1, 11):
                model = EnsembleVAE(
                    GaussianPrior(M),
                    [GaussianDecoder(new_decoder()) for _ in range(d)],
                    GaussianEncoder(new_encoder()),
                ).to(device)
                model.load_state_dict(torch.load(f"{d}decoder-{i}" + "/model.pt"))
                model.eval()

                with torch.no_grad():
                    x1, x2 = test_pairs[0]
                    pair_batch = torch.stack([x1, x2]).to(device)
                    z_mean = model.encoder(pair_batch).mean
                    p1, p2 = z_mean[0], z_mean[1]

                curve = model.optimize_geodesics(p1, p2, num_nodes=50, steps=70, lr=0.01)
                geodesics_distances.append(curve.get_length().cpu().detach().numpy())

                euc_dist = torch.norm(p1 - p2)
                euclidean_distances.append(euc_dist.cpu().numpy())

            euc_mu = np.mean(euclidean_distances)
            euc_sigma = np.std(euclidean_distances)

            geo_mu = np.mean(geodesics_distances)
            geo_sigma = np.std(geodesics_distances)

            euc_covs.append(euc_sigma / euc_mu)
            geo_covs.append(geo_sigma / geo_mu)
        
        print("Euclidean CoVs:")
        for cov in euc_covs:
            print(cov)
        
        print("Geodesics CoVs")
        for cov in geo_covs:
            print(cov)




        
