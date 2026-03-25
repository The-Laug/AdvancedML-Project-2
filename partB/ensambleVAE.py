import torch
import torch.nn as nn
from tqdm import tqdm
import torch.distributions as td

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


class EnsembleGaussianDecoder(nn.Module):
    def __init__(self, decoder_nets):
        """
        decoder_nets: A list of standard PyTorch nn.Sequential networks.
        """
        super(EnsembleGaussianDecoder, self).__init__()
        # nn.ModuleList tracks gradients for all ensemble members
        self.ensembles = nn.ModuleList(decoder_nets)

    def forward(self, z, idx=None):
        """
        If idx is None, it uniformly draws a random decoder from the ensemble.
        If idx is provided, it uses that specific decoder.
        """
        if idx is None:
            # Uniformly draw a random integer between 0 and M-1
            idx = torch.randint(0, len(self.ensembles), (1,)).item()
            
        means = self.ensembles[idx](z)
        return td.Independent(td.Normal(loc=means, scale=1e-1), 3)


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

        # This automatically draws a uniform decoder thanks to our updated forward pass!
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

    model.train()
    # NEW CODE
    for epoch in range(epochs):
        with tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for x, _ in pbar: # Unpack the batch properly here
                x = x.to(device) # Don't forget to move data to your device!
                
                # Uncomment this to improve your ensemble later!
                # x = noise(x) 

                optimizer.zero_grad()
                loss = model(x)
                loss.backward()
                optimizer.step()

                pbar.set_postfix(loss=f"{loss.item():.1f}")



latent_dim = 2
    # Define encoder and decoder networks
encoder_net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, latent_dim * 2),
)

num_decoders = 3

decoder_nets = []
for _ in range(num_decoders):
    net = nn.Sequential(
        nn.Linear(latent_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 784),
        nn.Sigmoid(), 
        nn.Unflatten(-1, (1, 28, 28)) 
    )
    decoder_nets.append(net)


from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
transform = transforms.Compose([transforms.ToTensor()])
mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
# Select only 3 classes and limit to 2048 observations
selected_classes = [0, 1, 2]
targets = mnist_train.targets
mask = torch.zeros_like(targets, dtype=torch.bool)
for c in selected_classes:
    mask = mask | (targets == c)
selected_idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
# Shuffle and pick 2048 examples reproducibly
gen = torch.Generator().manual_seed(0)
perm = torch.randperm(len(selected_idx), generator=gen)
pick = selected_idx[perm][:2048]
mnist_subset = Subset(mnist_train, pick.tolist())
data_loader = DataLoader(mnist_subset, batch_size=128, shuffle=True)

if __name__ == "__main__":
    training_model = False
    testing_model = True
    
    if training_model:
        model = VAE(GaussianPrior(latent_dim), EnsembleGaussianDecoder(decoder_nets), GaussianEncoder(encoder_net))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train(model, optimizer, data_loader, epochs=300, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        torch.save(model.state_dict(), "partB/ensamble_gauss_vae.pth")


    if testing_model:
        model = VAE(GaussianPrior(latent_dim), EnsembleGaussianDecoder(decoder_nets), GaussianEncoder(encoder_net))
        model.load_state_dict(torch.load("partB/ensamble_gauss_vae.pth", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
        model.eval()
        with torch.no_grad():
            samples = model.sample(10)
            print("Generated samples shape:", samples.shape)
            # Optionally visualize the generated samples
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 10, figsize=(15, 2))
            for i in range(10):
                import numpy as _np
                img = samples[i].cpu().numpy()
                # remove any singleton channel dims
                img = _np.squeeze(img)
                # if still 3-d and channel-first with 3 channels -> transpose to HWC
                if img.ndim == 3 and img.shape[0] == 3:
                    img = _np.transpose(img, (1, 2, 0))
                # ensure final shape is 2D or 3-channel HWC
                if img.ndim not in (2, 3):
                    raise ValueError(f"Can't display image with shape {img.shape}")
                axes[i].imshow(img, cmap="gray" if img.ndim == 2 else None, vmin=0, vmax=1)
                axes[i].axis("off")
            plt.show()
            #PCA plot of the latent space
            z = model.encoder(torch.stack([mnist_subset[i][0] for i in range(1000)]).to(next(model.parameters()).device)).mean.cpu().numpy()
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            z_2d = pca.fit_transform(z)
            #Use TSNE instead
            # from sklearn.manifold import TSNE
            # z_2d = TSNE(n_components=2, random_state=0).fit_transform(z)
            plt.scatter(z_2d[:, 0], z_2d[:, 1], c=mnist_subset.dataset.targets[mnist_subset.indices][:1000], cmap="tab10", alpha=0.7)
            #Only 3 colors in colorbar
            plt.colorbar(ticks=[0, 1, 2])
            plt.show()