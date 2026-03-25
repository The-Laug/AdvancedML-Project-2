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
        return td.Independent(td.Normal(loc=means, scale=1e-1), 2)


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
    #OLD CODE
    # with tqdm(range(num_steps)) as pbar:
    #     for step in pbar:
    #         try:
    #             x = next(iter(data_loader))[0]
    #             # x = noise(x.to(device))
    #             model = model
    #             optimizer.zero_grad()
    #             # from IPython import embed; embed()
    #             loss = model(x)
    #             loss.backward()
    #             optimizer.step()

    #             # Report
    #             if step % 5 == 0:
    #                 loss = loss.detach().cpu()
    #                 pbar.set_description(
    #                     f"total epochs ={epoch}, step={step}, loss={loss:.1f}"
    #                 )

    #             if (step + 1) % len(data_loader) == 0:
    #                 epoch += 1
    #         except KeyboardInterrupt:
    #             print(
    #                 f"Stopping training at total epoch {epoch} and current loss: {loss:.1f}"
    #             )
    #             break


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

# decoder_net = nn.Sequential(
#     nn.Linear(latent_dim, 512),
#     nn.ReLU(),
#     nn.Linear(512, 512),
#     nn.ReLU(),
#     nn.Linear(512, 784),
#     nn.Unflatten(-1, (28, 28))
# )


decoder_net = nn.Sequential(
    nn.Linear(latent_dim, 512),
    nn.ReLU(),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, 784),
    nn.Sigmoid(), # Keep this from the previous fix!
    nn.Unflatten(-1, (1, 28, 28)) # Added the '1' for the channel dimension
)

#mnist with 3 classes and 2048 observations
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
        model = VAE(GaussianPrior(latent_dim), GaussianDecoder(decoder_net), GaussianEncoder(encoder_net))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train(model, optimizer, data_loader, epochs=100, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        torch.save(model.state_dict(), "gauss_vae.pth")


    if testing_model:
        model = VAE(GaussianPrior(latent_dim), GaussianDecoder(decoder_net), GaussianEncoder(encoder_net))
        model.load_state_dict(torch.load("gauss_vae.pth", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
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