import torch


class RadialConformalManifold:
    """Manifold with metric G_x = (1 + ||x||^2) I."""

    def __init__(self, dim, dtype=torch.float32, device="cpu"):
        if dim < 1:
            raise ValueError("`dim` must be at least 1")
        self.dim = dim
        self.dtype = dtype
        self.device = torch.device(device)

    def _as_x(self, x):
        x = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        if x.shape[-1] != self.dim:
            raise ValueError(f"Expected last dimension {self.dim}, got {x.shape[-1]}")
        return x

    def conformal_factor(self, x):
        x = self._as_x(x)
        return 1.0 + torch.sum(x * x, dim=-1)

    def metric(self, x):
        x = self._as_x(x)
        phi = self.conformal_factor(x)
        identity = torch.eye(self.dim, dtype=self.dtype, device=self.device)
        return phi[..., None, None] * identity

    def inverse_metric(self, x):
        x = self._as_x(x)
        phi = self.conformal_factor(x)
        identity = torch.eye(self.dim, dtype=self.dtype, device=self.device)
        return (1.0 / phi)[..., None, None] * identity

    def inner_product(self, x, u, v):
        x = self._as_x(x)
        u = torch.as_tensor(u, dtype=self.dtype, device=self.device)
        v = torch.as_tensor(v, dtype=self.dtype, device=self.device)

        if u.shape != x.shape or v.shape != x.shape:
            raise ValueError("`u` and `v` must have same shape as `x`")

        phi = self.conformal_factor(x)
        return phi * torch.sum(u * v, dim=-1)

    def norm(self, x, v):
        return torch.sqrt(torch.clamp(self.inner_product(x, v, v), min=0.0))

    def christoffel_symbols(self, x):
        """
        Returns Gamma^k_{ij} with shape (..., d, d, d), index order (..., k, i, j).
        """
        x = self._as_x(x)
        phi = self.conformal_factor(x)
        dphi = 2.0 * x

        n = self.dim
        delta_ij = torch.eye(n, dtype=self.dtype, device=self.device).view(1, n, n)
        delta_jk = torch.eye(n, dtype=self.dtype, device=self.device).view(n, 1, n)
        delta_ik = torch.eye(n, dtype=self.dtype, device=self.device).view(n, n, 1)

        coeff = (1.0 / (2.0 * phi))[..., None, None, None]

        term1 = delta_jk[None, ...] * dphi[..., None, :, None]
        term2 = delta_ik[None, ...] * dphi[..., None, None, :]
        term3 = delta_ij[None, ...] * dphi[..., :, None, None]

        return coeff * (term1 + term2 - term3)

    def curve_length(self, points):
        """
        Discrete manifold length:
        L ≈ Σ ||x_{i+1} - x_i||_{(x_i+x_{i+1})/2}
        """
        points = self._as_x(points)
        if points.ndim != 2:
            raise ValueError("`points` must have shape (N, d)")
        if points.shape[0] < 2:
            raise ValueError("Need at least 2 points")

        deltas = points[1:] - points[:-1]
        mid = 0.5 * (points[1:] + points[:-1])
        seg = self.norm(mid, deltas)
        return seg.sum()


def straight_line_points(start, end, num_points=200, dtype=torch.float32, device="cpu"):
    start = torch.as_tensor(start, dtype=dtype, device=device)
    end = torch.as_tensor(end, dtype=dtype, device=device)
    if start.shape != end.shape:
        raise ValueError("`start` and `end` must have same shape")
    if num_points < 2:
        raise ValueError("`num_points` must be at least 2")

    t = torch.linspace(0.0, 1.0, num_points, dtype=dtype, device=device).reshape(-1, 1)
    return (1.0 - t) * start.reshape(1, -1) + t * end.reshape(1, -1)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    manifold = RadialConformalManifold(dim=2, device=device)

    x = torch.tensor([1.0, 2.0], device=device)
    print("x:", x.tolist())
    print("phi(x):", manifold.conformal_factor(x).item())
    print("G_x:\n", manifold.metric(x))
    print("G_x^{-1}:\n", manifold.inverse_metric(x))

    start = torch.tensor([-2.0, -2.0], device=device)
    end = torch.tensor([2.0, 2.0], device=device)
    points = straight_line_points(start, end, num_points=500, device=device)
    length = manifold.curve_length(points)
    print(f"Manifold length from {start.tolist()} to {end.tolist()}: {length.item():.6f}")
