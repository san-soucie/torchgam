import torch

def deboor(x: torch.Tensor, t: torch.Tensor, c: torch.Tensor, p: int):
    """Evaluates S(x).

    Arguments
    ---------
    x: Position. 
    t: Array of knot positions, unpadded (and potentially unsorted).
    c: Array of control points.
    p: Degree of B-spline. 
    """
    t, _ = torch.sort(t, dim=-1, stable=True)
    k = torch.gt(x, t.unsqueeze(-1)).sum(dim=-2) - 1 + p
    if len(t.shape) == 1:
        t = t.unsqueeze(0)
    t = torch.nn.functional.pad(t, (p, p), mode='replicate')
    d = c[torch.arange(-p, 1, 1).unsqueeze(-1) + k - p]

    for r in range(1, p + 1):
        for j in range(p, r - 1, -1):
            alpha = (x - t.index_select(-1, j + k - p)) / (t.index_select(-1, j + 1 + k - r) - t.index_select(-1, j + k - p)) 
            d[..., j, :] = (1.0 - alpha) * d[..., j - 1, :] + alpha * d[..., j, :]
    return d[..., p, :]

class BSpline(torch.nn.Module):
    def __init__(self, knots: torch.Tensor, order: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.knots = knots
        self.order = order
        self.p = self.order - 1
        self.n = self.knots.size(-1) - self.p
        self.control_points = torch.ones(
            (self.n,), device=self.knots.device, dtype=self.knots.dtype
        )
        

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return deboor(input, self.knots, self.control_points, self.p)


def main():
    s = BSpline(torch.arange(0.0, 1.1, 0.1), 2)
    x = torch.arange(0.05, 1.05, 0.05)
    print(s(x))


if __name__ == "__main__":
    main()
