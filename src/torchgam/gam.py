import torch


class BSpline(torch.nn.Module):
    def __init__(self, knots: torch.Tensor, order: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.knots = knots
        self.order = order
        self.p = torch.tensor(self.order - 1).long()
        self.n = self.knots.size(-1) - self.order
        self.nrange = torch.arange(0, self.n, 1, dtype=torch.long)
        self.weights = torch.ones(
            (self.n,), device=self.knots.device, dtype=self.knots.dtype
        )

    def eval_spline(self, x: torch.Tensor, i: torch.LongTensor) -> torch.Tensor:
        while len(i.shape) <= 1:
            i = i.unsqueeze(-1)
        ret = torch.zeros(*x.shape, *i.shape, device=x.device, dtype=x.dtype).squeeze()
        if not torch.is_nonzero(self.p):
            ret += (x < self.knots[i + 1]) & (x >= self.knots[i])
        else:
            # if self.knots[i + self.p] != self.knots[i]:
            ret += (
                (x - self.knots[i])
                / (self.knots[i + self.p] - self.knots[i])
                * self.eval_spline(x, i)
            )
            # if self.knots[i + self.p + 1] != self.knots[i + 1]:
            ret += (
                (self.knots[i + self.p + 1] - x)
                / (self.knots[i + self.p + 1] - self.knots[i + 1])
                * self.eval_spline(x, i + 1)
            )
        return ret

    def spline(self, x: torch.Tensor) -> torch.Tensor:
        return self.eval_spline(x, self.nrange)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.weights @ self.spline(input)


def main():
    s = BSpline(torch.arange(0.0, 1.1, 0.1), 3)
    x = torch.arange(0.0, 1.01, 0.01)
    print(s(x))


if __name__ == "__main__":
    main()
