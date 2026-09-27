import torch
import torch.nn as nn


def sym_eigh(x):
    batch_size, dim, _ = x.shape
    x_sym = 0.5 * (x + x.transpose(-1, -2))
    identity = torch.eye(dim, device=x.device).unsqueeze(0).expand(batch_size, -1, -1)
    x_sym = x_sym + 1e-4 * identity

    try:
        eigvals, eigvecs = torch.linalg.eigh(x_sym)
        return eigvals, eigvecs
    except (RuntimeError, torch._C._LinAlgError):
        try:
            eigvals, eigvecs = torch.linalg.eigh(x_sym.cpu())
            return eigvals.to(x.device), eigvecs.to(x.device)
        except Exception:
            eigvals = torch.ones(batch_size, dim, device=x.device) * 1e-4
            eigvecs = torch.eye(dim, device=x.device).unsqueeze(0).repeat(batch_size, 1, 1)
            return eigvals, eigvecs


def safe_log_map(x, epsilon=1e-5):
    eigvals, eigvecs = sym_eigh(x)
    eigvals = torch.clamp(eigvals, min=epsilon)
    eigvals_log = torch.diag_embed(torch.log(eigvals))
    return eigvecs @ eigvals_log @ eigvecs.transpose(-1, -2)


def safe_exp_map(x):
    eigvals, eigvecs = sym_eigh(x)
    eigvals = torch.clamp(eigvals, max=10.0)
    eigvals_exp = torch.diag_embed(torch.exp(eigvals))
    return eigvecs @ eigvals_exp @ eigvecs.transpose(-1, -2)


class SPDTransform(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(input_size, output_size), requires_grad=True)
        nn.init.orthogonal_(self.weight)

    def forward(self, x):
        q, _ = torch.linalg.qr(self.weight)
        w = q.unsqueeze(0).expand(x.shape[0], -1, -1)
        return torch.bmm(w.transpose(1, 2), torch.bmm(x, w))


class SPDRectified(nn.Module):
    def __init__(self, epsilon=1e-4):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        eigvals, eigvecs = sym_eigh(x)
        eigvals = torch.clamp(eigvals, min=self.epsilon)
        return eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-1, -2)


class SPDTangentSpace(nn.Module):
    def __init__(self, input_size, vectorize=True):
        super().__init__()
        self.vectorize = vectorize
        if vectorize:
            self.idx = torch.triu_indices(input_size, input_size)

    def forward(self, x):
        x_log = safe_log_map(x)
        if self.vectorize:
            row, col = self.idx
            return x_log[:, row, col]
        return x_log
