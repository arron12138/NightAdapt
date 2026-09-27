import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = float(margin)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2:
            raise ValueError(f'Expected embeddings [B, D], got {tuple(embeddings.shape)}')
        labels = labels.view(-1).long()
        if embeddings.size(0) != labels.size(0):
            raise ValueError('Embeddings and labels batch size mismatch.')

        dist = torch.cdist(embeddings, embeddings, p=2)
        same = labels.unsqueeze(0) == labels.unsqueeze(1)
        diff = ~same
        same.fill_diagonal_(False)

        inf = 1e9
        dist_pos = dist.clone()
        dist_pos[~same] = -inf
        hardest_pos, _ = dist_pos.max(dim=1)

        dist_neg = dist.clone()
        dist_neg[~diff] = inf
        hardest_neg, _ = dist_neg.min(dim=1)

        valid = (hardest_pos > -inf / 2) & (hardest_neg < inf / 2)
        if not torch.any(valid):
            return torch.zeros((), device=embeddings.device)

        loss = F.relu(hardest_pos - hardest_neg + self.margin)
        return loss[valid].mean()


class CenterLoss(nn.Module):
    def __init__(self, num_classes: int, feat_dim: int):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2:
            raise ValueError(f'Expected embeddings [B, D], got {tuple(embeddings.shape)}')
        labels = labels.view(-1).long()
        centers_batch = self.centers.index_select(0, labels)
        return 0.5 * torch.mean(torch.sum((embeddings - centers_batch) ** 2, dim=1))


class ModalitySeparationLoss(nn.Module):
    def __init__(self, margin: float = 1.0, normalize: bool = True):
        super().__init__()
        self.margin = float(margin)
        self.normalize = bool(normalize)

    def forward(self, feat_eeg: torch.Tensor, feat_eog: torch.Tensor, feat_emg: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            feat_eeg = F.normalize(feat_eeg, dim=1)
            feat_eog = F.normalize(feat_eog, dim=1)
            feat_emg = F.normalize(feat_emg, dim=1)

        d_eegeg = F.pairwise_distance(feat_eeg, feat_eog)
        d_eogemg = F.pairwise_distance(feat_eog, feat_emg)
        d_eegemg = F.pairwise_distance(feat_eeg, feat_emg)

        loss = (
            F.relu(self.margin - d_eegeg).mean() +
            F.relu(self.margin - d_eogemg).mean() +
            F.relu(self.margin - d_eegemg).mean()
        ) / 3.0
        return loss
