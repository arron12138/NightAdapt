import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .spd import SPDTransform, SPDRectified, SPDTangentSpace
except ImportError:  # pragma: no cover
    from spd import SPDTransform, SPDRectified, SPDTangentSpace


class TemporalAttention(nn.Module):
    """Temporal attention used by the verified stage_3_1_1 EEG MAtt_DA model."""

    def __init__(self, in_channels: int, hidden_dim: int = 8):
        super().__init__()
        self.query = nn.Conv1d(in_channels, hidden_dim, kernel_size=1)
        self.key = nn.Conv1d(in_channels, hidden_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query(x)
        k = self.key(x)
        score = torch.tanh(q + k).mean(dim=1, keepdim=True)
        weights = F.softmax(score, dim=-1)
        return x + self.gamma * (x * weights)


class E2R(nn.Module):
    """Euclidean signal to SPD covariance, following stage_3_1_1/models.py."""

    def __init__(self, input_channels: int = 1, num_electrodes: int = 3, temporal_filters: int = 8):
        super().__init__()
        self.num_electrodes = int(num_electrodes)
        self.temporal_filters = int(temporal_filters)
        self.conv_temporal = nn.Conv2d(
            input_channels,
            temporal_filters,
            kernel_size=(1, 50),
            padding=(0, 25),
            bias=False,
        )
        self.bn = nn.BatchNorm2d(temporal_filters)
        self.expanded_dim = self.num_electrodes * self.temporal_filters
        self.temporal_att = TemporalAttention(self.expanded_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4:
            raise ValueError(f"MAtt_DA expects input shape (B,C,T) or (B,1,C,T), got {tuple(x.shape)}")

        x = self.conv_temporal(x)
        x = self.bn(x)
        x = F.elu(x)
        x = torch.nan_to_num(x, nan=0.0, posinf=1e3, neginf=-1e3)

        b, c, h, w = x.shape
        x = x.contiguous().view(b, c * h, w)
        x = self.temporal_att(x)
        x = x - x.mean(dim=2, keepdim=True)

        cov = x @ x.transpose(1, 2) / max(w - 1, 1)
        cov = 0.5 * (cov + cov.transpose(1, 2))
        trace = torch.diagonal(cov, dim1=-2, dim2=-1).sum(-1)
        cov = cov / (trace.unsqueeze(-1).unsqueeze(-1) + 1e-6)

        identity = torch.eye(cov.shape[-1], device=cov.device, dtype=cov.dtype).unsqueeze(0)
        cov = cov + 1e-4 * identity
        return torch.nan_to_num(cov, nan=0.0, posinf=1e3, neginf=-1e3)


class AttentionManifold(nn.Module):
    """Tangent-space attention copied from the verified MAtt_DA design."""

    def __init__(self, dim: int, dim_k: int):
        super().__init__()
        self.q = nn.Linear(dim, dim_k)
        self.k = nn.Linear(dim, dim_k)
        self.v = nn.Linear(dim, dim)
        self.gate = nn.Sequential(nn.Linear(dim_k, dim), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = self.q(x)
        key = self.k(x)
        energy = torch.tanh(query + key)
        weights = self.gate(energy)
        return x + self.v(x) * weights


class MAttDABranch(nn.Module):
    """Generic MAtt_DA branch for EEG/EOG/EMG domain alignment."""

    def __init__(
        self,
        num_classes: int = 5,
        num_physical_channels: int = 3,
        input_channels: int = 1,
        temporal_filters: int = 8,
        spd_hidden: int = 18,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.num_physical_channels = int(num_physical_channels)
        self.e2r = E2R(input_channels, num_physical_channels, temporal_filters)
        self.spd_dim_in = self.num_physical_channels * int(temporal_filters)
        if spd_hidden > self.spd_dim_in:
            raise ValueError(f"spd_hidden={spd_hidden} cannot exceed SPD input dim={self.spd_dim_in}")

        self.trans1 = SPDTransform(self.spd_dim_in, spd_hidden)
        self.rect1 = SPDRectified()
        self.tangent_layer = SPDTangentSpace(spd_hidden)
        self.vec_dim = int(spd_hidden * (spd_hidden + 1) / 2)
        self.att_manifold = AttentionManifold(self.vec_dim, max(self.vec_dim // 2, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.vec_dim, num_classes)

    @property
    def feature_dim(self) -> int:
        return self.vec_dim

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        cov = self.e2r(x)
        cov = self.trans1(cov)
        cov = 0.5 * (cov + cov.transpose(1, 2))
        cov = self.rect1(cov)
        vec = self.tangent_layer(cov)
        vec = self.att_manifold(vec)
        vec = torch.nan_to_num(vec, nan=0.0, posinf=1e3, neginf=-1e3)
        logits = self.fc(self.dropout(vec))
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e3, neginf=-1e3)
        return logits, vec


class EuclideanBranch(nn.Module):
    """CNN/attention branch without SPD manifold or tangent-space mapping.

    The output dimension intentionally matches the corresponding MAttDABranch
    (`spd_hidden * (spd_hidden + 1) / 2`) so ablation checkpoints remain
    compatible with Part1 export/evaluation and downstream Part2 interfaces.
    """

    def __init__(
        self,
        num_classes: int = 5,
        num_physical_channels: int = 3,
        input_channels: int = 1,
        temporal_filters: int = 8,
        spd_hidden: int = 18,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.num_physical_channels = int(num_physical_channels)
        self.temporal_filters = int(temporal_filters)
        self.expanded_dim = self.num_physical_channels * self.temporal_filters
        self.vec_dim = int(spd_hidden * (spd_hidden + 1) / 2)
        self.conv_temporal = nn.Conv2d(
            input_channels,
            temporal_filters,
            kernel_size=(1, 50),
            padding=(0, 25),
            bias=False,
        )
        self.bn = nn.BatchNorm2d(temporal_filters)
        self.temporal_att = TemporalAttention(self.expanded_dim)
        self.project = nn.Sequential(
            nn.Linear(self.expanded_dim * 2, self.vec_dim),
            nn.LayerNorm(self.vec_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.vec_dim, num_classes)

    @property
    def feature_dim(self) -> int:
        return self.vec_dim

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4:
            raise ValueError(f"EuclideanBranch expects input shape (B,C,T) or (B,1,C,T), got {tuple(x.shape)}")

        x = self.conv_temporal(x)
        x = self.bn(x)
        x = F.elu(x)
        x = torch.nan_to_num(x, nan=0.0, posinf=1e3, neginf=-1e3)

        b, c, h, w = x.shape
        x = x.contiguous().view(b, c * h, w)
        x = self.temporal_att(x)
        mean = x.mean(dim=2)
        std = x.std(dim=2, unbiased=False)
        vec = self.project(torch.cat([mean, std], dim=1))
        vec = torch.nan_to_num(vec, nan=0.0, posinf=1e3, neginf=-1e3)
        logits = self.fc(self.dropout(vec))
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e3, neginf=-1e3)
        return logits, vec


class EEGOnlyStage1(nn.Module):
    """EEG MAtt_DA with separate public and extra branches.

    Keep the legacy Part2 interface:
    public EEG feature = 300 dims, extra EEG feature = 171 dims,
    full EEG feature = 471 dims.
    """

    def __init__(
        self,
        num_classes: int = 5,
        eeg_public_channels: int = 3,
        eeg_extra_channels: int = 16,
        eeg_public_temporal_filters: int = 8,
        eeg_extra_temporal_filters: int = 6,
        eeg_public_spd_hidden: int = 24,
        eeg_extra_spd_hidden: int = 18,
        use_conti_extra_proxy: bool = True,
        use_riemann_backbone: bool = True,
    ):
        super().__init__()
        self.eeg_public_channels = int(eeg_public_channels)
        self.eeg_extra_channels = int(eeg_extra_channels)
        self.use_conti_extra_proxy = bool(use_conti_extra_proxy)
        branch_cls = MAttDABranch if use_riemann_backbone else EuclideanBranch
        self.use_riemann_backbone = bool(use_riemann_backbone)
        self.eeg_public_branch = branch_cls(
            num_classes=num_classes,
            num_physical_channels=self.eeg_public_channels,
            temporal_filters=eeg_public_temporal_filters,
            spd_hidden=eeg_public_spd_hidden,
            dropout=0.5,
        )
        self.eeg_extra_branch = branch_cls(
            num_classes=num_classes,
            num_physical_channels=self.eeg_extra_channels,
            temporal_filters=eeg_extra_temporal_filters,
            spd_hidden=eeg_extra_spd_hidden,
            dropout=0.5,
        )
        self.eeg_public_dim = self.eeg_public_branch.feature_dim
        self.eeg_extra_dim = self.eeg_extra_branch.feature_dim
        self.eeg_dim = self.eeg_public_dim + self.eeg_extra_dim
        self.eeg_extra_project = nn.Linear(self.eeg_extra_dim, self.eeg_extra_dim)
        if self.use_conti_extra_proxy:
            self.conti_extra_proxy = nn.Sequential(
                nn.Linear(self.eeg_public_dim, self.eeg_extra_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
            )
        self.eeg_fuse = nn.Sequential(
            nn.Linear(self.eeg_dim, self.eeg_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.fc = nn.Linear(self.eeg_dim, num_classes)

    @property
    def feature_dim(self) -> int:
        return self.eeg_dim

    def forward(
        self,
        eeg_public_x: torch.Tensor,
        eeg_extra_x: torch.Tensor,
        eeg_public_mask: torch.Tensor = None,
        eeg_extra_mask: torch.Tensor = None,
        domain: str = None,
    ) -> dict:
        _, eeg_public_feat = self.eeg_public_branch(eeg_public_x, eeg_public_mask)
        if eeg_extra_mask is None:
            _, eeg_extra_feat = self.eeg_extra_branch(eeg_extra_x, None)
            eeg_extra_feat = self.eeg_extra_project(eeg_extra_feat)
        else:
            has_extra = eeg_extra_mask.sum(dim=1) > 0
            eeg_extra_feat = eeg_public_feat.new_zeros((eeg_public_feat.size(0), self.eeg_extra_dim))
            if bool(has_extra.any().item()):
                valid_idx = has_extra.nonzero(as_tuple=False).squeeze(1)
                _, eeg_extra_valid = self.eeg_extra_branch(
                    eeg_extra_x[valid_idx],
                    eeg_extra_mask[valid_idx],
                )
                eeg_extra_valid = self.eeg_extra_project(eeg_extra_valid)
                eeg_extra_feat[valid_idx] = eeg_extra_valid
            if self.use_conti_extra_proxy and domain == "conti":
                missing_idx = (~has_extra).nonzero(as_tuple=False).squeeze(1)
                if bool(missing_idx.numel() > 0):
                    # Conti has no real extra EEG channels in the current
                    # preprocessing output. Match the original Part1 strategy:
                    # learn a feature-level proxy from public EEG without
                    # pushing proxy gradients back into the public branch.
                    proxy_feat = self.conti_extra_proxy(eeg_public_feat[missing_idx].detach())
                    eeg_extra_feat[missing_idx] = proxy_feat

        feat = self.eeg_fuse(torch.cat([eeg_public_feat, eeg_extra_feat], dim=1))
        feat = torch.nan_to_num(feat, nan=0.0, posinf=1e3, neginf=-1e3)
        logits = self.fc(feat)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e3, neginf=-1e3)
        return {
            "logits": logits,
            "feat": feat,
            "eeg_feat": feat,
            "eeg_public_feat": eeg_public_feat,
            "eeg_extra_feat": eeg_extra_feat,
        }


class SingleModalStage1(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_channels: int,
        temporal_filters: int,
        spd_hidden: int,
        feature_name: str,
        use_riemann_backbone: bool = True,
    ):
        super().__init__()
        self.feature_name = feature_name
        branch_cls = MAttDABranch if use_riemann_backbone else EuclideanBranch
        self.use_riemann_backbone = bool(use_riemann_backbone)
        self.branch = branch_cls(
            num_classes=num_classes,
            num_physical_channels=num_channels,
            temporal_filters=temporal_filters,
            spd_hidden=spd_hidden,
            dropout=0.5,
        )
        self.feature_dim = self.branch.feature_dim

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> dict:
        logits, feat = self.branch(x, mask)
        return {
            "logits": logits,
            "feat": feat,
            self.feature_name: feat,
        }


class SeparateModalStage1(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        eeg_public_channels: int = 3,
        eeg_extra_channels: int = 16,
        eog_channels: int = 2,
        emg_channels: int = 1,
        eeg_public_temporal_filters: int = 8,
        eeg_extra_temporal_filters: int = 6,
        eog_temporal_filters: int = 6,
        emg_temporal_filters: int = 4,
        eeg_public_spd_hidden: int = 24,
        eeg_extra_spd_hidden: int = 18,
        eog_spd_hidden: int = 12,
        emg_spd_hidden: int = 4,
        use_domain_adapter: bool = False,
        use_conti_extra_proxy: bool = True,
        use_riemann_backbone: bool = True,
    ):
        super().__init__()
        del use_domain_adapter
        self.use_riemann_backbone = bool(use_riemann_backbone)
        self.eeg_model = EEGOnlyStage1(
            num_classes=num_classes,
            eeg_public_channels=eeg_public_channels,
            eeg_extra_channels=eeg_extra_channels,
            eeg_public_temporal_filters=eeg_public_temporal_filters,
            eeg_extra_temporal_filters=eeg_extra_temporal_filters,
            eeg_public_spd_hidden=eeg_public_spd_hidden,
            eeg_extra_spd_hidden=eeg_extra_spd_hidden,
            use_conti_extra_proxy=use_conti_extra_proxy,
            use_riemann_backbone=use_riemann_backbone,
        )
        self.eog_model = SingleModalStage1(
            num_classes=num_classes,
            num_channels=eog_channels,
            temporal_filters=eog_temporal_filters,
            spd_hidden=eog_spd_hidden,
            feature_name="eog_feat",
            use_riemann_backbone=use_riemann_backbone,
        )
        self.emg_model = SingleModalStage1(
            num_classes=num_classes,
            num_channels=emg_channels,
            temporal_filters=emg_temporal_filters,
            spd_hidden=emg_spd_hidden,
            feature_name="emg_feat",
            use_riemann_backbone=use_riemann_backbone,
        )

    def forward(
        self,
        eeg_public_x: torch.Tensor,
        eeg_extra_x: torch.Tensor,
        eog_x: torch.Tensor,
        emg_x: torch.Tensor,
        eeg_public_mask: torch.Tensor = None,
        eeg_extra_mask: torch.Tensor = None,
        eog_mask: torch.Tensor = None,
        emg_mask: torch.Tensor = None,
        domain: str = None,
    ) -> dict:
        eeg = self.eeg_model(eeg_public_x, eeg_extra_x, eeg_public_mask, eeg_extra_mask, domain=domain)
        eog = self.eog_model(eog_x, eog_mask)
        emg = self.emg_model(emg_x, emg_mask)
        logits = (eeg["logits"] + eog["logits"] + emg["logits"]) / 3.0
        fused_feat = torch.cat([eeg["eeg_feat"], eog["eog_feat"], emg["emg_feat"]], dim=1)
        return {
            "logits": logits,
            "fused_feat": fused_feat,
            "eeg_feat": eeg["eeg_feat"],
            "eeg_public_feat": eeg["eeg_public_feat"],
            "eeg_extra_feat": eeg["eeg_extra_feat"],
            "eog_feat": eog["eog_feat"],
            "emg_feat": emg["emg_feat"],
        }


MultiModalMAttDA = SeparateModalStage1
