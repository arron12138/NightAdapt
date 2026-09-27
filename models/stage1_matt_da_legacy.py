import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .spd import SPDTransform, SPDTangentSpace, SPDRectified
except ImportError:  # pragma: no cover
    from spd import SPDTransform, SPDTangentSpace, SPDRectified


class TemporalAttention(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 16):
        super().__init__()
        self.query = nn.Conv1d(in_channels, hidden_dim, kernel_size=1)
        self.key = nn.Conv1d(in_channels, hidden_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query(x)
        k = self.key(x)
        score = torch.tanh(q + k).mean(dim=1, keepdim=True)
        weight = F.softmax(score, dim=-1)
        out = x * weight
        return x + self.gamma * out


class ChannelMaskGate(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(1, num_channels, 1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = x * self.gain
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        return x


class SPDFeatureBranch(nn.Module):
    def __init__(
        self,
        num_channels: int,
        input_channels: int = 1,
        temporal_filters: int = 8,
        spd_hidden: int = 24,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.mask_gate = ChannelMaskGate(num_channels)

        self.conv_temporal = nn.Conv2d(
            input_channels,
            temporal_filters,
            kernel_size=(1, 50),
            padding=(0, 25),
            bias=False,
        )
        self.bn = nn.BatchNorm2d(temporal_filters)
        self.expanded_dim = num_channels * temporal_filters
        self.temporal_att = TemporalAttention(self.expanded_dim)

        if spd_hidden > self.expanded_dim:
            raise ValueError(
                f'SPDFeatureBranch illegal config: spd_hidden={spd_hidden} > expanded_dim={self.expanded_dim}'
            )

        self.trans = SPDTransform(self.expanded_dim, spd_hidden)
        self.rect = SPDRectified()
        self.tangent = SPDTangentSpace(spd_hidden)
        self.vec_dim = int(spd_hidden * (spd_hidden + 1) / 2)
        self.post = nn.Sequential(
            nn.Linear(self.vec_dim, self.vec_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if x.ndim == 3:
            x = self.mask_gate(x, mask).unsqueeze(1)
        elif x.ndim == 4:
            x = self.mask_gate(x.squeeze(1), mask).unsqueeze(1)
        else:
            raise ValueError(f'Unsupported input shape: {x.shape}')

        x = self.conv_temporal(x)
        x = self.bn(x)
        x = F.elu(x)
        x = torch.nan_to_num(x, nan=0.0, posinf=1e3, neginf=-1e3)

        b, c, h, w = x.shape
        x = x.contiguous().view(b, c * h, w)
        x = self.temporal_att(x)
        x = x - x.mean(dim=2, keepdim=True)
        x = torch.nan_to_num(x, nan=0.0, posinf=1e3, neginf=-1e3)

        denom = max(w - 1, 1)
        cov = x @ x.transpose(1, 2) / denom
        cov = 0.5 * (cov + cov.transpose(1, 2))

        trace = torch.diagonal(cov, dim1=-2, dim2=-1).sum(-1)
        cov = cov / (trace.unsqueeze(-1).unsqueeze(-1) + 1e-6)

        eye = torch.eye(cov.shape[-1], device=cov.device, dtype=cov.dtype).unsqueeze(0)
        cov = cov + 1e-3 * eye
        cov = torch.nan_to_num(cov, nan=0.0, posinf=1e3, neginf=-1e3)

        cov = self.trans(cov)
        cov = 0.5 * (cov + cov.transpose(1, 2))
        eye2 = torch.eye(cov.shape[-1], device=cov.device, dtype=cov.dtype).unsqueeze(0)
        cov = cov + 1e-3 * eye2

        cov = self.rect(cov)
        vec = self.tangent(cov)
        vec = self.post(vec)
        vec = torch.nan_to_num(vec, nan=0.0, posinf=1e3, neginf=-1e3)
        return vec


class DomainAdapter(nn.Module):
    def __init__(self, feature_dim: int, bottleneck_ratio: int = 4):
        super().__init__()
        hidden = max(feature_dim // bottleneck_ratio, 8)
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x + self.net(x)
        out = torch.nan_to_num(out, nan=0.0, posinf=1e3, neginf=-1e3)
        return out


class FusionHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.cls = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor):
        fused = self.fuse(x)
        logits = self.cls(fused)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e3, neginf=-1e3)
        return logits, fused


class MultiModalMAttDA(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        eeg_public_channels: int = 3,
        eeg_extra_channels: int = 12,
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
        use_domain_adapter: bool = True,
        use_conti_extra_proxy: bool = True,
    ):
        super().__init__()

        self.eeg_public_branch = SPDFeatureBranch(
            num_channels=eeg_public_channels,
            temporal_filters=eeg_public_temporal_filters,
            spd_hidden=eeg_public_spd_hidden,
        )
        self.eeg_extra_branch = SPDFeatureBranch(
            num_channels=eeg_extra_channels,
            temporal_filters=eeg_extra_temporal_filters,
            spd_hidden=eeg_extra_spd_hidden,
        )
        self.eog_branch = SPDFeatureBranch(
            num_channels=eog_channels,
            temporal_filters=eog_temporal_filters,
            spd_hidden=eog_spd_hidden,
        )
        self.emg_branch = SPDFeatureBranch(
            num_channels=emg_channels,
            temporal_filters=emg_temporal_filters,
            spd_hidden=emg_spd_hidden,
        )

        self.eeg_public_dim = self.eeg_public_branch.vec_dim
        self.eeg_extra_dim = self.eeg_extra_branch.vec_dim
        self.eog_dim = self.eog_branch.vec_dim
        self.emg_dim = self.emg_branch.vec_dim

        self.use_conti_extra_proxy = use_conti_extra_proxy
        self.eeg_extra_project = nn.Linear(self.eeg_extra_dim, self.eeg_extra_dim)
        if self.use_conti_extra_proxy:
            self.conti_extra_proxy = nn.Sequential(
                nn.Linear(self.eeg_public_dim, self.eeg_extra_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
            )
        self.eeg_fuse = nn.Sequential(
            nn.Linear(self.eeg_public_dim + self.eeg_extra_dim, self.eeg_public_dim + self.eeg_extra_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        self.use_domain_adapter = use_domain_adapter
        if use_domain_adapter:
            self.eeg_public_adapters = nn.ModuleDict({
                'conti': DomainAdapter(self.eeg_public_dim),
                'bp': DomainAdapter(self.eeg_public_dim),
                'mass': DomainAdapter(self.eeg_public_dim),
            })
            self.eog_adapters = nn.ModuleDict({
                'conti': DomainAdapter(self.eog_dim),
                'bp': DomainAdapter(self.eog_dim),
                'mass': DomainAdapter(self.eog_dim),
            })
            self.emg_adapters = nn.ModuleDict({
                'conti': DomainAdapter(self.emg_dim),
                'bp': DomainAdapter(self.emg_dim),
                'mass': DomainAdapter(self.emg_dim),
            })

        fused_dim = (self.eeg_public_dim + self.eeg_extra_dim) + self.eog_dim + self.emg_dim
        self.fusion_head = FusionHead(fused_dim, num_classes)

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
        eeg_public_feat = self.eeg_public_branch(eeg_public_x, eeg_public_mask)
        eog_feat = self.eog_branch(eog_x, eog_mask)
        emg_feat = self.emg_branch(emg_x, emg_mask)

        if eeg_extra_mask is None:
            eeg_extra_feat = self.eeg_extra_project(self.eeg_extra_branch(eeg_extra_x, None))
        else:
            has_extra = (eeg_extra_mask.sum(dim=1) > 0)
            eeg_extra_feat = eeg_public_feat.new_zeros((eeg_public_feat.size(0), self.eeg_extra_dim))
            if bool(has_extra.any().item()):
                valid_idx = has_extra.nonzero(as_tuple=False).squeeze(1)
                eeg_extra_valid = self.eeg_extra_branch(
                    eeg_extra_x[valid_idx],
                    eeg_extra_mask[valid_idx],
                )
                eeg_extra_valid = self.eeg_extra_project(eeg_extra_valid)
                eeg_extra_feat[valid_idx] = eeg_extra_valid
            if self.use_conti_extra_proxy and domain == 'conti':
                missing_idx = (~has_extra).nonzero(as_tuple=False).squeeze(1)
                if bool(missing_idx.numel() > 0):
                    # Conti has no real extra EEG channels in preprocessing output.
                    # Use a classification-only proxy feature and detach the input so
                    # this auxiliary path does not push gradients back into the
                    # public branch that is used for domain alignment.
                    proxy_feat = self.conti_extra_proxy(eeg_public_feat[missing_idx].detach())
                    eeg_extra_feat[missing_idx] = proxy_feat

        if self.use_domain_adapter and domain is not None:
            if domain not in self.eeg_public_adapters:
                raise KeyError(f'Unknown domain: {domain}')
            eeg_public_feat = self.eeg_public_adapters[domain](eeg_public_feat)
            eog_feat = self.eog_adapters[domain](eog_feat)
            emg_feat = self.emg_adapters[domain](emg_feat)

        eeg_feat = self.eeg_fuse(torch.cat([eeg_public_feat, eeg_extra_feat], dim=1))

        eeg_public_feat = torch.nan_to_num(eeg_public_feat, nan=0.0, posinf=1e3, neginf=-1e3)
        eeg_extra_feat = torch.nan_to_num(eeg_extra_feat, nan=0.0, posinf=1e3, neginf=-1e3)
        eeg_feat = torch.nan_to_num(eeg_feat, nan=0.0, posinf=1e3, neginf=-1e3)
        eog_feat = torch.nan_to_num(eog_feat, nan=0.0, posinf=1e3, neginf=-1e3)
        emg_feat = torch.nan_to_num(emg_feat, nan=0.0, posinf=1e3, neginf=-1e3)

        fused_in = torch.cat([eeg_feat, eog_feat, emg_feat], dim=1)
        logits, fused_feat = self.fusion_head(fused_in)

        return {
            'logits': logits,
            'fused_feat': fused_feat,
            'eeg_public_feat': eeg_public_feat,
            'eeg_extra_feat': eeg_extra_feat,
            'eeg_feat': eeg_feat,
            'eog_feat': eog_feat,
            'emg_feat': emg_feat,
        }


class MMDLoss(nn.Module):
    def __init__(self, kernel_mul: float = 2.0, kernel_num: int = 5):
        super().__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

    def guassian_kernel(self, source: torch.Tensor, target: torch.Tensor):
        ns = int(source.size(0))
        nt = int(target.size(0))
        if ns == 0 or nt == 0:
            return None

        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0)
        total1 = total.unsqueeze(1)
        l2_distance_sq = ((total0 - total1) ** 2).sum(2)

        l2_distance_sq = torch.nan_to_num(l2_distance_sq, nan=0.0, posinf=1e6, neginf=0.0)
        l2_distance_sq = torch.clamp(l2_distance_sq, min=0.0)

        if self.fix_sigma is not None:
            bandwidth = self.fix_sigma
        else:
            denom = max((ns + nt) * (ns + nt) - (ns + nt), 1)
            bandwidth = torch.sum(l2_distance_sq.detach()) / float(denom)

        if not torch.is_tensor(bandwidth):
            bandwidth = torch.tensor(float(bandwidth), device=source.device, dtype=source.dtype)
        else:
            bandwidth = bandwidth.to(device=source.device, dtype=source.dtype)

        bandwidth = torch.clamp(bandwidth, min=1e-6)
        bandwidth = bandwidth / (self.kernel_mul ** (self.kernel_num // 2))
        bandwidth_list = [
            torch.clamp(bandwidth * (self.kernel_mul ** i), min=1e-6)
            for i in range(self.kernel_num)
        ]

        kernel_val = []
        for band in bandwidth_list:
            kernel = torch.exp(-l2_distance_sq / band)
            kernel = torch.nan_to_num(kernel, nan=0.0, posinf=0.0, neginf=0.0)
            kernel_val.append(kernel)

        out = sum(kernel_val)
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ns = int(source.size(0))
        nt = int(target.size(0))
        if ns < 2 or nt < 2:
            return source.new_tensor(0.0)

        kernels = self.guassian_kernel(source, target)
        if kernels is None:
            return source.new_tensor(0.0)

        xx = kernels[:ns, :ns]
        yy = kernels[ns:ns + nt, ns:ns + nt]
        xy = kernels[:ns, ns:ns + nt]
        yx = kernels[ns:ns + nt, :ns]

        loss = xx.mean() + yy.mean() - xy.mean() - yx.mean()
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1e3, neginf=-1e3)
        return loss


class ClassConditionalMMDLoss(nn.Module):
    def __init__(self, kernel_mul: float = 2.0, kernel_num: int = 5):
        super().__init__()
        self.base_mmd = MMDLoss(kernel_mul=kernel_mul, kernel_num=kernel_num)

    def forward(
        self,
        source_feat: torch.Tensor,
        source_label: torch.Tensor,
        target_feat: torch.Tensor,
        target_label: torch.Tensor,
    ) -> torch.Tensor:
        classes = torch.unique(torch.cat([source_label, target_label], dim=0))
        losses = []

        for c in classes:
            s_idx = source_label == c
            t_idx = target_label == c
            ns = int(s_idx.sum().item())
            nt = int(t_idx.sum().item())
            if ns > 1 and nt > 1:
                sub_loss = self.base_mmd(source_feat[s_idx], target_feat[t_idx])
                if torch.isfinite(sub_loss):
                    losses.append(sub_loss)

        if len(losses) == 0:
            return source_feat.new_tensor(0.0)

        out = torch.stack(losses).mean()
        out = torch.nan_to_num(out, nan=0.0, posinf=1e3, neginf=-1e3)
        return out
