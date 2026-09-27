from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .stage2_flexible_fusion import FlexibleFusionModel
    from .stage2_graph import build_fusion_head
except ImportError:  # pragma: no cover
    from stage2_flexible_fusion import FlexibleFusionModel
    from stage2_graph import build_fusion_head


class TemporalConvBlock(nn.Module):
    """Residual temporal convolution block for epoch-level features."""

    def __init__(self, dim: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        padding = (int(kernel_size) // 2) * int(dilation)
        self.conv = nn.Conv1d(dim, dim, kernel_size=int(kernel_size), padding=padding, dilation=int(dilation))
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = x.transpose(1, 2)
        y = self.conv(y).transpose(1, 2)
        y = self.norm(y)
        y = F.relu(y, inplace=True)
        y = self.drop(y)
        return residual + y


class TemporalStage2Model(nn.Module):
    """Graph-fusion Stage2 model with an optional temporal context head.

    The model first applies ``FlexibleFusionModel`` to each epoch in a sequence,
    then predicts the target epoch using one of the supported temporal heads.

    ``window_mode="causal"`` uses the last element in the sequence as the target,
    which makes the model compatible with online inference where future epochs
    are unavailable.
    """

    def __init__(
        self,
        dims: Dict[str, int],
        temporal_kind: str,
        seq_len: int,
        model_kind: str = "dynamic",
        modalities: Sequence[str] = ("eeg", "eog", "emg"),
        num_classes: int = 5,
        hidden_dim: int = 512,
        modality_dim: int = 256,
        dropout: float = 0.05,
        top_k: int = 3,
        gcn_layers: int = 2,
        head_layers: int = 2,
        temporal_layers: int = 2,
        temporal_kernel: int = 3,
        window_mode: str = "causal",
    ):
        super().__init__()
        self.temporal_kind = str(temporal_kind).lower()
        self.seq_len = int(seq_len)
        self.window_mode = str(window_mode).strip().lower()
        if self.seq_len < 1:
            raise ValueError("seq_len must be positive.")
        if self.window_mode not in {"causal", "center"}:
            raise ValueError("window_mode must be 'causal' or 'center'.")
        if self.window_mode == "center" and self.seq_len % 2 == 0:
            raise ValueError("center window requires an odd seq_len.")

        self.target_index = self.seq_len - 1 if self.window_mode == "causal" else self.seq_len // 2
        self.model_kind = str(model_kind)
        self.modalities = tuple(modalities)
        self.base = FlexibleFusionModel(
            dims=dims,
            modalities=self.modalities,
            model_kind=self.model_kind,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            modality_dim=modality_dim,
            dropout=dropout,
            top_k=top_k,
            gcn_layers=gcn_layers,
            head_layers=head_layers,
        )

        if self.temporal_kind in {"none", "center", "identity"}:
            self.temporal = None
            self.pos_embed = None
        elif self.temporal_kind == "tcn":
            blocks = []
            for i in range(max(1, int(temporal_layers))):
                blocks.append(
                    TemporalConvBlock(
                        modality_dim,
                        kernel_size=temporal_kernel,
                        dilation=2**i,
                        dropout=dropout,
                    )
                )
            self.temporal = nn.Sequential(*blocks)
            self.pos_embed = None
        elif self.temporal_kind == "bigru":
            hidden = max(8, modality_dim // 2)
            self.temporal = nn.GRU(
                input_size=modality_dim,
                hidden_size=hidden,
                num_layers=max(1, int(temporal_layers)),
                dropout=dropout if int(temporal_layers) > 1 else 0.0,
                bidirectional=True,
                batch_first=True,
            )
            self.pos_embed = None
        elif self.temporal_kind == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=modality_dim,
                nhead=4,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.temporal = nn.TransformerEncoder(layer, num_layers=max(1, int(temporal_layers)))
            self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, modality_dim))
        else:
            raise ValueError(f"Unsupported temporal_kind={temporal_kind!r}")

        self.out_norm = nn.LayerNorm(modality_dim)
        self.out_drop = nn.Dropout(dropout)
        self.classifier = build_fusion_head(
            modality_dim,
            hidden_dim,
            num_classes,
            dropout,
            head_layers=head_layers,
        )

    def forward(self, eeg_seq: torch.Tensor, eog_seq: torch.Tensor, emg_seq: torch.Tensor):
        batch_size, seq_len, _ = eeg_seq.shape
        if seq_len != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {seq_len}.")

        eeg_flat = eeg_seq.reshape(batch_size * seq_len, -1)
        eog_flat = eog_seq.reshape(batch_size * seq_len, -1)
        emg_flat = emg_seq.reshape(batch_size * seq_len, -1)
        _logits_flat, aux_loss, aux = self.base(eeg_flat, eog_flat, emg_flat)
        z = aux["feat_bn"].reshape(batch_size, seq_len, -1)

        if self.temporal_kind in {"none", "center", "identity"}:
            feat = z[:, self.target_index, :]
        elif self.temporal_kind == "bigru":
            z_ctx, _hidden = self.temporal(z)
            center_raw = z[:, self.target_index, :]
            center_ctx = z_ctx[:, self.target_index, :]
            feat = center_raw + center_ctx
        else:
            z_in = z if self.pos_embed is None else z + self.pos_embed[:, :seq_len, :]
            z_ctx = self.temporal(z_in)
            center_raw = z[:, self.target_index, :]
            center_ctx = z_ctx[:, self.target_index, :]
            feat = center_raw + center_ctx

        feat = self.out_norm(feat)
        feat = self.out_drop(feat)
        logits = torch.nan_to_num(self.classifier(feat), nan=0.0, posinf=1e3, neginf=-1e3)
        out_aux = {
            "feat_mutual": feat,
            "feat_bn": feat,
        }
        for key in ("feat_eeg", "feat_eog", "feat_emg"):
            if key in aux:
                out_aux[key] = aux[key].reshape(batch_size, seq_len, -1)[:, self.target_index, :]
        return logits, aux_loss, out_aux


__all__ = ["TemporalConvBlock", "TemporalStage2Model"]
