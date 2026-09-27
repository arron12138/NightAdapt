from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .stage2_graph import (
        GatedDynamicGraphLayer,
        MutualGraphLayer,
        VectorEncoder,
        build_fusion_head,
    )
except ImportError:  # pragma: no cover
    from stage2_graph import (
        GatedDynamicGraphLayer,
        MutualGraphLayer,
        VectorEncoder,
        build_fusion_head,
    )


class FlexibleFusionModel(nn.Module):
    """Stage2 fusion model with configurable modality and graph variants.

    Parameters
    ----------
    dims:
        Feature dimensions, for example
        ``{"eeg_dim": 471, "eog_dim": 78, "emg_dim": 10}``.
    modalities:
        Modalities used by the model. Supported names are ``"eeg"``, ``"eog"``
        and ``"emg"``.
    model_kind:
        Fusion variant. Supported values:
        ``"dynamic"``, ``"gated_dynamic"``, ``"static"``, ``"concat"`` and
        ``"dynamic_residual"``.

    Notes
    -----
    This class was extracted from the Stage2 experiment runner and contains only
    the reusable model definition. Training loops, metric computation, data
    loading and grid-search logic are intentionally excluded.
    """

    def __init__(
        self,
        dims: Dict[str, int],
        modalities: Sequence[str],
        model_kind: str = "dynamic",
        num_classes: int = 5,
        hidden_dim: int = 256,
        modality_dim: int = 128,
        dropout: float = 0.2,
        top_k: int = 3,
        gcn_layers: int = 2,
        head_layers: int = 1,
    ):
        super().__init__()
        self.modalities = tuple(modalities)
        self.model_kind = str(model_kind)
        self.top_k = int(top_k)
        self.eps = 1e-8

        encoders = {}
        if "eeg" in self.modalities:
            encoders["eeg"] = VectorEncoder(dims["eeg_dim"], hidden_dim, modality_dim, dropout)
        if "eog" in self.modalities:
            encoders["eog"] = VectorEncoder(dims["eog_dim"], hidden_dim, modality_dim, dropout)
        if "emg" in self.modalities:
            encoders["emg"] = VectorEncoder(dims["emg_dim"], hidden_dim, modality_dim, dropout)
        if not encoders:
            raise ValueError("At least one modality must be selected.")
        self.encoders = nn.ModuleDict(encoders)

        if self.model_kind == "gated_dynamic" and len(self.modalities) > 1:
            self.gated_graph_layers = nn.ModuleList(
                [
                    GatedDynamicGraphLayer(modality_dim, num_nodes=len(self.modalities), dropout=dropout)
                    for _ in range(max(1, gcn_layers))
                ]
            )
            self.graph_layers = nn.ModuleList()
        elif self.model_kind in {"dynamic", "dynamic_residual", "static"} and len(self.modalities) > 1:
            self.gated_graph_layers = nn.ModuleList()
            self.graph_layers = nn.ModuleList(
                [MutualGraphLayer(modality_dim, dropout=dropout) for _ in range(max(1, gcn_layers))]
            )
        else:
            self.gated_graph_layers = nn.ModuleList()
            self.graph_layers = nn.ModuleList()

        if self.model_kind in {"concat", "dynamic_residual"} and len(self.modalities) > 1:
            self.concat_projector = nn.Sequential(
                nn.Linear(len(self.modalities) * modality_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, modality_dim),
                nn.LayerNorm(modality_dim),
                nn.ReLU(inplace=True),
            )
        else:
            self.concat_projector = None

        if self.model_kind == "dynamic_residual" and len(self.modalities) > 1:
            self.residual_gate = nn.Sequential(
                nn.Linear(modality_dim * 2, modality_dim),
                nn.Sigmoid(),
            )
        else:
            self.residual_gate = None

        self.bnneck = nn.LayerNorm(modality_dim)
        self.dropout = nn.Dropout(dropout)
        self.fusion_head = build_fusion_head(
            modality_dim,
            hidden_dim,
            num_classes,
            dropout,
            head_layers=head_layers,
        )

    def build_affinity(self, nodes: torch.Tensor) -> torch.Tensor:
        batch_size, n_nodes, _ = nodes.shape
        if self.model_kind == "static":
            return torch.full(
                (batch_size, n_nodes, n_nodes),
                fill_value=1.0 / float(n_nodes),
                dtype=nodes.dtype,
                device=nodes.device,
            )

        nodes_norm = F.normalize(nodes, dim=-1)
        affinity = torch.bmm(nodes_norm, nodes_norm.transpose(1, 2))
        diag_idx = torch.arange(n_nodes, device=affinity.device)
        affinity[:, diag_idx, diag_idx] = 1.0
        k = max(1, min(self.top_k, n_nodes))
        _, topk_idx = torch.topk(affinity, k=k, dim=2)
        mask = torch.zeros_like(affinity)
        mask.scatter_(2, topk_idx, 1.0)
        adj = affinity * mask
        return adj / (adj.sum(dim=2, keepdim=True) + self.eps)

    @staticmethod
    def consistency_loss(features: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(features) < 2:
            return torch.zeros((), device=features[0].device)
        target = torch.stack(list(features), dim=0).mean(dim=0)
        return torch.stack([F.mse_loss(feat, target) for feat in features]).mean()

    def forward(self, eeg_feat: torch.Tensor, eog_feat: torch.Tensor, emg_feat: torch.Tensor):
        raw = {"eeg": eeg_feat, "eog": eog_feat, "emg": emg_feat}
        feats = {name: self.encoders[name](raw[name]) for name in self.modalities}
        ordered = [feats[name] for name in self.modalities]
        aux_loss = self.consistency_loss(ordered)

        if len(ordered) == 1:
            feat_mutual = ordered[0]
            aux: Dict[str, torch.Tensor] = {"feat_mutual": feat_mutual}
        elif self.model_kind == "concat":
            feat_mutual = self.concat_projector(torch.cat(ordered, dim=1))
            aux = {"feat_mutual": feat_mutual}
        elif self.model_kind == "gated_dynamic":
            nodes = torch.stack(ordered, dim=1)
            adj = None
            for layer in self.gated_graph_layers:
                nodes, adj = layer(nodes)
            feat_mutual = nodes.mean(dim=1)
            aux = {"feat_mutual": feat_mutual, "graph_nodes": nodes, "adjacency": adj}
        else:
            nodes = torch.stack(ordered, dim=1)
            adj = self.build_affinity(nodes)
            for layer in self.graph_layers:
                nodes = layer(nodes, adj)
            graph_feat = nodes.mean(dim=1)
            aux = {"graph_nodes": nodes, "adjacency": adj, "feat_graph": graph_feat}

            if self.model_kind == "dynamic_residual":
                residual_feat = self.concat_projector(torch.cat(ordered, dim=1))
                gate = self.residual_gate(torch.cat([graph_feat, residual_feat], dim=1))
                feat_mutual = gate * graph_feat + (1.0 - gate) * residual_feat
                aux["feat_residual"] = residual_feat
                aux["residual_gate"] = gate
            else:
                feat_mutual = graph_feat
            aux["feat_mutual"] = feat_mutual

        for name, feat in feats.items():
            aux[f"feat_{name}"] = feat

        feat_bn = self.dropout(self.bnneck(feat_mutual))
        logits = torch.nan_to_num(self.fusion_head(feat_bn), nan=0.0, posinf=1e3, neginf=-1e3)
        aux["feat_bn"] = feat_bn
        return logits, aux_loss, aux


__all__ = ["FlexibleFusionModel"]
