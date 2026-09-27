import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MutualGraphLayer(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.msg = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, nodes: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        agg = torch.bmm(adj, nodes)
        out = self.msg(agg)
        out = self.norm(out)
        out = self.act(out)
        out = self.drop(out)
        return nodes + out


class GatedDynamicGraphLayer(nn.Module):
    """Dynamic modality graph layer kept for Experiment temporal runners."""

    def __init__(self, dim: int, num_nodes: int, dropout: float = 0.2):
        super().__init__()
        self.scale = float(dim) ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.msg_proj = nn.Linear(dim, dim)
        self.gate_proj = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.edge_bias = nn.Parameter(torch.zeros(num_nodes, num_nodes))

    def forward(self, nodes: torch.Tensor):
        q = self.q_proj(nodes)
        k = self.k_proj(nodes)
        logits = torch.bmm(q, k.transpose(1, 2)) * self.scale
        logits = logits + self.edge_bias.unsqueeze(0)
        adj = F.softmax(logits, dim=-1)
        msg = torch.bmm(adj, nodes)
        update = F.relu(self.msg_proj(msg), inplace=True)
        gate = torch.sigmoid(self.gate_proj(torch.cat([nodes, msg], dim=-1)))
        out = self.norm(nodes + self.drop(gate * update))
        return out, adj


def build_fusion_head(
    in_dim: int,
    hidden_dim: int,
    num_classes: int,
    dropout: float,
    head_layers: int = 1,
) -> nn.Sequential:
    layers = []
    current_dim = int(in_dim)
    for _ in range(max(1, int(head_layers))):
        layers.extend(
            [
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
        )
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, num_classes))
    return nn.Sequential(*layers)


class MutualConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feat_eeg: torch.Tensor, feat_eog: torch.Tensor, feat_emg: torch.Tensor) -> torch.Tensor:
        target = (feat_eeg + feat_eog + feat_emg) / 3.0
        loss = (
            F.mse_loss(feat_eeg, target) +
            F.mse_loss(feat_eog, target) +
            F.mse_loss(feat_emg, target)
        ) / 3.0
        return loss


class MutualTransferFusionModel(nn.Module):
    def __init__(
        self,
        eeg_dim: int,
        eog_dim: int,
        emg_dim: int,
        num_classes: int = 5,
        hidden_dim: int = 256,
        modality_dim: int = 128,
        dropout: float = 0.2,
        top_k: int = 3,
        gcn_layers: int = 2,
    ):
        super().__init__()
        self.top_k = int(top_k)
        self.eps = 1e-8

        self.eeg_encoder = VectorEncoder(eeg_dim, hidden_dim, modality_dim, dropout)
        self.eog_encoder = VectorEncoder(eog_dim, hidden_dim, modality_dim, dropout)
        self.emg_encoder = VectorEncoder(emg_dim, hidden_dim, modality_dim, dropout)

        self.graph_layers = nn.ModuleList(
            [MutualGraphLayer(modality_dim, dropout=dropout) for _ in range(max(1, gcn_layers))]
        )

        self.bnneck = nn.LayerNorm(modality_dim)
        self.dropout = nn.Dropout(dropout)
        # Keep the name `fusion_head` so Part3 online fine-tuning can still target it directly.
        self.fusion_head = nn.Sequential(
            nn.Linear(modality_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.aux_regularizer = MutualConsistencyLoss()

    def build_affinity(self, nodes: torch.Tensor) -> torch.Tensor:
        nodes = F.normalize(nodes, dim=-1)
        affinity = torch.bmm(nodes, nodes.transpose(1, 2))
        n_nodes = affinity.size(1)
        diag_idx = torch.arange(n_nodes, device=affinity.device)
        affinity[:, diag_idx, diag_idx] = 1.0
        k = max(1, min(self.top_k, n_nodes))
        _, topk_idx = torch.topk(affinity, k=k, dim=2)
        mask = torch.zeros_like(affinity)
        mask.scatter_(2, topk_idx, 1.0)
        adj = affinity * mask
        adj = adj / (adj.sum(dim=2, keepdim=True) + self.eps)
        return adj

    def forward(self, eeg_feat: torch.Tensor, eog_feat: torch.Tensor, emg_feat: torch.Tensor):
        feat_eeg = self.eeg_encoder(eeg_feat)
        feat_eog = self.eog_encoder(eog_feat)
        feat_emg = self.emg_encoder(emg_feat)

        aux_loss = self.aux_regularizer(feat_eeg, feat_eog, feat_emg)

        nodes = torch.stack([feat_eeg, feat_eog, feat_emg], dim=1)
        adj = self.build_affinity(nodes)
        for layer in self.graph_layers:
            nodes = layer(nodes, adj)

        feat_mutual = nodes.mean(dim=1)
        feat_bn = self.bnneck(feat_mutual)
        feat_bn = self.dropout(feat_bn)
        logits = self.fusion_head(feat_bn)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e3, neginf=-1e3)

        aux = {
            'feat_eeg': feat_eeg,
            'feat_eog': feat_eog,
            'feat_emg': feat_emg,
            'feat_mutual': feat_mutual,
            'feat_bn': feat_bn,
            'graph_nodes': nodes,
            'adjacency': adj,
        }
        return logits, aux_loss, aux


def build_multimodal_fusion_model(
    eeg_dim: int,
    eog_dim: int,
    emg_dim: int,
    num_classes: int = 5,
    hidden_dim: int = 256,
    modality_dim: int = 128,
    dropout: float = 0.2,
    top_k: int = 3,
    gcn_layers: int = 2,
):
    return MutualTransferFusionModel(
        eeg_dim=eeg_dim,
        eog_dim=eog_dim,
        emg_dim=emg_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        modality_dim=modality_dim,
        dropout=dropout,
        top_k=top_k,
        gcn_layers=gcn_layers,
    )
