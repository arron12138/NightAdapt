from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


def build_sleep_stage_transition_rule_mask(num_classes: int = 5) -> np.ndarray:
    """Return a rule mask for common sleep-stage transitions.

    Label order is ``0=W, 1=N1, 2=N2, 3=N3, 4=REM``. A value of 1 means the
    transition is allowed by the prior, while 0 means it is masked out.
    """

    if num_classes != 5:
        return np.ones((num_classes, num_classes), dtype=np.float64)
    return np.asarray(
        [
            [1, 1, 1, 0, 0],  # W -> W / N1 / N2
            [1, 1, 1, 0, 1],  # N1 -> W / N1 / N2 / REM
            [1, 1, 1, 1, 1],  # N2 -> all common neighbours
            [1, 1, 1, 1, 0],  # N3 -> W / N1 / N2 / N3
            [1, 1, 1, 0, 1],  # REM -> W / N1 / N2 / REM
        ],
        dtype=np.float64,
    )


def build_transition_matrix_from_labels(
    label_arrays: List[np.ndarray],
    num_classes: int = 5,
    smoothing: float = 1.0,
    self_bias: float = 0.5,
    rule_mask: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """Estimate a row-normalized transition matrix from historical labels."""

    mask = np.ones((num_classes, num_classes), dtype=np.float64) if rule_mask is None else np.asarray(rule_mask, dtype=np.float64)
    if mask.shape != (num_classes, num_classes):
        raise ValueError(f"transition rule mask shape mismatch: expected {(num_classes, num_classes)}, got {mask.shape}")
    np.fill_diagonal(mask, 1.0)

    mats = []
    for y in label_arrays:
        y = np.asarray(y, dtype=np.int64).reshape(-1)
        y = y[(y >= 0) & (y < num_classes)]
        if len(y) < 2:
            continue

        counts = np.zeros((num_classes, num_classes), dtype=np.float64)
        for a, b in zip(y[:-1], y[1:]):
            counts[int(a), int(b)] += 1.0
        counts *= mask
        counts += float(smoothing) * mask
        counts += np.eye(num_classes, dtype=np.float64) * float(self_bias)

        row_sum = counts.sum(axis=1, keepdims=True)
        zero_rows = np.where(row_sum.squeeze(1) <= 0)[0]
        for row_idx in zero_rows:
            counts[row_idx] = mask[row_idx]
            row_sum[row_idx, 0] = counts[row_idx].sum()
        row_sum[row_sum <= 0] = 1.0
        mats.append(counts / row_sum)

    if not mats:
        eye = np.eye(num_classes, dtype=np.float64)
        return torch.tensor(eye / eye.sum(axis=1, keepdims=True), dtype=torch.float32)
    return torch.tensor(np.mean(np.stack(mats, axis=0), axis=0), dtype=torch.float32)


def normalize_top2_margin_rules(rules: object, num_classes: int = 5) -> List[Dict[str, object]]:
    """Validate top-2 margin calibration rules."""

    out: List[Dict[str, object]] = []
    if not isinstance(rules, (list, tuple)):
        return out
    for item in rules:
        if not isinstance(item, dict):
            continue
        try:
            top1_class = int(item.get("top1_class"))
            top2_class = int(item.get("top2_class"))
            target_class = int(item.get("target_class", top2_class))
            margin_threshold = float(item.get("margin_threshold"))
        except (TypeError, ValueError):
            continue
        if not (0 <= top1_class < num_classes and 0 <= top2_class < num_classes and 0 <= target_class < num_classes):
            continue
        if top1_class == top2_class or target_class == top1_class or margin_threshold < 0.0:
            continue
        if target_class != top2_class:
            continue
        out.append(
            {
                "top1_class": top1_class,
                "top2_class": top2_class,
                "target_class": target_class,
                "margin_threshold": margin_threshold,
                "changed_count": int(item.get("changed_count", 0) or 0),
            }
        )
    return out


def normalize_n1_calibration(calibration: Optional[Dict[str, object]]) -> Optional[Dict[str, object]]:
    """Normalize optional N1 calibration metadata."""

    if not calibration:
        return None
    out = dict(calibration)
    out["enabled"] = bool(out.get("enabled", False))
    out["target_class"] = int(out.get("target_class", 1))
    out["delta"] = float(out.get("delta", 0.0))
    out["margin_threshold"] = float(out.get("margin_threshold", 0.0))
    out["require_top2"] = bool(out.get("require_top2", True))
    out["selected_acc"] = float(out.get("selected_acc", 0.0))
    out["selected_macro_f1"] = float(out.get("selected_macro_f1", 0.0))
    out["selected_n1_f1"] = float(out.get("selected_n1_f1", 0.0))
    out["baseline_acc"] = float(out.get("baseline_acc", 0.0))
    out["baseline_macro_f1"] = float(out.get("baseline_macro_f1", 0.0))
    out["baseline_n1_f1"] = float(out.get("baseline_n1_f1", 0.0))
    out["acc_floor"] = float(out.get("acc_floor", 0.0))

    class_bias = out.get("class_bias", None)
    if class_bias is None:
        class_bias_arr = np.zeros(5, dtype=np.float32)
    else:
        class_bias_arr = np.asarray(class_bias, dtype=np.float32).reshape(-1)
        if class_bias_arr.size < 5:
            class_bias_arr = np.pad(class_bias_arr, (0, 5 - class_bias_arr.size), mode="constant")
        elif class_bias_arr.size > 5:
            class_bias_arr = class_bias_arr[:5]
    out["class_bias"] = class_bias_arr.astype(float).tolist()
    out["logit_bias_enabled"] = bool(out.get("logit_bias_enabled", np.any(np.abs(class_bias_arr) > 1e-8)))
    out["logit_bias_selected_acc"] = float(out.get("logit_bias_selected_acc", out.get("selected_acc", 0.0)))
    out["logit_bias_selected_macro_f1"] = float(out.get("logit_bias_selected_macro_f1", out.get("selected_macro_f1", 0.0)))
    out["logit_bias_selected_n1_f1"] = float(out.get("logit_bias_selected_n1_f1", out.get("selected_n1_f1", 0.0)))
    out["logit_bias_baseline_acc"] = float(out.get("logit_bias_baseline_acc", out.get("baseline_acc", 0.0)))
    out["logit_bias_baseline_macro_f1"] = float(out.get("logit_bias_baseline_macro_f1", out.get("baseline_macro_f1", 0.0)))
    out["logit_bias_baseline_n1_f1"] = float(out.get("logit_bias_baseline_n1_f1", out.get("baseline_n1_f1", 0.0)))

    top2_rules = normalize_top2_margin_rules(out.get("top2_margin_rules", []), num_classes=len(class_bias_arr))
    out["top2_margin_rules"] = top2_rules
    out["top2_margin_enabled"] = bool(out.get("top2_margin_enabled", len(top2_rules) > 0)) and len(top2_rules) > 0
    out["top2_margin_selected_acc"] = float(out.get("top2_margin_selected_acc", out.get("selected_acc", 0.0)))
    out["top2_margin_selected_macro_f1"] = float(out.get("top2_margin_selected_macro_f1", out.get("selected_macro_f1", 0.0)))
    out["top2_margin_selected_n1_f1"] = float(out.get("top2_margin_selected_n1_f1", out.get("selected_n1_f1", 0.0)))
    out["top2_margin_baseline_acc"] = float(out.get("top2_margin_baseline_acc", out.get("selected_acc", 0.0)))
    out["top2_margin_baseline_macro_f1"] = float(out.get("top2_margin_baseline_macro_f1", out.get("selected_macro_f1", 0.0)))
    out["top2_margin_baseline_n1_f1"] = float(out.get("top2_margin_baseline_n1_f1", out.get("selected_n1_f1", 0.0)))
    return out


def apply_top2_margin_rules_to_logits(
    logits: torch.Tensor,
    rules: Sequence[Dict[str, object]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply conservative top-2 margin class flips to logits."""

    rules = normalize_top2_margin_rules(rules, num_classes=logits.size(1))
    if not rules:
        mask = torch.zeros(logits.size(0), dtype=torch.bool, device=logits.device)
        return logits, mask

    logits_work = logits
    changed_any = torch.zeros(logits.size(0), dtype=torch.bool, device=logits.device)
    eps = torch.as_tensor(1e-4, dtype=logits.dtype, device=logits.device)
    for rule in rules:
        k = min(2, logits_work.size(1))
        if k < 2:
            break
        topk = torch.topk(logits_work, k=k, dim=1)
        top1_idx = topk.indices[:, 0]
        top2_idx = topk.indices[:, 1]
        margin = topk.values[:, 0] - topk.values[:, 1]
        target_class = int(rule["target_class"])
        apply_mask = (
            (top1_idx == int(rule["top1_class"]))
            & (top2_idx == int(rule["top2_class"]))
            & (margin <= float(rule["margin_threshold"]))
        )
        if not torch.any(apply_mask):
            continue
        if logits_work is logits:
            logits_work = logits_work.clone()
        top1_vals = logits_work[apply_mask, top1_idx[apply_mask]]
        logits_work[apply_mask, target_class] = top1_vals + eps
        changed_any |= apply_mask
    return logits_work, changed_any


def apply_n1_bias_calibration_to_logits(
    logits: torch.Tensor,
    calibration: Optional[Dict[str, object]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply optional logit-bias and top-2 N1 calibration rules."""

    calibration = normalize_n1_calibration(calibration)
    if not calibration:
        mask = torch.zeros(logits.size(0), dtype=torch.bool, device=logits.device)
        return logits, mask

    class_bias = torch.as_tensor(
        calibration.get("class_bias", [0.0] * logits.size(1)),
        dtype=logits.dtype,
        device=logits.device,
    )
    if class_bias.numel() < logits.size(1):
        class_bias = F.pad(class_bias, (0, logits.size(1) - class_bias.numel()))
    elif class_bias.numel() > logits.size(1):
        class_bias = class_bias[: logits.size(1)]

    logits_work = logits
    if bool(calibration.get("logit_bias_enabled", False)) and torch.any(torch.abs(class_bias) > 1e-12):
        logits_work = logits_work + class_bias.view(1, -1)

    combined_mask = torch.zeros(logits.size(0), dtype=torch.bool, device=logits.device)
    if calibration.get("enabled", False):
        target_class = int(calibration.get("target_class", 1))
        delta = float(calibration.get("delta", 0.0))
        margin_threshold = float(calibration.get("margin_threshold", 0.0))
        require_top2 = bool(calibration.get("require_top2", True))
        if abs(delta) > 1e-12:
            k = min(2, logits_work.size(1))
            topk = torch.topk(logits_work, k=k, dim=1)
            topk_idx = topk.indices
            topk_vals = topk.values
            pred_raw = topk_idx[:, 0]
            margin = torch.zeros(logits.size(0), dtype=logits.dtype, device=logits.device)
            if k >= 2:
                margin = topk_vals[:, 0] - topk_vals[:, 1]

            if require_top2:
                target_in_top2 = (topk_idx == target_class).any(dim=1)
            else:
                target_in_top2 = torch.ones(logits.size(0), dtype=torch.bool, device=logits.device)
            apply_mask = target_in_top2 & (pred_raw != target_class) & (margin <= margin_threshold)
            if torch.any(apply_mask):
                logits_adj = logits_work.clone()
                logits_adj[apply_mask, target_class] += delta
                logits_work = logits_adj
                combined_mask |= apply_mask

    if calibration.get("top2_margin_enabled", False):
        logits_work, top2_mask = apply_top2_margin_rules_to_logits(logits_work, calibration.get("top2_margin_rules", []))
        combined_mask |= top2_mask
    return logits_work, combined_mask


@dataclass
class OnlineCfg:
    """Configuration for teacher-student pseudo-online adaptation."""

    conf_thres: float = 0.97
    margin_thres: float = 0.20
    entropy_thres: float = 0.60
    require_agreement: bool = True
    pseudo_prior_min_support: float = 0.05
    pseudo_temp: float = 2.0
    online_lr: float = 1e-4
    online_weight_decay: float = 1e-4
    l2sp_weight: float = 1e-3
    buffer_size: int = 256
    recent_buffer_size: int = 128
    update_every: int = 1
    steps_per_update: int = 1
    batch_size: int = 16
    lambda_pl_max: float = 1.0
    lambda_ramp_steps: int = 50
    cons_weight: float = 0.1
    aug_noise_std: float = 0.01
    aug_drop_prob: float = 0.0
    teacher_fallback: bool = True
    fallback_conf: float = 0.90
    warmup_steps: int = 0
    alpha_teacher_init: float = 0.80
    alpha_student_init: float = 0.20
    alpha_eta: float = 0.05
    use_transition_prior: bool = True
    transition_prior_strength: float = 0.2
    transition_prior_min_prob: float = 1e-4
    transition_prior_teacher_only: bool = False
    use_student_only: bool = False
    train_modules: Tuple[str, ...] = ("fusion_head", "classifier")
    n1_calibration: Optional[Dict[str, object]] = None


class OnlineFeatureAdapter:
    """Teacher-student pseudo-online adapter for chronological sleep staging.

    The adapter receives one feature window at a time, predicts the current
    sleep stage, optionally applies a transition prior, accepts high-confidence
    pseudo-labels, and updates the student model from a recent buffer.

    The wrapped model is expected to return ``(logits, aux_loss, aux_dict)``
    when called as ``model(eeg, eog, emg)``.
    """

    def __init__(self, teacher: nn.Module, student: nn.Module, device: torch.device, cfg: OnlineCfg, trans_mat: torch.Tensor):
        self.teacher = teacher.eval()
        self.student = student.eval()
        self.device = device
        self.cfg = cfg
        self.trans_mat = trans_mat.clone().float()
        self.n1_calibration = normalize_n1_calibration(cfg.n1_calibration)
        self.prev_pred = None
        self.buffer: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, int]] = deque(maxlen=cfg.buffer_size)
        self.online_step = 0
        self.alpha_t = float(cfg.alpha_teacher_init)
        self.alpha_s = float(cfg.alpha_student_init)

        self._freeze_student_for_online()
        params = [p for p in self.student.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError(f"No trainable student parameters matched train_modules={cfg.train_modules}.")
        self.optimizer = optim.AdamW(params, lr=cfg.online_lr, weight_decay=cfg.online_weight_decay)
        self.anchor = {n: p.detach().cpu().clone() for n, p in self.student.named_parameters() if p.requires_grad}

    def _freeze_student_for_online(self) -> None:
        for p in self.student.parameters():
            p.requires_grad = False
        trainable_prefixes = tuple(self.cfg.train_modules)
        for name, p in self.student.named_parameters():
            if any(name.startswith(prefix) for prefix in trainable_prefixes):
                p.requires_grad = True

    def _l2sp(self) -> torch.Tensor:
        loss = torch.zeros((), device=self.device)
        for name, p in self.student.named_parameters():
            if p.requires_grad and name in self.anchor:
                loss = loss + (p - self.anchor[name].to(self.device)).pow(2).mean()
        return self.cfg.l2sp_weight * loss

    def _current_lambda_pl(self) -> float:
        ramp = min(1.0, float(self.online_step) / (self.cfg.lambda_ramp_steps + 1e-12))
        return self.cfg.lambda_pl_max * ramp

    @staticmethod
    def _entropy(probs: torch.Tensor) -> torch.Tensor:
        return -(probs * torch.log(probs + 1e-12)).sum(dim=1)

    def _get_transition_prior(self, device: torch.device) -> Optional[torch.Tensor]:
        if (not self.cfg.use_transition_prior) or (self.prev_pred is None):
            return None
        prior = self.trans_mat[self.prev_pred].to(device).unsqueeze(0)
        allowed = prior > 0
        if not torch.any(allowed):
            return None
        prior = torch.where(
            allowed,
            torch.clamp(prior, min=float(self.cfg.transition_prior_min_prob)),
            torch.zeros_like(prior),
        )
        prior = prior / prior.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return prior

    def _apply_transition_prior_to_logits(self, logits: torch.Tensor, apply_prior: bool = True) -> torch.Tensor:
        prior = self._get_transition_prior(logits.device) if apply_prior else None
        if prior is None:
            return torch.softmax(logits, dim=1)
        allowed = prior > 0
        safe_prior = torch.where(allowed, prior, torch.ones_like(prior))
        log_prior = torch.log(safe_prior)
        logits_adj = logits + float(self.cfg.transition_prior_strength) * log_prior
        logits_adj = logits_adj.masked_fill(~allowed, -1e9)
        return torch.softmax(logits_adj, dim=1)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        y = x.clone()
        if self.cfg.aug_drop_prob > 0:
            keep = (torch.rand_like(y) > self.cfg.aug_drop_prob).float()
            y = y * keep
        if self.cfg.aug_noise_std > 0:
            y = y + self.cfg.aug_noise_std * torch.randn_like(y)
        return y

    @torch.no_grad()
    def _teacher_pseudo(self, p_t: torch.Tensor, p_s: torch.Tensor) -> Optional[Tuple[torch.Tensor, float, Dict[str, float]]]:
        conf_t, pred_t = torch.max(p_t, dim=1)
        conf_t = float(conf_t.item())
        pred_t = int(pred_t.item())
        pred_s = int(torch.argmax(p_s, dim=1).item())
        prior = self._get_transition_prior(p_t.device)
        prior_support = 1.0 if prior is None else float(prior[0, pred_t].item())

        top2 = torch.topk(p_t, k=2, dim=1).values.squeeze(0)
        margin = float((top2[0] - top2[1]).item())
        ent = float(self._entropy(p_t).item())

        if prior_support < self.cfg.pseudo_prior_min_support:
            return None
        if conf_t < self.cfg.conf_thres:
            return None
        if margin < self.cfg.margin_thres:
            return None
        if ent > self.cfg.entropy_thres:
            return None
        if self.cfg.require_agreement and pred_t != pred_s:
            return None

        q = torch.softmax(torch.log(p_t + 1e-12) / self.cfg.pseudo_temp, dim=1).squeeze(0).detach().cpu()
        wt = prior_support * conf_t * max(margin, 1e-3) * float(np.exp(-ent))
        wt = float(np.clip(wt, 0.05, 1.0))
        return q, wt, {
            "prior_support": prior_support,
            "teacher_conf": conf_t,
            "teacher_margin": margin,
            "teacher_entropy": ent,
            "teacher_pred": float(pred_t),
            "student_pred": float(pred_s),
        }

    def _train_from_buffer(self) -> None:
        if len(self.buffer) < max(8, self.cfg.batch_size):
            return
        recent = list(self.buffer)[-min(len(self.buffer), self.cfg.recent_buffer_size) :]
        if len(recent) < max(8, self.cfg.batch_size):
            return

        self.student.train()
        params = [p for p in self.student.parameters() if p.requires_grad]
        for _ in range(self.cfg.steps_per_update):
            replace = len(recent) < self.cfg.batch_size
            idx = np.random.choice(len(recent), size=self.cfg.batch_size, replace=replace)
            batch = [recent[i] for i in idx]

            eeg = torch.stack([b[0] for b in batch], dim=0).to(self.device)
            eog = torch.stack([b[1] for b in batch], dim=0).to(self.device)
            emg = torch.stack([b[2] for b in batch], dim=0).to(self.device)
            q = torch.stack([b[3] for b in batch], dim=0).to(self.device)
            wt = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=self.device)

            self.optimizer.zero_grad()
            logits_s, _, _ = self.student(eeg, eog, emg)
            temp = self.cfg.pseudo_temp
            log_p_s_t = F.log_softmax(logits_s / temp, dim=1)
            log_q = torch.log(q + 1e-12)
            loss_pl_each = (q * (log_q - log_p_s_t)).sum(dim=1) * (temp * temp)
            loss_pl = (loss_pl_each * wt).mean()

            eeg_1 = self._augment(eeg)
            eeg_2 = self._augment(eeg)
            logits_1, _, _ = self.student(eeg_1, eog, emg)
            logits_2, _, _ = self.student(eeg_2, eog, emg)
            p1 = torch.softmax(logits_1.detach(), dim=1)
            logp2 = F.log_softmax(logits_2, dim=1)
            loss_cons = (p1 * (torch.log(p1 + 1e-12) - logp2)).sum(dim=1).mean()

            loss = self._current_lambda_pl() * loss_pl + self.cfg.cons_weight * loss_cons + self._l2sp()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            self.optimizer.step()

        self.student.eval()

    def _update_alpha(self, p_t: torch.Tensor, p_s: torch.Tensor) -> None:
        loss_t = float(self._entropy(p_t).mean().item())
        loss_s = float(self._entropy(p_s).mean().item())
        self.alpha_t *= float(np.exp(-self.cfg.alpha_eta * loss_t))
        self.alpha_s *= float(np.exp(-self.cfg.alpha_eta * loss_s))
        z = self.alpha_t + self.alpha_s
        if z <= 1e-12:
            self.alpha_t, self.alpha_s = self.cfg.alpha_teacher_init, self.cfg.alpha_student_init
            z = self.alpha_t + self.alpha_s
        self.alpha_t /= z
        self.alpha_s /= z

    def _select_output_probs(self, p_t: torch.Tensor, p_s: torch.Tensor, t: int) -> torch.Tensor:
        if self.cfg.use_student_only:
            if self.cfg.teacher_fallback and (float(torch.max(p_s, dim=1).values.item()) < self.cfg.fallback_conf):
                return p_t
            return p_s

        if t < self.cfg.warmup_steps:
            return p_t

        conf_s = float(torch.max(p_s, dim=1).values.item())
        if self.cfg.teacher_fallback and conf_s < self.cfg.fallback_conf:
            return p_t

        p_out = self.alpha_t * p_t + self.alpha_s * p_s
        p_out = p_out / (p_out.sum(dim=1, keepdim=True) + 1e-12)
        return p_out

    @torch.no_grad()
    def step(self, eeg: torch.Tensor, eog: torch.Tensor, emg: torch.Tensor, t: int) -> Tuple[int, float, np.ndarray, Dict[str, object]]:
        eeg_b = eeg.unsqueeze(0).to(self.device)
        eog_b = eog.unsqueeze(0).to(self.device)
        emg_b = emg.unsqueeze(0).to(self.device)
        prev_pred_used = self.prev_pred

        logits_t, _, _ = self.teacher(eeg_b, eog_b, emg_b)
        logits_s, _, _ = self.student(eeg_b, eog_b, emg_b)
        logits_t, cal_mask_t = apply_n1_bias_calibration_to_logits(logits_t, self.n1_calibration)
        logits_s, cal_mask_s = apply_n1_bias_calibration_to_logits(logits_s, self.n1_calibration)

        p_t_raw = torch.softmax(logits_t, dim=1)
        p_s_raw = torch.softmax(logits_s, dim=1)
        p_t = self._apply_transition_prior_to_logits(logits_t, apply_prior=True)
        p_s = self._apply_transition_prior_to_logits(
            logits_s,
            apply_prior=not self.cfg.transition_prior_teacher_only,
        )
        prior = self._get_transition_prior(logits_t.device)
        self._update_alpha(p_t, p_s)

        p_out_raw = self._select_output_probs(p_t_raw, p_s_raw, t)
        p_out = self._select_output_probs(p_t, p_s, t)
        pred = int(torch.argmax(p_out, dim=1).item())
        conf = float(torch.max(p_out, dim=1).values.item())
        probs = p_out.squeeze(0).cpu().numpy().astype(np.float32)
        pred_raw = int(torch.argmax(p_out_raw, dim=1).item())
        conf_raw = float(torch.max(p_out_raw, dim=1).values.item())

        pseudo = self._teacher_pseudo(p_t, p_s)
        pseudo_meta: Optional[Dict[str, float]] = None
        if pseudo is not None:
            q_cpu, wt, pseudo_meta = pseudo
            self.buffer.append((eeg.cpu(), eog.cpu(), emg.cpu(), q_cpu, wt, self.online_step))

        if (t + 1) % self.cfg.update_every == 0:
            with torch.enable_grad():
                self._train_from_buffer()

        trace = {
            "prev_pred_used": -1 if prev_pred_used is None else int(prev_pred_used),
            "transition_prior_used": int(prior is not None),
            "n1_calibration_used": int(self.n1_calibration is not None and self.n1_calibration.get("enabled", False)),
            "teacher_n1_calibrated": int(torch.any(cal_mask_t).item()),
            "student_n1_calibrated": int(torch.any(cal_mask_s).item()),
            "teacher_raw_pred": int(torch.argmax(p_t_raw, dim=1).item()),
            "teacher_raw_conf": float(torch.max(p_t_raw, dim=1).values.item()),
            "teacher_prior_pred": int(torch.argmax(p_t, dim=1).item()),
            "teacher_prior_conf": float(torch.max(p_t, dim=1).values.item()),
            "student_raw_pred": int(torch.argmax(p_s_raw, dim=1).item()),
            "student_raw_conf": float(torch.max(p_s_raw, dim=1).values.item()),
            "student_prior_pred": int(torch.argmax(p_s, dim=1).item()),
            "student_prior_conf": float(torch.max(p_s, dim=1).values.item()),
            "output_raw_pred": pred_raw,
            "output_raw_conf": conf_raw,
            "output_prior_pred": pred,
            "output_prior_conf": conf,
            "changed_by_prior": int(pred_raw != pred),
            "teacher_prior_support": 1.0 if prior is None else float(prior[0, int(torch.argmax(p_t, dim=1).item())].item()),
            "student_prior_support": 1.0 if prior is None else float(prior[0, int(torch.argmax(p_s, dim=1).item())].item()),
            "output_prior_support": 1.0 if prior is None else float(prior[0, pred].item()),
            "pseudo_accepted": int(pseudo is not None),
            "pseudo_weight": 0.0 if pseudo is None else float(wt),
        }
        if pseudo_meta is not None:
            trace.update(
                {
                    "pseudo_teacher_conf": float(pseudo_meta["teacher_conf"]),
                    "pseudo_teacher_margin": float(pseudo_meta["teacher_margin"]),
                    "pseudo_teacher_entropy": float(pseudo_meta["teacher_entropy"]),
                    "pseudo_prior_support": float(pseudo_meta["prior_support"]),
                }
            )

        self.prev_pred = pred
        self.online_step += 1
        return pred, conf, probs, trace


__all__ = [
    "OnlineCfg",
    "OnlineFeatureAdapter",
    "apply_n1_bias_calibration_to_logits",
    "apply_top2_margin_rules_to_logits",
    "build_sleep_stage_transition_rule_mask",
    "build_transition_matrix_from_labels",
    "normalize_n1_calibration",
    "normalize_top2_margin_rules",
]
