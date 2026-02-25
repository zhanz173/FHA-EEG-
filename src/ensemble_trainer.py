"""
Deep Ensemble Trainer for Multi-Label Classification with Uncertainty Estimation

We treat each of the k label as an independent Bernoulli. For a Bernoulli with success probability p, the entropy is given by:
h(p) = -p log p - (1-p) log(1-p)

For a total m members in the ensemble, we compute:
Total (predictive) entropy per sample:
First compute per-label entropies from average predictions across members, then sum over labels:
H_total = sum_{k=1}^K h(p_k^bar) -> shape (N,)

Expected entropy (aleatoric uncertainty) per sample:
For each member m, compute per-label entropies, sum over labels, then average across members:
H_expected = 1/M sum_{m=1}^M H_total^{(m)} -> shape (N,)

Multi-label specifics
---------------------
- Assumes **K independent binary labels** (e.g., K=5). Outputs K logits; uses **sigmoid + BCEWithLogitsLoss**.
- Metrics: micro/macro precision/recall/F1; subset accuracy optional; ECE computed **per label** and averaged.
- Uncertainty: uses **Bernoulli entropy** per label `h(p) = -p log p - (1-p) log(1-p)`; total = sum over labels.

"""
from __future__ import annotations
import os
import math
import json
import random
from dataclasses import dataclass, asdict
from typing import Callable, Optional, Dict, Any, Tuple, List
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
import pandas as pd

# ----------------------------
# Utilities
# ----------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def to_device(obj: Any, device: torch.device) -> Any:
    """Recursively move tensors to device inside dicts, lists, tuples, and tensors.

    Keeps non-tensor objects unchanged.
    """
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(to_device(v, device) for v in obj)
    return obj


# ----------------------------
# Metrics (multi-label)
# ----------------------------

def bce_with_logits_loss(logits: torch.Tensor, targets: torch.Tensor, sample_weights: Optional[torch.Tensor] = None, positive_weight: Optional[torch.Tensor] = None, labels_mask: Optional[torch.Tensor] = None, bootstrapping: Optional[float] = 0.0) -> torch.Tensor:
    """Numerically safe BCE-with-logits.
    Args:
        logits: (B, K) raw logits.
        targets: (B, K) binary targets in {0,1} (will be clamped to [0,1]).
        sample_weights: (B, K) optional per-sample/per-label weights.
        positive_weight: (K,) or scalar, passed to BCE-with-logits as pos_weight.
        labels_mask: (B, K) mask for valid labels (0/1).
        bootstrapping: β in [0,1] for soft bootstrapping. 0 -> no bootstrapping; 1 -> targets replaced by model predictions.
            https://research.google/pubs/training-deep-neural-networks-on-noisy-labels-with-bootstrapping/
    Returns:
        Scalar BCE loss.
    """
    targets = targets.float()
    # Clip target range silently (dataset should already provide 0/1 but defensive).
    if (targets.min() < 0) or (targets.max() > 1):
        targets = targets.clamp(0.0, 1.0)

    if bootstrapping and bootstrapping > 0: # y = beta * y + (1-beta) * y_pred
        beta = max(min(bootstrapping, 1.0), 0.0)
        p_pred = torch.sigmoid(logits).detach()
        targets = beta * targets + (1 - beta) * p_pred

    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none', pos_weight=positive_weight) # shape (B,K)

    # apply labels mask if provided
    if labels_mask is not None:
        labels_mask = labels_mask.float().to(loss.device)
        assert labels_mask.shape == loss.shape, "Labels mask shape mismatch"
        loss = loss * labels_mask
    
    if sample_weights is not None:
        sample_weights = sample_weights.float().to(loss.device)
        assert sample_weights.shape == loss.shape, "Sample weights shape mismatch"
        loss = loss * sample_weights
        loss = loss.mean()
    else:
        loss = loss.mean()

    return loss


def focal_bce_with_logits_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sample_weights: Optional[torch.Tensor] = None,
    positive_weight: Optional[torch.Tensor] = None,
    labels_mask: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
    alpha: Optional[torch.Tensor] = None,
    bootstrapping: Optional[float] = 0.0
) -> torch.Tensor:
    """Numerically safe focal BCE-with-logits.

    Args:
        logits: (B, K) raw logits.
        targets: (B, K) binary targets in {0,1} (will be clamped to [0,1]).
        sample_weights: (B, K) optional per-sample/per-label weights.
        positive_weight: (K,) or scalar, passed to BCE-with-logits as pos_weight.
        labels_mask: (B, K) mask for valid labels (0/1).
        gamma: focusing parameter γ >= 0. 0 -> standard BCE.
        alpha: class balancing factor.
            - If scalar: α for positives, (1-α) for negatives.
            - If (K,): per-class α for positives; (1-α_k) for negatives.
        bootstrapping: β in [0,1] for soft bootstrapping. 0 -> no bootstrapping; 1 -> targets replaced by model predictions.
            https://research.google/pubs/training-deep-neural-networks-on-noisy-labels-with-bootstrapping/
    Returns:
        Scalar focal BCE loss.
    """
    targets = targets.float()
    # Defensive clamp
    if (targets.min() < 0) or (targets.max() > 1):
        targets = targets.clamp(0.0, 1.0)

    # Standard BCE-with-logits (per-element), still using pos_weight if given
    if bootstrapping and bootstrapping > 0: # y = beta * y + (1-beta) * y_pred
        beta = max(min(bootstrapping, 1.0), 0.0)
        p_pred = torch.sigmoid(logits).detach()
        targets = beta * targets + (1 - beta) * p_pred
    bce_loss = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=positive_weight
    )  # (B, K)

    # Probabilities for focal modulation term
    probs = torch.sigmoid(logits)
    # p_t = p if y=1 else (1-p)
    p_t = probs * targets + (1.0 - probs) * (1.0 - targets)

    # Focal factor: (1 - p_t)^gamma
    focal_factor = (1.0 - p_t).pow(gamma)

    # Optional alpha-balancing
    if alpha is not None:
        alpha = alpha.to(logits.device).float()
        if alpha.numel() == 1:
            # scalar alpha: α for positive, (1-α) for negative
            alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        else:
            # per-class alpha: shape (K,) or (1,K) -> broadcast to (B,K)
            if alpha.dim() == 1:
                alpha = alpha.view(1, -1)
            alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    else:
        alpha_t = 1.0

    loss = bce_loss * focal_factor * alpha_t  # (B, K)

    # Apply labels mask if provided
    if labels_mask is not None:
        labels_mask = labels_mask.float().to(loss.device)
        assert labels_mask.shape == loss.shape, "Labels mask shape mismatch"
        loss = loss * labels_mask

    # Apply sample weights if provided
    if sample_weights is not None:
        sample_weights = sample_weights.float().to(loss.device)
        assert sample_weights.shape == loss.shape, "Sample weights shape mismatch"
        loss = loss * sample_weights

    return loss.mean()


def sigmoid_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits)


def multilabel_prf1(probs: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Dict[str, Any]:
    """Compute micro/macro precision/recall/F1 for multi-label tasks.
    probs: (B,K) in [0,1]; targets: (B,K) in {0,1}
    """
    targets = targets.float()
    preds = (probs >= threshold).float()
    tp = (preds * targets).sum(dim=0)
    fp = (preds * (1 - targets)).sum(dim=0)
    fn = ((1 - preds) * targets).sum(dim=0)

    # per-label precision/recall/f1
    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)

    macro_p = prec.mean().item()
    macro_r = rec.mean().item()
    macro_f1 = f1.mean().item()

    # micro (sum over labels)
    TP = tp.sum(); FP = fp.sum(); FN = fn.sum()
    micro_p = (TP / (TP + FP + 1e-12)).item()
    micro_r = (TP / (TP + FN + 1e-12)).item()
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r + 1e-12))

    # PR_ROC AREA per label
    pr_aucs, roc_aucs = PR_ROC_curve(targets.detach().cpu().numpy(), probs.detach().cpu().numpy())
    # Convert to torch float for safe loading
    pr_aucs = torch.tensor(pr_aucs, dtype=torch.float32)
    roc_aucs = torch.tensor(roc_aucs, dtype=torch.float32)

    return {
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "micro_f1": float(micro_f1),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "pr_aucs": pr_aucs,
        "roc_aucs": roc_aucs
    }

def PR_ROC_curve(y_true, y_scores):
    """Compute precision-recall and ROC curves for binary classification.

    Args:
        y_true: Ground truth binary labels (N,K).
        y_scores: Predicted scores/probabilities (N,K).

    Returns:
        Area under PR curve and ROC curve for each label.

    """
    from sklearn.metrics import precision_recall_curve, roc_curve, auc
    N, K = y_true.shape
    pr_aucs = []
    roc_aucs = []
    for k in range(K):
        precision, recall, _ = precision_recall_curve(y_true[:, k], y_scores[:, k])
        fpr, tpr, _ = roc_curve(y_true[:, k], y_scores[:, k])
        pr_auc = auc(recall, precision)
        roc_auc = auc(fpr, tpr)
        pr_aucs.append(pr_auc)
        roc_aucs.append(roc_auc)
    return pr_aucs, roc_aucs

def ece_binary_per_label(probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 15) -> float:
    """Expected Calibration Error averaged over labels for binary tasks.
    We bin by predicted probability of the positive class and compare mean predicted prob vs. empirical frequency.
    """
    B, K = probs.shape
    targets = targets.float()
    eces = []
    bin_edges = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    for k in range(K):
        p = probs[:, k]
        y = targets[:, k]
        ece = torch.zeros(1, device=probs.device)
        for i in range(n_bins):
            start, end = bin_edges[i], bin_edges[i+1]
            in_bin = (p > start) * (p <= end)
            prop = in_bin.float().mean()
            if prop > 0:
                p_bin = p[in_bin].mean()
                y_bin = y[in_bin].mean()
                ece += torch.abs(p_bin - y_bin) * prop
        eces.append(ece)
    return torch.stack(eces).mean().item()


def bernoulli_entropy(p: torch.Tensor) -> torch.Tensor:
    eps = 1e-12
    return -(p.clamp(eps, 1 - eps) * (p + eps).log() + (1 - p).clamp(eps, 1).log() * (1 - p).clamp(eps, 1 - eps))


def bernoulli_entropy_sum(p: torch.Tensor) -> torch.Tensor:
    """Sum of Bernoulli entropies over labels: h(p) = -p log p - (1-p) log(1-p).
    p: (N,K) or (M,N,K); returns (N,) when input is (N,K), or (N,) when input is (M,N,K) after summing over K and averaging over M where appropriate.
    """
    eps = 1e-12
    return -(p.clamp(eps, 1 - eps) * (p + eps).log() + (1 - p).clamp(eps, 1) * (1 - p + eps).log()).sum(dim=-1)


def nll_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, targets)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def softmax_nll(probs: torch.Tensor, targets: torch.Tensor) -> float:
    # probs assume softmax already
    eps = 1e-12
    picked = probs[torch.arange(probs.size(0)), targets]
    return (-torch.log(picked + eps)).mean().item()


def expected_calibration_error(probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 15) -> float:
    """ECE with equal-width bins in confidence [0,1]."""
    confidences, predictions = probs.max(dim=1)
    accuracies = (predictions == targets).float()
    ece = torch.zeros(1, device=probs.device)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    for i in range(n_bins):
        start, end = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > start) * (confidences <= end)
        prop = in_bin.float().mean()
        if prop > 0:
            acc_in_bin = accuracies[in_bin].float().mean()
            conf_in_bin = confidences[in_bin].float().mean()
            ece += torch.abs(acc_in_bin - conf_in_bin) * prop
    return ece.item()


def entropy_from_probs(p: torch.Tensor, dim: int = -1) -> torch.Tensor:
    eps = 1e-12
    return -(p * (p + eps).log()).sum(dim=dim)


# ----------------------------
# Trainer
# ----------------------------
@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0
    bootstrapping_targets: float = 0.0 # beta parameter for soft bootstrapping (0=no bootstrapping, 1=targets replaced by model predictions)
    use_positive_weight: bool = False
    use_sample_weight: bool = False
    use_good_labels_only: bool = False
    cosine_schedule: bool = True
    num_workers: int = 4
    amp: bool = True
    early_stop_patience: int = 10
    grad_clip: float = 1.0


class DeepEnsembleTrainer:
    def __init__(
        self,
        num_classes: int,
        model_fn: Callable[[Dict[str, Any]], nn.Module],
        device: Optional[str] = None,
        checkpoint_dir: str = "checkpoints",
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.num_classes = num_classes
        self.model_fn = model_fn
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.model_kwargs = model_kwargs or {}

    def build_model(self) -> nn.Module:
        """Must provide a model builder"""
        return self.model_fn(self.model_kwargs)
    
    def get_positive_weights(self) -> Optional[torch.Tensor]:
        if hasattr(self, 'positive_weights'):
            return self.positive_weights.to(self.device) 
        return None
    
    def _balance_weights(self, ds: Dataset) -> torch.Tensor:
        # Compute per-sample weights to balance classes in multi-label dataset
        # ds assumed to return dicts with "y" key
        if hasattr(self, 'positive_weights'):
            return self.positive_weights  # reuse if already computed
        
        print("Computing sample weights for multi-label balancing...")
        # call dataset get sample weights method
        positive_weights = ds.get_positive_weights() # shape (K,), numpy array
        print(f"Sample weights computed. Stats: min={positive_weights.min().item():.4f}, max={positive_weights.max().item():.4f}, mean={positive_weights.mean().item():.4f}")
        self.positive_weights = torch.tensor(positive_weights, dtype=torch.float32) ## store for later use
        return self.positive_weights

    def _make_loader(self, ds: Dataset, batch_size: int, shuffle: bool, num_workers: int, sampler=None) -> DataLoader:
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle if sampler is None else False,
                          sampler=sampler, num_workers=num_workers, pin_memory=True, drop_last=False)

    def _bootstrap_sampler(self, ds: Dataset, seed: int) -> WeightedRandomSampler:
        # Uniform bootstrap with replacement over N samples
        gen = torch.Generator()
        gen.manual_seed(seed)
        n = len(ds)
        weights = torch.ones(n)
        # Sample n indices with replacement via WeightedRandomSampler
        return WeightedRandomSampler(weights=weights, num_samples=n, replacement=True, generator=gen)

    def _step(self, model: nn.Module, batch: Dict[str, Any], scaler: Optional[torch.cuda.amp.GradScaler] = None,
              optim: Optional[torch.optim.Optimizer] = None, per_sample_weight: bool = False, use_good_labels_only: bool = False, bootstrapping_targets: float = 0.0) -> torch.Tensor:
        x, y = batch["x"], batch["y"] # x: input, y: (B,K) multi-label targets
        sample_weights = batch.get("sample_weights", None) if per_sample_weight else None
        labels_mask = batch.get("labels_mask", None) if use_good_labels_only else None


        # ---- Non-AMP path ----    
        if scaler is None:
            logits, _ = model(x)
            loss_classification = focal_bce_with_logits_loss(logits, y, positive_weight=self.get_positive_weights(), sample_weights=sample_weights, labels_mask=labels_mask, bootstrapping=bootstrapping_targets)
            aux_loss = model.get_aux_loss() # put aux loss or l2 reg loss here if needed; default is zero
            loss = loss_classification + 0.01 * aux_loss
            if optim is not None:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optim.step(); optim.zero_grad(set_to_none=True)
        else:
        # ---- AMP path ----
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits, _= model(x)

            # Compute numerically sensitive loss in full fp32
            with torch.autocast('cuda', enabled=False):
                loss_classification = focal_bce_with_logits_loss(logits.float(), y.float(), positive_weight=self.get_positive_weights(), sample_weights=sample_weights, labels_mask=labels_mask, bootstrapping=bootstrapping_targets)
                aux_loss = model.get_aux_loss()
                loss = loss_classification + 0.01 * aux_loss

            scaler.scale(loss).backward()
            if optim is not None:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)

        return loss.detach()

    @torch.no_grad()
    def _eval(self, model: nn.Module, loader: DataLoader) -> Dict[str, float]:
        model.eval()
        all_logits, all_targets = [], []
        for batch in loader:
            batch = to_device(batch, self.device)
            logits, _ = model(batch["x"]) # ignore age pred here
            all_logits.append(logits)
            all_targets.append(batch["y"])
        logits = torch.cat(all_logits)
        targets = torch.cat(all_targets)
        # Mask out any rows with non-finite targets/logits to avoid NaN/Inf metrics
        row_mask = torch.isfinite(targets).all(dim=1) & torch.isfinite(logits).all(dim=1)
        if row_mask.sum() == 0:
            return {"bce": math.inf, "ece": math.inf, "micro_precision": 0.0, "micro_recall": 0.0, "micro_f1": 0.0, "macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0}
        logits = logits[row_mask]
        targets = targets[row_mask]
        # Multi-label evaluation: sigmoid probs, BCE loss, per-label ECE, PR/F1
        probs = torch.sigmoid(logits)
        bce_t = F.binary_cross_entropy_with_logits(logits, targets.float(), pos_weight=self.get_positive_weights())
        bce = float(bce_t.detach().item()) if torch.isfinite(bce_t) else float("inf")
        prf = multilabel_prf1(probs, targets)
        ece = ece_binary_per_label(probs, targets)
        out = {"bce": bce, "ece": ece}
        out.update(prf)
        return out

    def train_one(
        self,
        train_ds: Dataset,
        val_ds: Dataset,
        seed: int,
        cfg: TrainConfig,
        member_id: int,
        use_bootstrap_sampler: bool = True,
        lr: Optional[float] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        set_seed(seed)
        model = self.build_model().to(self.device)
        print(f"Model has {count_parameters(model)} parameters ({count_parameters(model, trainable_only=True)} trainable).")
        optim = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr or cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = None
        if cfg.cosine_schedule:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs)

        sampler = self._bootstrap_sampler(train_ds, seed) if use_bootstrap_sampler else None
        train_loader = self._make_loader(train_ds, cfg.batch_size, shuffle=(sampler is None), num_workers=cfg.num_workers, sampler=sampler)
        val_loader = self._make_loader(val_ds, cfg.batch_size*2, shuffle=False, num_workers=cfg.num_workers)

        
        scaler = torch.cuda.amp.GradScaler() if cfg.amp else None

        best_val = math.inf
        patience = cfg.early_stop_patience
        best_path = os.path.join(self.checkpoint_dir, f"member{member_id:02d}.pt")

        for epoch in range(cfg.epochs):
            model.train()
            for batch in train_loader:
                batch = to_device(batch, self.device)
                loss = self._step(model, batch, scaler, optim, per_sample_weight=cfg.use_sample_weight, use_good_labels_only=cfg.use_good_labels_only, bootstrapping_targets=cfg.bootstrapping_targets)
                if not torch.isfinite(loss):
                    print(f"[WARN] Non-finite training loss encountered (epoch {epoch}). Batch skipped.")

            if scheduler is not None:
                scheduler.step()
                if verbose:
                    print(f"Epoch {epoch}: Learning rate is now {scheduler.get_last_lr()}")

            # validation
            model.eval()
            with torch.no_grad():
                metrics = self._eval(model, val_loader) 
            val_score = metrics["bce"]
            if verbose:
                print(f"Epoch {epoch}: Validation loss {val_score:.4f}, micro_f1 {metrics['micro_f1']:.4f}, macro_f1 {metrics['macro_f1']:.4f}")
                if isinstance(metrics['pr_aucs'], float):
                    print(f"PR AUCs: {metrics['pr_aucs']:.3f}")
                else:
                    print(f"PR AUCs: {[f'{x:.3f}' for x in metrics['pr_aucs'] ]}")

                if isinstance(metrics['roc_aucs'], float):
                    print(f"ROC AUCs: {metrics['roc_aucs']:.3f}")
                else:
                    print(f"ROC AUCs: {[f'{x:.3f}' for x in metrics['roc_aucs']]}")

            if not math.isfinite(val_score):
                print(f"[WARN] Validation BCE is non-finite (value={val_score}). Ignoring for early stopping/comparison.")
                patience -= 1
                if patience <= 0:
                    break
                continue
            if val_score + 1e-6 < best_val:
                best_val = val_score
                patience = cfg.early_stop_patience
                torch.save({
                    "state_dict": model.state_dict(),
                    "seed": seed,
                    "member_id": member_id,
                    "metrics": metrics,
                }, best_path)
            else:
                patience -= 1
                if patience <= 0:
                    break

        return {"checkpoint": best_path, "best_val_bce": best_val}

    def fit_ensemble(
        self,
        train_ds: Dataset,
        val_ds: Dataset,
        ensemble_size: int = 5,
        cfg: TrainConfig = TrainConfig(),
        base_seed: int = 123,
        use_bootstrap_sampler: bool = True,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        ckpts = []
        # check if need to balance sample weights
        if cfg.use_positive_weight:
            print("Using positive weights for balancing classes in multi-label dataset.")
            _ = self._balance_weights(train_ds) # cache sample weights

        if cfg.use_sample_weight:
            print("Using per-sample weights for balancing classes in multi-label dataset.")


        for m in range(ensemble_size):
            if verbose:
                print(f"Training ensemble member {m+1}/{ensemble_size}...")
            seed = base_seed + m * 9973
            out = self.train_one(train_ds, val_ds, seed=seed, cfg=cfg, member_id=m, use_bootstrap_sampler=use_bootstrap_sampler, verbose=verbose)
            ckpts.append(out["checkpoint"])
            if verbose:
                bce_val = out['best_val_bce']
                if math.isfinite(bce_val):
                    print(f"  -> Best val BCE: {bce_val:.4f}, checkpoint saved to {out['checkpoint']}")
                else:
                    print(f"  -> Best val BCE not finite (value={bce_val}). Check warnings above. Checkpoint path (may be uncalibrated): {out['checkpoint']}")
        return {"checkpoints": ckpts, "checkpoint_dir": self.checkpoint_dir}

    @torch.no_grad()
    def _predict_member(self, model: nn.Module, loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor, List[Any]]:
        model.eval()
        probs_list, targets_list, ids_list = [], [], []
        for batch in loader:
            batch = to_device(batch, self.device)
            logits = model(batch["x"])[0]  # tuple of (logits, ...); we only need logits here
            probs = torch.sigmoid(logits)
            probs_list.append(probs)
            targets_list.append(batch.get("y", torch.full((probs.size(0),), -1, dtype=torch.long, device=probs.device)))
            ids = batch.get("id")
            if ids is None:
                ids_list += [None] * probs.size(0)
            else:
                ids_list += list(ids)
        return torch.cat(probs_list), torch.cat(targets_list), ids_list

    @torch.no_grad()
    def ensemble_uncertainty(
        self,
        ds: Dataset,
        checkpoint_dir: Optional[str] = None,
        checkpoints: Optional[List[str]] = None,
        batch_size: int = 256,
        num_workers: int = 4,
        save_path: Optional[str] = None,
    ):
        if checkpoints is None:
            checkpoint_dir = checkpoint_dir or self.checkpoint_dir
            checkpoints = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
            checkpoints.sort()
        loader = self._make_loader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        member_probs: List[torch.Tensor] = []
        targets = None
        ids = None
        for ck in checkpoints:
            state = torch.load(ck, map_location=self.device, weights_only=False)
            model = self.build_model().to(self.device)
            model.load_state_dict(state["state_dict"]) 
            probs, t, ids_list = self._predict_member(model, loader)
            member_probs.append(probs)
            targets = t if targets is None else targets
            ids = ids_list
        # Stack: (M, N, K)
        P = torch.stack(member_probs, dim=0)
        P_mean = P.mean(dim=0)  # (N,K)

        # --------------------------------------------
        # Entropy decomposition (per-label) using helper
        # --------------------------------------------
        H_total_labels = bernoulli_entropy(P_mean) # (N,K)
        H_expected_labels = bernoulli_entropy(P).mean(dim=0) # (N,K)
        H_epistemic_labels = H_total_labels - H_expected_labels # (N,K)

        var_per_label = P.var(dim=0)  # (N,K)
        var_sum = var_per_label.sum(dim=1)  # (N,)

        # Metrics if targets exist (>=0)
        metrics = {}
        if (targets is not None) and (targets.min().item() >= 0):
            prf1 = multilabel_prf1(P_mean, targets)
            bce = F.binary_cross_entropy(P_mean.clamp(1e-6,1-1e-6), targets).item()
            ece = ece_binary_per_label(P_mean, targets)
            metrics = {**prf1, "bce": bce, "ece": ece}

            
        N, K = P_mean.shape
        # Ensure ids is an iterable of length N; if None, create a list of None so enumerate(...) is safe.
        ids_filled = ids if ids is not None else [None] * N
        data = {
            "id": [i if i is not None else idx for idx, i in enumerate(ids_filled)],
            "conf_mean": P_mean.mean(dim=1).cpu().numpy(),
            "var_sum": var_sum.cpu().numpy(),
        }
        for k in range(K):
            data[f"p_label{k}"] = P_mean[:, k].cpu().numpy()
            data[f"var_label{k}"] = var_per_label[:, k].cpu().numpy()
            data[f"H_total_label{k}"] = H_total_labels[:, k].cpu().numpy()
            data[f"H_expected_label{k}"] = H_expected_labels[:, k].cpu().numpy()
            data[f"H_epistemic_label{k}"] = H_epistemic_labels[:, k].cpu().numpy()
        if (targets is not None) and (targets.min().item() >= 0):
            for k in range(K):
                data[f"y_label{k}"] = targets[:, k].cpu().numpy()
            preds = (P_mean >= 0.5).float()
            subset_acc = (preds.cpu().numpy() == targets.cpu().numpy()).all(axis=1)
            data["subset_acc"] = subset_acc
        
        df = pd.DataFrame(data)
        if save_path is not None:
            df.to_csv(save_path, index=False)
        return df, metrics
