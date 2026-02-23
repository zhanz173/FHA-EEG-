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
from utils import Scaler, TemperatureScalerPerLabel
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

def bce_with_logits_loss(logits: torch.Tensor, targets: torch.Tensor, sample_weights: Optional[torch.Tensor] = None, positive_weight: Optional[torch.Tensor] = None, labels_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Numerically safe BCE-with-logits.

    - Ensures targets are float in [0,1]. Any value outside is clipped and a warning can be logged once.

    """
    targets = targets.float()
    # Clip target range silently (dataset should already provide 0/1 but defensive).
    if (targets.min() < 0) or (targets.max() > 1):
        targets = targets.clamp(0.0, 1.0)

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

    Returns:
        Scalar focal BCE loss.
    """
    targets = targets.float()
    # Defensive clamp
    if (targets.min() < 0) or (targets.max() > 1):
        targets = targets.clamp(0.0, 1.0)

    # Standard BCE-with-logits (per-element), still using pos_weight if given
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



class ModelTrainer:
    """
    Lightweight trainer optimized for CORN-style ordinal supervision.

    Handles mixed precision, label-quality masking, and per-label sample
    weights without any external dependencies beyond PyTorch.
    """
    def __init__(
        self,
        model: nn.Module,
        num_labels: int = 5,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        grad_clip: float = 1.0,
        device: Optional[str] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        mixed_precision: bool = True,
        train_with_neurologist_ids: bool = False,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.grad_clip = grad_clip
        self.scheduler = scheduler
        self.num_labels = num_labels

        self.optimizer = self._setup_optimizer(lr, weight_decay)

        self.mixed_precision = mixed_precision and self.device.type == "cuda"
        self.autocast_dtype = torch.bfloat16 if (self.mixed_precision and torch.cuda.is_bf16_supported()) else torch.float16
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)
        self.train_with_neurologist_ids = train_with_neurologist_ids # learn an embedding per neurologist ID

        self.positive_weight_tensor: Optional[torch.Tensor] = None

    def _setup_optimizer(self, lr: float, weight_decay: float):
        base_params = []
        rater_params = []

        base_params_count = 0
        rater_params_count = 0

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if "rater_emb" in name or "rater_projector" in name:
                rater_params.append(p)
                rater_params_count += p.numel()
            else:
                base_params.append(p)
                base_params_count += p.numel()

        optimizer = torch.optim.AdamW(
            [
                {"params": base_params, "weight_decay": weight_decay},
                {"params": rater_params, "weight_decay": weight_decay * 10},  # stronger shrinkage
            ],
            lr=lr,
        )
        print(f"Optimizer setup: {base_params_count} base params, {rater_params_count} rater params.")

        return optimizer

    def _prepare_batch(self, batch: Dict[str, Any]) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        x_lbr, x_welch = batch["x"]
        x_lbr = x_lbr.to(self.device, non_blocking=True)
        x_welch = x_welch.to(self.device, non_blocking=True)
        y = batch["y"].to(self.device, non_blocking=True)
        if y.dtype != torch.long:
            y = y.long()
        sample_weight = batch.get("sample_weight", None)
        if sample_weight is not None:
            sample_weight = sample_weight.to(self.device, non_blocking=True).float()
        label_mask = batch.get("labels_mask", None)
        if label_mask is not None:
            label_mask = label_mask.to(self.device, non_blocking=True).bool()

        neurologist_ids = batch.get("neurologist_id", None)
        if neurologist_ids is not None:
            neurologist_index = neurologist_ids.to(self.device, non_blocking=True).long()

        return (x_lbr, x_welch), y, sample_weight, label_mask, neurologist_index
    
        
    def _weighted_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_weight: Optional[torch.Tensor],
        label_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        loss = focal_bce_with_logits_loss(
            logits,
            targets,
            sample_weights=sample_weight,
            positive_weight=self.positive_weight_tensor,
            labels_mask=label_mask,
        )
        return loss

    def _compute_metrics(self, logits: torch.Tensor, targets: torch.Tensor, label_mask: Optional[torch.Tensor]) -> Dict[str, Any]:
        '''
        total_valid: int - number of valid labels (after masking)
        total_correct: int - number of correct predictions (after masking)
        per_label_correct: torch.Tensor - number of correct predictions per label
        per_label_valid: torch.Tensor - number of valid labels per label
        '''
        preds = torch.sigmoid(logits) >= 0.5  # threshold at 0.5
        valid = label_mask.bool() if label_mask is not None else torch.ones_like(targets, dtype=torch.bool)
        correct_mask = (preds == targets) & valid
        per_label_correct = correct_mask.sum(dim=0).to(torch.float64).cpu()
        per_label_valid = valid.sum(dim=0).to(torch.float64).cpu()

        total_valid = per_label_valid.sum().item()
        total_correct = per_label_correct.sum().item()

        return {
            "total_valid": total_valid,
            "total_correct": total_correct,
            "per_label_correct": per_label_correct,
            "per_label_valid": per_label_valid,
        }

    def current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _aggregate_epoch(
        self,
        loss_total: float,
        sample_count: int,
        metric_sums: Dict[str, float],
        per_label_correct: Optional[torch.Tensor],
        per_label_valid: Optional[torch.Tensor],
        pr_auc_scores: Optional[List[float]] = None,
        roc_auc_scores: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        avg_loss = loss_total / max(sample_count, 1)
        total_valid = metric_sums["valid"]
        accuracy = metric_sums["correct"] / total_valid if total_valid > 0 else 0.0

        per_label_acc: List[float] = []
        if per_label_correct is not None and per_label_valid is not None:
            per_label_acc = (per_label_correct / per_label_valid.clamp(min=1)).tolist()
        
        pr_auc_scores = pr_auc_scores if pr_auc_scores is not None else []
        roc_auc_scores = roc_auc_scores if roc_auc_scores is not None else []

        return {"loss": avg_loss, "accuracy": accuracy,  "per_label_accuracy": per_label_acc, "lr": self.current_lr(), "pr_auc": pr_auc_scores, "roc_auc": roc_auc_scores}

    def train_epoch(self, loader: DataLoader, log_interval: int = 0, use_label_mask: bool = False, use_sample_weights: bool = False) -> Dict[str, Any]:
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        metric_sums = {"correct": 0.0, "valid": 0.0}
        per_label_correct = None
        per_label_valid = None

        for step, batch in enumerate(loader):
            (x_lbr, x_welch), y, weights, label_mask, neurologist_ids = self._prepare_batch(batch)
            neurologist_ids = neurologist_ids if self.train_with_neurologist_ids else None # mean rater embedding handled by model

            self.optimizer.zero_grad(set_to_none=True)
            if self.mixed_precision:
                with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                    logits = self.model((x_lbr, x_welch), neurologist_ids=neurologist_ids, use_mean_rater=True) #training mean rater embedding
                    loss = self._weighted_loss(logits, y, 
                                                    weights if use_sample_weights else None,
                                                    label_mask if use_label_mask else None)
                self.scaler.scale(loss).backward()
                if self.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model((x_lbr, x_welch), neurologist_ids=neurologist_ids, use_mean_rater=True) #training mean rater embedding
                loss = self._weighted_loss(logits, y, 
                                                weights if use_sample_weights else None, 
                                                label_mask if use_label_mask else None)
                loss.backward()
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            total_loss += loss.item() * y.size(0)
            total_samples += y.size(0)

            with torch.no_grad():
                metrics = self._compute_metrics(logits, y > 0.5, label_mask)
                metric_sums["correct"] += metrics["total_correct"]
                metric_sums["valid"] += metrics["total_valid"]
                if per_label_correct is None:
                    per_label_correct = metrics["per_label_correct"]
                    per_label_valid = metrics["per_label_valid"]
                else:
                    per_label_correct += metrics["per_label_correct"]
                    per_label_valid += metrics["per_label_valid"]

            if log_interval and ((step + 1) % log_interval == 0):
                step_acc = metrics["total_correct"] / max(metrics["total_valid"], 1)
                print(f"[train] step {step+1}: loss={loss.item():.4f}, acc={step_acc:.4f}")

        if self.scheduler is not None and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step()

        return self._aggregate_epoch(total_loss, total_samples, metric_sums, per_label_correct, per_label_valid)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, Any]:
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        metric_sums = {"correct": 0.0, "valid": 0.0}
        per_label_correct = None
        per_label_valid = None

        all_logits = []
        all_targets = []

        for batch in loader:
            (x_lbr, x_welch), y, weights, label_mask, neurologist_ids = self._prepare_batch(batch)
            logits = self.model((x_lbr, x_welch), neurologist_ids=None, use_mean_rater=True) # eval with mean rater embedding, ignore neurologist IDs
            loss = self._weighted_loss(logits, y, weights, label_mask)
            total_loss += loss.item() * y.size(0)
            total_samples += y.size(0)

            metrics = self._compute_metrics(logits, y>0.5, label_mask) # prevent continuous labels during metric computation
            metric_sums["correct"] += metrics["total_correct"]
            metric_sums["valid"] += metrics["total_valid"]
            if per_label_correct is None:
                per_label_correct = metrics["per_label_correct"]
                per_label_valid = metrics["per_label_valid"]
            else:
                per_label_correct += metrics["per_label_correct"]
                per_label_valid += metrics["per_label_valid"]

            all_logits.append(logits)
            all_targets.append(y)  # N, C

        all_logits = torch.cat(all_logits, dim=0)  # [N, L]

        y_targets = torch.cat(all_targets, dim=0)  # [N, L]
        y_probs = torch.sigmoid(all_logits)
        pr_auc_scores, roc_auc_scores = PR_ROC_curve(y_targets.cpu(), y_probs.cpu() )

        return self._aggregate_epoch(total_loss, total_samples, metric_sums, per_label_correct, per_label_valid, pr_auc_scores, roc_auc_scores)
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 1,
        log_interval: int = 0,
        use_label_mask: bool = False,
        use_sample_weights: bool = False,
        use_positive_weight: bool = False,
    ) -> List[Dict[str, Any]]:
        history: List[Dict[str, Any]] = []
        if use_positive_weight:
            positive_weight = train_loader.dataset.get_positive_weights()
            self.positive_weight_tensor = torch.tensor(positive_weight, dtype=torch.float32).to(self.device)
        else:
            self.positive_weight_tensor = None

        for epoch in range(1, epochs + 1):
            train_stats = self.train_epoch(train_loader, log_interval=log_interval, use_label_mask=use_label_mask, use_sample_weights=use_sample_weights)
            val_stats = self.evaluate(val_loader) if val_loader is not None else None
            if self.scheduler is not None and isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric = (val_stats or train_stats)["loss"]
                self.scheduler.step(metric)
            history.append({"epoch": epoch, "train": train_stats, "val": val_stats})
            if val_stats is not None:
                print(
                    f"[epoch {epoch}] train_loss={train_stats['loss']:.4f}, "
                    f"val_loss={val_stats['loss']:.4f}, val_acc={val_stats['accuracy']:.4f}",
                )
                print(f"per_label_acc={val_stats['per_label_accuracy']}")
                print(f"val_pr_auc={val_stats['pr_auc']}")
                print(f"val_roc_auc={val_stats['roc_auc']}")
            else:
                print(f"[epoch {epoch}] train_loss={train_stats['loss']:.4f}, train_acc={train_stats['accuracy']:.4f}")
        return history

    @torch.no_grad()
    def predict(
        self,
        loader: DataLoader,
        return_logits: bool = False,
    ) -> Dict[str, torch.Tensor]:
        self.model.eval()
        preds, targets, logits_store = [], [], []
        for batch in loader:
            (x_lbr, x_welch), y, _, _, _ = self._prepare_batch(batch)
            logits = self.model((x_lbr, x_welch),)
            preds.append(torch.sigmoid(logits).cpu())
            targets.append(y.cpu())
            if return_logits:
                logits_store.append(logits.cpu())
        out = {
            "predictions": torch.cat(preds) if preds else torch.empty(0, self.num_labels),
            "targets": torch.cat(targets) if targets else torch.empty(0, self.num_labels),
        }

        return out

    def save_checkpoint(self, path: str):
        import os
        # check if directory exists
        dir_name = os.path.dirname(path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "num_labels": self.num_labels,
            "embedding_dim": self.model.rater_emb.embedding_dim if hasattr(self.model, "rater_emb") else None,
            "num_neurologists": self.model.num_neurologists if hasattr(self.model, "num_neurologists") else None,
        }
        torch.save(payload, path)

    def load_checkpoint(self, path: str, strict: bool = True):
        payload = torch.load(path, map_location=self.device)
        self.model.load_state_dict(payload["model"], strict=strict)
        if "optimizer" in payload and payload["optimizer"] is not None:
            self.optimizer.load_state_dict(payload["optimizer"])
        if self.scheduler is not None and payload.get("scheduler") is not None:
            self.scheduler.load_state_dict(payload["scheduler"])


if __name__ == "__main__":
    from dataloader import EEGDatasetWithLabel
    from model import GRU_Classifier
    FHA_EEG_FEATURES_ROOT = r"H:\EEG_features\EEG_features_labram_welch"
    CURRENT_FILEPATH = os.path.dirname(os.path.abspath(__file__))
    train_ds = EEGDatasetWithLabel(root=FHA_EEG_FEATURES_ROOT, metadata=r"H:\Thesis_Project\Essembles\experiment\train_set.csv", return_ids=True, return_ordinal=False, return_neurologist_ids=True)
    val_ds = EEGDatasetWithLabel(root=FHA_EEG_FEATURES_ROOT, metadata=r"H:\Thesis_Project\Essembles\experiment\val_set.csv", return_ids=True, return_ordinal=False, return_neurologist_ids=True)
    n_neurologists = train_ds.get_valide_neurologist_counts()

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2)

    trainer = ModelTrainer(
        model=GRU_Classifier(hidden_size=128, num_layers=4, num_classes=5, num_neurologists=n_neurologists, rater_emb_dim=4),
        lr=3e-4,
        weight_decay=1e-3,
        grad_clip=1.0,
        device="cuda" if torch.cuda.is_available() else "cpu",
        mixed_precision=False,
        train_with_neurologist_ids=True,
    )

    history = trainer.fit(train_loader, val_loader, epochs=10, log_interval=100, use_sample_weights=True, use_positive_weight=False) # test run

    trainer.save_checkpoint(CURRENT_FILEPATH+"\\experiment\\gru_with_id_checkpoint_2.pth")

    