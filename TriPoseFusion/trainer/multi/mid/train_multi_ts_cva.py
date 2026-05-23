#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: train_multi_ts_cva.py
Project: project/trainer/multi
Created Date: 2026-02-09
Author: Kaixu Chen
-----
Comment:
Enhanced Trainer for Temporal-Synchronous Cross-View Attention (TS-CVA) model.

This trainer implements the training and validation loop for TS-CVA with
advanced features including:
- Comprehensive metrics logging and visualization
- Attention/gate weight visualization support
- Gradient accumulation for large batches
- Mixed precision training support
- Early stopping and model checkpointing

Have a good code time :)
-----
Copyright (c) 2026 The University of Tsukuba
-----
"""

from typing import Any, Dict, Optional, List
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
    MulticlassConfusionMatrix,
)

from project.models.ts_cva_model import TSCVAModel
from project.utils.helper import save_helper

logger = logging.getLogger(__name__)


class MultiTSCVATrainer(LightningModule):
    """
    Enhanced Trainer for Temporal-Synchronous Cross-View Attention (TS-CVA) model.
    
    This trainer provides comprehensive training, validation, and testing
    functionality for the TS-CVA model, with support for multiple views
    (front, left, right) and extensive monitoring capabilities.
    
    Expected batch format:
        batch["video"]["front"] : (B, C, T, H, W)
        batch["video"]["left"]  : (B, C, T, H, W)
        batch["video"]["right"] : (B, C, T, H, W)
        batch["label"]          : (B,)
        batch["info"]           : Optional[List[Dict]] - metadata
    
    Features:
        - Multi-view video action recognition with TS-CVA
        - Cross-view attention visualization
        - Gated view aggregation weight analysis
        - Comprehensive metrics tracking (accuracy, precision, recall, F1)
        - Confusion matrix logging
        - Learning rate scheduling
        - Gradient clipping for stable training
    """
    
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters()
        
        # Extract hyperparameters
        self.img_size = hparams.data.img_size
        self.lr = hparams.loss.lr
        self.num_classes = hparams.model.model_class_num
        self.grad_clip_val = getattr(hparams.train, 'grad_clip_val', 1.0)
        
        # Initialize TS-CVA model
        self.model = TSCVAModel(hparams)
        
        # Initialize metrics for training
        self.train_accuracy = MulticlassAccuracy(num_classes=self.num_classes)
        self.train_precision = MulticlassPrecision(num_classes=self.num_classes, average='macro')
        self.train_recall = MulticlassRecall(num_classes=self.num_classes, average='macro')
        self.train_f1_score = MulticlassF1Score(num_classes=self.num_classes, average='macro')
        
        # Initialize metrics for validation
        self.val_accuracy = MulticlassAccuracy(num_classes=self.num_classes)
        self.val_precision = MulticlassPrecision(num_classes=self.num_classes, average='macro')
        self.val_recall = MulticlassRecall(num_classes=self.num_classes, average='macro')
        self.val_f1_score = MulticlassF1Score(num_classes=self.num_classes, average='macro')
        self.val_confusion_matrix = MulticlassConfusionMatrix(num_classes=self.num_classes)
        
        # Initialize metrics for testing
        self.test_accuracy = MulticlassAccuracy(num_classes=self.num_classes)
        self.test_precision = MulticlassPrecision(num_classes=self.num_classes, average='macro')
        self.test_recall = MulticlassRecall(num_classes=self.num_classes, average='macro')
        self.test_f1_score = MulticlassF1Score(num_classes=self.num_classes, average='macro')
        self.test_confusion_matrix = MulticlassConfusionMatrix(num_classes=self.num_classes)
        
        # Storage for attention and gate weights visualization
        self.val_attention_weights = []
        self.val_gate_weights = []
        self.val_predictions = []
        self.val_labels = []
        
        # Test predictions and labels for saving results
        self.test_pred_list = []
        self.test_label_list = []
        
        # Training statistics
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
        # Save path for results (same root as other trainers, used by save_helper)
        self.save_root = getattr(hparams, "log_path", None) or getattr(hparams.train, "log_path", None)
        
        logger.info(f"MultiTSCVATrainer initialized with {self.num_classes} classes")
        logger.info(f"Learning rate: {self.lr}, Image size: {self.img_size}")
        
    def forward(
        self, 
        videos: Dict[str, torch.Tensor],
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through the TS-CVA model.
        
        Args:
            videos: Dictionary containing video tensors for each view
                - 'front': (B, C, T, H, W)
                - 'left': (B, C, T, H, W)
                - 'right': (B, C, T, H, W)
            return_attention: Whether to store attention weights for visualization
            
        Returns:
            logits: (B, num_classes) - classification logits
        """
        return self.model(videos, return_attention=return_attention)
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        Training step for a single batch.
        
        Args:
            batch: Batch data containing videos and labels
            batch_idx: Index of the current batch
            
        Returns:
            loss: Training loss for the batch
        """
        videos = batch["video"]
        labels = batch["label"].long()
        B = labels.size(0)
        
        # Forward pass (no attention weights during training for efficiency)
        logits = self.forward(videos, return_attention=False)
        
        # Compute loss
        loss = F.cross_entropy(logits, labels)
        
        # Compute predictions and probabilities
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        # Update metrics
        self.train_accuracy.update(probs, labels)
        self.train_precision.update(probs, labels)
        self.train_recall.update(probs, labels)
        self.train_f1_score.update(probs, labels)
        
        # Log loss
        self.log("train/loss", loss, on_step=True, on_epoch=True, 
                 batch_size=B, prog_bar=True, sync_dist=True)
        
        # Store outputs for epoch-level logging
        self.training_step_outputs.append({
            'loss': loss.detach(),
            'preds': preds.detach(),
            'labels': labels.detach(),
        })
        
        return loss
    
    def on_train_epoch_end(self) -> None:
        """
        Called at the end of the training epoch.
        Compute and log epoch-level metrics.
        """
        # Compute epoch metrics
        train_acc = self.train_accuracy.compute()
        train_prec = self.train_precision.compute()
        train_rec = self.train_recall.compute()
        train_f1 = self.train_f1_score.compute()
        
        # Log epoch metrics
        self.log_dict({
            "train/epoch_acc": train_acc,
            "train/epoch_precision": train_prec,
            "train/epoch_recall": train_rec,
            "train/epoch_f1": train_f1,
        }, sync_dist=True)
        
        # Reset metrics for next epoch
        self.train_accuracy.reset()
        self.train_precision.reset()
        self.train_recall.reset()
        self.train_f1_score.reset()
        
        # Clear stored outputs
        self.training_step_outputs.clear()
        
        logger.info(f"Train Epoch End - Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Validation step for a single batch.
        
        Args:
            batch: Batch data containing videos and labels
            batch_idx: Index of the current batch
            
        Returns:
            Dictionary containing validation metrics
        """
        videos = batch["video"]
        labels = batch["label"].long()
        B = labels.size(0)
        
        # Forward pass with attention weights for visualization
        logits = self.forward(videos, return_attention=True)
        
        # Compute loss
        loss = F.cross_entropy(logits, labels)
        
        # Compute predictions and probabilities
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        # Update metrics
        self.val_accuracy.update(probs, labels)
        self.val_precision.update(probs, labels)
        self.val_recall.update(probs, labels)
        self.val_f1_score.update(probs, labels)
        self.val_confusion_matrix.update(probs, labels)
        
        # Log loss
        self.log("val/loss", loss, on_step=False, on_epoch=True, 
                 batch_size=B, prog_bar=True, sync_dist=True)
        
        # Store attention and gate weights for visualization (first few batches only)
        if batch_idx < 10:
            attn_weights = self.model.get_attention_weights()
            gate_weights = self.model.get_gate_weights()
            
            if attn_weights is not None:
                self.val_attention_weights.append(attn_weights.detach().cpu())
            if gate_weights is not None:
                self.val_gate_weights.append(gate_weights.detach().cpu())
            
            self.val_predictions.append(preds.detach().cpu())
            self.val_labels.append(labels.detach().cpu())
        
        # Store outputs for epoch-level logging
        self.validation_step_outputs.append({
            'loss': loss.detach(),
            'preds': preds.detach(),
            'labels': labels.detach(),
        })
        
        return {
            "val_loss": loss,
            "val_preds": preds,
            "val_labels": labels,
        }
    
    def on_validation_epoch_end(self) -> None:
        """
        Called at the end of the validation epoch.
        Compute and log epoch-level metrics and visualizations.
        """
        # Compute epoch metrics
        val_acc = self.val_accuracy.compute()
        val_prec = self.val_precision.compute()
        val_rec = self.val_recall.compute()
        val_f1 = self.val_f1_score.compute()
        
        # Log epoch metrics
        self.log_dict({
            "val/epoch_acc": val_acc,
            "val/epoch_precision": val_prec,
            "val/epoch_recall": val_rec,
            "val/epoch_f1": val_f1,
        }, sync_dist=True)
        
        # Log confusion matrix if logger supports it
        if hasattr(self.logger, 'experiment'):
            confusion_mat = self.val_confusion_matrix.compute()
            logger.info(f"Confusion Matrix:\n{confusion_mat}")
        
        # Log attention and gate weight statistics
        if self.val_gate_weights:
            gate_weights_tensor = torch.cat(self.val_gate_weights, dim=0)  # (N, T, num_views)
            
            # Compute mean gate weights across all samples and timesteps
            mean_gate_weights = gate_weights_tensor.mean(dim=(0, 1))  # (num_views,)
            
            self.log_dict({
                "val/gate_weight_front": mean_gate_weights[0],
                "val/gate_weight_left": mean_gate_weights[1],
                "val/gate_weight_right": mean_gate_weights[2],
            }, sync_dist=True)
            
            logger.info(f"Val Gate Weights - Front: {mean_gate_weights[0]:.4f}, "
                       f"Left: {mean_gate_weights[1]:.4f}, "
                       f"Right: {mean_gate_weights[2]:.4f}")
        
        # Reset metrics for next epoch
        self.val_accuracy.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1_score.reset()
        self.val_confusion_matrix.reset()
        
        # Clear visualization data
        self.val_attention_weights.clear()
        self.val_gate_weights.clear()
        self.val_predictions.clear()
        self.val_labels.clear()
        
        # Clear stored outputs
        self.validation_step_outputs.clear()
        
        logger.info(f"Val Epoch End - Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
    
    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Test step for a single batch.
        
        Args:
            batch: Batch data containing videos and labels
            batch_idx: Index of the current batch
            
        Returns:
            Dictionary containing test metrics
        """
        videos = batch["video"]
        labels = batch["label"].long()
        B = labels.size(0)
        
        # Forward pass with attention weights
        logits = self.forward(videos, return_attention=True)
        
        # Compute loss
        loss = F.cross_entropy(logits, labels)
        
        # Compute predictions and probabilities
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        # Update metrics
        self.test_accuracy.update(probs, labels)
        self.test_precision.update(probs, labels)
        self.test_recall.update(probs, labels)
        self.test_f1_score.update(probs, labels)
        self.test_confusion_matrix.update(probs, labels)
        
        # Log loss
        self.log("test/loss", loss, on_step=False, on_epoch=True, 
                 batch_size=B, sync_dist=True)
        
        # Store outputs
        self.test_step_outputs.append({
            'loss': loss.detach(),
            'preds': preds.detach(),
            'labels': labels.detach(),
        })
        
        # Store predictions and labels for save_helper
        # Store probabilities instead of class indices for AUROC calculation
        self.test_pred_list.append(probs.detach())
        self.test_label_list.append(labels.detach())
        
        return {
            "test_loss": loss,
            "test_preds": preds,
            "test_labels": labels,
        }
    
    def on_test_epoch_end(self) -> None:
        """
        Called at the end of the test epoch.
        Compute and log final test metrics, then save results using save_helper.
        """
        # Compute test metrics
        test_acc = self.test_accuracy.compute()
        test_prec = self.test_precision.compute()
        test_rec = self.test_recall.compute()
        test_f1 = self.test_f1_score.compute()
        
        # Log test metrics
        self.log_dict({
            "test/epoch_acc": test_acc,
            "test/epoch_precision": test_prec,
            "test/epoch_recall": test_rec,
            "test/epoch_f1": test_f1,
        }, sync_dist=True)
        
        # Log confusion matrix
        confusion_mat = self.test_confusion_matrix.compute()
        logger.info(f"Test Confusion Matrix:\n{confusion_mat}")
        
        # Reset metrics
        self.test_accuracy.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_f1_score.reset()
        self.test_confusion_matrix.reset()
        
        # Clear stored outputs
        self.test_step_outputs.clear()
        
        logger.info(f"Test End - Acc: {test_acc:.4f}, Precision: {test_prec:.4f}, "
                   f"Recall: {test_rec:.4f}, F1: {test_f1:.4f}")
        
        # Save results using save_helper
        if self.test_pred_list and self.test_label_list:
            # Determine fold name from logger (expected: fold_0, fold_1, ...)
            fold = (
                Path(getattr(self.logger, "root_dir", "fold")).name
                if self.logger
                else "fold"
            )

            # Determine save path (root experiment log dir)
            save_path = self.save_root
            if save_path is None:
                if self.logger and hasattr(self.logger, "root_dir"):
                    save_path = str(Path(self.logger.root_dir).parent)
                else:
                    save_path = "./logs"
            
            logger.info(f"Saving test results to {save_path}")
            save_helper(
                all_pred=self.test_pred_list,
                all_label=self.test_label_list,
                fold=fold,
                save_path=save_path,
                num_class=self.num_classes,
            )
            logger.info("Test results saved successfully")
        
        # Clear test predictions and labels
        self.test_pred_list.clear()
        self.test_label_list.clear()
    
    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.
        
        Returns:
            Dictionary containing optimizer and scheduler configuration
        """
        # Adam optimizer with weight decay
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            weight_decay=1e-5
        )
        
        # ReduceLROnPlateau scheduler to reduce LR when validation accuracy plateaus
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            # verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/epoch_acc",
                "interval": "epoch",
                "frequency": 1,
            },
        }
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        """
        Override optimizer step to add gradient clipping.
        """
        # Clip gradients to prevent exploding gradients
        if self.grad_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip_val)
        
        # Update weights
        optimizer.step(closure=optimizer_closure)
    
    def get_progress_bar_dict(self):
        """
        Customize progress bar to show important metrics.
        """
        items = super().get_progress_bar_dict()
        # Remove version number from progress bar
        items.pop("v_num", None)
        return items
