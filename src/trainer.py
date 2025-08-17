import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import logging
import random
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import mlflow
import optuna.exceptions

from config import TrainingConfig
from dataset import prepare_dataloaders
from model import setup_model

# Check for mixed precision compatibility
try:
    from torch.cuda.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False
    print("Warning: Automatic Mixed Precision (AMP) not available. Install PyTorch 1.6+ for speedup on CUDA devices.")

logger = logging.getLogger(__name__)

class EfficientNetTrainer:
    """
    Handles the entire training lifecycle for the EfficientNetV2-L model,
    including setup, training, validation, checkpointing, and metric plotting.
    """
    def __init__(self, config: TrainingConfig, trial=None):
        self.config = config
        self.trial = trial

        self._set_seed()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        self.train_loader, self.val_loader, self.class_names = prepare_dataloaders(self.config)

        if not self.class_names:
            raise ValueError("Class names could not be determined from the dataset. Please check data directory.")
        self.num_classes = len(self.class_names)
        logger.info(f"Detected {self.num_classes} classes.")

        self.model = setup_model(self.num_classes, self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        self.optimizer = None
        self.scheduler = None

        self.best_val_accuracy = 0.0
        self.start_epoch = 0

        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

        if AMP_AVAILABLE:
            self.scaler = GradScaler()
            logger.info("Automatic Mixed Precision (AMP) is enabled.")
        else:
            self.scaler = None
            logger.warning("Automatic Mixed Precision (AMP) is NOT enabled. Training might be slower.")

    def _set_seed(self):
        """Sets random seeds for reproducibility across different libraries."""
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        random.seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def _reinitialize_optimizer_and_scheduler(self, lr, is_frozen_phase):
        if is_frozen_phase:
            optimizer_params = self.model.classifier.parameters()
        else:
            optimizer_params = self.model.parameters()
            
        self.optimizer = optim.AdamW(optimizer_params, lr=lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            verbose=True
        )

    def _load_best_checkpoint(self):
        """
        Scans for all available checkpoints and loads the one with the highest
        validation accuracy to resume training. Returns the starting epoch and best accuracy.
        """
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        checkpoint_files = [f for f in os.listdir(self.config.checkpoint_dir) if f.startswith('checkpoint_') and f.endswith('.pth')]

        if not checkpoint_files:
            logger.info("No checkpoints found, starting training from scratch.")
            return 0, 0.0

        best_checkpoint_path = None
        highest_accuracy = -1.0
        
        logger.info("Scanning for best checkpoint to resume from...")
        for filename in checkpoint_files:
            match = re.match(r'checkpoint_epoch_(\d+)_acc_([\d.]+)\.pth', filename)
            if match:
                epoch = int(match.group(1))
                accuracy = float(match.group(2))
                if accuracy > highest_accuracy:
                    highest_accuracy = accuracy
                    best_checkpoint_path = os.path.join(self.config.checkpoint_dir, filename)

        if not best_checkpoint_path:
            logger.warning("No valid checkpoints with accuracy metadata found. Starting from scratch.")
            return 0, 0.0

        logger.info(f"Loading best checkpoint from {best_checkpoint_path}...")
        try:
            checkpoint = torch.load(best_checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])

            # --- CORRECTION HERE ---
            # The checkpoint's epoch value is zero-indexed, representing the last completed epoch.
            # So, the next epoch to train is `epoch + 1`.
            start_epoch = checkpoint.get('epoch', -1) + 1
            best_val_accuracy = checkpoint.get('best_val_accuracy', 0.0)
            
            # Re-initialize optimizer/scheduler based on the checkpoint's progress
            is_frozen_phase = start_epoch <= self.config.freeze_backbone_epochs
            self._reinitialize_optimizer_and_scheduler(
                self.config.initial_head_lr if is_frozen_phase else self.config.fine_tune_full_lr,
                is_frozen_phase
            )

            if 'optimizer_state_dict' in checkpoint and len(self.optimizer.param_groups) == len(checkpoint['optimizer_state_dict']['param_groups']):
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                logger.info("Optimizer and scheduler state successfully loaded.")
            else:
                logger.warning("Optimizer state could not be loaded due to parameter group mismatch. Re-initializing.")
                
            self.train_losses = checkpoint.get('train_losses', [])
            self.train_accuracies = checkpoint.get('train_accuracies', [])
            self.val_losses = checkpoint.get('val_losses', [])
            self.val_accuracies = checkpoint.get('val_accuracies', [])
            
            logger.info(f"Successfully loaded checkpoint. Resuming training from epoch {start_epoch + 1} with best validation accuracy: {best_val_accuracy:.2f}%")
            return start_epoch, best_val_accuracy
        
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {best_checkpoint_path}: {e}. Starting training from scratch.")
            return 0, 0.0

    def _save_checkpoint(self, epoch: int, val_acc: float):
        """
        Saves the current training state as a unique checkpoint file
        and also saves the model weights if it's the best so far.
        """
        # --- CORRECTION HERE ---
        # Changed the epoch value in the filename to be consistent
        checkpoint_filename = f"checkpoint_epoch_{epoch+1}_acc_{val_acc:.2f}.pth"
        checkpoint_path = os.path.join(self.config.checkpoint_dir, checkpoint_filename)
        
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_accuracy': val_acc,
            'best_val_accuracy': self.best_val_accuracy,
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
        }
        torch.save(checkpoint_data, checkpoint_path)
        logger.info(f"Checkpoint saved for epoch {epoch+1} to {checkpoint_path}")
        
        if val_acc > self.best_val_accuracy:
            torch.save(checkpoint_data, self.config.best_model_path)
            logger.info(f"Updated best model checkpoint at {self.config.best_model_path}")

    def train_epoch(self, epoch: int):
        """Performs one training epoch."""
        self.model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}/{self.config.num_epochs}")
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            with autocast(enabled=AMP_AVAILABLE):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            if (batch_idx + 1) % self.config.log_interval == 0:
                progress_bar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    acc=f"{(100 * correct / total):.2f}%",
                    lr=self.optimizer.param_groups[0]['lr']
                )

        avg_train_loss = train_loss / len(self.train_loader)
        train_acc = 100 * correct / total
        return avg_train_loss, train_acc

    def validate_epoch(self, epoch: int):
        """Performs one validation epoch."""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc=f"Validation Epoch {epoch+1}/{self.config.num_epochs}")
            for images, labels in progress_bar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                with autocast(enabled=AMP_AVAILABLE):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{(100 * correct / total):.2f}%")

        avg_val_loss = val_loss / len(self.val_loader)
        val_acc = 100 * correct / total
        return avg_val_loss, val_acc, all_preds, all_labels

    def train(self):
        """Executes the main training loop across all epochs."""
        self.config.save_config()

        self.start_epoch, self.best_val_accuracy = self._load_best_checkpoint()
        
        is_frozen_phase = self.start_epoch <= self.config.freeze_backbone_epochs
        
        # We need to re-initialize the optimizer and scheduler here if a checkpoint wasn't loaded
        # or if the loaded checkpoint was from the frozen phase
        if not self.optimizer:
            self._reinitialize_optimizer_and_scheduler(self.config.initial_head_lr, is_frozen_phase=True)

        logger.info("Starting training...")

        for epoch in range(self.start_epoch, self.config.num_epochs):
            logger.info(f"\nEpoch {epoch+1}/{self.config.num_epochs}")

            if (epoch + 1) == (self.config.freeze_backbone_epochs + 1):
                logger.info("Unfreezing backbone for fine-tuning...")
                for param in self.model.parameters():
                    param.requires_grad = True

                self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config.fine_tune_full_lr, weight_decay=1e-4)
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=2, verbose=True)
                logger.info(f"Optimizer and scheduler re-initialized for fine-tuning with LR: {self.config.fine_tune_full_lr}")

            avg_train_loss, train_acc = self.train_epoch(epoch)
            self.train_losses.append(avg_train_loss)
            self.train_accuracies.append(train_acc)

            logger.info(f"---- Entering validation phase for Epoch {epoch+1}...")
            avg_val_loss, val_acc, all_preds, all_labels = self.validate_epoch(epoch)
            self.val_losses.append(avg_val_loss)
            self.val_accuracies.append(val_acc)
            logger.info(f"---- Exited validation phase for Epoch {epoch+1}.")

            self.scheduler.step(avg_val_loss)

            logger.info(f"--- Train Loss: {avg_train_loss:.4f}, Accuracy: {train_acc:.2f}%")
            logger.info(f"--- Val   Loss: {avg_val_loss:.4f}, Accuracy: {val_acc:.2f}%")

            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)

            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                os.makedirs(os.path.dirname(self.config.best_model_path), exist_ok=True)
                torch.save(self.model.state_dict(), self.config.best_model_path)
                logger.info(f"-------Saved best model at epoch {epoch+1} with val accuracy: {val_acc:.2f}% to {self.config.best_model_path}")
                mlflow.log_artifact(self.config.best_model_path)

            self._save_checkpoint(epoch, val_acc)

            if (epoch + 1) % self.config.confusion_matrix_interval == 0:
                self._plot_confusion_matrix(all_labels, all_preds, epoch)

            if self.trial:
                self.trial.report(val_acc, epoch)
                if self.trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        logger.info("🎉 Training complete!")
        self._plot_metrics()
        mlflow.log_artifact(os.path.join(self.config.run_dir, "training_metrics.png"))
        
        return self.best_val_accuracy

    def _plot_confusion_matrix(self, true_labels: list, predictions: list, epoch: int):
        """Plots and saves the confusion matrix for the given epoch."""
        cm = confusion_matrix(true_labels, predictions, labels=range(len(self.class_names)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.class_names)
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
        ax.set_title(f"Confusion Matrix (Epoch {epoch+1})")
        plt.tight_layout()
        plot_path = os.path.join(self.config.run_dir, f"confusion_matrix_epoch_{epoch+1}.png")
        plt.savefig(plot_path)
        logger.info(f"💾 Confusion matrix saved to {plot_path}")
        plt.close(fig)
        mlflow.log_artifact(plot_path)

    def _plot_metrics(self):
        """Plots and saves the training and validation loss and accuracy curves."""
        if not self.train_losses:
            logger.warning("No training data to plot metrics.")
            return

        epochs = range(1, len(self.train_losses) + 1)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_losses, 'b', label='Training loss')
        plt.plot(epochs, self.val_losses, 'r', label='Validation loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_accuracies, 'b', label='Training accuracy')
        plt.plot(epochs, self.val_accuracies, 'r', label='Validation accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plot_path = os.path.join(self.config.run_dir, "training_metrics.png")
        plt.savefig(plot_path)
        logger.info(f"Training metrics plot saved to {plot_path}")
        plt.close()