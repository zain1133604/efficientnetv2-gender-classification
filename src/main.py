import argparse
import sys
import os
import logging
from datetime import datetime

import mlflow
import optuna

from config import TrainingConfig
from trainer import EfficientNetTrainer
from tuning import objective

# Set up logging at the beginning
logger = logging.getLogger(__name__)

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Train an EfficientNetV2-L model for gender classification.")
    
    parser.add_argument(
        "--tune",
        action='store_true',
        help="Enable hyperparameter tuning with Optuna. Runs multiple trials."
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=20,
        help="Number of Optuna trials to run when --tune is enabled."
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Path to the dataset directory."
    )
    parser.add_argument(
        "--checkpoint_base_dir",
        type=str,
        help="Base directory to save models and logs."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name for this training run, used for checkpoint subdirectories."
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for training and validation."
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        help="Number of epochs to train for."
    )
    parser.add_argument(
        "--freeze_backbone_epochs",
        type=int,
        help="Number of initial epochs to train only the classifier head."
    )
    parser.add_argument(
        "--initial_head_lr",
        type=float,
        help="Learning rate for the classifier head when backbone is frozen."
    )
    parser.add_argument(
        "--fine_tune_full_lr",
        type=float,
        help="Learning rate when fine-tuning the entire model."
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        help="How often (in batches) to log training loss."
    )
    parser.add_argument(
        "--confusion_matrix_interval",
        type=int,
        help="How often (in epochs) to plot and save the confusion matrix."
    )
    parser.add_argument(
        "--train_split_ratio",
        type=float,
        help="Ratio of data to use for training (0.0 to 1.0)."
    )
    parser.add_argument(
        "--image_size",
        type=int,
        help="Input image size for the model (e.g., 224 for EfficientNet)."
    )
    args = parser.parse_args()
    return args

def run_single_session(args):
    """Executes a single training session with default or specified parameters."""
    config_kwargs = {k: v for k, v in args.__dict__.items() if v is not None}
    
    config = TrainingConfig(**config_kwargs)
    config.setup_logging()

    mlflow.set_experiment(config.model_name)
    with mlflow.start_run(run_name=f"single_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_params(config.to_dict())

        if not os.path.isdir(config.data_dir):
            logger.error(f"Error: Dataset directory not found or is not a directory: '{config.data_dir}'.")
            sys.exit(1)
        try:
            trainer = EfficientNetTrainer(config)
            trainer.train()
        except FileNotFoundError as fnf_e:
            logger.error(f"A file was not found during execution: {fnf_e}")
            sys.exit(1)
        except ValueError as ve:
            logger.error(f"Configuration or data error: {ve}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}", exc_info=True)
            sys.exit(1)

def main():
    """Main entry point for the training script."""
    args = parse_args()

    if args.tune:
        mlflow.set_experiment("EfficientNetV2-L Hyperparameter Tuning")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=args.n_trials)
        print("\n--- Best Trial Results ---")
        print(f"  Value: {study.best_trial.value:.2f}%")
        print(f"  Params: {study.best_trial.params}")
    else:
        run_single_session(args)

if __name__ == "__main__":
    main()
