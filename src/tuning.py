import mlflow
import optuna
import logging

from config import TrainingConfig
from trainer import EfficientNetTrainer

logger = logging.getLogger(__name__)

def objective(trial):
    """
    Optuna objective function to be optimized.
    It suggests hyperparameters, runs a training session, and returns the final validation accuracy.
    """
    # --- Suggest hyperparameters to be tuned ---
    initial_head_lr = trial.suggest_float("initial_head_lr", 1e-4, 1e-2, log=True)
    fine_tune_full_lr = trial.suggest_float("fine_tune_full_lr", 1e-6, 1e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    config_kwargs = {
        "initial_head_lr": initial_head_lr,
        "fine_tune_full_lr": fine_tune_full_lr,
        "batch_size": batch_size,
        "model_name": f"gender_classifier_efficientnet_v2_l_tuned_trial_{trial.number}"
    }
    config = TrainingConfig(**config_kwargs)
    config.setup_logging()

    with mlflow.start_run(run_name=f"optuna_trial_{trial.number}"):
        mlflow.log_params(config.to_dict())
        try:
            trainer = EfficientNetTrainer(config, trial=trial)
            final_accuracy = trainer.train()
            mlflow.log_metric("final_accuracy", final_accuracy)
            return final_accuracy
        except Exception as e:
            logger.error(f"Optuna Trial {trial.number} failed: {e}")
            raise e