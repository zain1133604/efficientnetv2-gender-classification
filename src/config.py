import os
import logging
import sys
import json
from datetime import datetime
from logging.handlers import RotatingFileHandler

class TrainingConfig:
    """
    Manages all configuration parameters for the training process.
    Provides methods to initialize from defaults, update from arguments,
    and save the current configuration.
    """
    def __init__(self, **kwargs):
        # Default configuration values
        # >>> IMPORTANT: SET YOUR DEFAULT DATASET PATH HERE <<<
        self.data_dir: str = "A:\\New folder\\project 1\\gender"
        # >>> END IMPORTANT SECTION <<<

        self.checkpoint_base_dir: str = "A:\\model_checkpoints"
        self.model_name: str = "gender_classifier_efficientnet_v2_l"
        self.seed: int = 42
        self.batch_size: int = 16
        self.num_epochs: int = 40
        self.freeze_backbone_epochs: int = 5
        self.initial_head_lr: float = 0.001
        self.fine_tune_full_lr: float = 0.00005
        self.log_interval: int = 5
        self.confusion_matrix_interval: int = 10
        self.train_split_ratio: float = 0.8
        self.image_size: int = 224

        # Update defaults with any provided keyword arguments
        for key, value in kwargs.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)
            elif value is not None:
                logging.warning(f"Unknown configuration parameter: {key}. Ignoring.")

        # Derived paths
        self.checkpoint_dir = os.path.join(self.checkpoint_base_dir, self.model_name)
        self.best_model_path = os.path.join(self.checkpoint_dir, "best_modeln.pth")
        
        # Create unique run directory for logs and plots
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.checkpoint_dir, f"run_{self.run_id}")
        os.makedirs(self.run_dir, exist_ok=True)
    
    def setup_logging(self):
        """Sets up logging to file and console for the entire application."""
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        file_handler = RotatingFileHandler(
            os.path.join(self.run_dir, "training.log"),
            maxBytes=10*1024*1024,
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.encoding = 'utf-8'

        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

        logging.getLogger(__name__).info(f"Configuration for run {self.run_id}:\n{self.to_json_string()}")

    def to_dict(self) -> dict:
        """Converts configuration attributes to a dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_') and not callable(getattr(self, k)) and k != 'logger'}

    def to_json_string(self) -> str:
        """Converts configuration to a pretty-printed JSON string."""
        return json.dumps(self.to_dict(), indent=4)

    def save_config(self):
        """Saves the current configuration to a JSON file in the run directory."""
        config_path = os.path.join(self.run_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
        logging.info(f"Configuration saved to {config_path}")