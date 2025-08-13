# efficientnetv2-gender-classification
High-accuracy gender classification model using EfficientNetV2-L, trained on a custom 181K+ image dataset with robust checkpointing, overfitting prevention, and power outage recovery.
# Gender classification with EfficientNetV2-L — 96.18% accuracy on 181K+ custom images
# Gender Classification with EfficientNetV2-L

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-red)
![Accuracy](https://img.shields.io/badge/Accuracy-96.18%25-green)
![Dataset Size](https://img.shields.io/badge/Dataset-181K%2B%20images-orange)

A **high-accuracy gender classification model** trained on a custom dataset of **181K+ images** of male and female faces.  
This project focuses on:
- Large-scale dataset management.
- Power outage recovery during training.
- Overfitting prevention with modern deep learning techniques.

---

## 📂 Dataset

1. **Initial Source:** 10K prebuilt Kaggle gender dataset.
2. **Custom Collection:** Expanded to 250K+ images.
3. **Cleaning Process:**
   - Deleted duplicates.
   - Bulk renamed files.
   - Balanced classes.
   - Reduced to **181,225 clean images**.
4. **Class Distribution:**
   - `female/`
   - `male/`

---

## ⚙️ Model Details

| Feature                  | Value                          |
|--------------------------|--------------------------------|
| Backbone                 | EfficientNetV2-L               |
| Image Size               | 224×224                        |
| Classes                  | Female, Male                   |
| Batch Size               | 16                             |
| Num Epochs               | 40                             |
| Mixed Precision (AMP)    | ✅ Enabled                      |
| Train/Val Split          | 80% / 20%                      |

---

## 🛡️ Power Outage Recovery

Implemented a **robust checkpointing system** to handle unexpected power loss:
- Saves **every epoch checkpoint** (model, optimizer, scheduler).
- Keeps **best accuracy model** separately.
- Auto-scans and resumes from highest-accuracy checkpoint.
- Preserves optimizer & scheduler state for continued training.

---

## 📉 Overfitting Prevention

Techniques applied:
- Weight Decay.
- Strong Data Augmentation.
- Learning Rate Scheduling.
- Freeze Backbone (first 5 epochs).
- Separate LR for backbone & head.

---

## 📝 Experiment Tracking

- **MLflow** for logging metrics & parameters.
- **Optuna** for hyperparameter tuning.
- **Confusion Matrix** generated every 10 epochs.

---

## 🔧 Training Config (Latest Run)

```json
{
  "model_name": "gender_classifier_efficientnet_v2_l",
  "seed": 42,
  "batch_size": 16,
  "num_epochs": 40,
  "freeze_backbone_epochs": 5,
  "initial_head_lr": 0.001,
  "fine_tune_full_lr": 5e-05,
  "train_split_ratio": 0.8,
  "image_size": 224
}
