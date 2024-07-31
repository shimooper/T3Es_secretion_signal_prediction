import os.path
from copy import deepcopy
from transformers import TrainerCallback
from sklearn.metrics import average_precision_score
from evaluate import load
import numpy as np

HF_CACHE_DIR = "/groups/pupko/yairshimony/.cache/huggingface"
HF_MODELS_CACHE_DIR = os.path.join(HF_CACHE_DIR, "hub")
HF_EVALUATE_CACHE_DIR = os.path.join(HF_CACHE_DIR, "evaluate")

mcc_metric = load("matthews_correlation", cache_dir=HF_EVALUATE_CACHE_DIR)

# Used to log metrics on train set during training at the end of each epoch (by default metrics are calculated only on evaluation/validation set)
# Solution taken from - https://stackoverflow.com/questions/67457480/how-to-get-the-accuracy-per-epoch-or-step-for-the-huggingface-transformers-train
# The calculated loss here appears in the logs as "train/train_loss" and is the loss of the train_set at the end of each epoch.
# This differs from the automatically logged "train/loss" which is the average loss of all steps during the epoch.
class CalcMetricsOnTrainSetCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy


def compute_metrics(eval_pred):
    predictions, true_labels = eval_pred
    predictions_labels = np.argmax(predictions, axis=1)
    predictions_scores = predictions[:, 1]  # probability estimates of the positive class

    scores = mcc_metric.compute(predictions=predictions_labels, references=true_labels)
    scores['auprc'] = average_precision_score(y_true=true_labels, y_score=predictions_scores)
    return scores
