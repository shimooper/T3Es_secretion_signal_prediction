from copy import deepcopy
from transformers import TrainerCallback


def get_class_name(obj):
    return obj.__class__.__name__


# Used to log metrics on train set during training (by default metrics are calculated only on evaluation/validation set)
# Solution taken from - https://stackoverflow.com/questions/67457480/how-to-get-the-accuracy-per-epoch-or-step-for-the-huggingface-transformers-train
class CustomCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy
