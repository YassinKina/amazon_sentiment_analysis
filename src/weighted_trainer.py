from transformers import Trainer
import torch.nn as nn

class WeightedTrainer(Trainer):
    def __init__(self, class_weights, nested_bar, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.nested_bar = nested_bar
        self._batch_counter = 0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")

        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        
        # Update batch progress if nested_bar exists
        if self.nested_bar:
            self._batch_counter += 1
            self.nested_bar.update_batch(self._batch_counter)
            
            
        return (loss, outputs) if return_outputs else loss