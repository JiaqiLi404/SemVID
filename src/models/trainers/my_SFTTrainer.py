from trl import SFTTrainer
import logging
from transformers.trainer_callback import PrinterCallback

from src.models.builder import TRAINERS
from src.models.utils.callbacks import CUDAMemoryCallback, LoggerMetricsCallback

logger = logging.getLogger(__name__)


@TRAINERS.register_module()
class MySFTTrainer(SFTTrainer):
    def __init__(self, model, *args, **kwargs):
        super().__init__(
            model=model,
            args=kwargs.get('args'),
            data_collator=kwargs.get('data_collator'),
            train_dataset=kwargs.get('train_dataset'),
            eval_dataset=kwargs.get('eval_dataset'),
            processing_class=kwargs.get('processing_class'),
            compute_loss_func=kwargs.get('compute_loss_func'),
            compute_metrics=kwargs.get('compute_metrics'),
            preprocess_logits_for_metrics=kwargs.get('preprocess_logits_for_metrics'),
            peft_config=kwargs.get('peft_config'),
            formatting_func=kwargs.get('formatting_func'),
            callbacks=[CUDAMemoryCallback(every_n_steps=1, reset_peak=False), LoggerMetricsCallback()],
        )
        self.remove_callback(PrinterCallback)
