from transformers import TrainerCallback


class ExperimentLoggerCallback(TrainerCallback):
    """Forward Trainer logs into the configured experiment logger."""

    def __init__(self, logger, model_ref, include_sample_correctness: bool = False):
        super().__init__()
        self.logger = logger
        self.model_ref = model_ref
        self.include_sample_correctness = include_sample_correctness

    def on_train_begin(self, args, state, control, **kwargs):
        self.logger.start()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        payload = dict(logs)
        model_losses = getattr(self.model_ref, "_last_losses", None)
        if model_losses is not None:
            payload.update(model_losses)

        if self.include_sample_correctness:
            sample = getattr(self.model_ref, "_last_sample", None)
            if isinstance(sample, dict) and "gt_norm" in sample and "pred_norm" in sample:
                payload["sample/is_correct"] = float(sample["gt_norm"] == sample["pred_norm"])

        self.logger.log(payload=payload, step=int(state.global_step))

    def on_train_end(self, args, state, control, **kwargs):
        self.logger.finish()
