"""
Code for managing experiments through the HParams dashboard.
"""
import os
from typing import Any, Dict, Optional

import tensorflow as tf  # type: ignore
from absl import flags  # type: ignore
from tensorboard.plugins.hparams import api as hp  # type: ignore

from universal_attention import models, train

FLAGS = flags.FLAGS

# You can't compare runs with different tap layers, target_epochs, or
# initializations, so turn these into flags.
flags.DEFINE_multi_string(
    "tap_layer",
    ["conv2_block2_out", "conv3_block3_out", "conv4_block5_out", "post_relu",],
    "Layers to tap into for attention.",
)
flags.DEFINE_integer(
    "target_epochs", 1, "Number of epochs to evaluate the target model with."
)
flags.DEFINE_boolean(
    "imagenet_init", True, "Initialize the network with ImageNet weights."
)

# Hyperparameters
HP_BATCH_SIZE = hp.HParam("batch_size", hp.IntInterval(1, 2048))
HP_ALIGNMENT_SIZE = hp.HParam("alignment_size", hp.IntInterval(1, 2048))
HP_QUERY_SIZE = hp.HParam("query_size", hp.IntInterval(1, 2048))
HP_NUM_SUBTASK_BATCHES = hp.HParam("num_subtask_batches", hp.IntInterval(1, 30))
HP_SUBTASK_LEARNING_RATE = hp.HParam(
    "subtask_learning_rate", hp.RealInterval(1e-6, 1.0)
)
HP_META_LEARNING_RATE = hp.HParam(
    "meta_learning_rate", hp.RealInterval(1e-6, 1.0)
)
HP_TARGET_LEARNING_RATE = hp.HParam(
    "target_learning_rate", hp.RealInterval(1e-6, 1.0)
)

# Default values for hyperparameters
DEFAULT_HPARAMS: Dict[hp.HParam, Any] = {
    HP_BATCH_SIZE: 64,
    HP_ALIGNMENT_SIZE: 256,
    HP_QUERY_SIZE: 512,
    HP_NUM_SUBTASK_BATCHES: 5,
    HP_SUBTASK_LEARNING_RATE: 1e-2,
    HP_META_LEARNING_RATE: 1e-3,
    HP_TARGET_LEARNING_RATE: 1e-4,
}


def run_reptile_experiment(
    run_name: str,
    hparams: Dict[hp.HParam, Any],
    max_steps: Optional[int] = None,
    initial_step: int = 0,
    initial_checkpoint: Optional[str] = None,
    subtask_summary_every: Optional[int] = None,
    checkpoint_every: Optional[int] = None,
    eval_every: Optional[int] = None,
) -> float:
    """
    Run a meta-learning experiment, logging hyperparameters.

    Parameters
    ----------
    run_name: str
        The name to use for this experiment.
    hyperparameters: {hp.HParam -> Any}
        Hyperparameters to override the defaults with.
        If any of the values are None, they are ignored.
    max_steps: int (default: None)
        Number of meta-steps to run this experiment for.
        If None (default), train forever.
    subtask_summary_every: int (default: None)
        Number of meta-steps between plotting summaries for a subtask.
        If 0 or None (default), do not plot summaries for subtasks.
    max_steps: int (default: None)
        Maximum number of steps to train for.
        If 0 or None (default), train forever.
    initial_step: int (default: 0)
        Initial step to begin plotting at. Used to continue training runs.
    initial_checkpoint: path (default: None)
        If provided, path to "meta-classifier" checkpoint to initialize encoder.
    subtask_summary_every: int (default: None)
        Number of meta-steps between plotting summaries for a subtask.
        If 0 or None (default), do not plot summaries for subtasks.
    checkpoint_every: int (default: None)
        Number of meta-steps between saving checkpoints.  If 0 or
        None (default), do not save checkpoints until the end of training.
    eval_every: int (default: None)
        Number of meta-steps between evaluation by transfer-learning to the
        target task. If 0 or None (default), do not evaluate until the end
        of training.

    Returns
    -------
    validation_accuracy: float
        This final validation accuracy for this run.
    """
    new_hparams = {k: v for k, v in hparams.items() if v is not None}
    run_hparams = {**DEFAULT_HPARAMS, **new_hparams}

    encoder = models.QueryAttentionEncoder(
        tap_layers=FLAGS.tap_layer,
        alignment_size=run_hparams[HP_ALIGNMENT_SIZE],
        query_size=run_hparams[HP_QUERY_SIZE],
        imagenet_init=FLAGS.imagenet_init,
    )

    with tf.summary.create_file_writer(FLAGS.tb_log_dir).as_default():
        hp.hparams_config(
            hparams=list(run_hparams.keys()),
            metrics=[
                hp.Metric(
                    "Target training accuracy", dataset_type=hp.Metric.TRAINING,
                ),
                hp.Metric(
                    "Target validation accuracy",
                    dataset_type=hp.Metric.VALIDATION,
                ),
            ],
        )

    log_dir = os.path.join(FLAGS.tb_log_dir, run_name, "meta")
    with tf.summary.create_file_writer(log_dir).as_default():
        hp.hparams(run_hparams)

    return train.train_reptile(
        meta_encoder=encoder,
        run_name=run_name,
        batch_size=run_hparams[HP_BATCH_SIZE],
        num_subtask_batches=run_hparams[HP_NUM_SUBTASK_BATCHES],
        subtask_learning_rate=run_hparams[HP_SUBTASK_LEARNING_RATE],
        meta_learning_rate=run_hparams[HP_META_LEARNING_RATE],
        target_learning_rate=run_hparams[HP_TARGET_LEARNING_RATE],
        max_steps=max_steps,
        initial_step=initial_step,
        initial_checkpoint=initial_checkpoint,
        subtask_summary_every=subtask_summary_every,
        checkpoint_every=checkpoint_every,
        eval_every=eval_every,
        target_epochs=FLAGS.target_epochs,
    )
