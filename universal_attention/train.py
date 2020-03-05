"""
Code for both meta-training and target training.
"""
import os
import random
from typing import Optional

import tensorflow as tf  # type: ignore
from absl import flags, logging  # type: ignore

# pylint:disable=import-error
from tensorflow.keras import (  # type: ignore
    callbacks,
    layers,
    losses,
    metrics,
    optimizers,
)
from universal_attention import data, models, summaries, utils

FLAGS = flags.FLAGS


flags.DEFINE_string(
    "checkpoint_dir",
    "./outputs/checkpoints",
    "Directory to store checkpoints in.",
)
utils.register_path_validator("checkpoint_dir", is_dir=True)

flags.DEFINE_string(
    "tb_log_dir", "./outputs/logs", "Directory to store TensorBoard logs in.",
)
utils.register_path_validator("tb_log_dir", is_dir=True)

flags.DEFINE_integer("random_seed", 1337, "Random seed to use.")


# Training API
def train_target(
    encoder: layers.Layer,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    run_name: Optional[str] = None,
    meta_step: Optional[int] = None,
    full_logs: bool = False,
) -> callbacks.History:
    """
    Use an encoder to create and train a new classifier on the meta-target
    dataset. Modifies the encoder's weights in place.

    Parameters
    ----------
    encoder: tf.keras.layers.Layer
        The encoder to use for this run.
    num_epochs: int
        How many epochs to train for.
    batch_size: int
        Batch size to use for target training.
    learning_rate: float
        Learning rate to use for target model.
    run_name: str (default: None)
        The name to store logs and checkpoints under. Must be provided if
        full_logs is True.
    meta_step: int (default: None)
        If meta-training, which meta-step we're on. Used for logs & checkpoints.
    full_logs: bool (default: False)
        Whether to plot a full set of logs for this training run.

    Returns
    -------
    history: tf.keras.callbacks.History
        The Keras History of this training run.
    """
    tf.random.set_seed(FLAGS.random_seed)

    dataset_and_info = data.load_target_dataset(batch_size)
    splits, info = dataset_and_info

    # Pre-select a fixed set of training and test images to plot for debugging
    train_imgs, train_labels = data.fetch_examples(
        splits["train"], FLAGS.num_debug_images
    )
    test_imgs, test_labels = data.fetch_examples(
        splits["test"], FLAGS.num_debug_images
    )

    classifier = models.attending_classifier(encoder, dataset_and_info)
    classifier.compile(
        optimizers.Adam(learning_rate),
        loss={
            "prediction": losses.SparseCategoricalCrossentropy(from_logits=True)
        },
        metrics={
            "prediction": [
                metrics.SparseCategoricalAccuracy(name="top_1_accuracy"),
                metrics.SparseTopKCategoricalAccuracy(
                    name="top_5_accuracy", k=5
                ),
            ]
        },
    )

    target_callbacks = []
    if full_logs:
        if not run_name:
            raise ValueError("If full_logs is True, run_name must be provided")

        log_dir = os.path.join(FLAGS.tb_log_dir, run_name, "target")
        if meta_step:
            log_dir = os.path.join(log_dir, f"step_{meta_step}")

        target_callbacks.extend(
            [
                callbacks.TensorBoard(log_dir=log_dir, write_graph=False),
                summaries.WatchImages(
                    train_imgs,
                    train_labels,
                    os.path.join(log_dir, "train"),
                    info,
                ),
                summaries.WatchImages(
                    test_imgs,
                    test_labels,
                    os.path.join(log_dir, "validation"),
                    info,
                ),
            ]
        )

    return classifier.fit(
        splits["train"].prefetch(tf.data.experimental.AUTOTUNE),
        validation_data=splits["test"].prefetch(tf.data.experimental.AUTOTUNE),
        epochs=num_epochs,
        callbacks=target_callbacks,
    )


def train_subtask(
    classifier: tf.keras.Model,
    task: data.DatasetAndInfo,
    num_batches: int,
    plot_summaries: bool = False,
    meta_step: Optional[int] = None,
) -> None:
    """
    Use an encoder to create and train a new classifier on a meta-training
    dataset. Modifies the encoder's weights in place.

    Parameters
    ----------
    classifier: tf.keras.Model (tf.Tensor -> models.AttendingClassifierOutput)
        A classifier for this dataset. Assumed to be compiled.
    task: data.DatasetAndInfo
        A dataset yielding training batches for this subtask, and its info.
    num_batches
        How many batches to train for.
    learning_rate
        Learning rate to use for this subtask.
    plot_summaries (default: False)
        Whether to plot summaries for the first batch of this subtask.
    meta_step (default: None)
        Which step of meta-optimization we're on.
        Must be provided if plot_summaries is True.
    """
    # Show plots for the first batch of a subtask at certain steps
    if plot_summaries:
        first_batch = True
        if meta_step is None:
            raise ValueError(
                "If plot_summaries is True, meta_step must be provided."
            )

    dataset, info = task
    for images, labels in dataset["train"].take(num_batches):
        if plot_summaries and first_batch:
            summaries.plot_summaries(
                images, labels, classifier, info, meta_step  # type: ignore
            )
            first_batch = False

        classifier.train_on_batch(images, labels)


def train_reptile(
    meta_encoder: layers.Layer,
    run_name: str,
    batch_size: int,
    num_subtask_batches: int,
    subtask_learning_rate: float,
    meta_learning_rate: float,
    target_learning_rate: float,
    max_steps: Optional[int] = None,
    initial_step: int = 0,
    initial_checkpoint: Optional[str] = None,
    subtask_summary_every: Optional[int] = None,
    checkpoint_every: Optional[int] = None,
    eval_every: Optional[int] = None,
    target_epochs: int = 1,
) -> float:
    """
    Meta-train encoder with Reptile, evaluating on the target task periodically
    and at the end of training.

    NOTE: not all arguments to this function should should be left at their
    default values! This will lead to an infinite training run with no logs,
    checkpoints, or evaluation results.

    Parameters
    ----------
    meta_encoder: tf.keras.layers.Layer
        The encoder to train.
    run_name: str
        The name to associate logs and checkpoints with.
    batch_size: int
        Batch size to use for subtask and target training.
    num_subtask_batches: int
        The number of subtask batches to train on per meta step.
        In the Reptile paper, "k" or "inner loop steps."
    subtask_learning_rate: float
        Learning rate to use for subtask training.
    meta_learning_rate: float
        Learning rate to use for meta-steps.
    target_learning_rate: float
        Learning rate to use when training on the target task.
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
    target_epochs: int (default: 1)
        Number of epochs to train the target model for during evaluation.

    Returns
    -------
    validation_accuracy: float
        The final target validation accuracy for this run.
    """
    print("=== Beginning a Reptile training run ===")
    logging.set_verbosity(logging.ERROR)  # Suppress spurious Keras warnings
    random.seed(FLAGS.random_seed)
    tf.random.set_seed(FLAGS.random_seed)

    checkpoint_dir = os.path.join(FLAGS.checkpoint_dir, run_name, "meta")
    os.makedirs(checkpoint_dir, exist_ok=True)
    meta_writer = tf.summary.create_file_writer(
        os.path.join(FLAGS.tb_log_dir, run_name, "meta")
    )

    def preprocess(splits):
        return {
            split: dataset.repeat().prefetch(tf.data.experimental.AUTOTUNE)
            for split, dataset in splits.items()
        }

    target_task = data.load_target_dataset(batch_size)
    subtasks = [
        (preprocess(splits), info)
        for splits, info in data.load_meta_datasets(batch_size)
    ]

    # Fill out a dummy "meta classifier" so that the two encoders' weights match
    meta_classifier = models.attending_classifier(meta_encoder, target_task)

    # See Reptile paper; momentum found to hurt meta-optimization
    meta_optimizer = optimizers.Adam(meta_learning_rate, beta_1=0.0)
    meta_step = initial_step

    if initial_checkpoint:
        meta_classifier.load_weights(initial_checkpoint)

    # A non-weight-sharing copy of the encoder, used in subtasks to optimize
    # the task network without impacting the meta-weights.
    task_encoder = meta_encoder.from_config(meta_encoder.get_config())

    # Make one full classifier with a persistent head per task
    subtask_models = [
        (models.attending_classifier(task_encoder, task), task)
        for task in subtasks
    ]

    for classifier, _ in subtask_models:
        classifier.compile(
            optimizers.SGD(subtask_learning_rate),
            loss={
                "prediction": losses.SparseCategoricalCrossentropy(
                    from_logits=True
                )
            },
        )

    def target_eval(step):
        print(f"=== Beginning target eval ===")
        task_encoder.set_weights(meta_encoder.get_weights())
        return train_target(
            encoder=task_encoder,
            num_epochs=target_epochs,
            batch_size=batch_size,
            learning_rate=target_learning_rate,
            run_name=run_name,
            meta_step=step,
        ).history

    def subtask_step(step):
        classifier, task = random.choice(subtask_models)
        print(f"[{meta_step}] Training on subtask: {task[1].name}")

        summary = subtask_summary_every and (step % subtask_summary_every == 0)
        task_encoder.set_weights(meta_encoder.get_weights())

        with meta_writer.as_default():
            train_subtask(
                classifier=classifier,
                task=task,
                num_batches=num_subtask_batches,
                plot_summaries=summary,
                meta_step=meta_step,
            )

    while (not max_steps) or (meta_step <= max_steps):
        # Periodic evaluation
        if eval_every and (meta_step % FLAGS.eval_every == 0):
            history = target_eval(meta_step)
            with meta_writer.as_default():
                tf.summary.scalar(
                    "Target training accuracy",
                    history["prediction_top_1_accuracy"][-1],
                    step=meta_step,
                )
                tf.summary.scalar(
                    "Target validation accuracy",
                    history["val_prediction_top_1_accuracy"][-1],
                    step=meta_step,
                )

        # Meta-training
        subtask_step(meta_step)
        meta_gradients = [
            orig - new
            for (orig, new) in zip(meta_encoder.weights, task_encoder.weights)
        ]
        meta_optimizer.apply_gradients(
            zip(meta_gradients, task_encoder.variables)
        )

        # Periodic checkpointing
        if (
            checkpoint_every
            and meta_step
            and (meta_step % checkpoint_every == 0)
        ):
            meta_classifier.save(
                os.path.join(
                    checkpoint_dir, f"{run_name}.meta:{meta_step}.hdf5"
                )
            )

        meta_step += 1

    # Final evaluation
    print("Done with meta-training!")
    eval_history = target_eval(meta_step)
    return eval_history["val_prediction_top_1_accuracy"][-1]
