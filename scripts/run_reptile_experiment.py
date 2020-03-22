#!/usr/bin/env python3

"""
Make a model with attention, meta-train it with Reptile, and evaluate on the
target task.
"""

from absl import app, flags  # type: ignore

from universal_attention import experiments, utils

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "run_name", None, "The name to group logs and checkpoints under."
)
flags.mark_flag_as_required("run_name")

# Experiment flags
flags.DEFINE_integer(
    "max_steps",
    2000,
    "How many meta-training steps to run for, or 0 to train forever.",
)
flags.DEFINE_string(
    "initial_checkpoint",
    None,
    "If provided, initial checkpoint to initialize encoder with.",
)
flags.DEFINE_integer("initial_step", 0, "Initial step to plot.")
flags.DEFINE_integer(
    "subtask_summary_every",
    30,
    "Number of meta-steps between plotting summaries for subtasks.",
)
flags.DEFINE_integer(
    "checkpoint_every", 200, "Number of meta-steps between saving checkpoints.",
)
flags.DEFINE_integer(
    "eval_every", 50, "Number of meta-steps between target evaluations.",
)

# Model flags
flags.DEFINE_integer(
    "alignment_size",
    256,
    "Dimension of each attention layer's parametric alignment vector.",
)
flags.DEFINE_integer(
    "query_size", 512, "Dimension of the global query vector.",
)

# Meta-training flags; defaults are not necessarily optimal
flags.DEFINE_integer(
    "num_subtask_batches",
    5,
    "How many optimization steps to perform on each sampled subtask ('k').",
)
flags.DEFINE_float(
    "meta_learning_rate",
    1e-3,
    "Learning rate to use for meta-optimization ('epsilon').",
)

# Training flags
flags.DEFINE_integer(
    "batch_size", 64, "Batch size to use for target and subtask training.",
)
flags.DEFINE_float(
    "subtask_learning_rate",
    1e-2,
    "Learning rate to use for subtask optimization.",
)
flags.DEFINE_float(
    "target_learning_rate",
    1e-4,
    "Learning rate to use when training on the target task during transfer.",
)


def main(_):
    """Run `run_attention_experiment.py`."""
    utils.initialize_hardware()

    experiments.run_reptile_experiment(
        run_name=FLAGS.run_name,
        hparams={
            experiments.HP_BATCH_SIZE: FLAGS.batch_size,
            experiments.HP_ALIGNMENT_SIZE: FLAGS.alignment_size,
            experiments.HP_QUERY_SIZE: FLAGS.query_size,
            experiments.HP_NUM_SUBTASK_BATCHES: FLAGS.num_subtask_batches,
            experiments.HP_SUBTASK_LEARNING_RATE: FLAGS.subtask_learning_rate,
            experiments.HP_META_LEARNING_RATE: FLAGS.meta_learning_rate,
            experiments.HP_TARGET_LEARNING_RATE: FLAGS.target_learning_rate,
        },
        max_steps=FLAGS.max_steps,
        initial_step=FLAGS.initial_step,
        initial_checkpoint=FLAGS.initial_checkpoint,
        subtask_summary_every=FLAGS.subtask_summary_every,
        checkpoint_every=FLAGS.checkpoint_every,
        eval_every=FLAGS.eval_every,
    )


if __name__ == "__main__":
    app.run(main)
