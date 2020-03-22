#!/usr/bin/env python3

"""
Run a hyperparameter search on meta-learning an attention model.
"""

import math

import hyperopt  # type: ignore
from absl import app, flags  # type: ignore

from universal_attention import experiments, utils

FLAGS = flags.FLAGS

# Required flags
flags.DEFINE_string(
    "run_name", None, "Name to group all optimization runs under."
)
flags.mark_flag_as_required("run_name")

# Optional experiment flags
flags.DEFINE_integer(
    "steps_per_experiment", 200, "Number of steps to run each experiment."
)
flags.DEFINE_integer(
    "target_learning_rate",
    1e-4,
    "Target learning rate to use for all experiments.",
)
flags.DEFINE_boolean(
    "imagenet_init", True, "Initialize the network with ImageNet weights."
)
flags.DEFINE_integer("batch_size", 64, "Batch size to use for all experiments.")

# Logistical flags
flags.DEFINE_integer("max_evals", 10000, "Maximum number of runs to initiate.")

# Flags to manually hold parts of the search space fixed
flags.DEFINE_integer(
    "alignment_size",
    None,
    "Dimension of each attention layer's parametric alignment vector.",
)
flags.DEFINE_integer(
    "query_size", None, "Dimension of the global query vector.",
)
flags.DEFINE_integer(
    "num_subtask_batches",
    None,
    "How many optimization steps to perform on each sampled subtask ('k').",
)
flags.DEFINE_float(
    "subtask_learning_rate",
    None,
    "Learning rate to use for subtask optimization.",
)
flags.DEFINE_float(
    "meta_learning_rate",
    None,
    "Learning rate to use for meta-optimization ('epsilon').",
)


GLOBAL_COUNT = 0  # Ensure that each run gets a unique name


def objective(args):
    """Objective of optimization."""
    global GLOBAL_COUNT  # pylint: disable=global-statement
    GLOBAL_COUNT += 1

    hparams = {
        experiments.HP_BATCH_SIZE: FLAGS.batch_size,
        experiments.HP_ALIGNMENT_SIZE: int(args["alignment_size"]),
        experiments.HP_QUERY_SIZE: int(args["query_size"]),
        experiments.HP_NUM_SUBTASK_BATCHES: int(args["num_subtask_batches"]),
        experiments.HP_SUBTASK_LEARNING_RATE: args["subtask_learning_rate"],
        experiments.HP_META_LEARNING_RATE: args["meta_learning_rate"],
    }

    return experiments.run_reptile_experiment(
        run_name=f"{FLAGS.run_name}_{GLOBAL_COUNT}",
        hparams=hparams,
        max_steps=FLAGS.steps_per_experiment,
    )


def main(_):
    """Run `run_attention_experiment.py`."""
    utils.initialize_hardware()

    search_space = {
        "alignment_size": FLAGS.alignment_size
        or hyperopt.hp.quniform("alignment_size", 4, 1024, 1),
        "query_size": FLAGS.query_size
        or hyperopt.hp.quniform("query_size", 4, 1024, 1),
        "num_subtask_batches": FLAGS.num_subtask_batches
        or hyperopt.hp.quniform("num_subtask_batches", 2, 15, 1),
        "subtask_learning_rate": FLAGS.subtask_learning_rate
        or hyperopt.hp.loguniform("subtask_learning_rate", math.log(1e-5), -1),
        "meta_learning_rate": FLAGS.meta_learning_rate
        or hyperopt.hp.loguniform("meta_learning_rate", math.log(1e-5), -1),
    }

    hyperopt.fmin(
        objective,
        search_space,
        algo=hyperopt.atpe.suggest,
        max_evals=FLAGS.max_evals,
    )


if __name__ == "__main__":
    app.run(main)
