#!/usr/bin/env python3

"""
Train and evaluate an encoder checkpoint on the target task.
"""

from absl import app, flags  # type: ignore

from universal_attention import models, train, utils

FLAGS = flags.FLAGS

# Required flags
flags.DEFINE_string(
    "encoder_checkpoint", None, "Encoder to initialize the target model with.",
)
utils.register_path_validator("encoder_checkpoint")

flags.DEFINE_string(
    "run_name", None, "Name to save logs and checkpoints under."
)
flags.mark_flags_as_required(["encoder_checkpoint", "run_name"])

# Optional flags
flags.DEFINE_integer(
    "num_epochs", 30, "Number of epochs to train on the target dataset."
)
flags.DEFINE_integer("batch_size", 64, "Batch size to use.")
flags.DEFINE_float("learning_rate", 1e-4, "Learning rate to use.")


def main(_):
    """Run `train_target.py`."""
    utils.initialize_hardware()
    model = models.load_checkpoint(FLAGS.encoder_checkpoint)
    encoder = model.get_layer("query_attention_encoder")
    train.train_target(
        encoder=encoder,
        num_epochs=FLAGS.num_epochs,
        batch_size=FLAGS.batch_size,
        learning_rate=FLAGS.learning_rate,
        run_name=FLAGS.run_name,
        full_logs=True,
    )


if __name__ == "__main__":
    app.run(main)
