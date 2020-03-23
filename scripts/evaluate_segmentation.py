#!/usr/bin/env python3

"""
Evaluate a model with attention on zero-shot segmentation.
"""

import pickle

import numpy as np  # type: ignore
from absl import app, flags  # type: ignore

from universal_attention import data, models, segmentation, utils

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "checkpoint", None, "Checkpoint to initialize encoder with.",
)
flags.mark_flag_as_required("checkpoint")

flags.DEFINE_string("save_path", None, "Path to save results to.")
flags.DEFINE_integer(
    "batch_size",
    64,
    "Batch size to use for evaluation. Should not affect results.",
)


def main(_):
    """Run `evaluate_segmentation.py`."""
    class_info = data.load_segmentation_class_info()
    utils.initialize_hardware()
    model = models.load_checkpoint(FLAGS.checkpoint)
    results = segmentation.evaluate_zero_shot(model, FLAGS.batch_size)

    if FLAGS.save_path:
        with open(FLAGS.save_path, "wb") as out_file:
            pickle.dump(results, out_file)

    print("\nRESULTS\n-------")
    for layer, (class_count, class_attention) in results.items():
        stuff_attention = 0.0
        nonstuff_attention = 0.0
        stuff_pixels = 0
        nonstuff_pixels = 0

        print(f"{layer}:")
        for idx, _, _, _, stuff, name in class_info:
            attention = class_attention[idx]
            count = class_count[idx]
            print(f"  {name}: {attention} / {count}")

            if stuff:
                stuff_attention += attention
                stuff_pixels += count
            else:
                nonstuff_attention += attention
                nonstuff_pixels += count

        print(f"  Stuff: {stuff_attention} / {stuff_pixels}")
        print(f"  Not stuff: {nonstuff_attention} / {nonstuff_pixels}")


if __name__ == "__main__":
    app.run(main)
