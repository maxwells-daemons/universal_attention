"""
Code for evaluating attention models on zero-shot multiclass segmentation.
"""

from typing import Dict, List

import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
from scipy.ndimage import zoom  # type: ignore

from universal_attention import data


def evaluate_zero_shot(
    model: tf.keras.models.Model, batch_size: int
) -> Dict[str, List[np.ndarray]]:
    """
    Evaluate an encoder with attention on zero-shot multiclass segmentation.

    Parameters
    ----------
    model: tf.keras.models.Model
        The (attending) classifier to evaluate.
    batch_size: int
        Batch size to use for evaluation. Should not affect results.

    Returns
    -------
    class_metrics: Dict[str, [np.ndarray, np.ndarray]]
        For each layer, the total attention and total true count assigned
        to each class.
    """
    attention_layers = model.output[1].keys()

    class_metrics = {
        layer: [
            np.zeros(data.SEGMENTATION_CLASSES, dtype=int),
            np.zeros(data.SEGMENTATION_CLASSES),
        ]
        for layer in attention_layers
    }

    splits, _ = data.load_segmentation_dataset(batch_size)

    for batch, (images, classes) in enumerate(splits["train"]):
        if batch % 10 == 0:
            print(f"On batch: {batch}")

        classes_flat = np.reshape(classes.numpy(), -1)
        _, attention_by_layer = model(images)

        for layer, attention in attention_by_layer.items():
            size_ratio = classes.shape[1] / attention.shape[1]
            attention_ = attention.numpy().astype("float32")[:, :, :, 0]
            attention_ = zoom(attention_, [1, size_ratio, size_ratio])
            attention_flat = np.reshape(attention_, -1) / (size_ratio ** 2)

            class_metrics[layer][0] += np.bincount(
                classes_flat, minlength=data.SEGMENTATION_CLASSES
            )
            class_metrics[layer][1] += np.bincount(
                classes_flat,
                weights=attention_flat,
                minlength=data.SEGMENTATION_CLASSES,
            )

    return class_metrics
