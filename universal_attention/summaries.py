"""
Utilities to make and display summaries.
"""

import io
from typing import Callable

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
import tensorflow_datasets as tfds  # type: ignore
from absl import flags  # type: ignore

FLAGS = flags.FLAGS


flags.DEFINE_integer(
    "num_debug_images",
    8,
    "Number of images to track for debugging, zero to disable plotting.",
)


def figure_to_image(figure: plt.Figure) -> tf.Tensor:
    """
    Convert a matplotlib figure to an image tensor.
    Borrowed from https://www.tensorflow.org/tensorboard/image_summaries.

    NOTE: consumes the figure window in the process.

    Parameters
    ----------
    figure: plt.Figure
        The figure to convert.

    Returns
    -------
    image_tensor: tf.Tensor
        A Tensor able to be used with tf.summary.image
        (but lacking a batch dimension).
    """
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png")
    plt.close(figure)
    buffer.seek(0)
    img = tf.image.decode_png(buffer.getvalue(), channels=3)
    return tf.expand_dims(img, 0)


def plot_summaries(
    images: np.ndarray,
    labels: np.ndarray,
    model: tf.keras.Model,
    ds_info: tfds.core.DatasetInfo,
    step: int,
) -> None:
    """
    Plot classification performance and attention on a set of images.

    Parameters
    ----------
    images: np.ndarray
        Images to use.
    labels: np.ndarray
        True integer labels for the images.
    model: np.ndarray
        A Model: images -> AttendingClassifierOutput.
    ds_info: tfds.core.DatasetInfo
        Info object for the dataset associated with these images.
    step: int
        X-axis for the summaries.
    """
    if not FLAGS.num_debug_images:
        return

    logits, attention_maps = model.predict_on_batch(
        images[: FLAGS.num_debug_images]
    )
    probs = tf.nn.softmax(logits, axis=-1)

    tf.summary.image(
        "Input images",
        _make_examples_summary(
            images=images,
            label_fn=ds_info.features["label"].int2str,
            true_labels=labels,
            pred_probs=probs.numpy(),
        ),
        step=step,
        max_outputs=FLAGS.num_debug_images,
    )

    for name, attention in attention_maps.items():
        tf.summary.histogram(
            f"Attention distribution ({name})", attention, step=step
        )

        tf.summary.image(
            f"Attention maps ({name})",
            _make_attention_summary(
                images,
                label_fn=ds_info.features["label"].int2str,
                true_labels=labels,
                pred_probs=probs.numpy(),
                attention=attention.numpy(),
            ),
            step=step,
            max_outputs=FLAGS.num_debug_images,
        )


# pylint: disable=too-few-public-methods
class WatchImages(tf.keras.callbacks.Callback):
    """
    A Callback which watches predictions and attention on a fixed set of images.

    Parameters
    ----------
    images: np.ndarray
        Images to watch.
    labels: np.ndarray
        True integer labels for the images.
    log_dir: str
        Directory to store logs in.
    ds_info: tfds.core.DatasetInfo
        Info object for the dataset associated with these images.
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        log_dir: str,
        ds_info: tfds.core.DatasetInfo,
    ):
        super(WatchImages, self).__init__()
        self.images = images
        self.labels = labels
        self.writer = tf.summary.create_file_writer(log_dir)
        self.ds_info = ds_info

    # pylint:disable=missing-function-docstring
    def on_epoch_end(self, epoch, _=None):
        with self.writer.as_default():
            plot_summaries(
                self.images, self.labels, self.model, self.ds_info, epoch,
            )


# Internals
def _make_examples_summary(
    images: np.ndarray,
    label_fn: Callable[[int], str],
    true_labels: np.ndarray,
    pred_probs: np.ndarray,
) -> tf.Tensor:
    def plot(image, true_label, predicted_label, predicted_prob):
        true_class = label_fn(true_label)
        predicted_class = label_fn(predicted_label)

        figure, axes = plt.subplots()
        axes.imshow((image + 1) / 2)
        axes.set_axis_off()
        figure.suptitle(
            f"True: {true_class} [{true_label}]\n"
            f"Predicted: {predicted_class} [{predicted_label}, "
            f"{predicted_prob:.3f}]"
        )
        return figure_to_image(figure)

    pred_labels = np.argmax(pred_probs, axis=1)
    pred_probs_matching = np.max(pred_probs, axis=1)
    plots = [
        plot(*args)
        for args in zip(images, true_labels, pred_labels, pred_probs_matching,)
    ]
    return tf.concat(plots, axis=0)


def _make_attention_summary(
    images: np.ndarray,
    label_fn: Callable[[int], str],
    true_labels: np.ndarray,
    pred_probs: np.ndarray,
    attention: np.ndarray,
) -> tf.Tensor:
    # See: https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/
    #      layer_images.html
    def plot(image, true_label, predicted_label, predicted_prob, attention):
        true_class = label_fn(true_label)
        predicted_class = label_fn(predicted_label)

        figure, axes = plt.subplots()
        extent = (0, 1, 0, 1)
        axes.imshow((image + 1) / 2, interpolation="bilinear", extent=extent)
        axes.imshow(
            attention.astype("float32"),
            cmap="viridis",
            alpha=0.5,
            interpolation="nearest",
            extent=extent,
        )
        axes.set_axis_off()
        figure.suptitle(
            f"True: {true_class} [{true_label}]\n"
            f"Predicted: {predicted_class} [{predicted_label}, "
            f"{predicted_prob:.3f}]"
        )
        return figure_to_image(figure)

    pred_labels = np.argmax(pred_probs, axis=1)
    pred_probs_matching = np.max(pred_probs, axis=1)
    plots = [
        plot(*args)
        for args in zip(
            images,
            true_labels,
            pred_labels,
            pred_probs_matching,
            attention[:, :, :, 0],
        )
    ]
    return tf.concat(plots, axis=0)
