"""
Code for loading data.
"""

import enum
import math
from typing import Callable, Dict, List, Tuple

import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
import tensorflow_datasets as tfds  # type: ignore
from absl import flags  # type: ignore

from universal_attention import utils

FLAGS = flags.FLAGS


# Optional flags
flags.DEFINE_string("data_dir", "./data", "Directory to download data to.")
utils.register_path_validator("data_dir", is_dir=True)

flags.DEFINE_integer(
    "shuffle_buffer_size",
    1024,
    "Size of the shuffle buffer for all datasets.",
    lower_bound=1,
)

DatasetSplits = Dict[str, tf.data.Dataset]
DatasetAndInfo = Tuple[DatasetSplits, tfds.core.DatasetInfo]
Example = Tuple[tf.Tensor, tf.Tensor]
PreprocessFunction = Callable[[tf.Tensor, tf.Tensor], Example]


class DatasetName(enum.Enum):
    """Lists the available datasets."""

    CALTECH_101 = "caltech101"
    BEANS = "beans"
    CALTECH_BIRDS_2011 = "caltech_birds2011"
    IMAGENETTE = "imagenette"
    STANFORD_DOGS = "stanford_dogs"
    OXFORD_IIIT_PET = "oxford_iiit_pet"
    OXFORD_FLOWERS102 = "oxford_flowers102"


# Resize all images to the same size because of speed & memory constraints
IMAGE_SIZES = {
    DatasetName.CALTECH_101: (128, 128),  # Orig: ~(300, 200)
    DatasetName.BEANS: (128, 128),  # Orig: (500, 500)
    DatasetName.CALTECH_BIRDS_2011: (128, 128),  # Orig: ~(300, 500)
    DatasetName.IMAGENETTE: (128, 128),  # Orig: ~(256, 256)
    DatasetName.STANFORD_DOGS: (128, 128),  # Orig: ~(300, 500)
    DatasetName.OXFORD_IIIT_PET: (128, 128),  # Orig: ~(300, 500)
    DatasetName.OXFORD_FLOWERS102: (128, 128),  # Orig: ~(500, 500)
}

TARGET_DATASET = DatasetName.CALTECH_101
META_DATASETS = [ds for ds in DatasetName if ds != TARGET_DATASET]

# Dataset loading API
def load_target_dataset(batch_size: int) -> DatasetAndInfo:
    """
    Load the dataset to be used as the target for meta-learning.
    NOTE: not prefetched!

    Parameters
    ----------
    batch_size: int
        Batch size to load the dataset with.

    Returns
    -------
    dataset, info: DatasetAndInfo
        The meta-target dataset and its information.
    """
    return _load_dataset(TARGET_DATASET, batch_size)


def load_meta_datasets(batch_size: int) -> List[DatasetAndInfo]:
    """
    Load the list of datasets to be used for meta-training.
    NOTE: not prefetched!

    Parameters
    ----------
    batch_size: int
        Batch size to load the dataset with.

    Returns
    -------
    list_of_ds_and_info: [DatasetAndInfo]
        The meta-training datasets and their information.
    """
    return [_load_dataset(ds, batch_size) for ds in META_DATASETS]


def fetch_examples(
    dataset: tf.data.Dataset, num_examples: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the first few examples from a dataset as in-memory ndarrays.

    Parameters
    ----------
    dataset: tf.data.Dataset
        The dataset to draw from.
    num_examples: int
        How many examples to draw.

    Returns
    -------
    examples: (np.ndarray, np.ndarray)
        The (image, label) batches.
    """
    num_batches = math.ceil(num_examples / FLAGS.batch_size)
    batches = list(iter(dataset.take(num_batches)))
    images_batched, labels_batched = zip(*batches)
    images = np.concatenate([batch.numpy() for batch in images_batched], axis=0)
    labels = np.concatenate([batch.numpy() for batch in labels_batched], axis=0)
    return images[:num_examples], labels[:num_examples]


# Internals
def _standardize(images: tf.Tensor, labels: tf.Tensor) -> Example:
    return (images - 128) / 128.0, labels


def _load_dataset(name: DatasetName, batch_size: int) -> DatasetAndInfo:
    if batch_size <= 0:
        raise ValueError("Batch size must be >= 0.")

    splits, info = tfds.load(
        name.value,
        data_dir=FLAGS.data_dir,
        as_supervised=True,
        with_info=True,
        shuffle_files=False,
    )

    def resize_image(image: tf.Tensor, label: tf.Tensor) -> Example:
        return tf.image.resize(image, IMAGE_SIZES[name]), label

    splits_processed = {
        key: ds.shuffle(FLAGS.shuffle_buffer_size)
        .map(resize_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .batch(batch_size)
        .map(_standardize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        for key, ds in splits.items()
    }

    return splits_processed, info
