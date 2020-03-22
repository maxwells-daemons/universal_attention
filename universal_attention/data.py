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


class ClassificationDataset(enum.Enum):
    """Lists the available datasets for classification."""

    CALTECH_101 = "caltech101"
    BEANS = "beans"
    CALTECH_BIRDS_2011 = "caltech_birds2011"
    IMAGENETTE = "imagenette"
    STANFORD_DOGS = "stanford_dogs"
    OXFORD_IIIT_PET = "oxford_iiit_pet"
    OXFORD_FLOWERS102 = "oxford_flowers102"


# Resize all images to the same size because of speed & memory constraints
IMAGE_SIZE = (128, 128)

TARGET_DATASET = ClassificationDataset.CALTECH_101
META_DATASETS = [ds for ds in ClassificationDataset if ds != TARGET_DATASET]

SEGMENTATION_DATASET = "scene_parse150"
SEGMENTATION_CLASSES = 151
SEGMENTATION_METADATA_PATH = "./data/objectInfo150.csv"

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
    return _load_classification_dataset(TARGET_DATASET, batch_size)


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
    return [
        _load_classification_dataset(ds, batch_size) for ds in META_DATASETS
    ]


def load_segmentation_dataset(batch_size: int) -> DatasetAndInfo:
    """
    Load the dataset used for zero-shot segmentation.

    Parameters
    ----------
    batch_size: int
        Batch size to load the dataset with.

    Returns
    -------
    dataset, info: DatasetAndInfo
        The segmentation dataset and its information.
    """
    splits, info = _load_dataset(SEGMENTATION_DATASET, batch_size)
    splits_processed = {
        key: ds.map(
            _resize_segmentation,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        .batch(batch_size)
        .map(
            _standardize_input, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        for key, ds in splits.items()
    }

    return splits_processed, info


def load_segmentation_class_info() -> np.ndarray:
    """
    Load the class information for the segmentation dataset.

    Returns
    -------
    class_info: np.ndarray
        Information for each of the 150 classes, plus the "other" class,
        as a structured array.

        Has fields:
            - Idx: class index (int)
            - Ratio: Fraction of dataset pixels occupied by this class (float)
            - Train: Count of training pixels occupied by this class (int).
                     -1 for the "other" class.
            - Val: Count of validation pixels occupied by this class (int).
                     -1 for the "other" class.
            - Stuff: Whether this class represents "stuffs" (1) or
                    "distinct objects" (0) (bool)
            - Name: Names for this class (string)

    """
    class_info = np.genfromtxt(
        "./data/objectInfo150.csv",
        names=True,
        delimiter=",",
        dtype=[
            ("Idx", "i8"),
            ("Ratio", "f8"),
            ("Train", "i8"),
            ("Val", "i8"),
            ("Stuff", "?"),
            ("Name", "U110"),
        ],
        converters={"Stuff": (lambda x: int(x) > 0)},
    )
    other = (0, 1 - class_info["Ratio"].sum(), -1, -1, True, "other")
    class_info = np.insert(class_info, 0, other)
    return class_info


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
def _standardize_input(images: tf.Tensor, labels: tf.Tensor) -> Example:
    return (images - 128) / 128.0, labels


def _resize_image_and_label(image: tf.Tensor, label: tf.Tensor) -> Example:
    return tf.image.resize(image, IMAGE_SIZE), label


def _resize_segmentation(image: tf.Tensor, target: tf.Tensor) -> Example:
    image = tf.image.resize(image, IMAGE_SIZE)
    target = tf.image.resize(
        target[:, :, :1],  # Extract segmentation mask, leaving channel axis
        IMAGE_SIZE,
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
    )
    return image, target[:, :, 0]  # Squeeze out channel axis


def _load_classification_dataset(
    dataset: ClassificationDataset, batch_size: int
) -> DatasetAndInfo:
    splits, info = _load_dataset(dataset.value, batch_size)
    splits_processed = {
        key: ds.shuffle(FLAGS.shuffle_buffer_size)
        .map(
            _resize_image_and_label,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        .batch(batch_size)
        .map(
            _standardize_input, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        for key, ds in splits.items()
    }

    return splits_processed, info


def _load_dataset(name: str, batch_size: int) -> DatasetAndInfo:
    if batch_size <= 0:
        raise ValueError("Batch size must be >= 0.")

    return tfds.load(
        name,
        data_dir=FLAGS.data_dir,
        as_supervised=True,
        with_info=True,
        shuffle_files=False,
    )
