"""
Various project-wide utilities that don't fit well anywhere else.
"""

import os

import tensorflow as tf  # type: ignore
from absl import flags  # type: ignore

# pylint: disable=import-error
from tensorflow.keras import mixed_precision  # type: ignore

FLAGS = flags.FLAGS


flags.DEFINE_boolean(
    "mixed_precision", True, "Whether to train with mixed precision."
)

_INITIALIZED = False


def register_path_validator(flag_name: str, is_dir: bool = False) -> None:
    """
    Register a validator ensuring that `flag_name` is an existing file.

    Parameters
    ----------
    flag_name: str
        Name of the flag to register a validator for.
    is_dir: bool (default: False)
        Whether the file must also be a directory.
    """
    if is_dir:
        flags.register_validator(
            flag_name,
            _dir_validator,
            f"--{flag_name} must be an existing directory.",
        )
    else:
        flags.register_validator(
            flag_name,
            _path_validator,
            f"--{flag_name} must be an existing file.",
        )


def initialize_hardware() -> None:
    """
    Initialize the hardware for training or evaluation.
    Can only be called once globally.
    """
    global _INITIALIZED  # pylint:disable=global-statement

    if _INITIALIZED:
        raise RuntimeError("Hardware has already been initialized.")

    device = tf.config.list_physical_devices("GPU")[0]
    tf.config.experimental.set_memory_growth(device, True)

    if FLAGS.mixed_precision:
        precision_policy = mixed_precision.experimental.Policy("mixed_float16")
        mixed_precision.experimental.set_policy(precision_policy)

    _INITIALIZED = True


def _path_validator(path: str) -> bool:
    return os.path.exists(os.path.expanduser(path))


def _dir_validator(path: str) -> bool:
    return os.path.isdir(os.path.expanduser(path))
