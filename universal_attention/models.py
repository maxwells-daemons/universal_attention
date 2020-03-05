"""
Code for making models.
"""

from typing import Dict, List, NamedTuple, Optional, Tuple

import tensorflow as tf  # type: ignore

# pylint: disable=import-error
from tensorflow.keras import layers  # type: ignore

from universal_attention import data

AttentionMaps = Dict[str, tf.Tensor]


# Encoders
class AttendingEncoderOutput(NamedTuple):
    """The outputs of an encoder that may have attention."""

    features: tf.Tensor
    attention: AttentionMaps


class Resnet50Encoder(layers.Layer):
    """
    An encoder using the trunk of ResNet50V2.

    Parameters
    ----------
    imagenet_init: bool (default: False)
        Whether to initialize weights with imagenet or randomly.
    """

    def __init__(self, imagenet_init: bool = False, **kwargs):
        super(Resnet50Encoder, self).__init__(**kwargs)
        self.imagenet_init = imagenet_init
        self.base_model = tf.keras.applications.ResNet50V2(
            include_top=False,
            weights="imagenet" if imagenet_init else None,
            pooling="avg",
        )

    # pylint: disable=missing-function-docstring
    def call(self, image: tf.Tensor) -> AttendingEncoderOutput:
        return AttendingEncoderOutput(self.base_model(image), {})

    def get_config(self):  # pylint: disable=missing-function-docstring
        return {"imagenet_init": self.imagenet_init}


class QueryFreeAttention(layers.Layer):
    """
    An attention layer based on a shallow model of pixel statistics.
    """

    def __init__(self, **kwargs):
        super(QueryFreeAttention, self).__init__(**kwargs)
        self.conv = layers.Conv2D(1, 1, name="scores")
        self.flatten_scores = layers.Flatten(name="scores_flat")

    # pylint: disable=missing-function-docstring
    def build(self, input_shape):
        # pylint: disable=attribute-defined-outside-init
        self.reshape_attention = layers.Reshape(
            (input_shape[1], input_shape[2], 1)
        )

    # pylint: disable=missing-function-docstring
    def call(self, inputs: tf.Tensor):
        scores = self.conv(inputs)
        attention_flat = tf.nn.softmax(self.flatten_scores(scores))
        return self.reshape_attention(attention_flat)


class QueryFreeEncoder(layers.Layer):
    """
    An encoder based on ResNet50, using QueryFreeAttention blocks tapped into
    one or more layers.

    Parameters
    ----------
    tap_layers: [str]
        Which layers to extract attention maps from.
    imagenet_init: bool (default: False)
        Whether to initialize weights with imagenet or randomly.
    """

    def __init__(
        self, tap_layers: List[str], imagenet_init: bool = False, **kwargs
    ):
        super(QueryFreeEncoder, self).__init__(**kwargs)
        self.tap_layers = tap_layers
        self.imagenet_init = imagenet_init

        base_model = tf.keras.applications.ResNet50V2(
            include_top=False,
            weights="imagenet" if imagenet_init else None,
            pooling=None,
        )
        tap_features = {
            layer: base_model.get_layer(layer).output for layer in tap_layers
        }
        self.tap_model = tf.keras.Model(
            inputs=base_model.input, outputs=tap_features,
        )

        self.attention_layers = {
            layer: QueryFreeAttention(name=f"attention_{layer}")
            for layer in tap_layers
        }

    # pylint:disable=missing-function-docstring
    def call(self, inputs: tf.Tensor):
        features = self.tap_model(inputs)
        output_features: List[tf.Tensor] = []
        attention: AttentionMaps = {}

        for layer in self.tap_layers:
            attention[layer] = self.attention_layers[layer](features[layer])
            weighted = attention[layer] * features[layer]
            collapsed = layers.GlobalAveragePooling2D(name=f"pool_{layer}")(
                weighted
            )
            output_features.append(collapsed)

        features = tf.concat(output_features, axis=-1)
        return AttendingEncoderOutput(features, attention)

    def get_config(self):  # pylint:disable=missing-function-docstring
        return {
            "tap_layers": self.tap_layers,
            "imagenet_init": self.imagenet_init,
        }


class BahdanauAttention(layers.Layer):
    """
    Compute the (learnable) "concat" or "additive" attention between a query
    and a feature map.

    Parameters
    ----------
    alignment_size: int
        Dimension of the alignment vector.
    name: str
        Name to use for this layer. Required to avoid conflicts during saving.
    """

    alignment_size: int

    def __init__(self, alignment_size: int, name: str, **kwargs):
        super(BahdanauAttention, self).__init__(name=name, **kwargs)
        self.alignment_size = alignment_size
        self.project = layers.Dense(
            alignment_size, activation="tanh", use_bias=False, name="project"
        )
        self.target = self.add_weight(
            name=f"{name}_target",
            shape=(alignment_size, 1),
            initializer="random_normal",
            trainable=True,
        )
        self.flatten_scores = layers.Flatten(name="scores_flat")

    # pylint:disable=missing-function-docstring
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]):
        """
        Get the attention map of a (features, query) tuple.

        Parameters
        ----------
        inputs: (tf.Tensor, tf.Tensor)
            A tuple of (features, query) tensors.

        Returns
        -------
        tf.Tensor
            The attention map for these features / query.
        """
        features, query = inputs
        _, rows, cols, _ = features.shape

        # Broadcast query to shape: [B, W, H, QUERY_SIZE]
        query_broadcast = tf.expand_dims(tf.expand_dims(query, 1), 1)
        query_tiled = tf.tile(query_broadcast, [1, rows, cols, 1])

        together = tf.concat([features, query_tiled], axis=-1)
        compressed = self.project(together)
        scores = tf.matmul(compressed, self.target)[:, :, :, 0]
        attention_flat = tf.nn.softmax(self.flatten_scores(scores))
        attention = tf.reshape(attention_flat, [-1, rows, cols, 1])
        return attention

    def get_config(self):  # pylint:disable=missing-function-docstring
        return {"alignment_size": self.alignment_size}


class QueryAttentionEncoder(layers.Layer):
    """
    The attention model proposed in Jetley et al.

    Parameters
    ----------
    tap_layers: [str]
        Which layers to compute attention and descriptors for.
    query_size: int
        Dimension of the global query vector.
    alignment_size: int
        Dimension of the parametric alignment vector.
    imagenet_init: bool (default: False)
        Whether to initialize weights with imagenet or randomly.
    """

    query_size: Optional[int]
    alignment_size: int

    def __init__(
        self,
        tap_layers: List[str],
        alignment_size: int,
        query_size: int,
        imagenet_init: bool = False,
        **kwargs,
    ):
        super(QueryAttentionEncoder, self).__init__(**kwargs)
        self.tap_layers = tap_layers
        self.query_size = query_size
        self.alignment_size = alignment_size
        self.imagenet_init = imagenet_init

        base_model = tf.keras.applications.ResNet50V2(
            include_top=False,
            weights="imagenet" if imagenet_init else None,
            pooling="avg",
        )
        tap_features = {
            layer: base_model.get_layer(layer).output for layer in tap_layers
        }
        tap_features.update({"global_rep": base_model.output})
        self.tap_model = tf.keras.Model(
            inputs=base_model.input, outputs=tap_features,
        )

        self.attention_layers = {
            layer: BahdanauAttention(alignment_size, name=f"attention_{layer}")
            for layer in tap_layers
        }

        self.query_project = layers.Dense(query_size, name="query")

    def call(self, inputs):  # pylint:disable=missing-function-docstring
        features = self.tap_model(inputs)
        query = self.query_project(features["global_rep"])
        output_features: List[tf.Tensor] = []
        attention: AttentionMaps = {}

        for layer in self.tap_layers:
            attention[layer] = self.attention_layers[layer](
                (features[layer], query)
            )
            weighted = attention[layer] * features[layer]
            collapsed = layers.GlobalAveragePooling2D(name=f"pool_{layer}")(
                weighted
            )
            output_features.append(collapsed)
        features = tf.concat(output_features, axis=-1)

        return AttendingEncoderOutput(features, attention)

    def get_config(self):  # pylint:disable=missing-function-docstring
        return {
            "tap_layers": self.tap_layers,
            "query_size": self.query_size,
            "alignment_size": self.alignment_size,
            "imagenet_init": self.imagenet_init,
        }


# Classification models
class AttendingClassifierOutput(NamedTuple):
    """The outputs of a classifier that may have attention."""

    class_probs: tf.Tensor
    attention: AttentionMaps


def attending_classifier(
    encoder: layers.Layer, dataset_and_info: data.DatasetAndInfo
) -> tf.keras.Model:
    """
    A classifier, made up of an encoder (possibly with attention) and a head
    to a fixed number of classes, run on fixed-size imagery.

    Parameters
    ----------
    encoder: tf.keras.layers.Layer
        A layer returning AttendingEncoderOutput.
    dataset_and_info: data.DatasetAndInfo
        A dataset and its information, used to decide shapes.

    Returns
    -------
    tf.keras.Model (tf.Tensor -> AttendingClassifierOutput)
        A model mapping images to class probabilities and attention maps.
    """
    splits, info = dataset_and_info
    img_shape = splits["train"].element_spec[0].shape[1:]
    num_classes = info.features["label"].num_classes

    image = tf.keras.Input(shape=(img_shape), name="image")
    features, attention = encoder(image)
    logits = layers.Dense(num_classes, name="prediction")(features)

    # Rename attention outputs
    attention = {
        key: layers.Lambda(lambda x: x, name=f"attention_{key}")(attention_map)
        for key, attention_map in attention.items()
    }

    return tf.keras.Model(image, AttendingClassifierOutput(logits, attention))
