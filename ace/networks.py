import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers as tfl


def proposal_network(
    num_features: int,
    context_units: int = 64,
    mixture_components: int = 10,
    residual_blocks: int = 4,
    hidden_units: int = 512,
    activation: str = "relu",
    dropout: float = 0.0,
    linear_output = None,
    **kwargs
):
    x_o = tfl.Input((num_features,), name="x_o")
    observed_mask = tfl.Input((num_features,), name="observed_mask")

    full_input = tfl.Concatenate()([x_o, observed_mask])
    h = tfl.Dense(hidden_units)(full_input)

    for _ in range(residual_blocks):
        res = tfl.Activation(activation)(h)
        res = tfl.Dense(hidden_units)(res)
        res = tfl.Activation(activation)(res)
        res = tfl.Dropout(dropout)(res)
        res = tfl.Dense(hidden_units)(res)
        h = tfl.Add()([h, res])

    h = tfl.Activation(activation)(h)
    if linear_output is None:
        linear_output = tfl.Dense(num_features * (3 * mixture_components + context_units))
    
    o = linear_output(h)
    o = tfl.Reshape([num_features, 3 * mixture_components + context_units])(o)
    context = o[..., :context_units]
    params = o[..., context_units:]

    def create_proposal_dist(t):
        logits = t[..., :mixture_components]
        means = t[..., mixture_components:-mixture_components]
        scales = tf.nn.softplus(t[..., -mixture_components:]) + 1e-3
        components_dist = tfp.distributions.Normal(
            loc=tf.cast(means, tf.float32), scale=tf.cast(scales, tf.float32)
        )
        return tfp.distributions.MixtureSameFamily(
            mixture_distribution=tfp.distributions.Categorical(
                logits=tf.cast(logits, tf.float32)
            ),
            components_distribution=components_dist,
        )

    proposal_dist = tfp.layers.DistributionLambda(create_proposal_dist)(params)

    return tf.keras.Model([x_o, observed_mask], [proposal_dist, context], **kwargs)


def proposal_and_feature_network(
    num_features: int,
    context_units: int = 64,
    mixture_components: int = 10,
    residual_blocks: int = 4,
    hidden_units: int = 512,
    activation: str = "relu",
    dropout: float = 0.0,
    linear_output = None,
    **kwargs
):
    print("Creating proposal network...")

    x_o = tfl.Input((num_features,), name="x_o")
    observed_mask = tfl.Input((num_features,), name="observed_mask")

    full_input = tfl.Concatenate()([x_o, observed_mask])
    h = tfl.Dense(hidden_units)(full_input)

    for _ in range(residual_blocks):
        res = tfl.Activation(activation)(h)
        res = tfl.Dense(hidden_units)(res)
        res = tfl.Activation(activation)(res)
        res = tfl.Dropout(dropout)(res)
        res = tfl.Dense(hidden_units)(res)
        h = tfl.Add()([h, res])

    print("Created residual blocks...")

    h = tfl.Activation(activation)(h)
    if linear_output is None:
        linear_output = tfl.Dense(num_features * (3 * mixture_components + context_units))
    
    o = linear_output(h)
    o = tfl.Reshape([num_features, 3 * mixture_components + context_units])(o)

    print("Created linear output...")

    context = o[..., :context_units]
    params = o[..., context_units:]

    def create_proposal_dist(t):
        logits = t[..., :mixture_components]
        means = t[..., mixture_components:-mixture_components]
        scales = tf.nn.softplus(t[..., -mixture_components:]) + 1e-3
        components_dist = tfp.distributions.Normal(
            loc=tf.cast(means, tf.float32), scale=tf.cast(scales, tf.float32)
        )
        return tfp.distributions.MixtureSameFamily(
            mixture_distribution=tfp.distributions.Categorical(
                logits=tf.cast(logits, tf.float32)
            ),
            components_distribution=components_dist,
        )

    proposal_dist = tfp.layers.DistributionLambda(create_proposal_dist)(params)

    print("Created proposal distribution...")

    return tf.keras.Model([x_o, observed_mask], [proposal_dist, context], **kwargs), tf.keras.Model([full_input], [h], **kwargs)


def energy_network(
    num_features: int,
    context_units: int,
    residual_blocks: int = 4,
    hidden_units: int = 128,
    activation: str = "relu",
    dropout: float = 0.0,
    energy_clip: float = 30.0,
    **kwargs
):
    x_u_i = tfl.Input((), name="x_u_i")
    u_i = tfl.Input((), name="u_i", dtype=tf.int32)
    context = tfl.Input((context_units,), name="context")

    u_i_one_hot = tf.one_hot(u_i, num_features)

    h = tfl.Concatenate()([tf.expand_dims(x_u_i, axis=-1), u_i_one_hot, context])
    h = tfl.Dense(hidden_units)(h)

    for _ in range(residual_blocks):
        res = tfl.Activation(activation)(h)
        res = tfl.Dense(hidden_units)(res)
        res = tfl.Activation(activation)(res)
        res = tfl.Dropout(dropout)(res)
        res = tfl.Dense(hidden_units)(res)
        h = tfl.Add()([h, res])

    h = tfl.Activation(activation)(h)
    h = tfl.Dense(1)(h)

    energies = tf.nn.softplus(h)
    energies = tf.clip_by_value(energies, 0.0, energy_clip)
    negative_energies = -energies

    return tf.keras.Model([x_u_i, u_i, context], negative_energies, **kwargs)
