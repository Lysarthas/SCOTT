import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Conv1D, LSTM, LayerNormalization, MultiHeadAttention, SpatialDropout1D, Flatten, GlobalAveragePooling1D

def cautrans_enc(input_shape, head_size, num_heads, num_f, dilations, k_size, dropout=0.0):
    inputs = Input(shape=input_shape)
    # Attention
    x = MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(inputs, inputs)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    x = res
    for d in dilations:
        x = Conv1D(filters=num_f, kernel_size=k_size, dilation_rate=d, padding='causal', activation='relu')(x)
        x = SpatialDropout1D(dropout)(x)
        x = LayerNormalization(epsilon=1e-06)(x)
    x = x + res
    # Map to latent space
    # x = Flatten()(x)
    # outputs = Dense(input_shape[0])(x)
    outputs = GlobalAveragePooling1D(data_format='channels_first')(x)
    return Model(inputs, outputs, name='encoder')

def transformer_block(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization

    x = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = Dropout(dropout)(x)
    x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    return x+res

def encoder(input_shape, num_block, head_size, num_heads, ff_dim, dropout):
    inputs = Input(shape=input_shape)
    x = inputs
    for _ in range(num_block):
        x = transformer_block(x, head_size, num_heads, ff_dim, dropout)
    output = GlobalAveragePooling1D(data_format="channels_first")(x)
    return Model(inputs, output)

def projector(input_shape, mlp_units, mlp_dropout=0):
    inputs = Input(shape=input_shape)
    x = inputs
    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(mlp_dropout)(x)
    outputs = Dense(8)(x)
    return Model(inputs, outputs)

def MLP_cl(input_dim, mlp_layers, n_class, bias = None):
    if bias is not None:
        bias = tf.keras.initializers.Constant(bias)
    inputs = Input(shape=input_dim)
    x = inputs
    for dim in mlp_layers:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(0.4)(x)
    outputs = Dense(n_class, activation='softmax', bias_initializer=bias)(x)
    return Model(inputs, outputs)