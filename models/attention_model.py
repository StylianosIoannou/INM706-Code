import tensorflow as tf

def build_attention_gru_model(input_shape):
    inputs = tf.keras.layers.Input(shape=input_shape)
    # GRU layer: processes the sequence and keeps outputs for each time step
    gru_output = tf.keras.layers.GRU(50, return_sequences=True)(inputs)
    # Use the last output from the GRU as the 'query' for attention, this tells the attention layer what to focus on
    query = tf.keras.layers.Lambda(lambda x: x[:, -1:, :])(gru_output)
    # Attention layer: computes the attention scores and applies them to the GRU outputs  
    attention_output = tf.keras.layers.Attention()([query, gru_output])
    attention_output = tf.keras.layers.Reshape((attention_output.shape[-1],))(attention_output)
    # Dropout: helps prevent overfitting (model relying too much on training data)
    x = tf.keras.layers.Dropout(0.2)(attention_output)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    # Create and compile the model
    model = tf.keras.models.Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model