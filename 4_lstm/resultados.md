LSTM simple: 3691; 3700  (Covariate augmented)
LSTM futuro simple
LSTM futuro b:  350; 3585; 3766 (Covariate augmented)
LSTM futuro b 2.0: 3356; 

learning_rate = 0.001
EPOCHS = 700

####################################################

# Encoder part (LSTM for past data)
past_data_layer = tf.keras.layers.Input(shape=past_data_shape, name="past_data")
encoder_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(21, return_sequences=False))(past_data_layer)

# Decoder part (LSTM for future exogenous features)
future_data_layer = tf.keras.layers.Input(shape=future_data_shape, name="future_data")
decoder_lstm = tf.keras.layers.LSTM(4, return_sequences=False)(future_data_layer)

# Combine the outputs of encoder and decoder (you can concatenate or merge them)
merged = tf.keras.layers.concatenate([encoder_lstm, decoder_lstm])

# Final output layer
output_units = target_shape # Output shape should match the target sequence
outputs = tf.keras.layers.Dense(output_units)(merged)
