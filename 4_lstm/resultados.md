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

#########################
BEST temp = lstm futuro b con 65bi + 4 fut (0.274 aprox)


##################
HUMIdITY
LSTM SIMPLE

32:
Average val_loss over 10 runs: 0.101360 ± 0.000658
Minimum val_loss over 10 runs: 0.100397

40:
Average val_loss over 10 runs: 0.101000 ± 0.001356
Minimum val_loss over 10 runs: 0.099342

42:
Average val_loss over 10 runs: 0.100912 ± 0.000764
Minimum val_loss over 10 runs: 0.099192

43: 
Average val_loss over 10 runs: 0.101115 ± 0.001006
Minimum val_loss over 10 runs: 0.100041

44:
Average val_loss over 10 runs: 0.101280 ± 0.001330
Minimum val_loss over 10 runs: 0.099277

50:
Average val_loss over 10 runs: 0.100929 ± 0.000608
Minimum val_loss over 10 runs: 0.100101

64: Average val_loss over 10 runs: 0.100971 ± 0.000404
Minimum val_loss over 10 runs: 0.100290

96:
Average val_loss over 10 runs: 0.101296 ± 0.000574
Minimum val_loss over 10 runs: 0.100351

BEST = 42 lstm simple; probadas unidades futuro; best = 4 (probado dense y lstm; mejor dense)
  past_data_layer = tf.keras.layers.Input(shape=past_data_shape, name="past_data")
    past_lstm = tf.keras.layers.LSTM(42, return_sequences=False)(past_data_layer)
    
    # past_lstm = CustomAttention(64)(past_lstm)
    # past_lstm = tf.keras.layers.MultiHeadAttention(
    # num_heads=3,
    # key_dim=16,               # so that 4*32 = 128 dims total
    # name="past_self_attn"
    # )(query=past_lstm, value=past_lstm, key=past_lstm)
    # past_lstm = tf.keras.layers.Flatten()(past_lstm) 
     
    future_data_layer = tf.keras.layers.Input(shape=future_data_shape, name="future_data")
    #future_lstm = tf.keras.layers.LSTM(4, return_sequences=False)(future_data_layer)
    future_lstm = tf.keras.layers.Flatten()(future_data_layer)
    future_lstm = tf.keras.layers.Dense(4, activation='relu')(future_lstm)
    #future_lstm = tf.keras.layers.Flatten()(future_lstm)

    merged = tf.keras.layers.concatenate([past_lstm, future_lstm])
    #merged= tf.keras.layers.Reshape((1, -1))(merged)

    outputs = tf.keras.layers.Dense(output_units)(merged)

    model = tf.keras.Model(inputs=[past_data_layer, future_data_layer], outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")



    ** El hibrido de humedad daria mejores resultados con lstm despues de convolucion **

    ###############################################
    F6 temp
        past_data_layer = tf.keras.layers.Input(shape=past_data_shape, name="past_data")
    past_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(65, return_sequences=False))(past_data_layer)
    
    # past_lstm = CustomAttention(64)(past_lstm)
    # past_lstm = tf.keras.layers.MultiHeadAttention(
    # num_heads=3,
    # key_dim=16,               # so that 4*32 = 128 dims total
    # name="past_self_attn"
    # )(query=past_lstm, value=past_lstm, key=past_lstm)
    # past_lstm = tf.keras.layers.Flatten()(past_lstm) 
     
    future_data_layer = tf.keras.layers.Input(shape=future_data_shape, name="future_data")
    future_lstm = tf.keras.layers.LSTM(6, return_sequences=False)(future_data_layer)
    # future_lstm = tf.keras.layers.Flatten()(future_data_layer)
    # future_lstm = tf.keras.layers.Dense(4, activation='relu')(future_lstm)
    #future_lstm = tf.keras.layers.Flatten()(future_lstm)

    merged = tf.keras.layers.concatenate([past_lstm, future_lstm])
    #merged= tf.keras.layers.Reshape((1, -1))(merged)

    merged = tf.keras.layers.Dense(2*output_units)(merged)

    outputs = tf.keras.layers.Dense(output_units)(merged)

    model = tf.keras.Model(inputs=[past_data_layer, future_data_layer], outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="huber", metrics=['mse'])

    return model

    4332
    -------------------------
    2852 tempf3
Average val_loss over 10 runs: 0.028951 ± 0.000268
Minimum val_loss over 10 runs: 0.028525

    past_data_layer = tf.keras.layers.Input(shape=past_data_shape, name="past_data")
    encoder_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(65, return_sequences=False))(past_data_layer)

    # Decoder part (LSTM for future exogenous features)
    future_data_layer = tf.keras.layers.Input(shape=future_data_shape, name="future_data")
    decoder_lstm = tf.keras.layers.LSTM(4, return_sequences=False)(future_data_layer)

    # Combine the outputs of encoder and decoder (you can concatenate or merge them)
    future_residue = tf.keras.layers.Flatten()(future_data_layer)
    merged = tf.keras.layers.concatenate([encoder_lstm, decoder_lstm, future_residue])

    # Final output layer
    #merged = tf.keras.layers.Dense(2* output_units)(merged)
    outputs = tf.keras.layers.Dense(output_units)(merged)

    model = tf.keras.Model(inputs=[past_data_layer, future_data_layer], outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")

