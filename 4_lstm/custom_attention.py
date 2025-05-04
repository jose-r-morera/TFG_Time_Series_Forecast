import tensorflow as tf

class CustomAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(CustomAttention, self).__init__()
        # These are trainable layers that compute the attention score
        self.W = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
        
    def call(self, inputs):
        # inputs shape: (batch_size, time_steps, hidden_units)
        # Compute score for each time step
        score = tf.nn.tanh(self.W(inputs))             # (batch_size, time_steps, units)
        attention_weights = tf.nn.softmax(self.V(score), axis=1)  # (batch_size, time_steps, 1)
        # Multiply each hidden state by its attention weight and sum over time steps
        context_vector = tf.reduce_sum(attention_weights * inputs, axis=1)  # (batch_size, hidden_units)
        return context_vector
    