import tensorflow as tf

class GRU_Cell(tf.keras.layers.AbstractRNNCell):

    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)

        self.units = units

        self.reset_gate_layer = tf.keras.layers.Dense(units=units, activation="sigmoid")
        self.update_gate_layer = tf.keras.layers.Dense(units=units, activation="sigmoid")
     
        self.hidden_state_candiate_layer = tf.keras.layers.Dense(units=units, activation="tanh")

    @property
    def state_size(self):
        return tf.TensorShape(self.units)
    
    @property
    def output_size(self):
        return tf.TensorShape(self.units)
    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        # hidden_state
        return tf.zeros(shape=(batch_size, self.units))
    

    def call(self, inputs, prev_hidden_state):
        
        prev_hidden_state = prev_hidden_state[0]


        concat_inputs_hidden = tf.concat([inputs, prev_hidden_state], axis=-1)

        #
        # Preparing
        #

        r = self.reset_gate_layer(concat_inputs_hidden)
        z = self.update_gate_layer(concat_inputs_hidden)

        #
        # Hidden state candiate
        #

        concat_resetedHiddenState_inputs = tf.concat([r * prev_hidden_state, inputs], axis=-1)
        hidden_state_candiate = self.hidden_state_candiate_layer(concat_resetedHiddenState_inputs)

        #
        # New hidden state
        #

        hidden_state = (1 - z) * prev_hidden_state + z * hidden_state_candiate

        return hidden_state, hidden_state