import tensorflow as tf

class BasicRNN_Cell(tf.keras.layers.AbstractRNNCell):

    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)

        self.units = units # aka hidden size
       
    def build(self, input_shape):

        input_size = input_shape[-1]

        #
        # Input -> hidden
        #
        self.weight_input_hidden = self.add_weight(shape=(input_size, self.units),
                        initializer='random_normal',
                        trainable=True)

        #
        # Hidden -> hidden
        #

        self.weight_hidden_hidden = self.add_weight(shape=(self.units, self.units),
                        initializer='random_normal',
                        trainable=True)
        
        self.bias_hidden = self.add_weight(shape=(self.units,),
                        initializer='random_normal',
                        trainable=True)
        
    @property
    def state_size(self):
        return tf.TensorShape(self.units)
    @property
    def output_size(self):
        return tf.TensorShape(self.units)
    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return tf.zeros(shape=(batch_size, self.units))
    

    def call(self, inputs, prev_hidden_state):
        
        prev_hidden_state = prev_hidden_state[0]

        hidden_state =  tf.math.sigmoid(
            inputs @ self.weight_input_hidden + \
            prev_hidden_state @ self.weight_hidden_hidden + self.bias_hidden
        )

        return hidden_state, hidden_state