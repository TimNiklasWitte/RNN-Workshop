import sys
sys.path.append("../")

import tensorflow as tf

from LSTM_Cell import *

class LSTM_Model(tf.keras.Model):
    def __init__(self, num_classes):
        super().__init__()

        self.lstm_cell = LSTM_Cell(units=20)
     
        self.layer_list = [
            tf.keras.layers.RNN(self.lstm_cell, return_sequences=False, unroll=True),
            tf.keras.layers.Dense(units=10, activation="tanh"),
            tf.keras.layers.Dense(units=5, activation="tanh"),
            tf.keras.layers.Dense(units=num_classes, activation="softmax")
        ]
 
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.bce_loss = tf.keras.losses.CategoricalCrossentropy()
        
        self.metric_loss = tf.keras.metrics.Mean(name="loss")
        self.metric_accuracy = tf.keras.metrics.Accuracy(name="accuracy")


    @tf.function 
    def call(self, x):
    
        for layer in self.layer_list:
            x = layer(x)

        return x
    
    @tf.function
    def train_step(self, x, target):

        with tf.GradientTape() as tape:
            prediction = self(x)
            loss = self.bce_loss(target, prediction)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.metric_loss.update_state(loss)

        prediction = tf.argmax(prediction, axis=-1)
        label = tf.argmax(target, axis=-1)
        self.metric_accuracy.update_state(label, prediction)
    

    def test_step(self, dataset):
          
        self.metric_loss.reset_states()
        self.metric_accuracy.reset_states()

        for x, target in dataset:
            prediction = self(x)

            loss = self.bce_loss(target, prediction)
            self.metric_loss.update_state(loss)

            prediction = tf.argmax(prediction, axis=-1)
            label = tf.argmax(target, axis=-1)
            self.metric_accuracy.update_state(label, prediction)
        