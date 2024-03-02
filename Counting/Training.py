import tensorflow as tf
import tqdm
import numpy as np

from BasicRNN_Model import *
from GRU_Model import *
from LSTM_Model import *


DATASET_SIZE = 5000
TEST_SPLIT = 0.1

MAX_NUMBER = 10
SEQ_LEN = 10

NUM_EPOCHS = 50
BATCH_SIZE = 32

def dataset_generator():

    for i in range(DATASET_SIZE):
        
        cnts = np.zeros(shape=(4,))

        seq = np.zeros(shape=(SEQ_LEN, 1))
        targets = np.zeros(shape=(SEQ_LEN, 1))

        for i in range(SEQ_LEN):
            x = np.random.randint(1, 3)
            seq[i] = x

            cnts[x] += 1

            if cnts[x] == 3:
                targets[i] = x
          

        yield seq, targets

def main():

    #
    # Create dataset
    #   

    ds = tf.data.Dataset.from_generator(
                dataset_generator,
                output_signature=(
                        # Use None for variable sequence length
                        tf.TensorSpec(shape=(SEQ_LEN, 1), dtype=tf.uint8),
                        tf.TensorSpec(shape=(SEQ_LEN, 1), dtype=tf.uint8)
                    )
                )
    
    train_ds_size = int(DATASET_SIZE - (DATASET_SIZE * TEST_SPLIT))
    train_ds = ds.take(train_ds_size)
    train_ds = train_ds.apply(prepare_data)

    test_ds = ds.skip(train_ds_size)
    test_ds = test_ds.apply(prepare_data)


    #
    # Logging
    #
    model_list = [BasicRNN_Model(), LSTM_Model(), GRU_Model()]

    for model in model_list:

        model_name = model.__class__.__name__
 
        file_path = f"logs/{model_name}"
        train_summary_writer = tf.summary.create_file_writer(file_path)

        #
        # Build model.
        #

        model.build(input_shape=(1, SEQ_LEN, 3))
        model.summary()

        #
        # Train and test loss/accuracy
        #
        print(f"Epoch 0")
        log(train_summary_writer, model, train_ds, test_ds, 0)

        #
        # Train loop
        #
        for epoch in range(1, NUM_EPOCHS + 1):
                
            print(f"Epoch {epoch}")

            for x, target in tqdm.tqdm(train_ds, position=0, leave=True): 
                model.train_step(x, target)

            log(train_summary_writer, model, train_ds, test_ds, epoch)



def log(train_summary_writer, model, train_ds, test_ds, epoch):

    # Epoch 0 = no training steps are performed 
    # test based on train data
    # -> Determinate initial train_loss and train_accuracy
    if epoch == 0:
        model.test_step(train_ds.take(5000))

    #
    # Train
    #
    train_loss = model.metric_loss.result()
    train_accuracy = model.metric_accuracy.result()

    model.metric_loss.reset_states()
    model.metric_accuracy.reset_states()

    #
    # Test
    #

    model.test_step(test_ds)

    test_loss = model.metric_loss.result()
    test_accuracy = model.metric_accuracy.result()

    model.metric_loss.reset_states()
    model.metric_accuracy.reset_states()

    #
    # Write to TensorBoard
    #
    with train_summary_writer.as_default():
        tf.summary.scalar(f"train_loss", train_loss, step=epoch)
        tf.summary.scalar(f"train_accuracy", train_accuracy, step=epoch)

        tf.summary.scalar(f"test_loss", test_loss, step=epoch)
        tf.summary.scalar(f"test_accuracy", test_accuracy, step=epoch)

    #
    # Output
    #
    print(f"    train_loss: {train_loss}")
    print(f"     test_loss: {test_loss}")
    print(f"train_accuracy: {train_accuracy}")
    print(f" test_accuracy: {test_accuracy}")
 
 
def prepare_data(dataset):

    # One hot target
    dataset = dataset.map(lambda x, target: (tf.one_hot(x, depth=3, axis=-1), target))
    dataset = dataset.map(lambda x, target: (tf.squeeze(x, axis=1), target))

    dataset = dataset.map(lambda x, target: (x, tf.one_hot(target, depth=4, axis=-1)))
    dataset = dataset.map(lambda x, target: (x, tf.squeeze(target, axis=1)))

    # Cache
    dataset = dataset.cache()
    
    #
    # Shuffle, batch, prefetch
    #
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")