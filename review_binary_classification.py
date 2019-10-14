from keras.datasets import imdb # 50,000 highly polarized reviews. 25,000 for training, 25,000 for testing. 50% postiive, 50% negative.
from keras import models
from keras import layers
import numpy as np

# Visualize the input
def reverse_word(i):
    word_index = imdb.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[i]])
    return decoded_review


# Can't feed list of integers to the network. Need to turn these lists into tensors. 
# One hot encode the word sequences.
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print(train_labels.shape)
print(train_labels[0])
x_train = vectorize_sequences(train_data)
x_test  = vectorize_sequences(test_data)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# The input data is vectors
# The labels are scalars (0 and 1s)
# Easiest setup. Stack of fully connected Dense layers with relu activation functions are good for this problem.
# Dense Layer : relu(dot(W, input) + b) -> Dot multiplication of weights and input, add bias and do relu on the result.

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,))) # Expects the 0th axis to be 10000 dimensional.
model.add(layers.Dense(16, activation='relu')) # Each intermediate layer has 16 hidden layers which basically decides the structure of the weight matrix.
model.add(layers.Dense(1, activation='sigmoid')) # The last layer uses sigmoid to squash the output between 0 and 1 to be able to make a prediction to positive/negative.

# Crossentropy is a quantity from the field of Information Theory that measures the distance between probability distributions
# or, in this case, between the ground-truth distribution and your predictions

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train, partial_y_train, epochs=4, batch_size=512, validation_data=(x_val, y_val))
result = model.evaluate(x_test, y_test)
model.predict(x_test)
print(result)

history_dict = history.history

import matplotlib.pyplot as plt

# Plotting the training and validation loss
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training Loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation Loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plotting the training and validation accuracy
acc = history_dict['acc']
val_acc = history_dict['val_acc']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
