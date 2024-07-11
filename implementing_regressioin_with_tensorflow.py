
# Deep Learning with Regression

import tensorflow as tf
from tensorflow import keras

#Data Definton

x = tf.constant([-1.0, -2.0, 0.0, 1.0, 2.0, 5.0, 7.0], dtype=tf.float32)
y=x*3-5
x.numpy()
y.numpy()

#model Defintion

model = keras.Sequential([
    keras.layers.Dense(units=1, input_shape=[1], activation=None)
])

# units=1 : This layer has one neuron.
#input_shape=[1]: This layer expects input data with  one-dimensional array with one element.

#model compile

model.compile(optimizer='sgd', loss='mean_squared_error')

#model summary
model.summary()

#train model

model.fit(x,y,batch_size=1,epochs=500)

#model test

x_test = tf.constant([-4.0, 11.0, 20.0], dtype=tf.float32)
y_test = x_test * 3 - 5
x_test.numpy()
y_test.numpy()

#how well does our model work?

y_predict = model.predict(x_test).flatten()

import matplotlib.pyplot as plt

# Plot the results
plt.scatter(x_test.numpy(), y_test.numpy(), c='green', s=80)
plt.scatter(x_test.numpy(), y_predict, c='red', s=20)


plt.show()
