import tensorflow as tf
import numpy as np


class RNN:
    # ...
    def step(self, x):
        # update the hidden state
        self.h = np.tanh(np.dot(self.W_hh, self.h) + np.dot(self.W_xh, x))
        # compute the output vector
        y = np.dot(self.W_hy, self.h)
        return y


x = 3

rnn = RNN()
y = rnn.step(x)  # x is an input vector, y is the RNN's output vector

# y1 = rnn1.step(x)
# y = rnn2.step(y1)
