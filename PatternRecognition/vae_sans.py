import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


class VAE:
    def __init__(self, n_in):
        self.n_in = n_in

    def build(self):
        self.x = tf.placeholder(tf.float32, [None, self.n_in])
        self.lr = tf.placeholder(tf.float32, [])

        W_en = tf.Variable(tf.random_normal([self.n_in, 256], stddev=0.001))
        b_en = tf.Variable(tf.zeros([256]))
        W_mu = tf.Variable(tf.random_normal([256, 2], stddev=0.001))
        b_mu = tf.Variable(tf.zeros([2]))

        W_sig = tf.Variable(tf.random_normal([256, 2], stddev=0.001))
        b_sig = tf.Variable(tf.zeros([2]))

        h_en = tf.nn.relu(tf.matmul(self.x, W_en) + b_en)
        mu = tf.matmul(h_en, W_mu) + b_mu
        sig = tf.matmul(h_en, W_sig) + b_sig
        e = tf.random_normal(tf.shape(mu))

        z = mu + tf.multiply(e, tf.exp(0.5 * sig))

        W_de = tf.Variable(tf.random_normal([2, 256], stddev=0.001))
        b_de = tf.Variable(tf.zeros([256]))
        W_out = tf.Variable(tf.random_normal([256, 784], stddev=0.001))
        b_out = tf.Variable(tf.random_normal([784]))

        h_de = tf.nn.relu(tf.matmul(z, W_de) + b_de)
        x_ = tf.matmul(h_de, W_out) + b_out

        KLD = -0.5 * tf.reduce_sum(1 + sig - tf.pow(mu, 2) - tf.exp(sig), reduction_indices=1)
        BCE = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_, labels=self.x), reduction_indices=1)
        self.loss = tf.reduce_mean(KLD + BCE)

        self.train = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        self.z_in = tf.placeholder(tf.float32, [None, 2])
        h_de_ = tf.nn.relu(tf.matmul(self.z_in, W_de) + b_de)
        self.x_ = tf.nn.sigmoid(tf.matmul(h_de_, W_out) + b_out)

def main():
    mnist = input_data.read_data_sets('/home/horo90/MNIST-data/', one_hot=True)
    n_in = 784
    lr = 1e-3

    with tf.Session() as sess:
        vae = VAE(n_in)
        vae.build()
        sess.run(tf.global_variables_initializer())

        xs = list()
        losses = list()
        for i in range(10000):
            tr_x, _ = mnist.train.next_batch(100)
            loss, _ = sess.run([vae.loss, vae.train], feed_dict={vae.x:tr_x, vae.lr:lr})

            if i % 100 == 0:
                xs.append(i)
                losses.append(loss)
                print('[',i,'] loss: ', loss)

        fig = plt.figure()
        plt.plot(xs, losses, label='loss')
        plt.legend()
        fig.show()

        n = 20
        x = np.linspace(-3., 3., n)
        y = np.linspace(-3., 3., n)
        fig = plt.figure(figsize=(8, 10))
        canvas = np.empty((n * 28, n * 28))
        for xi in range(len(x)):
            for yi in range(len(y)):
                z = np.array([x[xi], y[yi]]).reshape((1, 2))
                x_ = sess.run(vae.x_, feed_dict={vae.z_in:z})
                canvas[(n - xi - 1) * 28:(n - xi) * 28, yi * 28:(yi + 1) * 28] = x_[0].reshape((28, 28))
        plt.imshow(canvas, cmap='gray')
        plt.show()

if __name__ == '__main__':
    main()