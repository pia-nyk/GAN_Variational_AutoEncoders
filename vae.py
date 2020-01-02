import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_probability as tfp
import util
import numpy as np
import matplotlib.pyplot as plt

class DenseLayer(object):
    def __init__ (self, M1, M2, f=tf.nn.relu):

        self.W = tf.Variable(tf.random.normal(shape=(M1,M2))) * 2 / np.sqrt(M1)
        self.b = tf.Variable(np.zeros(M2).astype(np.float32))
        self.f = f

    #activation(WX + b)
    def forward(self, X):
        return self.f(tf.matmul(X, self.W) + self.b)

class VariationalAutoencoder:
    def __init__(self, D, hidden_layers_sizes):
        #hidden layer sizes have all the shapes of the network till the output layer
        #decoder will have the reverse shape as this

        self.X = tf.placeholder(tf.float32, shape=(None,D))

        #encoder code
        self.encoder_layers = []
        M_in = D #input layer
        for M_out in hidden_layers_sizes[:-1]: #iterate over all layers but the last as thats the output layer
            layer = DenseLayer(M_in, M_out)
            self.encoder_layers.append(layer)
            M_in = M_out

        M = hidden_layers_sizes[-1] #final layer size, also the input to the decoder layer
        #need 2* specified output layer size for means and std devs, no activation on the output layer
        layer = DenseLayer(M_in, 2*M, f=lambda x: x)
        self.encoder_layers.append(layer)

        #get means and std devs of Z.
        #std dev cannot be -ve, so use softplus activation and add a small number for smoothening purposes
        current_layer = self.X
        print("Layers", self.encoder_layers)
        #iterate all the hidden layers to get outputs of final layer
        for layer in self.encoder_layers:
            current_layer = layer.forward(current_layer)
        self.means = current_layer[:,:M] #get mean as the last M node values
        self.stddev = current_layer[:,M:] + 1e-6 #get mean as the first M node values

        Normal = tfp.distributions.Normal
        Bernoulli = tfp.distributions.Bernoulli

        #get a sample of Z
        standard_normal = Normal(loc=np.zeros(M, dtype=np.float32),scale=np.ones(M, dtype=np.float32))
        e = standard_normal.sample(tf.shape(self.means)[0])
        self.Z = e * self.stddev + self.means

        # decoder
        self.decoder_layers = []
        M_in = M
        for M_out in reversed(hidden_layers_sizes[:-1]):
          h = DenseLayer(M_in, M_out)
          self.decoder_layers.append(h)
          M_in = M_out

        # the decoder's final layer should technically go through a sigmoid
        # so that the final output is a binary probability (e.g. Bernoulli)
        # but Bernoulli accepts logits (pre-sigmoid) so we will take those
        # so no activation function is needed at the final layer
        h = DenseLayer(M_in, D, f=lambda x: x)
        self.decoder_layers.append(h)

        # get the logits
        current_layer_value = self.Z
        for layer in self.decoder_layers:
          current_layer_value = layer.forward(current_layer_value)
        logits = current_layer_value
        posterior_predictive_logits = logits # save for later

        # get the output
        self.X_hat_distribution = Bernoulli(logits=logits)

        # take samples from X_hat
        # we will call this the posterior predictive sample
        self.posterior_predictive = self.X_hat_distribution.sample()
        self.posterior_predictive_probs = tf.nn.sigmoid(logits)

        # take sample from a Z ~ N(0, 1)
        # and put it through the decoder
        # we will call this the prior predictive sample
        standard_normal = Normal(
          loc=np.zeros(M, dtype=np.float32),
          scale=np.ones(M, dtype=np.float32)
        )

        Z_std = standard_normal.sample(1)
        current_layer_value = Z_std
        for layer in self.decoder_layers:
          current_layer_value = layer.forward(current_layer_value)
        logits = current_layer_value

        prior_predictive_dist = Bernoulli(logits=logits)
        self.prior_predictive = prior_predictive_dist.sample()
        self.prior_predictive_probs = tf.nn.sigmoid(logits)


        # prior predictive from input
        # only used for generating visualization
        self.Z_input = tf.placeholder(tf.float32, shape=(None, M))
        current_layer_value = self.Z_input
        for layer in self.decoder_layers:
          current_layer_value = layer.forward(current_layer_value)
        logits = current_layer_value
        self.prior_predictive_from_input_probs = tf.nn.sigmoid(logits)
        #build the cost - kl divergence
        # kl = tf.reduce_sum(
        #     tfp.distributions.kl_divergence(
        #         self.Z.distribution, standard_normal
        #     ), 1
        # )
        #
        # #build the cost - log likelihood

    # now build the cost
        kl = -tf.log(self.stddev) + 0.5*(self.stddev**2 + self.means**2) - 0.5
        kl = tf.reduce_sum(kl, axis=1)

        expected_log_likelihood = tf.reduce_sum(self.X_hat_distribution.log_prob(self.X),1)



        self.elbo = tf.reduce_sum(expected_log_likelihood - kl)
        self.train_op = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(-self.elbo)

        # set up session and variables for later
        self.init_op = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(self.init_op)

    #fitting the data and running the session
    def fit(self, X, epochs=30, batch_sz=64):
      costs = []
      n_batches = len(X) // batch_sz
      print("n_batches:", n_batches)
      for i in range(epochs):
        print("epoch:", i)
        np.random.shuffle(X)
        for j in range(n_batches):
          batch = X[j*batch_sz:(j+1)*batch_sz]
          _, c, = self.sess.run((self.train_op, self.elbo), feed_dict={self.X: batch})
          c /= batch_sz # just debugging
          costs.append(c)
          if j % 100 == 0:
            print("iter: %d, cost: %.3f" % (j, c))
      plt.plot(costs)
      plt.show()

    def transform(self, X):
      return self.sess.run(
        self.means,
        feed_dict={self.X: X}
      )

    def prior_predictive_with_input(self, Z):
      return self.sess.run(
        self.prior_predictive_from_input_probs,
        feed_dict={self.Z_input: Z}
      )

    def posterior_predictive_sample(self, X):
      # returns a sample from p(x_new | X)
      return self.sess.run(self.posterior_predictive, feed_dict={self.X: X})

    def prior_predictive_sample_with_probs(self):
      # returns a sample from p(x_new | z), z ~ N(0, 1)
      return self.sess.run((self.prior_predictive, self.prior_predictive_probs))

def main():
    X, Y = util.get_mnist()
    # convert X to binary variable
    X = (X > 0.5).astype(np.float32)

    vae = VariationalAutoencoder(784, [200, 100])
    vae.fit(X)

    # plot reconstruction

    i = np.random.choice(len(X))
    x = X[i]
    im = vae.posterior_predictive_sample([x]).reshape(28, 28)
    plt.subplot(1,2,1)
    plt.imshow(x.reshape(28, 28), cmap='gray')
    plt.title("Original")
    plt.subplot(1,2,2)
    plt.imshow(im, cmap='gray')
    plt.title("Sampled")
    plt.show()



    im, probs = vae.prior_predictive_sample_with_probs()
    im = im.reshape(28, 28)
    probs = probs.reshape(28, 28)
    plt.subplot(1,2,1)
    plt.imshow(im, cmap='gray')
    plt.title("Prior predictive sample")
    plt.subplot(1,2,2)
    plt.imshow(probs, cmap='gray')
    plt.title("Prior predictive probs")
    plt.show()


if __name__ == '__main__':
    main()
