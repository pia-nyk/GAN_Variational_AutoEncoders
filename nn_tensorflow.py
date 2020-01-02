import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import matplotlib.pyplot as plt
from util2 import get_normalized_data, y2indicator

def error_rate(p, t):
    return np.mean(p != t)

def main():
    X_train, X_test, Y_train, Y_test = get_normalized_data()

    max_iter = 10
    print_period = 10
    learning_rate = 0.00004
    regularization = 0.01

    Ytrain_ind = y2indicator(Y_train)
    Ytest_ind = y2indicator(Y_test)

    N = X_train.shape[0]
    D = X_train.shape[1]
    batch_size = 500
    n_batches = N/batch_size

    M1 = 300
    M2 = 100
    K = 10
    W1_init = np.random.randn(D, M1)/28
    b1_init = np.zeros(M1)
    W2_init = np.random.randn(M1, M2)/np.sqrt(M1)
    b2_init = np.zeros(M2)
    W3_init = np.random.randn(M2, K)/np.sqrt(M2)
    b3_init = np.zeros(K)

    X = tf.placeholder(tf.float32, shape=(None, D), name='X')
    T = tf.placeholder(tf.float32, shape=(None, K), name='T')
    W1 = tf.Variable(W1_init.astype(np.float32)) # if this doesnt work, make it np.float32 and find diff between both
    b1 = tf.Variable(b1_init.astype(np.float32))
    W2 = tf.Variable(W2_init.astype(np.float32))
    b2 = tf.Variable(b2_init.astype(np.float32))
    W3 = tf.Variable(W3_init.astype(np.float32))
    b3 = tf.Variable(b3_init.astype(np.float32))

    Z1 = tf.nn.relu(tf.matmul(X, W1) + b1)
    Z2 = tf.nn.relu(tf.matmul(Z1, W2) + b2)
    Y_temp = tf.matmul(Z2, W3) + b3

    cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(Y_temp, T))
    train_op = tf.train.RMSPropOptimizer(learning_rate, decay= 0.99, momentum=0.9).minimize(cost)
    predict_op = tf.argmax(Y_temp, 1) #why this??

    LL = []
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)

        for i in range(max_iter):
            for j in range(int(n_batches)):
                X_batch = X_train[j*batch_size: (j+1)*batch_size,]
                Y_batch = Ytrain_ind[j*batch_size: (j+1)*batch_size,]

                sess.run(train_op, feed_dict={X:X_batch, T:Y_batch})


                if print_period%10 == 0:
                    test_cost = sess.run(cost, feed_dict={X:X_test, T: Ytest_ind})
                    prediction = sess.run(predict_op, feed_dict={X:X_test})
                    err = error_rate(prediction, Y_test)
                    print ("Cost/err at iteration i=%d j=%d: %.3f / %.3f" % (i,j,test_cost,err))
                    LL.append(test_cost)
        plt.plot(LL)
        plt.show()


if __name__ == '__main__':
    main()
