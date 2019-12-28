
import numpy as np
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.mixture import BayesianGaussianMixture

class BayesClassifier:
    def __init__(self,X,Y):
        self.X = X
        self.Y = Y
        self.K = len(set(Y))
        self.gaussian = []
        self.gmm = None

# find the mean and covariance of the data based on labels for gaussian model
    def fit(self):
        for k in range(self.K):
            Xk = self.X[self.Y == k]
            self.gmm = BayesianGaussianMixture()
            self.gmm.fit(Xk)
            self.gaussian.append(self.gmm)

# multi modal sampling using GMM
    def sample_data_for_label(self,y):
        local_gaussian = self.gaussian[y]
        sample = self.gmm.sample()
        return sample[0].reshape(8,8)

    def mean_data_for_label(self, y):
        local_gaussian = self.gaussian[y]
        sample = self.gmm.sample()
        mean = self.gmm.means_[sample[1]]
        return mean.reshape(8,8)

# method used for random sampling
    def sample(self):
        y = random.randrange(self.K)
        return self.sample_data_for_label(y)

if __name__ == '__main__':
    #load the mnist dataset and split it to separate X and y training data
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.5, shuffle=False) # X_train.shape = (898,64) y_train.shape = (898,)

    classifier = BayesClassifier(X_train,y_train)
    classifier.fit()

    for k in range(classifier.K): #perform random sampling using the function written above for all labels
        sample = classifier.sample_data_for_label(k)
        mean = classifier.mean_data_for_label(k)

        #plot the sampled data as well as mean for a particular label
        plt.subplot(1,2,1)
        plt.imshow(sample, cmap='gray')
        plt.title("Sample")
        plt.subplot(1,2,2)
        plt.imshow(mean, cmap='gray')
        plt.title("Mean")
        plt.show()
