
import numpy as np
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

class BayesClassifier:
    def __init__(self,X,Y):
        self.X = X
        self.Y = Y
        self.K = len(set(Y))
        self.gaussian = []

# find the mean and covariance of the data based on labels for gaussian model
    def fit(self):
        for k in range(self.K):
            Xk = self.X[self.Y == k]
            mean = Xk.mean(axis=0)
            covariance = np.cov(Xk.T)
            local_gaussian = {'mean': mean, 'covariance': covariance}
            self.gaussian.append(local_gaussian)

# random sampling using the mean and covaiance for given label
    def sample_data_for_label(self,y):
        local_gaussian = self.gaussian[y]
        return multivariate_normal.rvs(local_gaussian['mean'], local_gaussian['covariance'])

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
        sample = classifier.sample_data_for_label(k).reshape(8,8)
        mean = classifier.gaussian[k]['mean'].reshape(8,8)

        #plot the sampled data as well as mean for a particular label
        plt.subplot(1,2,1)
        plt.imshow(sample, cmap='gray')
        plt.title("Sample")
        plt.subplot(1,2,2)
        plt.imshow(mean, cmap='gray')
        plt.title("Mean")
        plt.show()
