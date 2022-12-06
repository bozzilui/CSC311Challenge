import numpy as np
import re
class BernoulliNB:
    def __init__(self):
        self.priors = None
        self.likelihoods = None
        self.num_classes = None
        self.num_features = None

    # Fit the model to the training data
    def fit(self, X, y):
        # Get the number of classes and the number of features
        self.num_classes = len(np.unique(y))
        self.num_features = X.shape[1]

        # Compute the prior probabilities for each class
        self.priors = np.zeros(self.num_classes)
        for i in range(self.num_classes):
            self.priors[i] = np.mean(y == (i+1))

        # Compute the likelihood probabilities for each class
        self.likelihoods = np.zeros((self.num_classes, self.num_features))
        for i in range(self.num_classes):
            for j in range(self.num_features):
                # Compute the likelihood of feature j in class i
                self.likelihoods[i, j] = np.mean(X[y == (i+1), j] == 1)


    # Predict the class labels for a set of samples
    def predict(self, X):
        # Get the number of samples and the number of features
        num_samples = X.shape[0]
        num_features = X.shape[1]
        y = 0
        # Initialize the predicted class labels
        pred = np.zeros(num_samples)

        # Compute the log probabilities for each class
        log_probs = np.zeros((num_samples, self.num_classes))
        for i in range(self.num_classes):
            log_prior = np.log(self.priors[i])
            # Compute the log likelihood probabilities for each feature in class i
            log_likelihoods = np.zeros(num_samples)
            for k in range(num_samples):
                for j in range(num_features):
                    if y == num_samples-1:
                        y = 0

                    if np.any(X[k, j] == 1):
                        log_likelihoods[y] = np.log(0.0000000001+self.likelihoods[i, j])
                    else:
                        log_likelihoods[y] = np.log(1 - 0.0000000001+self.likelihoods[i, j])
                    y += 1
            # Compute the log probability for class i

            log_probs[:, i] = log_prior + log_likelihoods
        # Predict the class labels
        pred = np.argmax(log_probs, axis=1)
        return pred + 1
