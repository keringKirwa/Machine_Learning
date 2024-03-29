import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

""" 
Support Vector Classifier.(for classification tasks)
By using a random seed and adding  or subtracting [2,2]  [-2,-2] , data is centered around 2,2 and (-2,-2) points. 
The numpy.random.randn() function creates an array of specified shape and fills it with random values as per standard 
NORMAL distribution.
the array  is a 2D array , with 100x and 100y pints.eg ([ 1.67567504  3.12640631]), not necessarily integers.
 
"""
# Set seed for reproducibility
np.random.seed(42)

# Generate random data points
num_points = 100
A = np.random.randn(num_points, 2) + np.array([2, 2])
B = np.random.randn(num_points, 2) + np.array([-2, -2])

if __name__ == "__main__":
    plt.show()
    from sklearn import svm

    # Combine the data points and labels
    X = np.vstack((A, B))
    y = np.concatenate((np.ones(num_points), -np.ones(num_points)))

    # Create the SVM classifier
    clf = svm.SVC(kernel='linear', C=1)

    # Fit the classifier to the data
    clf.fit(X, y)
    """
    Note that the classes A and B are all 2D vectors , meaning that the elements in this class will be sth like :
    [[21,2],[78,3]], meaning that we can access all the rows, and then the first elements in that row uaing the following funtion 
    A[:, 0],  
    support_vectors in this case is  a matrix that represents the data point of the dots/data points if
    which will be used to draw the line separating the data points .the aim is to find an equation that matches the input to output .
    """

    # Get the support vectors and plot the decision boundary
    support_vectors = clf.support_vectors_
    print("the As", A)
    plt.scatter(A[:, 0], A[:, 1], color='red', label='Class A')
    plt.scatter(B[:, 0], B[:, 1], color='blue', label='Class B')
    plt.scatter(support_vectors[:, 0], support_vectors[:, 1], color='green', label='Support Vectors')
    weights = clf.coef_
    print("The weights are ", weights)
    a = -weights[0] / weights[1]
    xx = np.linspace(-5, 5)
    yy = a * xx - (clf.intercept_[0]) / weights[1]
    plt.plot(xx, yy, 'k-')
    plt.legend()
    plt.show()
