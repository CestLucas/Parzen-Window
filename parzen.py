import numpy as np
iris = np.loadtxt('iris.txt')

######## DO NOT MODIFY THIS FUNCTION ########
def draw_rand_label(x, label_list):
    seed = abs(np.sum(x))
    while seed < 1:
        seed = 10 * seed
    seed = int(1000000 * seed)
    np.random.seed(seed)
    return np.random.choice(label_list)
#############################################

class Q1:

    def feature_means(self, iris):
        return np.mean(iris, axis=0)[:4]

    def covariance_matrix(self, iris):
        return np.cov(iris[:,:4].T)

    def feature_means_class_1(self, iris):
        mat = iris[np.where(iris[:,4] == 1)]
        return np.mean(mat, axis=0)[:4]

    def covariance_matrix_class_1(self, iris):
        mat = iris[np.where(iris[:,4] == 1)]
        return np.cov(mat[:,:4].T)

class HardParzen:
    def __init__(self, h):
        self.h = h  # "radius" of the Parzen window

    def train(self, train_inputs, train_labels):
        self.label_list = np.unique(train_labels)  # all labels in the training set
        self.train_inputs = train_inputs
        self.train_labels = train_labels 
     
    def minkowski_mat(self, x, Y, p=2):
        return (np.sum((np.abs(x - Y)) ** p, axis=1)) ** (1.0 / p)

    def compute_predictions(self, test_data):
        num_test = test_data.shape[0]  # number of items in the test data
        num_labels = len(self.label_list)  # number of training labels
        counts = np.ones((num_test, num_labels))  
        classes_pred = np.zeros(num_test) # record predictions
    
        for (i, ex) in enumerate(test_data):
            # Calculate distances to each each training input
            distances = self.minkowski_mat(ex, self.train_inputs)
            ind_neighbors = []  # list to store neighbor indexes
            radius = self.h
            
            # Find neighbors in Hard Parzen Window
            ind_neighbors = np.array([j for j in range(len(distances)) if distances[j] < radius])
            
            # if no neighbors within the Hard Parzen Window
            if len(ind_neighbors) == 0:
                ind_neighbors = np.array([int(draw_rand_label(ex, self.label_list))])
        
            # Calculate the number of neighbors belonging to each class
            cl_neighbors = list(self.train_labels[ind_neighbors] - 1)
            num_neighbors = len(cl_neighbors)  
            for j in range(min(num_neighbors, self.train_inputs.shape[0])):
                counts[i, int(cl_neighbors[j])] += 1
                
            # define classes_pred[i]
            classes_pred[i] = np.argmax(counts[i, :]) + 1
            
        return classes_pred


class SoftRBFParzen:
    def __init__(self, sigma):
        self.sigma  = sigma
        self.d = 2

    def train(self, train_inputs, train_labels):
        self.label_list = np.unique(train_labels)  # all labels in the training set
        self.train_inputs = train_inputs
        self.train_labels = train_labels 
           
    def kernel(self, xi, x):  # multivariate gaussian function
        return 1.0/((2*np.pi)**(self.d/2) * self.sigma**self.d) * np.e**(-1/2 * np.linalg.norm(xi - x)**2 / self.sigma**2)
    
    def compute_predictions(self, test_data):
        num_test = test_data.shape[0]  # number of items in the test data
        num_labels = len(self.label_list)  # number of training labels
        counts = np.ones((num_test, num_labels))  
        classes_pred = np.zeros(num_test) # record predictions
        
        for (i, ex) in enumerate(test_data):
            
            # iterate through training inputs
            for (j, train_item) in enumerate(self.train_inputs):
                weight = self.kernel(ex, train_item)
                counts[i, int(self.train_labels[j]) - 1] += weight # define similarity to the train label
            
            classes_pred[i] = np.argmax(counts[i,:]) + 1
        
        return classes_pred

    
def split_dataset(iris):
    indices = np.arange(0, iris.shape[0])
    training_set = iris[np.where(indices % 5 < 3)[0],:]
    validation_set = iris[np.where(indices % 5 == 3)[0],:]
    test_set = iris[np.where(indices % 5 == 4)[0],:]

    return training_set, validation_set, test_set


class ErrorRate:
    def __init__(self, x_train, y_train, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def hard_parzen(self, h):
        h_parzen = HardParzen(h)
        h_parzen.train(self.x_train, self.y_train)  # x_train = training set, x_val = validation set, y_train = t label
        h_parzen_pred = h_parzen.compute_predictions(self.x_val)
        return np.sum(h_parzen_pred != self.y_val) / float(self.y_val.size)

    def soft_parzen(self, sigma):
        s_parzen = SoftRBFParzen(sigma)
        s_parzen.train(self.x_train, self.y_train)
        s_parzen_pred = s_parzen.compute_predictions(self.x_val)
        return np.sum(s_parzen_pred != self.y_val) / float(self.y_val.size)


def get_test_errors(iris):
    set_h = [.001, .01, .1, .3, 1.0, 3.0, 10.0, 15.0, 20.0]
    set_sigma = [.001, .01, .1, .3, 1.0, 3.0, 10.0, 15.0, 20.0]
    
    training_set, validation_set, test_set = split_dataset(iris)
    x_train = training_set[:,:-1]
    y_train = training_set[:,-1]
    x_val = validation_set[:,:-1]
    y_val = validation_set[:,-1]
    x_test = test_set[:,:-1]
    y_test = test_set[:,-1]
    
    test_h = []
    for h in set_h:
        err_rate = ErrorRate(x_train, y_train, x_val, y_val)
        test_h.append((h, err_rate.hard_parzen(h)))

    test_sigma = []
    for sigma in set_sigma:
        err_rate = ErrorRate(x_train, y_train, x_val, y_val)
        test_sigma.append((sigma, err_rate.soft_parzen(sigma)))
    
    optimal_h = min(test_h, key = lambda t: t[1])[0]
    optimal_sigma = min(test_sigma, key = lambda t: t[1])[0]
    
    err_rate = ErrorRate(x_train, y_train, x_test, y_test)
    err_hard = err_rate.hard_parzen(optimal_h)
    err_soft = err_rate.soft_parzen(optimal_sigma)
    return [err_hard, err_soft]


def random_projections(X, A):
    return (2**-0.5)*np.dot(X, A)