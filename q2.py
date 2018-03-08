import numpy as np 

from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt

np.random.seed(1847)

class BatchSampler(object):
    '''
    A (very) simple wrapper to randomly sample batches without replacement.

    You shouldn't need to touch this.
    '''
    
    def __init__(self, data, targets, batch_size):
        self.num_points = data.shape[0]
        self.features = data.shape[1]
        self.batch_size = batch_size

        self.data = data
        self.targets = targets

        self.indices = np.arange(self.num_points)

    def random_batch_indices(self, m=None):
        '''
        Get random batch indices without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        if m is None:
            indices = np.random.choice(self.indices, self.batch_size, replace=False)
        else:
            indices = np.random.choice(self.indices, m, replace=False)
        return indices 

    def get_batch(self, m=None):
        '''
        Get a random batch without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        indices = self.random_batch_indices(m)
        X_batch = np.take(self.data, indices, 0)
        y_batch = self.targets[indices]
        return X_batch, y_batch  

class GDOptimizer(object):
    '''
    A gradient descent optimizer with momentum
    '''

    def __init__(self, lr, beta=0.0):
        self.lr = lr
        self.beta = beta
        self.vel = 0.0

    def update_params(self, params, grad):
        # Update parameters using GD with momentum and return
        # the updated parameters
        self.vel = np.multiply(self.vel,self.beta)
        self.vel =  self.vel + np.multiply(self.lr,grad)
        return params-self.vel

class SVM(object):
    '''
    A Support Vector Machine
    '''

    def __init__(self, c, feature_count):
        self.c = c
        self.w = np.random.normal(0.0, 0.1, feature_count)

        
        
    def hinge_loss(self, X, y):
        '''
        Compute the hinge-loss for input data X (shape (n, m)) with target y (shape (n,)).

        Returns a length-n vector containing the hinge-loss per data point.
        '''
        hinge_loss = np.zeros((len(y),1))
        for i in range(len(X)):
            term = 1-(y[i]*(np.dot(self.w.T,X[i])))
            hinge_loss[i] = max(term,0)
        # Implement hinge loss
        return hinge_loss

    def grad(self, X, y):
        '''
        Compute the gradient of the SVM objective for input data X (shape (n, m))
        with target y (shape (n,))

        Returns the gradient with respect to the SVM parameters (shape (m,)).
        '''
        hingeloss = self.hinge_loss(X,y)
        sum_ = 0
        for i in range(len(hingeloss)):
            if hingeloss[i] != 0:
                sum_ += y[i]*X[i]
                
        grad = self.w - ((self.c)/len(X))*sum_
        
        return grad

    def classify(self, X):
        '''
        Classify new input data matrix (shape (n,m)).

        Returns the predicted class labels (shape (n,))
        '''
        # Classify points as +1 or -1
        labels = []
        for i in range(len(X)):
            num = np.dot(X[i],self.w)
            #print(np.size(num))
            if num < 0:
                labels.append(-1)
            else:
                labels.append(1)
        return labels

def load_data():
    '''
    Load MNIST data (4 and 9 only) and split into train and test
    '''
    mnist = fetch_mldata('MNIST original', data_home='./data')
    label_4 = (mnist.target == 4)
    label_9 = (mnist.target == 9)

    data_4, targets_4 = mnist.data[label_4], np.ones(np.sum(label_4))
    data_9, targets_9 = mnist.data[label_9], -np.ones(np.sum(label_9))

    data = np.concatenate([data_4, data_9], 0)
    data = data / 255.0
    targets = np.concatenate([targets_4, targets_9], 0)

    permuted = np.random.permutation(data.shape[0])
    train_size = int(np.floor(data.shape[0] * 0.8))

    train_data, train_targets = data[permuted[:train_size]], targets[permuted[:train_size]]
    test_data, test_targets = data[permuted[train_size:]], targets[permuted[train_size:]]
    print("Data Loaded")
    print("Train size: {}".format(train_size))
    print("Test size: {}".format(data.shape[0] - train_size))
    print("-------------------------------")
    return train_data, train_targets, test_data, test_targets

def optimize_test_function(optimizer, w_init=10.0, steps=200):
    '''
    Optimize the simple quadratic test function and return the parameter history.
    '''
    def func(x):
        return 0.01 * x * x

    def func_grad(x):
        return 0.02 * x

    w = w_init
    w_history = [w_init]
    y = [func(w)]
    for i in range(steps):
        grad = func_grad(w)
        w = optimizer.update_params(w,grad)
        w_history.append(w)
        y.append(func(w))

    return w_history

def optimize_svm(train_data, train_targets, penalty, optimizer, batchsize, iters):
    '''
    Optimize the SVM with the given hyperparameters. Return the trained SVM.
    '''
    num_targets = train_data.shape
    hinge_losses = []
    svm = SVM(penalty,num_targets[1])
    batch_sampler = BatchSampler(train_data,train_targets,batchsize)
    for i in range(iters):
        data,targets = batch_sampler.get_batch()
        gradient = svm.grad(data,targets)

        svm.w = optimizer.update_params(svm.w,gradient)
        
    #svm.w = np.mean(s_w,axis=0)
        
    return svm

def classification_accuracy(true_labels, predict_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''
    size_data = np.size(true_labels)
    sum = 0
    for i in range(len(true_labels)):
        if true_labels[i] == predict_labels[i]:
            sum+=1
    accuracy = sum/size_data
    return accuracy*100

def plot_w(svm):
    reshaped = np.reshape(svm.w[1:].T,(28,28))
    plt.imshow(reshaped, cmap='gray')
    plt.show()
    

if __name__ == '__main__':
    np.random.seed(1847)
    #2.3
    train_data, train_targets, test_data, test_targets = load_data()
    train_data = np.insert(train_data,0,1,axis=1)
    test_data = np.insert(test_data,0,1,axis=1)
    beta_zero_opt = GDOptimizer(0.05,0.0)
    beta_point1_opt = GDOptimizer(0.05,0.1)
    svm_beta0 = optimize_svm(train_data,train_targets,1.0,beta_zero_opt,100,500)
    predict_train_0 = svm_beta0.classify(train_data)
    print("Classification accuracy on train data, beta=0.0: {}".format(classification_accuracy(train_targets,predict_train_0)))
    predict_test_0 = svm_beta0.classify(test_data)
    print("Classification accuracy on test data, beta=0.0: {}".format(classification_accuracy(test_targets,predict_test_0)))
    
    train_hingeloss_0 = svm_beta0.hinge_loss(train_data,train_targets)
    print("Train average hinge loss, beta = 0: {}".format(np.average(train_hingeloss_0)))
    test_hingeloss_0 = svm_beta0.hinge_loss(test_data,test_targets)
    print("Test average hinge loss, beta = 0: {}".format(np.average(test_hingeloss_0)))
    
    plot_w(svm_beta0)
    
    svm_beta1 = optimize_svm(train_data,train_targets,1.0,beta_point1_opt,100,500)
    predict_train_1 = svm_beta1.classify(train_data)
    #print(np.shape(predict_train_1))
    #print(np.shape(train_targets))
    print("Classification accuracy on train data, beta=0.1: {}".format(classification_accuracy(train_targets,predict_train_1)))
    predict_test_1 = svm_beta1.classify(test_data)
    print("Classification accuracy on test data, beta=0.1: {}".format(classification_accuracy(test_targets,predict_test_1)))
    
    plot_w(svm_beta1)
    
    train_hingeloss_1 = svm_beta1.hinge_loss(train_data,train_targets)
    print("Train average hinge loss, beta = 0: {}".format(np.average(train_hingeloss_1)))
    test_hingeloss_1 = svm_beta1.hinge_loss(test_data,test_targets)
    print("Test average hinge loss, beta = 0: {}".format(np.average(test_hingeloss_1)))
    
        #2.1
    beta_zero_opt = GDOptimizer(1.0,0.0)
    beta_point9_opt = GDOptimizer(1.0,0.9)
    param_hist_0 = optimize_test_function(beta_zero_opt)
    param_hist_9 = optimize_test_function(beta_point9_opt)

    zero, = plt.plot(range(201),param_hist_0,'r',label = 'beta = 0.0')
    point9, = plt.plot(range(201),param_hist_9,'b', label = 'beta = 0.9')
    plt.legend(handles = [zero,point9])
    plt.ylabel("w")
    plt.xlabel("steps")
    plt.show