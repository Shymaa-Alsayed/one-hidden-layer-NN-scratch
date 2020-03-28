import numpy as np
from scipy import optimize


class NeuralNet:
    m = 0
    lamda = 1
    y_matrix = None
    y = None
    X = None
    opt_theta1 = None
    opt_theta2 = None
    labels_dict = {}
    labels = None

    def __init__(self, inp_nodes=0, hidden_nodes=0, output_nodes=0):
        self.inp_nodes = inp_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

    def sigmoid(self, z):
        """return sigmoid function of z whether it is a number, vector, or matrix"""
        return 1 / (1 + np.exp(-z))

    def initialize_weights(self):
        """initialize weight matrices theta1 and theta2 via choosing a good epsilon value to break the symmetry"""
        eps = np.sqrt(6) / (np.sqrt(self.inp_nodes + self.output_nodes))
        theta1 = np.random.rand(self.hidden_nodes, self.inp_nodes + 1) * (2 * eps) - eps
        theta2 = np.random.rand(self.output_nodes, self.hidden_nodes + 1) * (2 * eps) - eps
        return theta1, theta2

    def architecture(self, inp_nodes, hidden_nodes, output_nodes):
        """set number of input, hidden, and output nodes that define the network and set dictionary of y labels"""
        self.inp_nodes = inp_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.labels = np.eye(self.output_nodes, dtype=int)

    def encode_label(self, label):
        """encode each element in y vector into a vector of length output nodes """
        if label not in self.labels_dict:
            self.labels_dict[label] = self.labels[0].tolist()
            self.labels = self.labels[1:]
        return self.labels_dict[label]

    def create_y_matrix(self):
        """create matrix that represents each y label as a row vector in this matrix"""
        self.y_matrix = np.zeros((self.m, self.output_nodes), dtype=int)
        for i in range(self.m):
            label = int(self.y[i])
            self.y_matrix[i] = self.encode_label(label)
        self.labels = np.eye(self.output_nodes, dtype=int)

    def fit_on_data(self, X, y):
        """ create X matrix of features and y vector"""
        self.X = X
        self.y = y
        self.m = X.shape[0]
        self.create_y_matrix()

    def sigmoid_grad(self, z):
        """return the gradient of the sigmoid of z whether it is a number, vector or a matrix """
        return np.multiply(self.sigmoid(z), 1 - self.sigmoid(z))

    def forward_prop(self,init_theta1,init_theta2):
        """perform forward propagation algorithm for one hidden and one output layer and return needed values for
         calculating cost function and back propagation algorithm"""
        X = self.X
        a1 = np.append(arr=np.ones((self.m, 1)).astype(int), values=X, axis=1)
        z2 = a1.dot(init_theta1.T)
        a2 = np.append(arr=np.ones((self.m, 1)).astype(int), values=self.sigmoid(z2), axis=1)
        z3 = a2.dot(init_theta2.T)
        h = self.sigmoid(z3)
        return a1,a2,z2,h

    def cost(self,x):
        """ regularized cost function and gradients  needed for optimization of parameter """
        init_theta1, init_theta2 = x[:self.hidden_nodes * (self.inp_nodes + 1)].reshape(
            (self.hidden_nodes, self.inp_nodes + 1)), x[self.hidden_nodes * (self.inp_nodes + 1):].reshape(
            (self.output_nodes, self.hidden_nodes + 1))
        a1,a2,z2,h=self.forward_prop(init_theta1,init_theta2)
        # compute cost for all examples
        total_cost = []
        for i in range(self.m):
            cost = (-1 / self.m) * sum(
                np.multiply(self.y_matrix[i, :], np.log(h[i, :])) + np.multiply(1 - self.y_matrix[i, :],
                                                                                np.log(1 - h[i, :])))
            total_cost.append(cost)

        # compute cost regularization value for the whole network
        reg = (self.lamda / (2 * self.m)) * (
                sum(sum(np.square(init_theta1[:, 1:]))) + sum(sum(np.square(init_theta2[:, 1:]))))
        cost=sum(total_cost) + reg

        # Back propagation
        delta3 = h - self.y_matrix
        delta2 = np.multiply(delta3.dot(init_theta2[:, 1:]), self.sigmoid_grad(z2))
        D1 = delta2.T.dot(a1)
        D2 = delta3.T.dot(a2)
        grad_theta1 = (1 / self.m) * D1
        grad_theta2 = (1 / self.m) * D2

        # regularization of gradients
        init_theta1[:,0]=0
        init_theta2[:,0]=0
        grad_theta1=grad_theta1+init_theta1*(self.lamda/self.m)
        grad_theta2=grad_theta2+init_theta2*(self.lamda/self.m)
        grad_vect = np.append(grad_theta1.reshape(-1), grad_theta2.reshape(-1), axis=0)
        return cost, grad_vect

    def compile(self):
        """return optimum parameters by compiling the network optimization algorithm"""
        init_theta1, init_theta2 = self.initialize_weights()
        print('Compiling  and optimization in progress....')
        x = np.append(arr=init_theta1.reshape(-1), values=init_theta2.reshape(-1), axis=0)
        fmin = optimize.minimize(fun=self.cost, x0=x, jac=True, method='TNC', options={'maxiter': 250})
        self.opt_theta1, self.opt_theta2 = fmin.x[:self.hidden_nodes * (self.inp_nodes + 1)].reshape(
            (self.hidden_nodes, self.inp_nodes + 1)), fmin.x[self.hidden_nodes * (self.inp_nodes + 1):].reshape(
            (self.output_nodes, self.hidden_nodes + 1))
        print('Optimum weights calculated successfully')

    def predict(self, instance):
        """return predicted class for new instance"""
        instance = np.array([[1] + instance])
        z2 = instance.dot(self.opt_theta1.T)
        a2 = np.append(arr=np.array([[1]]), values=self.sigmoid(z2), axis=1)
        z3 = a2.dot(self.opt_theta2.T)
        a3 = self.sigmoid(z3)
        h = a3[0].tolist()
        idx = h.index(max(h))
        out_class = len(h) * [0]
        out_class[idx] = 1
        for k, v in self.labels_dict.items():
            if v == out_class:
                return k

    def evaluate(self, x_test, y_test):
        """evaluate network performnce by printing performance profile of accuracy, precision, recall, f1-score"""
        y_pred = []
        for x in x_test:
            prediction = self.predict(x.tolist())
            y_pred.append(prediction)
        n_true = sum([True if y_pred[i] == y_test[i][0] else False for i in range(len(y_test))])
        accuracy = n_true / len(y_test) * 100
        return print('ACCURACY = {:.2f}'.format(accuracy))


