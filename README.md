# one-hidden-layer NN framework from scratch
This implementation is in python and is based upon core concepts
of neural networks as explained by Andrew Ng  as part of Stanford Machine Learning course.

## Setup as a package 
### Anaconda
* Create a new folder with your framework name in your Anaconda installation Lib folder, destination will be like: Anaconda3/Lib/framework
* add neuralnet.py to this folder
* add an empty python file to this folder and name it \_\_init\_\_.py

## Using it in python scripts

* to import the neural net into your script: 
```python
from framework.neuralnet import NeuralNet
```
* initializing by creating an instance of NeuralNet, lets call it ann
```python
ann=NeuralNet()
```
* architecture; specify number of nodes in each one of the three layers ordered as : input-hidden-output
```python
ann.architecture(5,10,2)
```
* fit_on_data; takes 2 arguments, first is matrix of features X (ndarray), second is vector of labels y (usually these are training data)
```python
ann.fit_on_data(x_train,y_train)
```
* compile; train the network by running the optimization algorithm which computes optimum weights
```python
ann.compile()
```
* predict; predict the label of a new instance; takes an instance x
```python
ann.predict(x_instance)
```
* evaluate; evaluate performance of the network by calculating accuracy; takes x_test, y_test
```python
ann.evaluate(x_test,y_test)
```

