# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 06:34:04 2015

@author: tanay
"""
from lasagne.layers import InputLayer, DropoutLayer, DenseLayer
from lasagne.updates import nesterov_momentum
from lasagne.objectives import binary_crossentropy
from nolearn.lasagne import NeuralNet
import theano
from theano import tensor as T
from theano.tensor.nnet import sigmoid
from sklearn import metrics
from sklearn.utils import shuffle
import numpy as np

learning_rate = theano.shared(np.float32(0.1))

input_size=Xtrh.shape

class AdjustVariable(object):
    def __init__(self, variable, target, half_life=20):
        self.variable = variable
        self.target = target
        self.half_life = half_life
    def __call__(self, nn, train_history):
        delta = self.variable.get_value() - self.target
        delta /= 2**(1.0/self.half_life)
        self.variable.set_value(np.float32(self.target + delta))


net = NeuralNet(
layers=[  
    ('input', InputLayer),
     ('hidden1', DenseLayer),
    ('dropout1', DropoutLayer),
    ('hidden2', DenseLayer),
    ('dropout2', DropoutLayer),
    ('output', DenseLayer),
    ],
# layer parameters:
input_shape=(None, input_size), 
hidden1_num_units=400, 
dropout1_p=0.4,
hidden2_num_units=200, 
dropout2_p=0.4,
output_nonlinearity=sigmoid, 
output_num_units=4, 

# optimization method:
update=nesterov_momentum,
update_learning_rate=learning_rate,
update_momentum=0.899,

# Decay the learning rate
on_epoch_finished=[
        AdjustVariable(learning_rate, target=0, half_life=4),
        ],

# This is silly, but we don't want a stratified K-Fold here
# To compensate we need to pass in the y_tensor_type and the loss.
regression=True,
y_tensor_type = T.imatrix,
objective_loss_function = binary_crossentropy,
 
max_epochs=75, 
eval_size=0.1,
verbose=1,
)

X, y = shuffle(Xtrh, y, random_state=123)
net.fit(X, y)

_, X_valid, _, y_valid = net.train_test_split(X, y, net.eval_size)
probas = net.predict_proba(X_valid)[:,0]
print("ROC score", metrics.roc_auc_score(y_valid, probas))

