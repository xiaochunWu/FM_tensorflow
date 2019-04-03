# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 19:29:45 2019

@author: ims
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.sparse import csr
from collections import defaultdict

# Convert list to sparse matrix

def vectorize_dic(dic, ix=None, p=None):
    '''
    Creates a scipy csr matrix from a list of lists(each inner list is a set of values corresponding to a feature)
    
    Args:
    dic: dictionary of feature lists. Keys are the name of features
    ix: index generator(default None)
    p: dimension of feature space(number of columns in the sparse matrix)(default None)
    '''
    if ix == None:
#        d = count(0)
        ix = defaultdict(lambda:0)
        
    n = len(list(dic.values())[0])
    g = len(list(dic.keys()))
    nz = n * g
    print('n:{},g:{},nz:{}'.format(n,g,nz))
    col_ix = np.empty(nz, dtype=int)
    
    i = 0
    for k, lis in dic.items():
        col_ix[i::g] = [ix[str(el) + str(k)] for el in lis]
        i += 1
        
    row_ix = np.repeat(np.arange(0, n), g)
    data = np.ones(nz)
    
    if p == None:
        p = len(ix)
        
    ixx = np.where(col_ix<p)
    return csr.csr_matrix((data[ixx],(row_ix[ixx],col_ix[ixx])), shape=(n,p)), ix

# Load data
cols = ['user', 'item', 'rating', 'timestamp']
train = pd.read_csv('../data/ua.base', delimiter='\t', names=cols)
test = pd.read_csv('../data/ua.test', delimiter='\t', names=cols)

# Vectorise data and convert them to csr matrix
X_train, ix = vectorize_dic({'users':train.user.values, 'items':train.item.values})
X_test, ix = vectorize_dic({'users':test.user.values, 'items':test.item.values}, ix, X_train.shape[1])
y_train = train.rating.values
y_test = test.rating.values

X_train = X_train.todense()
X_test = X_test.todense()

print(X_train.shape)
print(X_train.shape[0])
print(X_test.shape)
print(X_test.shape[0])

n, p = X_train.shape

k = 10

X = tf.placeholder('float', shape=[None, p])
y = tf.placeholder('float', shape=[None, 1])

w0 = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.zeros([p]))

V = tf.Variable(tf.random_normal([k, p], stddev=0.01))

y_hat = tf.Variable(tf.zeros([n, 1]))

# Calculate output with FM equation
linear_terms = tf.add(w0, tf.reduce_sum(tf.multiply(W, X), 1, keep_dims=True))
pair_interactions = (tf.multiply(0.5,
                                 tf.reduce_sum(
                                         tf.subtract(
                                                 tf.pow(tf.matmul(X, tf.transpose(V)), 2),
                                                 tf.matmul(tf.pow(X, 2), tf.transpose(tf.pow(V,2)))),
                                                 1, keep_dims=True)))
y_hat = tf.add(linear_terms, pair_interactions)
                    
lambda_w = tf.constant(0.001, name='lambda_w')
lambda_v = tf.constant(0.001, name='lambda_v')

l2_norm = (tf.reduce_sum(
                        tf.add(tf.multiply(lambda_w, tf.pow(W, 2)),
                               tf.multiply(lambda_v, tf.pow(V, 2)))))
error = tf.reduce_mean(tf.square(tf.subtract(y,y_hat)))
loss = tf.add(error, l2_norm)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

def batcher(X_, y_=None, batch_size=-1):
    n_samples = X_.shape[0]
    
    if batch_size == -1:
        batch_size = n_samples
    if batch_size < 1:
        raise ValueError('Parameter batch_size={} is unsupported'.format(batch_size))
        
    for i in range(0, n_samples, batch_size):
        upper_bound = min(i+batch_size, n_samples)
        ret_x = X_[i:upper_bound]
        ret_y = None
        if y_ is not None:
            ret_y = y_[i:i+batch_size]
            # yield是一个类似return的关键字，迭代一次遇到yield时就返回yield后面的值。下一次迭代时，从上一次迭代遇到的yield后面的代码开始执行
            # 简要理解：yield就是return返回一个值，并且记住这个返回的位置，下次迭代就从这个位置后开始
            yield (ret_x, ret_y)
#            return (ret_x, ret_y)
    
epochs = 10
batch_size = 1000

init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init)

#for epoch in tqdm(range(epochs), unit='epoch'):
for epoch in range(epochs):
    # shuffle
    perm = np.random.permutation(X_train.shape[0])
    for bX, bY in batcher(X_train[perm], y_train[perm], batch_size):
        sess.run(optimizer, feed_dict={X:bX.reshape(-1, p), y:bY.reshape(-1, 1)})
 
errors = []
for bX, bY in batcher(X_test, y_test):
    errors.append(sess.run(error, feed_dict={X:bX.reshape(-1,p), y:bY.reshape(-1,1)}))
       
RMSE = np.sqrt(np.array(errors).mean())
print(RMSE)