# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 13:40:45 2018

@author: scott
"""

import tensorflow as tf
import numpy as np
from datetime import datetime

class LogisticRegression(object):
    
    def __init__(self, 
                 initialiser=tf.keras.initializers.glorot_uniform(), 
                 l2_reg=0.0, 
                 optimiser=tf.train.AdamOptimizer(),
                 checkpoint_path='/tmp/my_model_final.ckpt'):
        '''
        Class initialiser
        
        Inputs:
            initialiser - (tf op) tensorflow initialiser
                          default=tf.keras.initializers.glorot_uniform()
            l2_reg - (float >= 0) l2 regularisation parameter
                     default=0.0
            optimiser - (tf op) tensorflow optimiser
                        default=tf.train.AdamOptimizer()
            checkpoint_path - (string) path for saving model checkpoint
                              default='/tmp/my_model_final.ckpt'
        '''
        
        self.initialiser = initialiser
        self.l2_reg = l2_reg
        self.optimiser = optimiser
        self.checkpoint_path = checkpoint_path
        self.weights = None
        self.biases = None
        self.tf_ops = {}
        self.first_fit = True
        
    
    def fit(self,
            X,
            y,
            batch_size=32, 
            num_epochs=100,
            verbose=True):
        '''
        Trains the model
        
        Inputs:
            X - (np array) training features
            y - (np vector) training labels (0/1)
            batch_size - (int > 0) training batch size
                         default=32
            num_epochs - (int > 0) number of training epochs
                         default=100
            verbose - (bool) whether or not to print updates after every epoch
                      default=True
        '''
    
        # get the data dimensions
        num_x, input_dim = X.shape
        
        # build the computational graph
        self.build_graph(input_dim)
        
        # run computation
        with tf.Session() as sess:
    
            if self.first_fit:
                # initialise variables
                sess.run(self.tf_ops['init'])
                self.first_fit = False 
            else:
                # restore variables
                self.tf_ops['saver'].restore(sess, self.checkpoint_path)

            # train for num_epochs
            num_batches = int(np.ceil(num_x / batch_size))
            for epoch in np.arange(1, num_epochs + 1):
                
                if verbose:
                    print('+-----------------------------------------------------------+')
                    print('Running epoch', epoch, 'of', num_epochs)

                if epoch % 10 == 0:
                    # save model checkpoint
                    save_path = self.tf_ops['saver'].save(sess, self.checkpoint_path)
                    
                # shuffle training data
                if epoch == 1:
                    shuffle = np.random.choice(num_x, num_x, replace=False)
                else:
                    shuffle = shuffle[np.random.choice(num_x, num_x, replace=False)]
                X = X[shuffle]
                y = y[shuffle]

                # train in batches                
                for batch in np.arange(num_batches):

                    if batch % 10 == 0:
                        # add loss to summary
                        summary_str = self.tf_ops['loss_summary'].eval(feed_dict={self.tf_ops['X']: X, 
                                                                                  self.tf_ops['y']: y})
                        step = epoch * num_batches + batch
                        self.tf_ops['file_writer'].add_summary(summary_str, step)

                    # get data in batch
                    i_first = batch * batch_size
                    i_last = min(num_x, (batch + 1) * batch_size)
                    X_batch = X[i_first:i_last]
                    y_batch = y[i_first:i_last]

                    # run training step
                    sess.run(self.tf_ops['training_op'], 
                             feed_dict={self.tf_ops['X']: X_batch, 
                                        self.tf_ops['y']: y_batch})
                    
                if verbose:
                    # compute training loss
                    xentropy = self.tf_ops['xentropy'].eval(feed_dict={self.tf_ops['X']: X, 
                                                                       self.tf_ops['y']: y})
                    print('Training loss =', xentropy)

            # save model checkpoint
            save_path = self.tf_ops['saver'].save(sess, self.checkpoint_path)

        self.tf_ops['file_writer'].close()
        
        
    def build_graph(self, input_dim):
        '''
        Builds the tensorflow computational graph

        Inputs:
            input_dim - (int > 0) number of input features        
        '''

        tf.reset_default_graph()

        # create placeholders for training data
        X = tf.placeholder(tf.float32, shape=(None, input_dim), name='X')
        y = tf.placeholder(tf.float32, shape=(None), name='y')

        # forward pass
        with tf.name_scope('forward_pass'):
            # define weights and bias term
            W = tf.Variable(self.initialiser(shape=(input_dim, 1)), name='weights', dtype=tf.float32)
            b = tf.Variable(0.0, name='bias', dtype=tf.float32)

            # compute logits
            logits = tf.squeeze(tf.matmul(X, W) + b)
            p = tf.sigmoid(logits)

        # loss
        with tf.name_scope('loss'):
            # cross entropy
            xentropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, 
                                                                              logits=logits))            
            if self.l2_reg > 0:
                # add regularisation loss
                reg_loss = tf.nn.l2_loss(W) * self.l2_reg
            else:
                reg_loss = 0.0

            loss = xentropy + reg_loss

        # optimiser
        with tf.name_scope('optimiser'):
            training_op = self.optimiser.minimize(loss)

        # variable initialiser
        init = tf.global_variables_initializer() 

        # model checkpoint saver
        saver = tf.train.Saver()  

        # summary for tensorboard
        now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        root_logdir = 'tf_logs'
        logdir = '{}/run-{}'.format(root_logdir, now)
        loss_summary = tf.summary.scalar('xentropy', loss)
        file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

        # store tf operations we will need to access from other methods
        self.tf_ops['X'] = X
        self.tf_ops['y'] = y
        self.tf_ops['p'] = p
        self.tf_ops['xentropy'] = xentropy
        self.tf_ops['training_op'] = training_op
        self.tf_ops['init'] = init
        self.tf_ops['saver'] = saver
        self.tf_ops['loss_summary'] = loss_summary
        self.tf_ops['file_writer'] = file_writer


    def predict(self, X, batch_size=32):
        '''
        Predicts probability of positive class for each input

        Inputs:
            X - (np array) features
            batch_size - (int > 0) batch size for computing probabilities
                         default = 32

        Outputs:
            p - (np vector) probability of positive class for each input
        '''

        # build computational graph
        num_x, input_dim = X.shape
        self.build_graph(input_dim)

        with tf.Session() as sess:

            # restore variables
            self.tf_ops['saver'].restore(sess, self.checkpoint_path)

            # compute probabilities
            p  = self.tf_ops['p'].eval(feed_dict={self.tf_ops['X']: X})
            
        return p                