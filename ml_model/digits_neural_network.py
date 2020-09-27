import numpy as np
import pandas as pd 
import tensorflow as tf
import math

class DigitNeuralNetwork:
    def __init__(self, epochs:int = 100, batch_size:int = 32):
        '''Initialize Digit NN'''
        self.epochs = epochs
        self.batch_size = batch_size
    
    def _activation(self, activation: str, y_matrix: tf.Tensor):
        '''Activation Function
        
        Args:
            activation (string): Name of activation function to use
            y_matrix (tf.Tensor): Tensor to have activation function applied too
        
        Returns:
            activation_tensor (tf.Tensor): Tensor that had non-linearity activation function applied
        '''
        if activation == 'Sigmoid' or activation == 'sigmoid':
            activation_tensor = tf.compat.v1.sigmoid(y_matrix)
            return activation_tensor
        elif activation == 'ReLu' or activation == 'relu':
            activation_tensor = tf.compat.v1.nn.relu(y_matrix)
            return activation_tensor
        elif activation == 'Tanh' or activation == 'tanh':
            activation_tensor = tf.compat.v1.tanh(y_matrix)
            return activation_tensor
        elif activation == 'LeakyReLu':
            activation_tensor = tf.compat.v1.nn.leaky_relu(y_matrix)
            return activation_tensor
        elif activation == 'Softmax' or activation == 'softmax':
            activation_tensor = tf.compat.v1.nn.softmax(y_matrix)
            return activation_tensor
        else:
            print("Invalid Activation function, returning original tensor")
            return y_matrix

    def _create_weights_and_bias(self, weights_shape: list, bias_shape: list) -> tf.Tensor:
        '''Creates a randomized weight and bias tensor to be used for NN calculations
        
        Args:
            weights_shape (list): Weight tensor shape
            bias_shape (list): Bias tensor shape
        
        Returns:
            W_n (tf.Tensor): Weight tensor
            b_n (tf.Tensor): Bias tensor
        '''
        W_n = tf.Variable(tf.random.truncated_normal(shape=weights_shape, stddev=0.03))
        b_n = tf.Variable(tf.random.truncated_normal(shape=bias_shape))

        return W_n, b_n
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_valid: np.ndarray, y_valid:np.ndarray, X_test, y_test):
        '''Trains the ML Model utilizing a 6 hidden layer CNN
        
        Args:
            X_train (np.ndarray): Training dataset
            y_train (np.ndarray): Training Label dataset
            X_valid (np.ndarray): Validation dataset
            y_valid (np.ndarray): Validation label dataset
        '''
        # Input Layer
        x = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 100, 100, 3], name='x_input')
        y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 10], name='y_labels')
        is_training = tf.compat.v1.placeholder(dtype=tf.bool, name='is_training')

        # Conv Hidden Layer 1 - Weights/Bias -> Layer Calculation
        W1, b1 = self._create_weights_and_bias([3, 3, 3, 32], [32])
        b1_conv = tf.add(tf.nn.conv2d(input=x, filter=W1, strides=[1,1,1,1], padding='SAME'), b1)
        b1_conv_batch_norm = tf.compat.v1.layers.batch_normalization(b1_conv, training=is_training)
        b1_conv_activation = self._activation('ReLu', b1_conv_batch_norm)

        # Conv Hidden Layer 2 - Weights/Bias -> Layer Calculation
        W2, b2 = self._create_weights_and_bias([3, 3, 32, 64], [64])
        b2_conv = tf.add(tf.nn.conv2d(input=b1_conv_activation, filter=W2, strides=[1,1,1,1], padding='SAME'), b2)
        b2_conv_batch_norm = tf.compat.v1.layers.batch_normalization(b2_conv, training=is_training)
        b2_conv_activation = self._activation('ReLu', b2_conv_batch_norm)

        # Conv Hidden Layer 3 - Weights/Bias -> Layer Calculation
        W3, b3 = self._create_weights_and_bias([3, 3, 64, 128], [128])
        b3_conv = tf.add(tf.nn.conv2d(input=b2_conv_activation, filter=W3, strides=[1,1,1,1], padding='SAME'), b3)
        b3_conv_batch_norm = tf.compat.v1.layers.batch_normalization(b3_conv, training=is_training)
        b3_conv_activation = self._activation('ReLu', b3_conv_batch_norm)

        # Flatten Conv Layer
        conv_shape = b3_conv_activation.get_shape()
        num_features = conv_shape[1:4].num_elements()
        flatten_conv = tf.reshape(b3_conv_activation, [-1, num_features])

        # FC Hidden Layer 4 - Weights/Bias -> Layer Calculation
        W4, b4 = self._create_weights_and_bias([flatten_conv.get_shape()[1:4].num_elements(), 256], [256])
        fc_layer4 = tf.add(tf.matmul(flatten_conv, W4), b4)
        fc_layer4_batch_norm = tf.compat.v1.layers.batch_normalization(fc_layer4, training=is_training)
        fc_layer4_activation = self._activation('ReLu', fc_layer4_batch_norm)

        # FC Hidden Layer 5 - Weights/Bias -> Layer Calculation
        W5, b5 = self._create_weights_and_bias([256, 64], [64])
        fc_layer5 = tf.add(tf.matmul(fc_layer4_activation, W5), b5)
        fc_layer5_batch_norm = tf.compat.v1.layers.batch_normalization(fc_layer5, training=is_training)
        fc_layer5_activation = self._activation('ReLu', fc_layer5_batch_norm)

        # FC Hidden Layer 6 - Weights/Bias -> Layer Calculation
        W6, b6 = self._create_weights_and_bias([64, 10], [10])
        fc_layer6 = tf.add(tf.matmul(fc_layer5_activation, W6), b6)
        fc_layer6_batch_norm = tf.compat.v1.layers.batch_normalization(fc_layer6, training=is_training)
        y_preds = self._activation('Softmax', fc_layer6_batch_norm)

        # Loss and Optimization
        loss_tensor = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_preds))
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)
        correct_preds = tf.equal(tf.argmax(y_preds, -1), tf.argmax(y, -1))
        accuracy_tensor = tf.reduce_mean(tf.cast(correct_preds, tf.float32))

        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        update_weights = optimizer.minimize(loss_tensor)
        update_weights = tf.group([update_weights, update_ops])

        # Training, Validation, and Test
        with tf.compat.v1.Session() as sess:
            init = tf.compat.v1.global_variables_initializer()
            sess.run(init)

            total_batch_size = math.ceil(X_train.shape[0] / self.batch_size)

            for epoch in range(1, self.epochs):
                train_loss = train_acc = 0.0
                valid_loss = valid_acc = 0.0

                for i in range(total_batch_size):
                    x_train_batch = X_train[i * self.batch_size:i * self.batch_size + self.batch_size]
                    y_train_batch = y_train[i * self.batch_size:i * self.batch_size + self.batch_size]

                    _, loss, acc = sess.run([update_weights, loss_tensor, accuracy_tensor],
                        feed_dict={x:x_train_batch, y:y_train_batch, is_training:True})

                    train_loss += loss
                    train_acc += acc

                train_loss /= total_batch_size
                train_acc /= total_batch_size

                valid_loss, valid_acc = sess.run([loss_tensor, accuracy_tensor],
                    feed_dict={x:X_valid, y:y_valid, is_training:False})

                print("Epoch: {0}".format(epoch))
                print("Train Accuracy: {0}%".format(train_acc * 100))
                print("Train Loss: {0}".format(train_loss))
                print("Valid Accuracy: {0}%".format(valid_acc * 100))
                print("Valid Loss: {0}".format(valid_loss))
                print()
            
            test_loss, test_acc = sess.run([loss_tensor, accuracy_tensor],
                feed_dict={x:X_test, y:y_test, is_training:False})

            print('Testing Results')
            print('---------------')
            print('Test Accuracy: {0}'.format(test_acc * 100))
            print('Test Loss: {0}'.format(test_loss))