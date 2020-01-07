import tensorflow as tf
import numpy as np


class MLP(object):
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, hidden_size, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
#            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            self.embedded_chars_avg = tf.reduce_mean(self.embedded_chars, axis=1)

        num_layers_0 = hidden_size
        num_layers_1 = hidden_size
        keep_prob=0.5

        ## Weights initialized by random normal function with std_dev = 1/sqrt(number of input features)
        weights_0 = tf.Variable(
            tf.random_normal([embedding_size, num_layers_0], stddev=(1 / tf.sqrt(float(embedding_size)))))
        bias_0 = tf.Variable(tf.random_normal([num_layers_0]))
        weights_1 = tf.Variable(
            tf.random_normal([num_layers_0, num_layers_1], stddev=(1 / tf.sqrt(float(num_layers_0)))))
        bias_1 = tf.Variable(tf.random_normal([num_layers_1]))

        ## Initializing weigths and biases
        with tf.name_scope("relu_0"):
            hidden_output_0 = tf.nn.relu(tf.matmul(self.embedded_chars_avg, weights_0) + bias_0)
        with tf.name_scope("relu_1"):
            hidden_output_1 = tf.nn.relu(tf.matmul(hidden_output_0, weights_1) + bias_1)
        with tf.name_scope("dropout"):
            hidden_output_1_1 = tf.nn.dropout(hidden_output_1, keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_layers_1, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(hidden_output_1_1, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
