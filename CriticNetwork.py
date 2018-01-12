import numpy as np
import tensorflow as tf

HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600

class CriticNetwork(object):
    def __init__(self, sess, state_shape, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size

        #Now create the model
        self.state = tf.placeholder(tf.float32, shape=[None]+state_shape)
        self.action = tf.placeholder(tf.float32, shape=[None]+action_size)
        self.y = tf.placeholder(tf.float32, shape=[None, 1])

        self.output, self.weights = self.create_critic_network(state_shape, action_size)
        self.target_output, self.target_weights = self.create_critic_network(state_size, action_size)
        self.action_grads = tf.gradients(self.output, self.action)  #GRADIENTS for policy update
        self.sess.run(tf.global_variables_initializer())

        self.loss = tf.losses.mean_squared_error(self.y, self.output)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)


    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

    def predict(self, states, actions):
        self.sess.run(self.output, feed_dict={
            self.state: states,
            self.action:actions
        })

    def target_predict(self, states, actions):
        self.sess.run(self.target_output, feed_dict={
            self.state: states,
            self.action:actions
        })
    def train(self, states, actions, y):
        n_iterations = 10
        for _ in range(n_iterations):
            self.sess.run(self.optimize, feed_dict={
                self.state: states,
                self.action:actions
            })

    def target_train(self):
        critic_weights = self.sess.run(self.weights)
        critic_target_weights = self.sess.run(self.target_weights)
        for i in xrange(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU)* critic_target_weights[i]
        self.sess.run(tf.assign(self.target_weights,critic_target_weights))

    def create_critic_network(self, state_shape, action_dim): #assuming state is just the image obtained from the observation
        print("Now we build the critic model")
        conv1 = tf.layers.conv2d(self.state, 32, kernel_size=(3,3),padding='valid', activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(conv1, 64, kernel_size=(3,3),padding='valid', activation=tf.nn.relu)
        conv3 = tf.layers.conv2d(conv2, 128, kernel_size=(3,3),padding='valid', activation=tf.nn.relu)
        conv3_flat = tf.layers.flatten(conv3)

        dense1 = tf.layers.dense(conv3_flat, HIDDEN1_UNITS, activation=tf.nn.relu)

        a_dense1 = tf.layers.dense(self.action, HIDDEN1_UNITS, activation=tf.nn.relu)
        new_concat = tf.layers.flatten(tf.concat([dense1, a_dense1], axis=1))
        dense2 = tf.layers.dense(new_concat, HIDDEN2_UNITS, activation=tf.nn.relu)
        output = tf.layers.dense(dense2, 1)

        return output, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
