import numpy as np
import tensdorflow as tf


#making a policy network to predict action probs when fed with state variables

class PolicyNetwork(object):
    def __init__(self, sess, state_shape, action_dim, batch_size, LR):
        self.sess = sess
        self.LR = LR
        self.batch_size = batch_size

        self.state = tf.placeholder(tf.float32, shape = [None, state_shape])
        self.y = tf.placeholder(tf.float32, shape = [None, action_dim])
        self.output, self.weights = self.create_policy_network(sate_shape, action_dim)
        self.loss = tf.losses.mean_squared_error(self.y, self.output)
        self.optimize = tf.AdamOptimizer(self.LR).minimize(self.loss, var_list=self.weights)
        self.sess.run(tf.global_variables_initializer())

    def train(self, states, actions):
        n_iterations = 10
        for _ in range(iterations):
            batch_indices = np.random.randint(states.shape[0], size=self.batch_size)
            batch_states = states[batch_indices]
            batch_actions = actions[batch_indices]
            self.sess.run(self.optimize, feed_dict{
            self.state : batch_states,
            self.y : batch_actions
            })

    def create_policy_network(self, state_shape, action_dim): #assuming state is just the image obtained from the observation

            conv1 = tf.layers.conv2d(self.state, 32, kernel_size=(3,3),padding='valid', activation=tf.nn.relu)
            conv2 = tf.layers.conv2d(conv1, 64, kernel_size=(3,3),padding='valid', activation=tf.nn.relu)
            conv3 = tf.layers.conv2d(conv2, 128, kernel_size=(3,3),padding='valid', activation=tf.nn.relu)
            conv3_flat = tf.contrib.layers.flatten(conv3)
            dense1 = tf.layers.dense(conv3_flat, HIDDEN1_UNITS, activation=tf.nn.relu)
            dense2 = tf.layers.dense(dense1, HIDDEN2_UNITS, activation=tf.nn.relu)

            Steering = tf.layers.dense(dense2, 1, activation=tf.nn.tanh)
            Acceleration = tf.layers.dense(dense2, 1, activation=tf.nn.sigmoid)
            Brake = tf.layers.dense(dense2, 1, activation=tf.nn.sigmoid)

            output = tf.concat([Steering,Acceleration,Brake], axis=1)

            return output, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
