import numpy as np
import tensorflow as tf

HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600

class ActorNetwork(object):
    def __init__(self, sess, state_shape, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        self.state = tf.placeholder(tf.float32, shape = [None]+ state_shape)

        #Now create the model
        # self.model , self.weights, self.state = self.create_actor_network(state_size, action_size)
        self.output, self.weights = self.create_actor_network(state_shape, action_size)
        self.target_output, self.target_weights = self.create_actor_network(state_shape, action_size)
        # self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size, action_size)
        self.action_gradient = tf.placeholder(tf.float32,[None,action_size])
        self.params_grad = tf.gradients(self.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        self.sess.run(tf.global_variables_initializer())


    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    def predict(self, states):
        self.sess.run(self.output, feed_dict={
            self.state: states
        })

    def target_predict(self, states):
        self.sess.run(self.target_output, feed_dict={
            self.state: states
        })

    def target_train(self):
        actor_weights = self.sess.run(self.weights)
        actor_target_weights = self.sess.run(self.target_weights)
        for i in xrange(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        self.sess.run(tf.assign(self.target_weights,actor_target_weights))

    def create_actor_network(self, state_shape, action_dim): #assuming state is just the image obtained from the observation

        conv1 = tf.layers.conv2d(self.state, 32, kernel_size=(3,3),padding='valid', activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(conv1, 64, kernel_size=(3,3),padding='valid', activation=tf.nn.relu)
        conv3 = tf.layers.conv2d(conv2, 128, kernel_size=(3,3),padding='valid', activation=tf.nn.relu)
        conv3_flat = tf.layers.flatten(conv3)
        dense1 = tf.layers.dense(conv3_flat, HIDDEN1_UNITS, activation=tf.nn.relu)
        dense2 = tf.layers.dense(dense1, HIDDEN2_UNITS, activation=tf.nn.relu)

        Steering = tf.layers.dense(dense2, 1, activation=tf.nn.tanh)
        Acceleration = tf.layers.dense(dense2, 1, activation=tf.nn.sigmoid)
        Brake = tf.layers.dense(dense2, 1, activation=tf.nn.sigmoid)

        output = tf.concat([Steering,Acceleration,Brake], axis=1)

        return output, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

# if __name__ == "__main__":
#     x_img = np.random.rand(16*16*3).reshape([1,16,16,3])
#     # print x_img
#     y_grads = np.random.rand(3).reshape([1,3])
#     # print y_grads
#     with tf.Session() as sess:
#         actor = ActorNetwork(sess, [16,16,3], 3, 1, 0.01, 0.001)
#         actor.train(x_img, y_grads)
