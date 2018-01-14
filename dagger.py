import numpy as np
import tensorflow as tf
from PolicyNetwork import PolicyNetwork

#dagger
class Dagger():
    def __init__(sess, state_shape, action_dim, batch_size, LR, buffer_size):

        print "Building a Policy Network."
        policy_network = PolicyNetwork(sess, state_shape, action_dim, batch_size, LR)

        #now we need to generate data to train the policy using master policy actor Network
        #loading the saved agent.
        trained_agent =

        self.buffer_size = buffer_size
