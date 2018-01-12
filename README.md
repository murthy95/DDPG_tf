## Using Tensorflow and Deep Deterministic Policy Gradient to play TORCS

This repository contains tensorflow implementation of Deep Deterministic policy gradients explained in the paper http://proceedings.mlr.press/v32/silver14.pdf. 
The Actor Critic agent is trained to drive autonomously in the TORCS environment.

For more details and documentation please refer to the blog 
https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html

# Installation Dependencies:

* Python 2.7
* Tensorflow r0.10
* [gym_torcs](https://github.com/ugo-nama-kun/gym_torcs) Pls. refer to (http://cicolink.blogspot.in/2012/10/how-to-compile-and-install-torcs-on.html) an excellent tutorial on how to install TORCS in Ubuntu.

# How to Run?

```
git clone
cd DDPG_tf
python ddpg.py 
```

(Change :the flag **train_indicator**=1 in ddpg.py if you want to train the network)
