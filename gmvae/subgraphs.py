import numpy as np
import tensorflow as tf

from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers import batch_norm
from tensorflow import initializers

def gaussian_sample(mean, var, scope=None):
    with tf.variable_scope(scope, 'gaussian_sample'):
        sample = tf.random_normal(tf.shape(mean), mean, tf.sqrt(var))
        sample.set_shape(mean.get_shape())
        return sample
		
def log_normal(x, mu, var, eps=0.0, axis=-1):
    if eps > 0.0:
        var = tf.add(var, eps, name='clipped_var')
    return -0.5 * tf.reduce_sum(
        tf.log(2 * np.pi) + tf.log(var) + tf.square(x - mu) / var, axis)
		
# n_x is number input features
# n_z is dim of latent space
# k is number of clusters/categories in gaussian mix

n_h=16
use_batch_norm = False

# vae subgraphs
def qy_graph(x, k, hidden_dims=[16, 16], phase=True):
    """q(y|x) computation subgraph generator function.
    
    Args:
        x (tf.Tensor): x tensor.
        k (int): Number of mixtures in the distribution.
        hidden_dims (iterable of int): Hidden layer dimensions as an iterable.
        phase (bool): True if in training phase, False otherwise.
    """
    reuse = len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='qy')) > 0
    hs = []
    with tf.variable_scope('qy'):
        # Add hidden layers
        hs.append(fully_connected(x, hidden_dims[0], scope='layer1', activation_fn=tf.nn.relu, reuse=reuse, weights_initializer=None))
        if use_batch_norm:
            hs[-1] = batch_norm(hs[-1], center=True, scale=True, is_training=phase, reuse=reuse, scope='bn1')
        for i in range(1, len(hidden_dims)):
            hs.append(fully_connected(hs[-1], hidden_dims[i], scope=f'layer{i+1}', activation_fn=tf.nn.relu, reuse=reuse, weights_initializer=None))
            if use_batch_norm:
                hs[-1] = batch_norm(hs[-1], center=True, scale=True, is_training=phase, reuse=reuse, scope=f'bn{i+1}')
        
        # Construct q(y|x) logits and softmax.
        qy_logit = fully_connected(hs[-1], k, scope='logit', activation_fn=tf.nn.relu, reuse=reuse, weights_initializer=None)
        qy = tf.nn.softmax(qy_logit, name='prob')
    return qy_logit, qy

def qz_graph(x, y, n_z, hidden_dims=[16,16], phase=True):
    """q(z|x,y) computation subgraph generator function.
    
    Args:
        x (tf.Tensor): x tensor.
        y (tf.Tensor): y tensor.
        n_z (int): Number of hidden dimensions.
        hidden_dims (iterable of int): Hidden layer dimensions as an iterable.
        phase (bool): True if training phase, False otherwise.
    """
    reuse = len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='qz')) > 0
    hs = []
    with tf.variable_scope('qz'):
        # Initial y transformation.
        h0 = fully_connected(y, int(y.get_shape()[-1]), scope='layer0', activation_fn=None, reuse=reuse, weights_initializer=None)
        # Concatenate x and transformed y.
        xy = tf.concat((x, h0), 1, name='xy/concat')
        # Add hidden layers.
        hs.append(fully_connected(xy, hidden_dims[0], scope='layer1', activation_fn=tf.nn.relu, reuse=reuse, weights_initializer=None))
        if use_batch_norm:
            hs[-1] = batch_norm(hs[-1], center=True, scale=True, is_training=phase, reuse=reuse, scope='bn1')
        for i in range(1, len(hidden_dims)):
            hs.append(fully_connected(hs[-1], hidden_dims[i], scope=f'layer{i+1}', activation_fn=tf.nn.relu, reuse=reuse, weights_initializer=None))
            if use_batch_norm:
                hs[-1] = batch_norm(hs[-1], center=True, scale=True, is_training=phase, reuse=reuse, scope=f'bn{i+1}')
        
        zm = fully_connected(hs[-1], n_z, scope='zm', activation_fn=None, reuse=reuse, weights_initializer=None)
        zv = fully_connected(hs[-1], n_z, scope='zv', activation_fn=tf.nn.softplus, reuse=reuse, weights_initializer=None)+1e-5
        z = z_graph(zm,zv)
    return z, zm, zv

def z_graph(zm,zv):
    """p(z) is computed here."""
    with tf.variable_scope('z'):
        z = gaussian_sample(zm, zv, 'z')
        # Used to feed into z when sampling
        z = tf.identity(z, name='z_sample')
    return z

def pz_graph(y, n_z, hidden_dims=[16], phase=True):
    """p(z|y) is computed here."""
    reuse = len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pz')) > 0
    hs = []
    with tf.variable_scope('pz'):
        hs.append(fully_connected(y, hidden_dims[0], scope='layer1', activation_fn=tf.nn.relu, reuse=reuse, weights_initializer=None))   
        if use_batch_norm:
            hs[-1] = batch_norm(hs[-1], center=True, scale=True, is_training=phase, reuse=reuse, scope='bn1')
        for i in range(1, len(hidden_dims)):
            hs.append(fully_connected(hs[-1], hidden_dims[i], scope=f'layer{i+1}', activation_fn=tf.nn.relu, reuse=reuse, weights_initializer=None))
            if use_batch_norm:
                hs[-1] = batch_norm(hs[-1], center=True, scale=True, is_training=phase, reuse=reuse, scope=f'bn{i+1}')
        
        zm = fully_connected(hs[-1], n_z, scope='zm', activation_fn=None, reuse=reuse, weights_initializer=None)
        zv = fully_connected(hs[-1], n_z, scope='zv', activation_fn=tf.nn.softplus, reuse=reuse, weights_initializer=None)+1e-5
    return y, zm, zv

def px_fixed_graph(z, n_x):
    """p(x|z) is computed here."""
    reuse = len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='px_fixed')) > 0
    with tf.variable_scope('px_fixed'):
        h = fully_connected(z, n_h, scope='layer1', activation_fn=tf.nn.relu, reuse=reuse, weights_initializer=None)
        px_logit = fully_connected(h, n_x, scope='output', activation_fn=None, reuse=reuse, weights_initializer=None)
        #px_logit = tf.identity(px_logit,name='x')
    return px_logit

def px_graph(z, n_x, hidden_dims=[16,16], phase=True):
    """p(x|z) is computed here."""
    reuse = len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='px')) > 0
    hs = []
    with tf.variable_scope('px'):
        hs.append(fully_connected(z, hidden_dims[0], scope='layer1', activation_fn=tf.nn.relu, reuse=reuse, weights_initializer=None))
        if use_batch_norm:
            hs[-1] = batch_norm(hs[-1], center=True, scale=True, is_training=phase, reuse=reuse, scope='bn1')
        for i in range(1, len(hidden_dims)):
            hs.append(fully_connected(hs[-1], hidden_dims[i], scope=f'layer{i+1}', activation_fn=tf.nn.relu, reuse=reuse, weights_initializer=None))
            if use_batch_norm:
                hs[-1] = batch_norm(hs[-1], center=True, scale=True, is_training=phase, reuse=reuse, scope=f'bn{i+1}')
        
        xm = fully_connected(hs[-1], n_x, scope='xm', activation_fn=None, reuse=reuse, weights_initializer=None)
        xv = fully_connected(hs[-1], n_x, scope='xv', activation_fn=tf.nn.softplus, reuse=reuse, weights_initializer=None)+1e-5
        #px_logit = tf.identity(px_logit,name='x')
    return xm, xv

def labeled_loss(k, x, xm, xv, z, zm, zv, zm_prior, zv_prior):
    """Variational loss for the mixture VAE given for each given q(y=i|x, z), hence the
        name labeled_loss."""
    return -log_normal(x, xm, xv) + log_normal(z, zm, zv) - log_normal(z, zm_prior, zv_prior) - np.log(1/k) 
