import os
import itertools
import sys
import numpy as np
import tensorflow as tf
from subgraphs import qy_graph, qz_graph, px_graph, pz_graph, labeled_loss
from fncs import show_default_graph, progbar

def constant(value, dtype='float32', name=None):
    return tf.constant(value, dtype, name=name)

def placeholder(shape, dtype='float32', name=None):
    return tf.placeholder(dtype, shape, name=name)	

class GMVAE():
    def __init__(self,
                 model_folder,
                 k=10, 
                 n_x=784, 
                 n_z = 64,
                 qy_dims = [16,16],
                 qz_dims = [16,16],
                 pz_dims = [16,16],
                 px_dims = [16,16],
                 r_nent = 1, # 0.5 was good.
                 batch_size=1000, 
                 lr=0.00001):
        """Build a GMM VAE model.
        
        Args:
            k (int): Number of mixture components.
            n_x (int): Number of observable dimensions.
            n_z (int): Number of hidden dimensions.
            qy_dims (iterable of int): Iterable of hidden dimensions in qy subgraph.
            qz_dims (iterable of int): Iterable of hidden dimensions in qz subgraph.
            pz_dims (iterable of int): Iterable of hidden dimensions in pz subgraph.
            px_dims (iterable of int): Iterable of hidden dimensions in px subgraph.
            r_nent (float): A constant for weighting negative entropy term in the loss.
            batch_size (int): Number of samples in each batch.
            lr (float): Learning rate.
        """
        self.model_folder = model_folder
        self.model_path = os.path.join(model_folder, "model.ckpt")
        self.k = k
        self.n_x = n_x
        self.n_z = n_z
        self.qy_dims = qy_dims
        self.qz_dims = qz_dims
        self.pz_dims = pz_dims
        self.px_dims = px_dims
        self.r_nent = r_nent
        self.batch_size = batch_size
        self.lr = lr
        self.last_epoch = 0
        self.build()

    def build(self):
        tf.reset_default_graph()
        
        self.x = placeholder((None, self.n_x), name='x')
        self.phase = tf.placeholder(tf.bool, name='phase')

        # create a y "placeholder"
        with tf.name_scope('y_'):
            self.y_ = tf.fill(tf.stack([tf.shape(self.x)[0], self.k]), 0.0)

        # propose distribution over y
        self.qy_logit, self.qy = qy_graph(self.x, self.k, self.qy_dims, self.phase)

        # for each proposed y, infer z and reconstruct x
        self.z, self.zm, self.zv, self.zm_prior, self.zv_prior, \
        self.xm, self.xv, self.y = [[None] * self.k for i in range(8)]
        for i in range(self.k):
            with tf.name_scope(f'graphs/hot_at{i:d}'):
                y = tf.add(self.y_, constant(np.eye(self.k)[i], name=f'hot_at_{i:d}'))
                self.z[i], self.zm[i], self.zv[i] = qz_graph(self.x, y, self.n_z, self.qz_dims, self.phase)
                self.y[i], self.zm_prior[i], self.zv_prior[i] = pz_graph(y, self.n_z, self.pz_dims, self.phase)
                self.xm[i], self.xv[i] = px_graph(self.z[i], self.n_x, self.px_dims, self.phase)

        with tf.name_scope('loss'):
            with tf.name_scope('neg_entropy'):
                self.nent = -tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.qy, logits=self.qy_logit) #YB: _v2
            losses = [None] * self.k
            for i in range(self.k):
                with tf.name_scope('loss_at{:d}'.format(i)):
                    losses[i] = labeled_loss(self.k, self.x, self.xm[i], self.xv[i],
                                             self.z[i], self.zm[i], self.zv[i],
                                             self.zm_prior[i], self.zv_prior[i])
            with tf.name_scope('final_loss'):
                self.loss = tf.add_n(
                    [self.nent*self.r_nent] +
                    [self.qy[:, i] * losses[i] for i in range(self.k)])

        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        show_default_graph()

    def train(self, dataset, sess, epochs, n_train_eval=10000,
              n_test_eval=10000, save_parameters=True, is_labeled=False,
              track_losses=True, verbose=True):
        history = {"iters":[], "ent":[], "loss":[], "acc":[], 
                   "val_ent":[], "val_loss":[], "val_acc":[]}
        
        saver = tf.train.Saver()
        
        batch_size = self.batch_size
        batches = dataset.train.data.shape[0] // batch_size
        
        if track_losses:
            data_train_eval = dataset.train.data[np.random.choice(len(dataset.train.data), n_train_eval)]
            data_test_eval  = dataset.test.data [np.random.choice(len(dataset.test.data ), n_test_eval )]
        every_n_epochs = 50
        for i_epoch in range(1,epochs+1):
            a, b = np.zeros(batches*batch_size), np.zeros(batches*batch_size)
            for i_batch in range(batches):
                batch = dataset.train.next_batch(batch_size)
                a_i, b_i, _ = sess.run([self.nent, self.loss, self.train_step], feed_dict={'x:0':batch, 'phase:0':True})
                a[i_batch*batch_size:(i_batch+1)*batch_size-max(0,batch_size-len(a_i))] = a_i
                b[i_batch*batch_size:(i_batch+1)*batch_size-max(0,batch_size-len(b_i))] = b_i
                if verbose:
                    progbar(i_batch, batches)

            if track_losses and i_epoch % every_n_epochs == 1:
                c, d = sess.run([self.nent, self.loss], feed_dict={'x:0':data_test_eval, 'phase:0':False})
                a, b, c, d = -a.mean(), b.mean(), -c.mean(), d.mean()
                history['iters'].append(i_epoch)
                history['ent'].append(a)
                history['loss'].append(b)
                history['val_ent'].append(c)
                history['val_loss'].append(d)
                #history['val_acc'].append(e)
                if verbose:
                    msg = f'{"tr_ent":>10s},{"tr_loss":>10s},{"t_ent":>10s},{"t_loss":>10s},{"epoch":>10s}'
                    print(msg)
                    msg = f'{a:10.2e},{b:10.2e},{c:10.2e},{d:10.2e},{i_epoch:10d}'
                    print(msg)
                    qy = sess.run(self.qy, feed_dict={'x:0':data_test_eval[:5], 'phase:0':False})
                    print('Sample of qy')
                    print((("%.2f "*qy.shape[1])+"\n")*5 % tuple(qy.reshape(-1,)))
            elif verbose and i_epoch % every_n_epochs == 0:
                print(f"epoch: {i_epoch:10d}")
            
            self.last_epoch = i_epoch

        # Saves parameters every epochs
        saver.save(sess, self.model_path, global_step=i_epoch)
            

        qy = sess.run(self.qy, feed_dict={'x:0':dataset.test.data, 'phase:0':False})
        zm = sess.run(self.zm, feed_dict={'x:0':dataset.test.data, 'phase:0':False})

        return history, qy, zm
    
    def encode_y(self, data):
        # tf.reset_default_graph()
        saver = tf.train.Saver()
        with tf.Session() as sess:
          saver.restore(sess, self.model_path+f"-{self.last_epoch}")
          return self.qy.eval(feed_dict={'x:0': data, 'phase:0': False})
    
    def encode_zs(self, data):
        # tf.reset_default_graph()
        saver = tf.train.Saver()
        with tf.Session() as sess:
          saver.restore(sess, self.model_path+f"-{self.last_epoch}")
          return [self.z[i].eval(feed_dict={'x:0': data, 'phase:0': False}) for i in range(len(self.z))]
    
    def encode_zm(self, data):
        # tf.reset_default_graph()
        saver = tf.train.Saver()
        with tf.Session() as sess:
          saver.restore(sess, self.model_path+f"-{self.last_epoch}")
          return [self.zm[i].eval(feed_dict={'x:0': data, 'phase:0': False}) for i in range(len(self.z))]
    
    def encode_z(self, data):
        zs = self.encode_zs(data)
        ys = self.encode_y(data)
        z = np.zeros(zs[0].shape)
        for z_i in range(zs[0].shape[1]):
            for y_i in range(ys.shape[1]):
                z[:, z_i] += zs[y_i][:,z_i]*ys[:,y_i]
        return z

    def reconstruct_xs(self, data):
        # tf.reset_default_graph()
        saver = tf.train.Saver()
        with tf.Session() as sess:
          saver.restore(sess, self.model_path+f"-{self.last_epoch}")
          return [self.xm[i].eval(feed_dict={'x:0': data, 'phase:0': False}) for i in range(len(self.xm))]
    
    def reconstruct(self, data):
        xs = self.reconstruct_xs(data)
        ys = self.encode_y(data)
        x = np.zeros(xs[0].shape)
        for x_i in range(xs[0].shape[1]):
            for y_i in range(ys.shape[1]):
                x[:, x_i] += xs[y_i][:,x_i]*ys[:,y_i]
        return x
