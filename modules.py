import tensorflow as tf
import numpy as np
import pdb

def ph(shape, dtype=tf.float32, name=None):
    return tf.placeholder(dtype, shape, name=name)

def batch_norm(n_out, input_rank, phase_train, beta_trainable=True, gamma_trainable=True, scope='bn'):
    """
        n_out:       integer, depth of input maps
        input_rank:  rank of input x
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                    name='beta', trainable=beta_trainable)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                    name='gamma', trainable=gamma_trainable)
        pool_axes = np.arange(input_rank-1).tolist()
    def call(x):
        """
            Batch normalization on convolutional maps.
            Args:
                x:           Tensor, 4D BHWD input maps
        """
        with tf.variable_scope(scope):
            batch_mean, batch_var = tf.nn.moments(x, pool_axes, name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.5)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(phase_train,
                                mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)

            setattr(normed, 'batch_mean', batch_mean)
            setattr(normed, 'batch_var', batch_var)
            setattr(normed, 'mean', mean)
            setattr(normed, 'var', var)
            setattr(normed, 'call', call)
        return normed

    setattr(call, 'beta', beta)
    setattr(call, 'gamma', gamma)
    return call


def linear(n_in, n_out):
    W = tf.Variable(tf.random_normal([n_in, n_out]) * np.sqrt(2./n_in))
    b = tf.Variable(tf.zeros([n_out]))

    def call(x):
        out = tf.matmul(x, W) + b
        setattr(out, 'call', call)
        return out

    setattr(call, 'W', W)
    setattr(call, 'b', b)
    return call


def prelu(_x):
  alphas = tf.Variable(tf.zeros(_x.get_shape()[-1], dtype=tf.float32))
  # alphas = tf.get_variable('alpha', _x.get_shape()[-1],
  #                      initializer=tf.constant_initializer(0.0),
  #                      dtype=tf.float32)
  pos = tf.nn.relu(_x)
  neg = alphas * (_x - abs(_x)) * 0.5
  return pos + neg


def mlp(sizes, act_fn=tf.nn.relu, bn=None):
    linears = [linear(sizes[li], sizes[li+1]) for li in xrange(len(sizes)-1)]

    def call(x):
        activations = []
        bns = [None] * len(sizes)
        out = x
        activations.append(out)
        for li, layer in enumerate(linears[:-1]):
            out = layer(out)
            activations.append(out)
            if bn is not None:
                out = batch_norm(out.get_shape()[1], 2, bn)(out)
                bns[li] = out
            out = act_fn(out)
        out = linears[-1](out)
        activations.append(out)

        setattr(out, 'activations', activations)
        setattr(out, 'bns', bns)
        setattr(out, 'call', call)
        return out

    setattr(call, 'linears', linears)
    setattr(call, 'act_fn', act_fn)
    return call


def sample_poisson(rate):
    L = tf.exp(-tf.constant(rate))
    k = tf.constant(0.)
    p = tf.constant(1.)
    cond = lambda k, p: tf.greater(p, L)
    def inc(k, p):
        k = tf.add(k, 1.)
        p = tf.multiply(p, tf.random_uniform(()))
        return k, p
    kk, pp = tf.while_loop(cond, inc, [k, p])
    return kk - 1.


def static_size(x, d):
    out = x.get_shape()[d].value
    assert out is not None, 'shape of dim d is not static'
    return out


def tf_repeat(tensor, n):
    """behaves like np.repeat, except only runs over the leading dimension n times"""
    out = tf.reshape(
               tf.transpose(tf.tile(tf.transpose(tensor), tf.stack([n, 1]))),
               tf.stack([-1, tf.shape(tensor)[1]]))
    return out


def cartesian(reverse_order=False):
    def call(s0=None, s1=None):
        assert (s0 is not None) or (s1 is not None), 's0 and s1 cannot both be None'
        if s1 is None and s0 is None:
            return None
        elif s1 is None:
            return s0
        elif s0 is None:
            return s1
        else:
            with tf.name_scope('cartesian') as scope:
                ns0 = tf.shape(s0)[0]
                ns1 = tf.shape(s1)[0]
                if reverse_order:
                    rs0 = tf.tile(s0, tf.stack([ns1,1]))
                    rs1 = tf_repeat(s1, ns0)
                else:
                    rs0 = tf_repeat(s0, ns1)
                    rs1 = tf.tile(s1, tf.stack([ns0,1]))

                out = tf.concat([rs0, rs1], 1)
                ds0 = static_size(s0,1)
                ds1 = static_size(s1,1)
                out.set_shape((None, ds0+ds1))
            return out
    return call
