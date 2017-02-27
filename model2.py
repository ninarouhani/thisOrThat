import tensorflow as tf
import numpy as np
rng = np.random
from tensorflow.contrib.distributions import Normal, Categorical, Bernoulli, kl
from modules import mlp, linear, prelu, batch_norm, cartesian
import pdb

N_PRODUCT = 100
N_USER = 100

D_PRODUCT = 10
D_USER = 10

D_USER_CODE = 5
D_PRODUCT_CODE = 5
d_preference = D_USER_CODE + D_PRODUCT_CODE


#### MODEL
ph = lambda shape, dtype=tf.float32, name=None: tf.placeholder(shape=shape, dtype=dtype, name=name)

product_embeddings = tf.Variable(tf.random_normal([N_PRODUCT, D_PRODUCT]))
user_embeddings = tf.Variable(tf.random_normal([N_USER, D_USER]))

i_users = ph((None), dtype=tf.int32, name='i_users')
n_per_user = ph((None), dtype=tf.int32, name='n_per_user')
i_products = ph((None), dtype=tf.int32, name='i_products')
y = ph((None), dtype=tf.bool, name='y')
n_users = tf.shape(i_users)[0]

# # STATICS FOR TESTING
# user_embeddings = tf.cast(tf.expand_dims(tf.range(20) * 100, 1), tf.float32)
# product_embeddings = tf.cast(tf.expand_dims(tf.range(20) * 10, 1), tf.float32)
# i_users = tf.constant([1, 2, 3, 0])
# n_per_user = tf.constant([3,7,4,6])
# i_products = tf.range(20)
# n_users = tf.shape(i_users)[0]

# most of the model is prepping the data for arbitrary size input
def build_user_input(ii):
    """grab the products for the ith user in i_users and concat with i_user embedding"""
    n_user_prods = n_per_user[ii]
    i_start = tf.reduce_sum(n_per_user[:ii])
    i_end = i_start + n_user_prods
    i_user_prods = i_products[i_start:i_end]
    user_prods = tf.gather(product_embeddings, i_user_prods)
    user = user_embeddings[i_users[ii]:i_users[ii]+1]

    up = tf.concat(1, [tf.tile(user, [n_user_prods, 1]), user_prods])
    up.set_shape((None, D_USER+D_PRODUCT))

    return up


def build_network_input():
    network_input = build_user_input(0)
    ii = tf.constant(1)
    not_done = lambda ii, *args: ii < n_users
    def fn(ii, up):
        return ii + 1, tf.concat(0, [up, build_user_input(ii)])
    ii, network_input = tf.while_loop(not_done, fn, [ii, network_input])
    return network_input

network_input = build_network_input()
user_input = network_input[:, :D_USER]
product_input = network_input[:, D_USER:]

# and the actual model!
user_mls = mlp([D_USER, D_USER_CODE*2], act_fn=prelu, bn=train_flag)(user_input)
user_mu, user_ls = tf.unpack(tf.reshape(user_mls, [-1, D_USER_CODE, 2]), axis=2)
product_mls = mlp([D_PRODUCT, D_PRODUCT_CODE*2], act_fn=prelu, bn=train_flag)(product_input)
product_mu, product_ls = tf.unpack(tf.reshape(product_mls, [-1, D_PRODUCT_CODE, 2]), axis=2)

user_q = Normal(user_mu, tf.nn.softplus(user_ls))
product_q = Normal(product_mu, tf.nn.softplus(product_ls))

# user_z = user_q.sample()
user_z = user_mu
# product_z = product_q.sample()
product_z = product_mu
z_concat = tf.concat(1, [user_z, product_z])

yhat0 = mlp([D_USER_CODE+D_PRODUCT_CODE,  20, 1], act_fn=prelu, bn=train_flag)(z_concat)
yhat = tf.squeeze(yhat0, [1])


#### OPTIMIZATION
user_loss = kl(user_q, Normal(0., 1.))
product_loss = kl(product_q, Normal(0., 1.))
pref_loss = tf.nn.sigmoid_cross_entropy_with_logits(yhat, tf.cast(y, tf.float32))

n_correct = tf.reduce_sum(tf.cast(tf.equal(tf.greater(tf.nn.sigmoid(yhat),0.5), y), tf.float32))
p_correct = n_correct / tf.cast(tf.shape(i_products)[0], tf.float32)

loss = tf.reduce_sum(pref_loss) +\
       tf.reduce_sum(product_loss) +\
       tf.reduce_sum(user_loss)

# for lin in yhat0.call.linears:
#     loss = loss + tf.reduce_sum(tf.square(lin.W))

trainer = tf.train.AdamOptimizer(1e-3).minimize(loss)


#### DATA
sigmoid = lambda x: 1. / (1. + np.exp(-x))
# synthetic true feature prefs
user_prod_feat_prefs = rng.randn(N_USER, D_PRODUCT)
# synthetic true product features
prod_feats = rng.randn(N_PRODUCT, D_PRODUCT)
user_prefs = np.dot(user_prod_feat_prefs, prod_feats.T)
# random binary choice proportional to pref
user_y = rng.rand(*user_prefs.shape) < sigmoid(user_prefs)

# random split into train and test
i_pool = [rng.permutation(N_PRODUCT) for _ in xrange(N_USER)]

i_product_train = [i_pool[i_user][:N_PRODUCT/2] for i_user in xrange(N_USER)]
i_product_test = [i_pool[i_user][N_PRODUCT/2:] for i_user in xrange(N_USER)]

y_train = [user_y[i_user][:N_PRODUCT/2] for i_user in xrange(N_USER)]
y_test = [user_y[i_user][N_PRODUCT/2:] for i_user in xrange(N_USER)]


#### EXPERIMENT
def prep_fd(i_users0, i_products0, y0, training):
    n_per, ip_flat, y_flat = [], [], []
    for ip, y00 in zip(i_products0, y0):
        n_per.append(len(ip))
        ip_flat += ip.tolist()
        y_flat += y00.tolist()
    return {i_users: i_users0,
            i_products: np.array(ip_flat),
            n_per_user: np.array(n_per),
            y: np.array(y_flat),
            train_flag: training}


def train(sess):
    fd = prep_fd(np.arange(N_USER), i_product_train, y_train, True)
    sess.run([trainer], feed_dict=fd)


def test(sess, i_step):
    fd = prep_fd(np.arange(N_USER), i_product_train, y_train, False)
    l0 = sess.run(loss, feed_dict=fd)
    p0 = sess.run(p_correct, feed_dict=fd)
    print 'train loss step %d: %03f' % (i_step, l0)
    print 'train acc step %d: %03f' % (i_step, p0)
    fd = prep_fd(np.arange(N_USER), i_product_test, y_test, False)
    l0 = sess.run(loss, feed_dict=fd)
    p0 = sess.run(p_correct, feed_dict=fd)
    print 'test loss step %d: %03f' % (i_step, l0)
    print 'test acc step %d: %03f' % (i_step, p0)


N_UPDATE = int(1e4)
# BS = 100
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i_step in xrange(N_UPDATE):
        if i_step % 100 == 0:
            test(sess, i_step)
        else:
            train(sess)

# a = tf.constant(0)
# aa = tf.ones([4,5])
# b = lambda a, *args: a < tf.constant(4)
# def c(a, aa):
#     return a+1, tf.concat(0, [aa, aa])
# # c = lambda a, aa: a + 1, aa + 1
# d, dd = tf.while_loop(b, c, [a, aa], shape_invariants=[tf.TensorShape(None), tf.TensorShape((None, 5))])
